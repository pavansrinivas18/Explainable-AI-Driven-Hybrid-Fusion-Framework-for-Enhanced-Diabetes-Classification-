"""
FT-Transformer: Feature Tokenizer Transformer for Tabular Data.
Based on Gorishniy et al., NeurIPS 2021.
Transforms numerical clinical biomarkers into token embeddings,
applies Multi-Head Self-Attention to capture complex pairwise cross-feature dependencies,
and classifies disease risk using the [CLS] token representation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from src.config import DEVICE, FT_TRANSFORMER_PARAMS, RANDOM_SEED


class FeatureTokenizer(nn.Module):
    """
    Transforms continuous numerical features into d-dimensional embeddings.
    e_i = x_i * W_i + b_i
    """
    def __init__(self, n_features: int, d_token: int):
        super().__init__()
        self.n_features = n_features
        self.d_token = d_token

        # Weight and bias vectors for each feature
        self.weights = nn.Parameter(torch.randn(n_features, d_token) * (1.0 / np.sqrt(d_token)))
        self.biases = nn.Parameter(torch.zeros(n_features, d_token))

    def forward(self, x):
        # x shape: (batch_size, n_features)
        # Output shape: (batch_size, n_features, d_token)
        tokens = x.unsqueeze(-1) * self.weights.unsqueeze(0) + self.biases.unsqueeze(0)
        return tokens


class TransformerBlock(nn.Module):
    """
    Standard Pre-LN Transformer encoder block with Multi-Head Self-Attention and FFN.
    """
    def __init__(self, d_token: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_token)
        self.mha = nn.MultiheadAttention(
            embed_dim=d_token,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.ln2 = nn.LayerNorm(d_token)
        self.ffn = nn.Sequential(
            nn.Linear(d_token, d_token * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_token * 2, d_token),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Pre-LN Self-Attention
        norm_x = self.ln1(x)
        attn_out, attn_weights = self.mha(norm_x, norm_x, norm_x)
        x = x + attn_out

        # Pre-LN Feed-Forward
        x = x + self.ffn(self.ln2(x))
        return x, attn_weights


class FT_Transformer_Module(nn.Module):
    """
    Full FT-Transformer network.
    """
    def __init__(self, n_features: int, d_token: int = 64, n_blocks: int = 3,
                 n_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, d_token)

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_token, n_heads, dropout) for _ in range(n_blocks)
        ])

        # Final LayerNorm and Head
        self.final_ln = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, 2)

    def forward(self, x):
        batch_size = x.shape[0]

        # Tokenize features: (batch_size, n_features, d_token)
        feat_tokens = self.tokenizer(x)

        # Prepend [CLS] token: (batch_size, n_features + 1, d_token)
        cls_expanded = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_expanded, feat_tokens], dim=1)

        # Pass through Transformer encoder layers
        for block in self.blocks:
            tokens, last_attn_weights = block(tokens)

        # Extract [CLS] representation (index 0)
        cls_rep = self.final_ln(tokens[:, 0, :])
        logits = self.head(cls_rep)
        return logits, last_attn_weights


class FT_Transformer_Classifier:
    """
    Scikit-learn compatible wrapper for FT-Transformer.
    """
    def __init__(self, n_features: int, d_token=None, n_blocks=None,
                 n_heads=None, dropout=None, lr=None, epochs=None,
                 batch_size=None, device=DEVICE):
        self.n_features = n_features
        self.d_token = d_token or FT_TRANSFORMER_PARAMS["d_token"]
        self.n_blocks = n_blocks or FT_TRANSFORMER_PARAMS["n_blocks"]
        self.n_heads = n_heads or FT_TRANSFORMER_PARAMS["n_heads"]
        self.dropout = dropout if dropout is not None else FT_TRANSFORMER_PARAMS["dropout"]
        self.lr = lr or FT_TRANSFORMER_PARAMS["lr"]
        self.epochs = epochs or FT_TRANSFORMER_PARAMS["epochs"]
        self.batch_size = batch_size or FT_TRANSFORMER_PARAMS["batch_size"]
        self.device = device
        self.model = None

    def fit(self, X, y):
        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

        X_tensor = torch.tensor(np.array(X), dtype=torch.float32)
        y_tensor = torch.tensor(np.array(y), dtype=torch.long)

        # Class weights for clinical sensitivity
        num_pos = (y_tensor == 1).sum().item()
        num_neg = (y_tensor == 0).sum().item()
        total = len(y_tensor)
        weight_0 = total / (2.0 * max(num_neg, 1))
        weight_1 = total / (2.0 * max(num_pos, 1))
        class_weights = torch.tensor([weight_0, weight_1], dtype=torch.float32).to(self.device)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = FT_Transformer_Module(
            n_features=self.n_features,
            d_token=self.d_token,
            n_blocks=self.n_blocks,
            n_heads=self.n_heads,
            dropout=self.dropout
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=FT_TRANSFORMER_PARAMS["weight_decay"]
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        self.model.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                logits, _ = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
            scheduler.step()

        return self

    def predict_proba(self, X):
        self.model.eval()
        X_tensor = torch.tensor(np.array(X), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits, _ = self.model(X_tensor)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
        return probs

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return (probs[:, 1] >= threshold).astype(int)
