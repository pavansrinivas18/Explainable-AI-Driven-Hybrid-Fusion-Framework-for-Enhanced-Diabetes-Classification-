"""
KCCAM-DNN: Kendall's Correlation Coefficient Attention Mechanism in Deep Neural Network.
Implements a novel neural architecture that uses statistical rank correlation (Kendall's tau)
as an inductive prior to guide dynamic feature attention for clinical disease diagnosis.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from src.config import DEVICE, KCCAM_PARAMS, RANDOM_SEED


class FeatureAttention(nn.Module):
    """
    Kendall's Correlation-Guided Feature Attention Layer.
    Combines learned instance-level attention weights with the global Kendall's Tau vector:
    alpha = Softmax( W_a * x + b_a + lambda * |tau| )
    """
    def __init__(self, in_features: int, kendall_tau: np.ndarray, lambda_k: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.lambda_k = lambda_k

        # Register Kendall Tau as a non-trainable inductive prior buffer
        tau_tensor = torch.tensor(kendall_tau, dtype=torch.float32).abs().unsqueeze(0)
        self.register_buffer("kendall_prior", tau_tensor)

        # Learnable attention projection
        self.attn_proj = nn.Linear(in_features, in_features)
        self.temperature = nn.Parameter(torch.ones(1) * np.sqrt(in_features))

    def forward(self, x):
        # x shape: (batch_size, in_features)
        raw_scores = self.attn_proj(x)
        # Fuse learned score with Kendall rank correlation prior
        guided_scores = (raw_scores + self.lambda_k * self.kendall_prior) / self.temperature
        attn_weights = F.softmax(guided_scores, dim=-1)
        # Apply attention gating: attended features
        attended_x = x * attn_weights
        return attended_x, attn_weights


class KCCAM_DNN_Module(nn.Module):
    """
    Deep Neural Network backbone with residual skip-connections, LayerNorm, and GELU.
    """
    def __init__(self, in_features: int, kendall_tau: np.ndarray,
                 hidden_dims=[128, 64, 32], dropout_rate=0.25, lambda_k=0.5):
        super().__init__()
        self.attention = FeatureAttention(in_features, kendall_tau, lambda_k=lambda_k)

        # First dense projection
        self.fc1 = nn.Linear(in_features, hidden_dims[0])
        self.ln1 = nn.LayerNorm(hidden_dims[0])
        self.drop1 = nn.Dropout(dropout_rate)

        # Residual Block
        self.res_fc1 = nn.Linear(hidden_dims[0], hidden_dims[0])
        self.res_ln1 = nn.LayerNorm(hidden_dims[0])
        self.res_fc2 = nn.Linear(hidden_dims[0], hidden_dims[0])
        self.res_ln2 = nn.LayerNorm(hidden_dims[0])
        self.res_drop = nn.Dropout(dropout_rate)

        # Transition layers
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.ln2 = nn.LayerNorm(hidden_dims[1])
        self.drop2 = nn.Dropout(dropout_rate * 0.8)

        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.ln3 = nn.LayerNorm(hidden_dims[2])

        # Binary classification output head
        self.classifier = nn.Linear(hidden_dims[2], 2)

    def forward(self, x):
        # Feature Attention
        attended, weights = self.attention(x)

        # Layer 1
        h = self.drop1(F.gelu(self.ln1(self.fc1(attended))))

        # Residual Skip Block
        res = h
        h = F.gelu(self.res_ln1(self.res_fc1(h)))
        h = self.res_drop(self.res_ln2(self.res_fc2(h)))
        h = F.gelu(h + res)  # Skip connection

        # Layer 2 & 3
        h = self.drop2(F.gelu(self.ln2(self.fc2(h))))
        h = F.gelu(self.ln3(self.fc3(h)))

        # Logits
        logits = self.classifier(h)
        return logits, weights


class KCCAM_DNN_Classifier:
    """
    Scikit-learn compatible wrapper for KCCAM-DNN.
    Supports fit, predict, and predict_proba.
    """
    def __init__(self, in_features: int, kendall_tau: np.ndarray,
                 hidden_dims=None, dropout_rate=None, lr=None,
                 epochs=None, batch_size=None, lambda_k=None, device=DEVICE):
        self.in_features = in_features
        self.kendall_tau = kendall_tau
        self.hidden_dims = hidden_dims or KCCAM_PARAMS["hidden_dims"]
        self.dropout_rate = dropout_rate if dropout_rate is not None else KCCAM_PARAMS["dropout_rate"]
        self.lr = lr or KCCAM_PARAMS["lr"]
        self.epochs = epochs or KCCAM_PARAMS["epochs"]
        self.batch_size = batch_size or KCCAM_PARAMS["batch_size"]
        self.lambda_k = lambda_k or KCCAM_PARAMS["lambda_kendall"]
        self.device = device
        self.model = None

    def fit(self, X, y):
        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

        X_tensor = torch.tensor(np.array(X), dtype=torch.float32)
        y_tensor = torch.tensor(np.array(y), dtype=torch.long)

        # Compute cost-sensitive class weights to protect clinical sensitivity
        num_pos = (y_tensor == 1).sum().item()
        num_neg = (y_tensor == 0).sum().item()
        total = len(y_tensor)
        weight_0 = total / (2.0 * max(num_neg, 1))
        weight_1 = total / (2.0 * max(num_pos, 1))
        class_weights = torch.tensor([weight_0, weight_1], dtype=torch.float32).to(self.device)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = KCCAM_DNN_Module(
            in_features=self.in_features,
            kendall_tau=self.kendall_tau,
            hidden_dims=self.hidden_dims,
            dropout_rate=self.dropout_rate,
            lambda_k=self.lambda_k
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=KCCAM_PARAMS["weight_decay"]
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

    def get_attention_weights(self, X):
        """
        Extracts sample-wise feature attention weights for XAI validation.
        """
        self.model.eval()
        X_tensor = torch.tensor(np.array(X), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            _, weights = self.model(X_tensor)
        return weights.cpu().numpy()
