"""
Neural Network Architecture for Predictive Litigation Analytics.

v2.0 — Structured data only, no embedding encoders.

Single-input architecture using structured legal features (~77-dim):
1. StructuredEncoder: multi-layer MLP to encode legal features
2. Fusion MLP for final classification
3. Output: 2-class binary (Unterliegen / Obsiegen)

Much more compact than the embedding-based architecture (~28K vs ~2.56M params).
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import NN_CONFIG


class LitigationClassifier(nn.Module):
    """
    Classification model for Austrian civil case outcome prediction.

    Architecture (optimiert für strukturierte Daten, 50–10.000 Fälle):
    - StructuredEncoder: Multi-layer MLP encoding legal features
    - Fusion layer(s) before classification
    - 2-class binary output (0=Unterliegen, 1=Obsiegen)

    Gesamtparameter: ~28 K (bei Standardkonfiguration mit ~77 Input-Features)
    """

    def __init__(self, input_dim: int, config: dict = NN_CONFIG):
        super().__init__()

        self.config = config
        hidden_dims = config["hidden_dims"]
        fusion_dims = config["fusion_dims"]
        num_classes = config["num_classes"]
        dropout = config["dropout"]

        # Structured feature encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Fusion / classification head
        fusion_layers = []
        for dim in fusion_dims:
            fusion_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = dim

        self.fusion = nn.Sequential(*fusion_layers)
        self.classifier = nn.Linear(prev_dim, num_classes)

        # Store input_dim for checkpoint compatibility
        self.input_dim = input_dim

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, structured: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            structured: (batch, input_dim) tensor of encoded features

        Returns:
            logits: (batch, num_classes)
            probs: (batch, num_classes) — softmax probabilities
        """
        encoded = self.encoder(structured)
        fused = self.fusion(encoded)
        logits = self.classifier(fused)
        probs = F.softmax(logits, dim=-1)
        return logits, probs

    def predict_proba(self, structured: torch.Tensor) -> torch.Tensor:
        """Convenience method returning probabilities only."""
        with torch.no_grad():
            _, probs = self.forward(structured)
        return probs

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_architecture_summary(self) -> dict:
        return {
            "total_parameters": self.count_parameters(),
            "input_dim": self.input_dim,
            "hidden_dims": self.config["hidden_dims"],
            "fusion_dims": self.config["fusion_dims"],
            "num_classes": self.config["num_classes"],
            "dropout": self.config["dropout"],
        }


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in litigation outcomes.
    Focuses training on hard examples.
    """

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()
