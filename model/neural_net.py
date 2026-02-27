"""
Neural Network Architecture for Predictive Litigation Analytics.

Embedding-only architecture:
1. Text embeddings (3 sections × 3072-dim) via per-section encoders
   - klaegervorbringen, beklagtenvorbringen, aufgenommene_beweise
2. Flat fusion MLP (one hidden layer) for final classification

Output: 3-class (Unterliegen / Teilweise / Obsiegen)
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import EMBEDDING_DIM, EMBEDDING_SECTIONS, NN_CONFIG


class GaussianNoise(nn.Module):
    """
    Training-time Gaussian noise for embedding regularization.

    Acts as a stochastic data-augmentation layer on the raw embedding vectors:
    adds ε ~ N(0, std²) per-element during training, identity during eval.
    Even small std (0.01–0.03) over 3072-dim inputs provides strong implicit
    regularization without distorting the embedding geometry.
    """

    def __init__(self, std: float = 0.0):
        super().__init__()
        self.std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.std > 0:
            return x + torch.randn_like(x) * self.std
        return x


class EmbeddingEncoder(nn.Module):
    """
    Per-section embedding encoder.

    hidden_dim > 0 → two-layer MLP (original behavior, good for small datasets).
    hidden_dim = 0 → single linear projection directly to output_dim.
                     Halves the dominant parameter count (3072 × hidden vs
                     3072 × output), which drastically reduces overfitting for
                     larger datasets where the pretrained embeddings are already
                     rich enough.

    noise_std > 0 → Gaussian noise injected at input (training only).
    """

    def __init__(
        self,
        input_dim: int = EMBEDDING_DIM,
        hidden_dim: int = NN_CONFIG["embedding_hidden_dim"],
        output_dim: int = NN_CONFIG["embedding_output_dim"],
        dropout: float = NN_CONFIG["dropout_embedding"],
        noise_std: float = 0.0,
    ):
        super().__init__()
        layers = []

        # Optional input noise (training only)
        if noise_std > 0:
            layers.append(GaussianNoise(noise_std))

        if hidden_dim > 0:
            # Two-layer MLP: input → hidden → output
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
                nn.Dropout(dropout * 0.7),
            ])
        else:
            # Single-layer direct projection: input → output
            # ~50 % fewer parameters than the two-layer version at 256 hidden.
            layers.extend([
                nn.Linear(input_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])

        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class LitigationClassifier(nn.Module):
    """
    Main classification model for Austrian civil case outcome prediction.

    Architecture (optimiert für kleine Datensätze, 50–300 Fälle):
    - 3 EmbeddingEncoders (je ein Encoder pro Textabschnitt):
        klaegervorbringen, beklagtenvorbringen, aufgenommene_beweise
    - Flaches Fusion-MLP (ein Hidden Layer)
    - 3-class output (0=Unterliegen, 1=Teilweise, 2=Obsiegen)

    Rein embedding-basiert: keine strukturierten Merkmale (Streitwert,
    Anspruchsart etc.) — Training und Inference verwenden identische Inputs.
    """

    def __init__(self, config: dict = NN_CONFIG, **kwargs):
        super().__init__()

        self.config = config
        n_sections = len(EMBEDDING_SECTIONS)
        emb_output_dim = config["embedding_output_dim"]
        fusion_dims = config["fusion_dims"]
        num_classes = config["num_classes"]
        dropout_fusion = config["dropout_fusion"]

        # Per-section embedding encoders
        _noise_std = config.get("embedding_noise_std", 0.0)
        self.embedding_encoders = nn.ModuleList([
            EmbeddingEncoder(
                input_dim=EMBEDDING_DIM,
                hidden_dim=config["embedding_hidden_dim"],
                output_dim=emb_output_dim,
                dropout=config["dropout_embedding"],
                noise_std=_noise_std,
            )
            for _ in range(n_sections)
        ])

        # Fusion network: concatenation of all section encodings
        fusion_input_dim = n_sections * emb_output_dim
        layers = []
        prev_dim = fusion_input_dim
        for dim in fusion_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.GELU(),
                nn.Dropout(dropout_fusion),
            ])
            prev_dim = dim

        self.fusion = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        embeddings: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            embeddings: list of (batch, EMBEDDING_DIM) tensors, one per section

        Returns:
            logits: (batch, 3)
            probs: (batch, 3) — softmax probabilities
        """
        # Encode each text section
        encoded_sections = [
            encoder(emb)
            for encoder, emb in zip(self.embedding_encoders, embeddings)
        ]  # List of (batch, emb_output_dim)

        # Fuse all section encodings
        fused = torch.cat(encoded_sections, dim=-1)  # (batch, n*emb_out)
        fused = self.fusion(fused)

        # Classify
        logits = self.classifier(fused)
        probs = F.softmax(logits, dim=-1)

        return logits, probs

    def predict_proba(
        self,
        embeddings: list[torch.Tensor],
    ) -> torch.Tensor:
        """Convenience method returning probabilities only."""
        with torch.no_grad():
            _, probs = self.forward(embeddings)
        return probs

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_architecture_summary(self) -> dict:
        return {
            "total_parameters": self.count_parameters(),
            "embedding_sections": len(EMBEDDING_SECTIONS),
            "embedding_dim_input": EMBEDDING_DIM,
            "embedding_dim_output": self.config["embedding_output_dim"],
            "fusion_dims": self.config["fusion_dims"],
            "num_classes": self.config["num_classes"],
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
