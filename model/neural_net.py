"""
Neural Network Architecture for Predictive Litigation Analytics.

Multi-input architecture combining:
1. Text embeddings (5 sections × 3072-dim) via per-section encoders
2. Structured legal features via a compact encoder
3. Fusion network for final classification

Output: 3-class (Unterliegen / Teilweise / Obsiegen)
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import EMBEDDING_DIM, EMBEDDING_SECTIONS, NN_CONFIG


class EmbeddingEncoder(nn.Module):
    """
    Per-section embedding encoder.
    Reduces 3072-dim embedding to compact representation.
    Uses LayerNorm + Dropout for regularization.
    """

    def __init__(
        self,
        input_dim: int = EMBEDDING_DIM,
        hidden_dim: int = NN_CONFIG["embedding_hidden_dim"],
        output_dim: int = NN_CONFIG["embedding_output_dim"],
        dropout: float = NN_CONFIG["dropout_embedding"],
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.7),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class StructuredEncoder(nn.Module):
    """
    Encodes structured legal features (streitwert, claim type, defenses, etc.)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = NN_CONFIG["structured_hidden_dim"],
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class LitigationClassifier(nn.Module):
    """
    Main classification model for Austrian civil case outcome prediction.

    Architecture:
    - 5 EmbeddingEncoders (one per text section)
    - 1 StructuredEncoder
    - Fusion MLP
    - 3-class output (0=loss, 1=partial, 2=win)
    """

    def __init__(self, structured_dim: int, config: dict = NN_CONFIG):
        super().__init__()

        self.config = config
        n_sections = len(EMBEDDING_SECTIONS)
        emb_output_dim = config["embedding_output_dim"]
        struct_hidden_dim = config["structured_hidden_dim"]
        fusion_dims = config["fusion_dims"]
        num_classes = config["num_classes"]
        dropout_fusion = config["dropout_fusion"]

        # Per-section embedding encoders
        self.embedding_encoders = nn.ModuleList([
            EmbeddingEncoder(
                input_dim=EMBEDDING_DIM,
                hidden_dim=config["embedding_hidden_dim"],
                output_dim=emb_output_dim,
                dropout=config["dropout_embedding"],
            )
            for _ in range(n_sections)
        ])

        # Structured feature encoder
        self.structured_encoder = StructuredEncoder(
            input_dim=structured_dim,
            hidden_dim=struct_hidden_dim,
        )

        # Attention over sections (learn which sections matter most)
        self.section_attention = nn.Sequential(
            nn.Linear(emb_output_dim, 1),
            nn.Softmax(dim=0),
        )

        # Fusion network
        fusion_input_dim = n_sections * emb_output_dim + struct_hidden_dim
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
        structured: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            embeddings: list of (batch, EMBEDDING_DIM) tensors, one per section
            structured: (batch, structured_dim) tensor

        Returns:
            logits: (batch, 3)
            probs: (batch, 3) — softmax probabilities
        """
        # Encode each text section
        encoded_sections = [
            encoder(emb)
            for encoder, emb in zip(self.embedding_encoders, embeddings)
        ]  # List of (batch, emb_output_dim)

        # Concatenate all section encodings
        section_concat = torch.cat(encoded_sections, dim=-1)  # (batch, n*emb_out)

        # Encode structured features
        struct_encoded = self.structured_encoder(structured)  # (batch, struct_hidden)

        # Fuse
        fused = torch.cat([section_concat, struct_encoded], dim=-1)
        fused = self.fusion(fused)

        # Classify
        logits = self.classifier(fused)
        probs = F.softmax(logits, dim=-1)

        return logits, probs

    def predict_proba(
        self,
        embeddings: list[torch.Tensor],
        structured: torch.Tensor,
    ) -> torch.Tensor:
        """Convenience method returning probabilities only."""
        with torch.no_grad():
            _, probs = self.forward(embeddings, structured)
        return probs

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_architecture_summary(self) -> dict:
        return {
            "total_parameters": self.count_parameters(),
            "embedding_sections": len(EMBEDDING_SECTIONS),
            "embedding_dim_input": EMBEDDING_DIM,
            "embedding_dim_output": self.config["embedding_output_dim"],
            "structured_hidden_dim": self.config["structured_hidden_dim"],
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
