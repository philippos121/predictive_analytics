"""
Neural Network Architecture for Predictive Litigation Analytics.

Embedding-only architecture:
1. Text embeddings (2 sections × 3072-dim) via per-section encoders
   - klaegervorbringen, beklagtenvorbringen
2. Optional cross-section attention (SectionAttention) to weight sections
3. Flat fusion MLP for final classification

Output: 3-class (Unterliegen / Teilweise / Obsiegen)
"""

import sys
from pathlib import Path
from typing import Optional

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


class SectionAttention(nn.Module):
    """
    Learnable attention weights over the encoded text sections.

    Instead of treating klaegervorbringen and beklagtenvorbringen as
    equally important, this module learns which
    section is most predictive for each case.  The output is an attended
    summary vector concatenated alongside the per-section encodings in
    the fusion step, giving the model both the raw detail and the
    weighted highlight.
    """

    def __init__(self, section_dim: int):
        super().__init__()
        self.score = nn.Linear(section_dim, 1, bias=True)

    def forward(self, encoded_sections: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            encoded_sections: list of (batch, section_dim) tensors

        Returns:
            attended: (batch, section_dim) weighted-sum representation
        """
        stacked = torch.stack(encoded_sections, dim=1)       # (batch, n, dim)
        scores = self.score(stacked).squeeze(-1)             # (batch, n)
        weights = F.softmax(scores, dim=-1)                  # (batch, n)
        attended = (stacked * weights.unsqueeze(-1)).sum(dim=1)  # (batch, dim)
        return attended


class LitigationClassifier(nn.Module):
    """
    Main classification model for Austrian civil case outcome prediction.

    Architecture:
    - 2 EmbeddingEncoders (one per text section):
        klaegervorbringen, beklagtenvorbringen
    - Optional SectionAttention (learnable weights over sections)
    - Optional structured feature branch (disabled when structured_dim=0)
    - Flat fusion MLP
    - 3-class output (0=Unterliegen, 1=Teilweise, 2=Obsiegen)
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

        # Optional cross-section attention
        self.use_section_attention = config.get("use_section_attention", False)
        if self.use_section_attention:
            self.section_attention = SectionAttention(section_dim=emb_output_dim)

        # Optional structured feature branch
        # A small dedicated MLP lets the model learn interactions among the
        # structured fields before they are mixed with the embedding features.
        self.structured_dim = config.get("structured_dim", 0)
        struct_encoded_dim = 0
        if self.structured_dim > 0:
            struct_hidden = config.get("struct_encoder_dim", 16)
            self.struct_encoder = nn.Sequential(
                nn.Linear(self.structured_dim, struct_hidden),
                nn.LayerNorm(struct_hidden),
                nn.GELU(),
                nn.Dropout(dropout_fusion),
            )
            struct_encoded_dim = struct_hidden

        # Fusion: per-section encodings + optional attention + optional struct
        # With attention: (n_sections + 1) * emb_output_dim + struct_encoded_dim
        # Without:        n_sections * emb_output_dim + struct_encoded_dim
        attn_extra = emb_output_dim if self.use_section_attention else 0
        fusion_input_dim = n_sections * emb_output_dim + attn_extra + struct_encoded_dim
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
        struct_features: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            embeddings: list of (batch, EMBEDDING_DIM) tensors, one per section
            struct_features: (batch, structured_dim) tensor or None

        Returns:
            logits: (batch, 3)
            probs: (batch, 3) — softmax probabilities
        """
        # Encode each text section
        encoded_sections = [
            encoder(emb)
            for encoder, emb in zip(self.embedding_encoders, embeddings)
        ]  # List of (batch, emb_output_dim)

        # Concatenate per-section encodings
        section_concat = torch.cat(encoded_sections, dim=-1)  # (batch, n*emb_out)

        # Optional attended summary over sections
        if self.use_section_attention:
            attended = self.section_attention(encoded_sections)  # (batch, emb_out)
            section_concat = torch.cat([section_concat, attended], dim=-1)

        # Optional structured feature branch
        if (self.structured_dim > 0
                and struct_features is not None
                and struct_features.size(-1) > 0):
            struct_encoded = self.struct_encoder(struct_features)  # (batch, 32)
            section_concat = torch.cat([section_concat, struct_encoded], dim=-1)

        # Fuse and classify
        fused = self.fusion(section_concat)
        logits = self.classifier(fused)
        probs = F.softmax(logits, dim=-1)

        return logits, probs

    def predict_proba(
        self,
        embeddings: list[torch.Tensor],
        struct_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Convenience method returning probabilities only."""
        with torch.no_grad():
            _, probs = self.forward(embeddings, struct_features)
        return probs

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_architecture_summary(self) -> dict:
        return {
            "total_parameters": self.count_parameters(),
            "embedding_sections": len(EMBEDDING_SECTIONS),
            "embedding_dim_input": EMBEDDING_DIM,
            "embedding_dim_output": self.config["embedding_output_dim"],
            "use_section_attention": self.use_section_attention,
            "structured_dim": self.structured_dim,
            "struct_encoder_dim": self.config.get("struct_encoder_dim", 16),
            "fusion_dims": self.config["fusion_dims"],
            "num_classes": self.config["num_classes"],
        }


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in litigation outcomes.
    Focuses training on hard examples.

    label_smoothing > 0 prevents overconfident predictions — useful for
    small legal datasets where ground-truth labels carry inherent uncertainty.
    Focal weighting is still computed on hard targets for correct emphasis.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor = None,
        label_smoothing: float = 0.0,
        num_classes: int = 3,
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)

        if self.label_smoothing > 0.0:
            # Build soft targets: (1 - ε) * one_hot + ε / K
            with torch.no_grad():
                smooth_val = self.label_smoothing / num_classes
                soft_targets = torch.full_like(logits, smooth_val)
                soft_targets.scatter_(
                    1,
                    targets.unsqueeze(1),
                    1.0 - self.label_smoothing + smooth_val,
                )
            log_probs = F.log_softmax(logits, dim=-1)
            ce_smooth = -(soft_targets * log_probs).sum(dim=-1)   # (batch,)

            # Focal weight still based on hard targets for correct emphasis
            ce_hard = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
            pt = torch.exp(-ce_hard)
            focal_weight = (1 - pt) ** self.gamma

            return (focal_weight * ce_smooth).mean()
        else:
            ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
            pt = torch.exp(-ce_loss)
            focal_loss = ((1 - pt) ** self.gamma) * ce_loss
            return focal_loss.mean()
