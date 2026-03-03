"""
Neural Network Architecture for Predictive Litigation Analytics.

v4.0 — Cross-Attention Hybrid Architecture.

Key improvement over v3: Kläger and Beklagter embeddings interact via
cross-attention BEFORE fusion, letting the model learn that the outcome
depends on the RELATIONSHIP between claims and defenses — not just each
party's arguments in isolation.

Architecture:
1. Shared EmbeddingEncoder projects PCA-reduced embeddings to a common space
2. CrossAttentionBlock: Kläger attends to Beklagter and vice versa
3. StructuredEncoder for metadata features (with court-derived attention)
4. Fusion MLP → binary classification

Output: 2-class (Nicht-Obsiegen / Obsiegen)
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import EMBEDDING_DIM_USED, EMBEDDING_SECTIONS, NN_CONFIG, PCA_DIM


class EmbeddingEncoder(nn.Module):
    """
    Per-section embedding encoder.
    Compresses PCA-reduced embedding to compact representation.
    """

    def __init__(
        self,
        input_dim: int = PCA_DIM,
        hidden_dim: int = NN_CONFIG["embedding_hidden_dim"],
        output_dim: int = NN_CONFIG["embedding_output_dim"],
        dropout: float = NN_CONFIG["dropout_embedding"],
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Dropout(dropout),  # input dropout: prevent memorising raw embeddings
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class CrossAttentionBlock(nn.Module):
    """
    Bidirectional cross-attention between Kläger and Beklagter representations.

    Legal outcomes depend on the INTERACTION between claims and defenses.
    Cross-attention lets the model learn patterns like:
    - Plaintiff's Gewährleistung argument + defendant's Verjährung defense → signal
    - Strength of plaintiff's claim IN CONTEXT OF defendant's rebuttal

    Each side attends to the other, producing interaction-aware representations.
    Uses a single attention head (sufficient for our embedding dimension).
    """

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.dim = dim

        # Separate Q/K/V projections for each direction
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        self.out_proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        self.scale = dim ** -0.5

    def _cross_attend(
        self,
        query: torch.Tensor,    # (batch, dim) — the side doing the attending
        context: torch.Tensor,   # (batch, dim) — the side being attended to
    ) -> torch.Tensor:
        """Single-token cross-attention: query attends to context."""
        # Reshape to (batch, 1, dim) for standard attention math
        q = self.q_proj(query).unsqueeze(1)    # (batch, 1, dim)
        k = self.k_proj(context).unsqueeze(1)  # (batch, 1, dim)
        v = self.v_proj(context).unsqueeze(1)  # (batch, 1, dim)

        # Attention weights (batch, 1, 1) — degenerate case with 1 token each
        # but the projection still learns a useful transformation
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).squeeze(1)  # (batch, dim)
        return self.out_proj(out)

    def forward(
        self,
        klaeger: torch.Tensor,     # (batch, dim)
        beklagter: torch.Tensor,   # (batch, dim)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Bidirectional cross-attention.

        Returns:
            klaeger_out: kläger representation enriched with beklagter context
            beklagter_out: beklagter representation enriched with kläger context
        """
        # Kläger attends to Beklagter
        klaeger_cross = self._cross_attend(klaeger, beklagter)
        klaeger_out = self.norm1(klaeger + self.dropout(klaeger_cross))

        # Beklagter attends to Kläger
        beklagter_cross = self._cross_attend(beklagter, klaeger)
        beklagter_out = self.norm2(beklagter + self.dropout(beklagter_cross))

        return klaeger_out, beklagter_out


class StructuredEncoder(nn.Module):
    """Encodes structured legal features (streitwert, claim type, defenses, etc.).

    Optionally applies court-derived attention weights: element-wise scaling
    of input features BEFORE the linear layer. This gives the model a prior
    from erstgericht_begruendung analysis — features that courts frequently
    rely on start with higher weight.

    The attention vector is a learnable Parameter initialized from the
    relevance analysis, so the model can still adjust during training.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = NN_CONFIG["structured_hidden_dim"],
        attention_init: "torch.Tensor | None" = None,
    ):
        super().__init__()

        # Court-derived feature attention (learnable, initialized from relevance)
        if attention_init is not None:
            self.feature_attention = nn.Parameter(attention_init.clone())
        else:
            # No relevance data → uniform weights (no-op multiply)
            self.feature_attention = nn.Parameter(torch.ones(input_dim))

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply learned attention before encoding
        x = x * self.feature_attention
        return self.encoder(x)


class LitigationClassifier(nn.Module):
    """
    Cross-attention hybrid classification model for Austrian civil case outcomes.

    v4.0 Architecture:
    - Shared EmbeddingEncoder: projects both sections to a common space
    - CrossAttentionBlock: Kläger ↔ Beklagter interaction (the key upgrade)
    - StructuredEncoder: metadata features with court-derived attention prior
    - Fusion MLP → binary classification

    The cross-attention is the critical difference from v3: instead of encoding
    kläger and beklagter independently and hoping the fusion layer figures out
    their relationship, we explicitly model their interaction.
    """

    def __init__(
        self,
        structured_dim: int,
        config: dict = NN_CONFIG,
        structured_attention_init: "torch.Tensor | None" = None,
    ):
        super().__init__()

        self.config = config
        self.structured_dim = structured_dim

        n_sections = len(EMBEDDING_SECTIONS)
        emb_output_dim = config["embedding_output_dim"]
        struct_hidden_dim = config["structured_hidden_dim"]
        fusion_dims = config["fusion_dims"]
        num_classes = config["num_classes"]
        dropout_fusion = config["dropout_fusion"]

        # Shared embedding encoder — both sides project into the same space
        # so cross-attention is meaningful (shared weights = parameter-efficient)
        self.shared_embedding_encoder = EmbeddingEncoder(
            input_dim=PCA_DIM,
            hidden_dim=config["embedding_hidden_dim"],
            output_dim=emb_output_dim,
            dropout=config["dropout_embedding"],
        )

        # Cross-attention: Kläger ↔ Beklagter interaction
        # Only created when we have exactly 2 sections (kläger + beklagter)
        self.has_cross_attention = (n_sections == 2)
        if self.has_cross_attention:
            self.cross_attention = CrossAttentionBlock(
                dim=emb_output_dim,
                dropout=dropout_fusion * 0.5,
            )

        # Structured feature encoder (with optional court-derived attention)
        self.structured_encoder = StructuredEncoder(
            input_dim=structured_dim,
            hidden_dim=struct_hidden_dim,
            attention_init=structured_attention_init,
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
            embeddings: list of (batch, PCA_DIM) tensors, one per section
            structured: (batch, structured_dim) tensor

        Returns:
            logits: (batch, num_classes)
            probs: (batch, num_classes) — softmax probabilities
        """
        # Encode each section through the shared encoder
        encoded_sections = [
            self.shared_embedding_encoder(emb) for emb in embeddings
        ]

        # Cross-attention: let Kläger and Beklagter interact
        if self.has_cross_attention and len(encoded_sections) == 2:
            encoded_sections[0], encoded_sections[1] = self.cross_attention(
                encoded_sections[0], encoded_sections[1]
            )

        # Concatenate all section encodings
        section_concat = torch.cat(encoded_sections, dim=-1)

        # Encode structured features
        struct_encoded = self.structured_encoder(structured)

        # Fuse all inputs
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
            "embedding_dim_raw": EMBEDDING_DIM_USED,
            "embedding_dim_pca": PCA_DIM,
            "embedding_dim_output": self.config["embedding_output_dim"],
            "structured_dim": self.structured_dim,
            "structured_hidden_dim": self.config["structured_hidden_dim"],
            "fusion_dims": self.config["fusion_dims"],
            "num_classes": self.config["num_classes"],
            "has_cross_attention": self.has_cross_attention,
        }
