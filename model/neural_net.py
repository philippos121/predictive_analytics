"""
Neural Network Architecture for Predictive Litigation Analytics.

v4.0 — Cross-Attention Hybrid Architecture (100 K+ cases, no PCA).

Full 1024-dim text embeddings → shared encoder → multi-head cross-attention
→ fusion with structured features → binary classification.

The cross-attention is the key architectural choice: legal outcomes depend
on the INTERACTION between Kläger claims and Beklagter defenses.  Each side
attends to the other before fusion, letting the model learn patterns like
"Gewährleistung claim + Verjährung defense → specific outcome signal."

Architecture:
1. Shared EmbeddingEncoder: 1024 → 256 (both sections, shared weights)
2. MultiHeadCrossAttention: 4-head bidirectional attention + FFN
3. StructuredEncoder: 77 features → 64-dim
4. Fusion MLP: (2×256 + 64 = 576) → 256 → 128 → 2

Output: 2-class (Nicht-Obsiegen / Obsiegen)
Total params: ~780 K
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import EMBEDDING_DIM_USED, EMBEDDING_SECTIONS, NN_CONFIG, PCA_DIM, USE_PCA


def _embedding_input_dim() -> int:
    """Return the actual input dimension for the embedding encoder."""
    return PCA_DIM if USE_PCA else EMBEDDING_DIM_USED


class EmbeddingEncoder(nn.Module):
    """
    Shared embedding encoder for both Kläger and Beklagter sections.
    Projects raw embeddings (1024-dim or PCA-reduced) into a compact space.
    """

    def __init__(
        self,
        input_dim: int = None,
        hidden_dim: int = NN_CONFIG["embedding_hidden_dim"],
        output_dim: int = NN_CONFIG["embedding_output_dim"],
        dropout: float = NN_CONFIG["dropout_embedding"],
    ):
        super().__init__()
        if input_dim is None:
            input_dim = _embedding_input_dim()
        self.encoder = nn.Sequential(
            nn.Dropout(dropout),
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
    Multi-head bidirectional cross-attention between Kläger and Beklagter.

    Standard transformer cross-attention pattern:
    1. Multi-head attention (query attends to context)
    2. Residual + LayerNorm
    3. Feed-forward network
    4. Residual + LayerNorm

    Applied bidirectionally: Kläger → Beklagter and Beklagter → Kläger.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = NN_CONFIG.get("n_attention_heads", 4),
        dropout: float = 0.1,
        ff_mult: int = 2,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        assert dim % n_heads == 0, f"dim {dim} must be divisible by n_heads {n_heads}"

        # Cross-attention (shared projections — both directions use same weights)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        # Post-attention norms
        self.norm_k = nn.LayerNorm(dim)
        self.norm_b = nn.LayerNorm(dim)

        # Feed-forward network (applied after attention to each side)
        ff_dim = dim * ff_mult
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout),
        )
        self.norm_ffn_k = nn.LayerNorm(dim)
        self.norm_ffn_b = nn.LayerNorm(dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.scale = (dim // n_heads) ** -0.5

    def _cross_attend(
        self,
        query: torch.Tensor,    # (batch, dim)
        context: torch.Tensor,  # (batch, dim)
    ) -> torch.Tensor:
        """Multi-head cross-attention: query attends to context."""
        B = query.shape[0]
        head_dim = self.dim // self.n_heads

        # Project and reshape for multi-head: (B, n_heads, 1, head_dim)
        q = self.q_proj(query).view(B, self.n_heads, 1, head_dim)
        k = self.k_proj(context).view(B, self.n_heads, 1, head_dim)
        v = self.v_proj(context).view(B, self.n_heads, 1, head_dim)

        # Attention: (B, n_heads, 1, 1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # Apply attention to values
        out = (attn @ v).view(B, self.dim)  # (B, dim)
        return self.out_proj(out)

    def forward(
        self,
        klaeger: torch.Tensor,    # (batch, dim)
        beklagter: torch.Tensor,  # (batch, dim)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Bidirectional multi-head cross-attention + FFN.

        Returns interaction-enriched representations for both sides.
        """
        # Cross-attention + residual + norm
        k_attn = self._cross_attend(klaeger, beklagter)
        klaeger = self.norm_k(klaeger + k_attn)

        b_attn = self._cross_attend(beklagter, klaeger)
        beklagter = self.norm_b(beklagter + b_attn)

        # FFN + residual + norm
        klaeger = self.norm_ffn_k(klaeger + self.ffn(klaeger))
        beklagter = self.norm_ffn_b(beklagter + self.ffn(beklagter))

        return klaeger, beklagter


class StructuredEncoder(nn.Module):
    """Encodes structured legal features (streitwert, claim type, defenses, etc.).

    Straightforward MLP. No hand-crafted attention priors — with 100K cases
    the model learns feature importance end-to-end through backpropagation.
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
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class LitigationClassifier(nn.Module):
    """
    Cross-attention hybrid classifier for Austrian civil case outcomes.

    v4.0 Architecture (100 K+ cases):
    - Shared EmbeddingEncoder: full 1024-dim → 256-dim
    - MultiHeadCrossAttention: 4-head bidirectional + FFN (standard transformer)
    - StructuredEncoder: 77 features → 64-dim (learns importance via backprop)
    - Fusion MLP: 576 → 256 → 128 → 2
    """

    def __init__(
        self,
        structured_dim: int,
        config: dict = NN_CONFIG,
    ):
        super().__init__()

        self.config = config
        self.structured_dim = structured_dim

        n_sections = len(EMBEDDING_SECTIONS)
        emb_input_dim = _embedding_input_dim()
        emb_output_dim = config["embedding_output_dim"]
        struct_hidden_dim = config["structured_hidden_dim"]
        fusion_dims = config["fusion_dims"]
        num_classes = config["num_classes"]
        dropout_fusion = config["dropout_fusion"]

        # Shared embedding encoder
        self.shared_embedding_encoder = EmbeddingEncoder(
            input_dim=emb_input_dim,
            hidden_dim=config["embedding_hidden_dim"],
            output_dim=emb_output_dim,
            dropout=config["dropout_embedding"],
        )

        # Cross-attention (only for 2-section setup: kläger + beklagter)
        self.has_cross_attention = (n_sections == 2)
        if self.has_cross_attention:
            self.cross_attention = CrossAttentionBlock(
                dim=emb_output_dim,
                n_heads=config.get("n_attention_heads", 4),
                dropout=dropout_fusion * 0.5,
            )

        # Structured feature encoder
        self.structured_encoder = StructuredEncoder(
            input_dim=structured_dim,
            hidden_dim=struct_hidden_dim,
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
            embeddings: list of (batch, emb_dim) tensors, one per section
            structured: (batch, structured_dim) tensor

        Returns:
            logits: (batch, num_classes)
            probs: (batch, num_classes)
        """
        # Encode through shared encoder
        encoded_sections = [
            self.shared_embedding_encoder(emb) for emb in embeddings
        ]

        # Cross-attention: Kläger ↔ Beklagter interaction
        if self.has_cross_attention and len(encoded_sections) == 2:
            encoded_sections[0], encoded_sections[1] = self.cross_attention(
                encoded_sections[0], encoded_sections[1]
            )

        section_concat = torch.cat(encoded_sections, dim=-1)
        struct_encoded = self.structured_encoder(structured)

        fused = torch.cat([section_concat, struct_encoded], dim=-1)
        fused = self.fusion(fused)

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
            "embedding_input_dim": _embedding_input_dim(),
            "embedding_output_dim": self.config["embedding_output_dim"],
            "structured_dim": self.structured_dim,
            "structured_hidden_dim": self.config["structured_hidden_dim"],
            "fusion_dims": self.config["fusion_dims"],
            "num_classes": self.config["num_classes"],
            "has_cross_attention": self.has_cross_attention,
            "n_attention_heads": self.config.get("n_attention_heads", 4),
            "use_pca": USE_PCA,
        }
