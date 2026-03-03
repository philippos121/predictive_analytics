"""
Feature Engineer: Transforms raw case data into numeric tensors
suitable for the neural network.

v3.0 — Hybrid: embeddings + structured metadata.
Handles:
- Embedding loading, truncation, and stacking from HDF5
- Structured feature encoding (claim types, defenses, streitwert, etc.)
- Train/val split
- Feature normalization
"""

import json
import math
import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CLAIM_TYPES,
    DEFENSE_TYPES,
    EMBEDDING_DIM_USED,
    EMBEDDING_SECTIONS,
    FEATURE_RELEVANCE_FILE,
    LEGAL_ANALYSIS_BOOL_FIELDS,
    LEGAL_ANALYSIS_LIST_FIELDS,
    NUM_CLASSES,
    PCA_DIM,
    PCA_FILE,
    SCALER_FILE,
    TRAINING_CONFIG,
    map_outcome_label,
)


class FeatureEngineer:
    """
    Transforms case metadata into numeric feature vectors.

    Structured features include:
    - log(streitwert)                         [1]
    - claim_type one-hot                      [len(CLAIM_TYPES)+1]
    - defense flags                           [len(DEFENSE_TYPES)]
    - plaintiff_evidence_count                [1]
    - defendant_evidence_count                [1]
    - legal_basis_count                       [1]
    - court_level one-hot (BG/LG/OLG/OGH)    [4]
    - sachverstaendiger                       [1]
    - legal_analysis bool fields              [~42]
    - legal_analysis list counts              [2]
    """

    INSTANZ_CLASSES = ["BG", "LG", "OLG", "OGH"]

    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False
        self._feature_dim = None

    @property
    def feature_dim(self) -> int:
        if self._feature_dim is None:
            self._feature_dim = self._compute_feature_dim()
        return self._feature_dim

    def _compute_feature_dim(self) -> int:
        n = 0
        n += 1                          # log_streitwert
        n += len(CLAIM_TYPES) + 1       # claim_type one-hot (+1 for "other")
        n += len(DEFENSE_TYPES)         # defense flags
        n += 1                          # klaeger_beweismittel count
        n += 1                          # beklagter_beweismittel count
        n += 1                          # anspruchsgruende count
        n += len(self.INSTANZ_CLASSES)  # court level one-hot
        n += 1                          # sachverstaendiger
        # Legal analysis features
        for fields in LEGAL_ANALYSIS_BOOL_FIELDS.values():
            n += len(fields)
        for fields in LEGAL_ANALYSIS_LIST_FIELDS.values():
            n += len(fields)
        return n

    def encode_case(self, case: dict) -> np.ndarray:
        """Encode a single case's structured data to a feature vector."""
        s = case.get("structured", {})
        features = []

        # 1. Log-normalized Streitwert
        sw = s.get("streitwert_eur")
        if sw and sw > 0:
            features.append(math.log1p(sw))
        else:
            features.append(0.0)

        # 2. Claim type (one-hot)
        claim_type = s.get("anspruchsart", "Andere")
        claim_vec = [0.0] * (len(CLAIM_TYPES) + 1)
        if claim_type in CLAIM_TYPES:
            claim_vec[CLAIM_TYPES.index(claim_type)] = 1.0
        else:
            claim_vec[-1] = 1.0
        features.extend(claim_vec)

        # 3. Defense flags
        einwendungen = s.get("einwendungen", {})
        for defense in DEFENSE_TYPES:
            features.append(1.0 if einwendungen.get(defense) else 0.0)

        # 4. Evidence counts
        features.append(float(len(s.get("klaeger_beweismittel", []))))
        features.append(float(len(s.get("beklagter_beweismittel", []))))

        # 5. Legal basis count
        features.append(float(len(s.get("anspruchsgruende", []))))

        # 6. Court level (one-hot)
        instanz = s.get("instanz", "")
        instanz_vec = [0.0] * len(self.INSTANZ_CLASSES)
        if instanz in self.INSTANZ_CLASSES:
            instanz_vec[self.INSTANZ_CLASSES.index(instanz)] = 1.0
        features.extend(instanz_vec)

        # 7. Expert witness
        features.append(1.0 if s.get("sachverstaendiger_bestellt") else 0.0)

        # 8. Legal analysis boolean fields (~42 features)
        la = case.get("legal_analysis", {})
        for section, fields in LEGAL_ANALYSIS_BOOL_FIELDS.items():
            section_data = la.get(section, {})
            for field in fields:
                features.append(1.0 if section_data.get(field) else 0.0)

        # 9. Legal analysis list counts (zitierte Normen)
        for section, fields in LEGAL_ANALYSIS_LIST_FIELDS.items():
            section_data = la.get(section, {})
            for field in fields:
                features.append(float(len(section_data.get(field, []))))

        return np.array(features, dtype=np.float32)

    def encode_batch(self, cases: list[dict]) -> np.ndarray:
        """Encode multiple cases. Returns (N, feature_dim) array."""
        return np.stack([self.encode_case(c) for c in cases])

    def fit_transform(self, cases: list[dict]) -> np.ndarray:
        """Fit scaler on training data, then transform."""
        X_raw = self.encode_batch(cases)
        X_scaled = self.scaler.fit_transform(X_raw)
        self.is_fitted = True
        self._feature_dim = X_raw.shape[1]
        return X_scaled

    def transform(self, cases: list[dict]) -> np.ndarray:
        """Transform using fitted scaler."""
        if not self.is_fitted:
            raise RuntimeError("FeatureEngineer not fitted. Call fit_transform first.")
        X_raw = self.encode_batch(cases)
        return self.scaler.transform(X_raw)

    def encode_single_transform(self, case: dict) -> np.ndarray:
        """Encode and scale a single case (for inference)."""
        if not self.is_fitted:
            raise RuntimeError("FeatureEngineer not fitted.")
        x = self.encode_case(case)
        return self.scaler.transform(x.reshape(1, -1))[0]

    def build_attention_weights(
        self,
        relevance_path: Path = FEATURE_RELEVANCE_FILE,
    ) -> Optional[np.ndarray]:
        """
        Build a per-feature attention weight vector from court-derived relevance data.

        Returns shape (feature_dim,) with weights >=0.5 for all features.
        Non-legal-analysis features (streitwert, claim_type, etc.) get weight 1.0.
        Legal-analysis boolean features get their court-derived weight.
        Returns None if the relevance file doesn't exist.
        """
        if not relevance_path.exists():
            return None

        with open(relevance_path, "r") as f:
            data = json.load(f)

        attention_weights_map = data.get("attention_weights", {})
        if not attention_weights_map:
            return None

        dim = self.feature_dim
        weights = np.ones(dim, dtype=np.float32)

        # Compute offset to the legal_analysis bool section
        offset = 0
        offset += 1                          # log_streitwert
        offset += len(CLAIM_TYPES) + 1       # claim_type one-hot
        offset += len(DEFENSE_TYPES)         # defense flags
        offset += 1                          # klaeger_beweismittel
        offset += 1                          # beklagter_beweismittel
        offset += 1                          # anspruchsgruende count
        offset += len(self.INSTANZ_CLASSES)  # court level one-hot
        offset += 1                          # sachverstaendiger

        # Fill in legal_analysis bool weights (same order as encode_case)
        idx = offset
        for section, fields in LEGAL_ANALYSIS_BOOL_FIELDS.items():
            for field in fields:
                if field in attention_weights_map:
                    weights[idx] = attention_weights_map[field]
                idx += 1

        # List count features keep weight 1.0 (already at default)

        return weights

    def save(self, path: Path = SCALER_FILE) -> None:
        """Persist the fitted scaler."""
        with open(path, "wb") as f:
            pickle.dump({
                "scaler": self.scaler,
                "feature_dim": self._feature_dim,
            }, f)

    def load(self, path: Path = SCALER_FILE) -> None:
        """Load a previously fitted scaler."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.scaler = data["scaler"]
        self._feature_dim = data.get("feature_dim")
        self.is_fitted = True


# ─── Embedding PCA ───────────────────────────────────────────────────────────────

class EmbeddingPCA:
    """
    PCA dimensionality reduction for text embeddings.

    Fits one shared PCA across all embedding sections (kläger + beklagter),
    reducing from EMBEDDING_DIM_USED (1024) to PCA_DIM (128).
    This addresses the curse of dimensionality: with ~6400 training samples
    and 2×1024 input dims, the model has too few examples per dimension.
    PCA reduces to 2×128 = 256 dims → ~25 examples per dimension.
    """

    def __init__(self, n_components: int = PCA_DIM):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.is_fitted = False
        self.explained_variance_ratio_sum: float = 0.0

    def fit(
        self,
        embeddings_dict: dict[str, dict[str, np.ndarray]],
        cases: list[dict],
    ) -> "EmbeddingPCA":
        """
        Fit PCA on all training embeddings (pooled across sections).

        All sections share one PCA so that kläger and beklagter embeddings
        live in the same reduced space.
        """
        all_vecs = []
        for case in cases:
            cid = case["case_id"]
            if cid not in embeddings_dict:
                continue
            emb = embeddings_dict[cid]
            for section in EMBEDDING_SECTIONS:
                if section in emb:
                    vec = truncate_embedding(emb[section])
                    all_vecs.append(vec)

        if len(all_vecs) < self.n_components:
            raise ValueError(
                f"Zu wenige Embeddings ({len(all_vecs)}) für PCA "
                f"mit {self.n_components} Komponenten."
            )

        X = np.stack(all_vecs)  # (N_total, 1024)
        self.pca.fit(X)
        self.is_fitted = True
        self.explained_variance_ratio_sum = float(
            self.pca.explained_variance_ratio_.sum()
        )
        return self

    def transform(self, vec: np.ndarray) -> np.ndarray:
        """Transform a single 1024-dim embedding to PCA_DIM dimensions."""
        if not self.is_fitted:
            raise RuntimeError("EmbeddingPCA not fitted. Call fit() first.")
        return self.pca.transform(vec.reshape(1, -1))[0].astype(np.float32)

    def save(self, path: Path = PCA_FILE) -> None:
        with open(path, "wb") as f:
            pickle.dump({
                "pca": self.pca,
                "n_components": self.n_components,
                "explained_variance_ratio_sum": self.explained_variance_ratio_sum,
            }, f)

    def load(self, path: Path = PCA_FILE) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.pca = data["pca"]
        self.n_components = data["n_components"]
        self.explained_variance_ratio_sum = data.get(
            "explained_variance_ratio_sum", 0.0
        )
        self.is_fitted = True


# ─── Embedding Utilities ─────────────────────────────────────────────────────────

def truncate_embedding(vec: np.ndarray, dim: int = EMBEDDING_DIM_USED) -> np.ndarray:
    """Truncate embedding to first `dim` dimensions (Matryoshka property)."""
    if len(vec) >= dim:
        return vec[:dim].astype(np.float32)
    # Pad if shorter
    padded = np.zeros(dim, dtype=np.float32)
    padded[:len(vec)] = vec
    return padded


def prepare_embeddings_for_case(
    emb_dict: dict[str, np.ndarray],
    dim: int = EMBEDDING_DIM_USED,
    pca: Optional[EmbeddingPCA] = None,
) -> list[np.ndarray]:
    """
    Prepare embedding vectors for a single case.
    Returns list of embedding arrays, one per EMBEDDING_SECTIONS entry.
    If pca is provided, applies PCA reduction after truncation.
    """
    result = []
    for section in EMBEDDING_SECTIONS:
        if section in emb_dict:
            vec = truncate_embedding(emb_dict[section], dim)
            if pca is not None and pca.is_fitted:
                vec = pca.transform(vec)
            result.append(vec)
        else:
            out_dim = pca.n_components if (pca is not None and pca.is_fitted) else dim
            result.append(np.zeros(out_dim, dtype=np.float32))
    return result


# ─── Dataset ─────────────────────────────────────────────────────────────────────

class LitigationDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for litigation cases (embeddings + structured features).

    Each item returns:
    - embeddings: list of tensors (one per section), each shape (emb_dim,)
    - structured: tensor of shape (feature_dim,)
    - label: int (0, 1, 2)
    """

    def __init__(
        self,
        cases: list[dict],
        embeddings_dict: dict[str, dict[str, np.ndarray]],
        structured_features: np.ndarray,
        pca: Optional[EmbeddingPCA] = None,
    ):
        self.cases = cases
        self.embeddings_dict = embeddings_dict
        self.structured_features = structured_features
        self.pca = pca

        # Filter to cases that have all required data
        self.valid_indices = [
            i for i, c in enumerate(cases)
            if c["case_id"] in embeddings_dict
            and c["structured"].get("outcome") is not None
        ]

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> tuple:
        case_idx = self.valid_indices[idx]
        case = self.cases[case_idx]
        case_id = case["case_id"]

        # Load, truncate, and optionally PCA-reduce embeddings
        emb_data = self.embeddings_dict[case_id]
        embeddings = [
            torch.tensor(vec, dtype=torch.float32)
            for vec in prepare_embeddings_for_case(emb_data, pca=self.pca)
        ]

        # Structured features
        structured = torch.tensor(
            self.structured_features[case_idx], dtype=torch.float32
        )

        # Label (mapped to NUM_CLASSES scheme)
        raw_label = int(case["structured"]["outcome"])
        label = map_outcome_label(raw_label)

        return embeddings, structured, label


def prepare_dataset(
    cases: list[dict],
    embeddings_dict: dict[str, dict],
    feature_engineer: FeatureEngineer,
    val_split: float = TRAINING_CONFIG["val_split"],
    random_seed: int = TRAINING_CONFIG["random_seed"],
    pca: Optional[EmbeddingPCA] = None,
) -> tuple["LitigationDataset", "LitigationDataset", "LitigationDataset"]:
    """
    Prepare train, validation, and full datasets.
    Returns: train_dataset, val_dataset, full_dataset
    """
    np.random.seed(random_seed)

    # Encode structured features
    structured_features = feature_engineer.fit_transform(cases)

    full_dataset = LitigationDataset(
        cases, embeddings_dict, structured_features, pca=pca,
    )

    if len(full_dataset) == 0:
        raise ValueError("No valid labeled cases with embeddings found.")

    # Stratified split (uses mapped labels for NUM_CLASSES scheme)
    indices = list(range(len(full_dataset)))
    labels = [
        map_outcome_label(int(
            full_dataset.cases[full_dataset.valid_indices[i]]["structured"]["outcome"]
        ))
        for i in indices
    ]

    label_to_indices: dict[int, list[int]] = {c: [] for c in range(NUM_CLASSES)}
    for i, lbl in zip(indices, labels):
        label_to_indices[lbl].append(i)

    train_indices, val_indices = [], []
    for lbl, idxs in label_to_indices.items():
        np.random.shuffle(idxs)
        n_val = max(1, int(len(idxs) * val_split)) if len(idxs) > 1 else 0
        val_indices.extend(idxs[:n_val])
        train_indices.extend(idxs[n_val:])

    train_dataset = _SubsetDataset(full_dataset, train_indices)
    val_dataset = _SubsetDataset(full_dataset, val_indices)

    return train_dataset, val_dataset, full_dataset


class _SubsetDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: LitigationDataset, indices: list[int]):
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        return self.dataset[self.indices[idx]]


def collate_fn(batch: list) -> tuple:
    """Collate for hybrid embedding + structured data."""
    n_sections = len(EMBEDDING_SECTIONS)

    # Stack embeddings per section
    embeddings_per_section = [
        torch.stack([item[0][s] for item in batch])
        for s in range(n_sections)
    ]

    structured_batch = torch.stack([item[1] for item in batch])
    labels_batch = torch.tensor([item[2] for item in batch], dtype=torch.long)

    return embeddings_per_section, structured_batch, labels_batch


def compute_class_weights(cases: list[dict]) -> torch.Tensor:
    """Compute inverse-frequency class weights for imbalanced datasets."""
    mapped_labels = [
        map_outcome_label(int(c["structured"]["outcome"]))
        for c in cases
        if c["structured"].get("outcome") is not None
    ]
    n_cls = NUM_CLASSES
    counts = [mapped_labels.count(i) for i in range(n_cls)]
    total = sum(counts)

    if total == 0 or any(c == 0 for c in counts):
        return torch.ones(n_cls)

    weights = [total / (n_cls * c) for c in counts]
    return torch.tensor(weights, dtype=torch.float32)
