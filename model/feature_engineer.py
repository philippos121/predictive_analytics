"""
Feature Engineer: Transforms raw case data into numeric tensors
suitable for the neural network.

Handles:
- Structured feature encoding (claim types, defenses, streitwert, etc.)
- Embedding loading and stacking
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
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CLAIM_TYPES,
    DEFENSE_TYPES,
    EMBEDDING_DIM,
    EMBEDDING_SECTIONS,
    ENCODER_FILE,
    SCALER_FILE,
    TRAINING_CONFIG,
)


class FeatureEngineer:
    """
    Transforms case metadata into numeric feature vectors.

    Structured features include:
    - log(streitwert) — normalized  [1]
    - claim_type — one-hot          [len(CLAIM_TYPES)+1]
    - defense flags                 [len(DEFENSE_TYPES)]
    - plaintiff_evidence_count      [1]
    - defendant_evidence_count      [1]
    - legal_basis_count             [1]
    - court_level                   [4] (BG/LG/OLG/OGH one-hot, bei OGH-Datenbasis = Erstgericht)
    - sachverstaendiger             [1]
    - has_aufrechnung               [1] (already in defense, redundant but useful)
    ─────────────────────────────────────────────────────
    Total: 1 + (len(CLAIM_TYPES)+1) + len(DEFENSE_TYPES) + 1 + 1 + 1 + 4 + 1 = varies
    """

    # Österreichische Gerichtsinstanzen: Bezirksgericht, Landesgericht, OLG, OGH
    # Bei OGH-Datenbasis: "instanz" = Erstgericht (BG oder LG)
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
        return n

    def encode_case(self, case: dict) -> np.ndarray:
        """
        Encode a single case's structured data to a feature vector.
        """
        s = case.get("structured", {})
        features = []

        # 1. Log-normalized Streitwert
        sw = s.get("streitwert_eur")
        if sw and sw > 0:
            features.append(math.log1p(sw))
        else:
            features.append(0.0)  # 0 = unknown/zero

        # 2. Claim type (one-hot)
        claim_type = s.get("anspruchsart", "Andere")
        claim_vec = [0.0] * (len(CLAIM_TYPES) + 1)
        if claim_type in CLAIM_TYPES:
            claim_vec[CLAIM_TYPES.index(claim_type)] = 1.0
        else:
            claim_vec[-1] = 1.0  # "Andere"
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

        return np.array(features, dtype=np.float32)

    def encode_batch(self, cases: list[dict]) -> np.ndarray:
        """Encode multiple cases. Returns (N, feature_dim) array."""
        return np.stack([self.encode_case(c) for c in cases])

    def fit_transform(self, cases: list[dict]) -> np.ndarray:
        """Fit scaler on training data and transform."""
        X = self.encode_batch(cases)
        X_scaled = self.scaler.fit_transform(X)
        self.is_fitted = True
        self._feature_dim = X.shape[1]
        return X_scaled

    def transform(self, cases: list[dict]) -> np.ndarray:
        """Transform using fitted scaler."""
        if not self.is_fitted:
            raise RuntimeError("FeatureEngineer not fitted. Call fit_transform first.")
        X = self.encode_batch(cases)
        return self.scaler.transform(X)

    def encode_single_transform(self, case: dict) -> np.ndarray:
        """Encode and scale a single case (for inference)."""
        if not self.is_fitted:
            raise RuntimeError("FeatureEngineer not fitted.")
        x = self.encode_case(case)
        return self.scaler.transform(x.reshape(1, -1))[0]

    def save(self, path: Path = SCALER_FILE) -> None:
        """Persist the fitted scaler."""
        with open(path, "wb") as f:
            pickle.dump({"scaler": self.scaler, "feature_dim": self._feature_dim}, f)

    def load(self, path: Path = SCALER_FILE) -> None:
        """Load a previously fitted scaler."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.scaler = data["scaler"]
        self._feature_dim = data.get("feature_dim")
        self.is_fitted = True


class LitigationDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for litigation cases.

    Each item returns:
    - embeddings: list of 5 tensors (one per section), each shape (EMBEDDING_DIM,)
    - structured: tensor of shape (feature_dim,)
    - label: int (0, 1, 2)
    """

    def __init__(
        self,
        cases: list[dict],
        embeddings_dict: dict[str, dict[str, np.ndarray]],
        structured_features: np.ndarray,
    ):
        self.cases = cases
        self.embeddings_dict = embeddings_dict
        self.structured_features = structured_features

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

        # Load embeddings for each section
        emb_data = self.embeddings_dict[case_id]
        embeddings = []
        for section in EMBEDDING_SECTIONS:
            if section in emb_data:
                vec = emb_data[section].astype(np.float32)
            else:
                vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)
            embeddings.append(torch.tensor(vec, dtype=torch.float32))

        # Structured features
        structured = torch.tensor(
            self.structured_features[case_idx], dtype=torch.float32
        )

        # Label
        label = int(case["structured"]["outcome"])

        return embeddings, structured, label


def prepare_dataset(
    cases: list[dict],
    embeddings_dict: dict[str, dict],
    feature_engineer: FeatureEngineer,
    val_split: float = TRAINING_CONFIG["val_split"],
    random_seed: int = TRAINING_CONFIG["random_seed"],
) -> tuple["LitigationDataset", "LitigationDataset", "LitigationDataset"]:
    """
    Prepare train, validation, and full datasets.

    Returns:
        train_dataset, val_dataset, full_dataset
    """
    np.random.seed(random_seed)

    # Encode structured features
    structured_features = feature_engineer.fit_transform(cases)

    full_dataset = LitigationDataset(cases, embeddings_dict, structured_features)

    if len(full_dataset) == 0:
        raise ValueError("No valid labeled cases with embeddings found.")

    # Stratified split (try to keep outcome distribution)
    indices = list(range(len(full_dataset)))
    labels = [
        full_dataset.cases[full_dataset.valid_indices[i]]["structured"]["outcome"]
        for i in indices
    ]

    # Group by label
    label_to_indices: dict[int, list[int]] = {0: [], 1: [], 2: []}
    for i, lbl in zip(indices, labels):
        label_to_indices[int(lbl)].append(i)

    train_indices, val_indices = [], []
    for lbl, idxs in label_to_indices.items():
        np.random.shuffle(idxs)
        n_val = max(1, int(len(idxs) * val_split)) if len(idxs) > 1 else 0
        val_indices.extend(idxs[:n_val])
        train_indices.extend(idxs[n_val:])

    # Create subset datasets
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
    """Custom collate for multi-section embeddings."""
    embeddings_batch = [item[0] for item in batch]
    structured_batch = torch.stack([item[1] for item in batch])
    labels_batch = torch.tensor([item[2] for item in batch], dtype=torch.long)

    # Transpose: (batch, n_sections, dim) → list of (batch, dim)
    n_sections = len(EMBEDDING_SECTIONS)
    embeddings_by_section = [
        torch.stack([embeddings_batch[b][s] for b in range(len(batch))])
        for s in range(n_sections)
    ]

    return embeddings_by_section, structured_batch, labels_batch


def compute_class_weights(cases: list[dict]) -> torch.Tensor:
    """Compute inverse-frequency class weights for imbalanced datasets."""
    labels = [
        c["structured"].get("outcome")
        for c in cases
        if c["structured"].get("outcome") is not None
    ]
    counts = [labels.count(i) for i in range(3)]
    total = sum(counts)

    if total == 0 or any(c == 0 for c in counts):
        return torch.ones(3)

    weights = [total / (3 * c) for c in counts]
    return torch.tensor(weights, dtype=torch.float32)
