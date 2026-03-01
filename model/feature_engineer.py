"""
Feature Engineer: Transforms raw case data into numeric tensors
suitable for the neural network.

v2.0 — Structured data only, no embeddings.
Encodes both the original case metadata AND the detailed legal analysis
schema into a single feature vector (~77 dimensions).
"""

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
    LEGAL_ANALYSIS_BOOL_FIELDS,
    LEGAL_ANALYSIS_LIST_FIELDS,
    RECHTSGEBIET_CATEGORIES,
    SCALER_FILE,
    TRAINING_CONFIG,
)


class FeatureEngineer:
    """
    Transforms case metadata + legal analysis into numeric feature vectors.

    Feature groups:
    A) From original structured data:
       - log(streitwert)                          [1]
       - claim_type one-hot                        [len(CLAIM_TYPES)+1]
       - defense flags (legacy)                    [len(DEFENSE_TYPES)]
       - plaintiff_evidence_count                  [1]
       - defendant_evidence_count                  [1]
       - legal_basis_count                         [1]
       - court_level one-hot (BG/LG/OLG/OGH)      [4]
       - sachverstaendiger                         [1]

    B) From legal_analysis:
       - rechtsgebiet_hauptkategorie one-hot       [len(RECHTSGEBIET_CATEGORIES)]
       - all boolean fields from schema            [~41]
       - zitierte_normen_klaeger count             [1]
       - zitierte_normen_beklagter count           [1]

    Total: ~77 features (exact count depends on config lists)
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

        # A) Original structured features
        n += 1                          # log_streitwert
        n += len(CLAIM_TYPES) + 1       # claim_type one-hot (+1 for "other")
        n += len(DEFENSE_TYPES)         # defense flags
        n += 1                          # klaeger_beweismittel count
        n += 1                          # beklagter_beweismittel count
        n += 1                          # anspruchsgruende count
        n += len(self.INSTANZ_CLASSES)  # court level one-hot
        n += 1                          # sachverstaendiger

        # B) Legal analysis features
        n += len(RECHTSGEBIET_CATEGORIES)  # rechtsgebiet one-hot
        for section_fields in LEGAL_ANALYSIS_BOOL_FIELDS.values():
            n += len(section_fields)
        for section_fields in LEGAL_ANALYSIS_LIST_FIELDS.values():
            n += len(section_fields)  # count per list field

        return n

    def encode_case(self, case: dict) -> np.ndarray:
        """
        Encode a single case's structured data + legal analysis to a feature vector.
        """
        s = case.get("structured", {})
        la = case.get("legal_analysis", {})
        features = []

        # ── A) Original structured features ──────────────────────────────────────

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

        # 3. Defense flags (legacy)
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

        # ── B) Legal analysis features ───────────────────────────────────────────

        # 8. Rechtsgebiet one-hot
        fm = la.get("fall_metadaten", {})
        rg = fm.get("rechtsgebiet_hauptkategorie", "Sonstiges")
        rg_vec = [0.0] * len(RECHTSGEBIET_CATEGORIES)
        if rg in RECHTSGEBIET_CATEGORIES:
            rg_vec[RECHTSGEBIET_CATEGORIES.index(rg)] = 1.0
        else:
            # Default to "Sonstiges" (last element)
            rg_vec[-1] = 1.0
        features.extend(rg_vec)

        # 9. All boolean fields from legal analysis schema
        for section_key, fields in LEGAL_ANALYSIS_BOOL_FIELDS.items():
            section = la.get(section_key, {})
            for field in fields:
                features.append(1.0 if section.get(field) else 0.0)

        # 10. List fields (count of items)
        for section_key, fields in LEGAL_ANALYSIS_LIST_FIELDS.items():
            section = la.get(section_key, {})
            for field in fields:
                val = section.get(field, [])
                features.append(float(len(val)) if isinstance(val, list) else 0.0)

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
    PyTorch Dataset for litigation cases (structured features only).

    Each item returns:
    - structured: tensor of shape (feature_dim,)
    - label: int (0, 1, 2)
    """

    def __init__(
        self,
        cases: list[dict],
        structured_features: np.ndarray,
    ):
        self.cases = cases
        self.structured_features = structured_features

        # Filter to cases that have outcome label
        self.valid_indices = [
            i for i, c in enumerate(cases)
            if c["structured"].get("outcome") is not None
        ]

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> tuple:
        case_idx = self.valid_indices[idx]
        case = self.cases[case_idx]

        # Structured features
        structured = torch.tensor(
            self.structured_features[case_idx], dtype=torch.float32
        )

        # Label
        label = int(case["structured"]["outcome"])

        return structured, label


def prepare_dataset(
    cases: list[dict],
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

    full_dataset = LitigationDataset(cases, structured_features)

    if len(full_dataset) == 0:
        raise ValueError("No valid labeled cases with legal analysis found.")

    # Stratified split
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
    """Collate for structured-only data."""
    structured_batch = torch.stack([item[0] for item in batch])
    labels_batch = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return structured_batch, labels_batch


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
