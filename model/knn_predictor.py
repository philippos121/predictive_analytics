"""
k-Nearest-Neighbour predictor for small training sets (< KNN_THRESHOLD cases).

v2.0 — Uses structured feature vectors instead of embedding concatenation.
Cosine similarity on the encoded feature vector for outcome prediction.

Design:
  - Encode each case via FeatureEngineer → ~77-dim feature vector
  - L2-normalize for cosine similarity
  - k = min(5, n_training) nearest neighbours
  - Weighted soft-voting: weight = cosine similarity (clipped to [0, 1])
"""

import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import NUM_CLASSES, OUTCOME_LABELS, map_outcome_label


class KNNLitigationPredictor:
    """
    k-Nearest-Neighbour classifier for Austrian civil case outcome prediction.

    Designed for datasets with fewer than KNN_THRESHOLD labeled cases, where
    neural network training would overfit. Uses structured feature vectors
    from FeatureEngineer — no embeddings required.
    """

    DEFAULT_K = 5

    def __init__(self, k: int = DEFAULT_K):
        self.k = k
        self.train_features: Optional[np.ndarray] = None  # (N, feature_dim)
        self.train_labels: Optional[np.ndarray] = None      # (N,)
        self.n_classes = NUM_CLASSES
        self.is_fitted = False
        self._feature_engineer = None

    # ── Normalization ──────────────────────────────────────────────────────────

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        """L2-normalize rows (for cosine similarity via dot product)."""
        norms = np.linalg.norm(X, axis=-1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        return X / norms

    # ── Fit ──────────────────────────────────────────────────────────────────

    def fit(self, cases: list[dict], feature_engineer) -> None:
        """Index all labeled training cases using structured features."""
        self._feature_engineer = feature_engineer

        labeled = [
            c for c in cases
            if c["structured"].get("outcome") is not None
        ]

        if not labeled:
            raise ValueError("Keine gültigen Trainingsfälle für kNN gefunden.")

        # Encode features — use fitted feature selection + scaler
        features = feature_engineer.transform(labeled)

        labels = [map_outcome_label(int(c["structured"]["outcome"])) for c in labeled]

        self.train_features = self._normalize(features)
        self.train_labels = np.array(labels, dtype=np.int64)
        self.is_fitted = True

    # ── Predict ──────────────────────────────────────────────────────────────

    def predict_proba_from_features(self, features: np.ndarray) -> np.ndarray:
        """
        Return class probability vector [p_loss, p_partial, p_win]
        from a pre-encoded feature vector.
        """
        if not self.is_fitted:
            raise RuntimeError("KNN-Modell nicht trainiert.")

        vec = self._normalize(features.reshape(1, -1))  # (1, D)

        sims = (self.train_features @ vec.T).ravel()  # (N,)
        k = min(self.k, len(self.train_labels))
        top_idx = np.argsort(sims)[-k:][::-1]
        top_sims = np.clip(sims[top_idx], 0.0, 1.0)

        probs = np.zeros(self.n_classes, dtype=np.float32)
        weight_sum = float(top_sims.sum())

        if weight_sum < 1e-8:
            probs[:] = 1.0 / self.n_classes
        else:
            for sim, lbl in zip(top_sims, self.train_labels[top_idx]):
                probs[lbl] += sim
            probs /= weight_sum

        return probs

    def predict_case(self, case_dict: dict) -> dict:
        """Predict outcome for a single case dict (with structured + legal_analysis)."""
        if self._feature_engineer is None:
            raise RuntimeError("kNN not fitted (no feature engineer).")

        features = self._feature_engineer.encode_single_transform(case_dict)
        probs = self.predict_proba_from_features(features)
        predicted_class = int(probs.argmax())

        result = {
            "predicted_outcome": predicted_class,
            "predicted_label": OUTCOME_LABELS[predicted_class],
            "probabilities": {
                OUTCOME_LABELS[i]: float(probs[i]) for i in range(NUM_CLASSES)
            },
            "confidence": float(probs.max()),
        }

        if NUM_CLASSES == 2:
            result["p_win"] = float(probs[1])
            result["p_partial"] = 0.0
            result["p_loss"] = float(probs[0])
        else:
            result["p_win"] = float(probs[2])
            result["p_partial"] = float(probs[1])
            result["p_loss"] = float(probs[0])

        return result

    # ── Persistence ──────────────────────────────────────────────────────────

    def get_n_training_cases(self) -> int:
        return len(self.train_labels) if self.is_fitted else 0

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "train_features": self.train_features,
                    "train_labels": self.train_labels,
                    "k": self.k,
                    "feature_engineer": self._feature_engineer,
                },
                f,
            )

    def load(self, path: Path) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.train_features = data["train_features"]
        self.train_labels = data["train_labels"]
        self.k = data.get("k", self.DEFAULT_K)
        self._feature_engineer = data.get("feature_engineer")
        self.is_fitted = True
