"""
k-Nearest-Neighbour predictor for small training sets (< KNN_THRESHOLD cases).

Uses cosine similarity on concatenated section embeddings for outcome prediction.
No gradient-based training — just index the training cases and query at inference.

Design:
  - Concatenate 3 section embeddings: 3 × 3072 = 9216-dim feature vector
  - Normalize to unit length (cosine similarity = dot product)
  - k = min(5, n_training) nearest neighbours
  - Weighted soft-voting: weight = cosine similarity (clipped to [0, 1])
"""

import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import EMBEDDING_DIM, EMBEDDING_SECTIONS, OUTCOME_LABELS


class KNNLitigationPredictor:
    """
    k-Nearest-Neighbour classifier for Austrian civil case outcome prediction.

    Designed for datasets with fewer than KNN_THRESHOLD labeled cases, where
    neural network training would overfit. Uses pre-computed OpenAI embeddings
    directly — no additional training required.
    """

    DEFAULT_K = 5

    def __init__(self, k: int = DEFAULT_K):
        self.k = k
        self.train_embeddings: Optional[np.ndarray] = None  # (N, 3*EMBEDDING_DIM)
        self.train_labels: Optional[np.ndarray] = None       # (N,)
        self.n_classes = 3
        self.is_fitted = False

    # ── Feature Construction ──────────────────────────────────────────────────

    def _concat_embeddings(self, embeddings: dict) -> np.ndarray:
        """Concatenate all section embeddings into one 9216-dim vector."""
        parts = []
        for section in EMBEDDING_SECTIONS:
            vec = embeddings.get(section)
            if vec is not None:
                arr = np.asarray(vec, dtype=np.float32).ravel()
                if arr.shape[0] != EMBEDDING_DIM:
                    arr = np.zeros(EMBEDDING_DIM, dtype=np.float32)
            else:
                arr = np.zeros(EMBEDDING_DIM, dtype=np.float32)
            parts.append(arr)
        return np.concatenate(parts)  # (3 * EMBEDDING_DIM,)

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        """L2-normalize rows (for cosine similarity via dot product)."""
        norms = np.linalg.norm(X, axis=-1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        return X / norms

    # ── Fit ──────────────────────────────────────────────────────────────────

    def fit(self, cases: list[dict], embeddings_dict: dict) -> None:
        """Index all labeled training cases."""
        vecs, labels = [], []
        for case in cases:
            case_id = case["case_id"]
            outcome = case["structured"].get("outcome")
            if outcome is None or case_id not in embeddings_dict:
                continue
            vec = self._concat_embeddings(embeddings_dict[case_id])
            vecs.append(vec)
            labels.append(int(outcome))

        if not vecs:
            raise ValueError("Keine gültigen Trainingsfälle für kNN gefunden.")

        self.train_embeddings = self._normalize(np.stack(vecs))  # (N, D)
        self.train_labels = np.array(labels, dtype=np.int64)
        self.is_fitted = True

    # ── Predict ──────────────────────────────────────────────────────────────

    def predict_proba(self, embeddings: dict) -> np.ndarray:
        """
        Return class probability vector [p_loss, p_partial, p_win].

        Each of the k nearest neighbours contributes its class label with
        weight equal to its cosine similarity (clipped to [0, 1]).
        """
        if not self.is_fitted:
            raise RuntimeError("KNN-Modell nicht trainiert.")

        vec = self._normalize(
            self._concat_embeddings(embeddings).reshape(1, -1)
        )  # (1, D)

        sims = (self.train_embeddings @ vec.T).ravel()  # (N,)
        k = min(self.k, len(self.train_labels))
        top_idx = np.argsort(sims)[-k:][::-1]
        top_sims = np.clip(sims[top_idx], 0.0, 1.0)

        probs = np.zeros(self.n_classes, dtype=np.float32)
        weight_sum = float(top_sims.sum())

        if weight_sum < 1e-8:
            # Uniform fallback when all similarities are zero
            probs[:] = 1.0 / self.n_classes
        else:
            for sim, lbl in zip(top_sims, self.train_labels[top_idx]):
                probs[lbl] += sim
            probs /= weight_sum

        return probs

    def predict(self, embeddings: dict) -> dict:
        """Predict outcome dict for a single case (same format as NN predictor)."""
        probs = self.predict_proba(embeddings)
        predicted_class = int(probs.argmax())
        return {
            "predicted_outcome": predicted_class,
            "predicted_label": OUTCOME_LABELS[predicted_class],
            "probabilities": {
                OUTCOME_LABELS[i]: float(probs[i]) for i in range(3)
            },
            "confidence": float(probs.max()),
            "p_win": float(probs[2]),
            "p_partial": float(probs[1]),
            "p_loss": float(probs[0]),
        }

    # ── Persistence ──────────────────────────────────────────────────────────

    def get_n_training_cases(self) -> int:
        return len(self.train_labels) if self.is_fitted else 0

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "train_embeddings": self.train_embeddings,
                    "train_labels": self.train_labels,
                    "k": self.k,
                },
                f,
            )

    def load(self, path: Path) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.train_embeddings = data["train_embeddings"]
        self.train_labels = data["train_labels"]
        self.k = data.get("k", self.DEFAULT_K)
        self.is_fitted = True
