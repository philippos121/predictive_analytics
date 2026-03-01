"""
Linear Probe Baseline for Predictive Litigation Analytics.

Fits a Logistic Regression directly on the concatenated raw OpenAI embeddings
(all sections concatenated) to establish the maximum accuracy achievable by a
purely linear classifier over the embedding space.

Interpretation:
  If MLP val acc ≈ linear probe val acc → the MLP adds no generalisation over
  linear separation; the bottleneck is the signal in the embeddings, not the
  architecture complexity.

  If MLP val acc >> linear probe val acc → the MLP is genuinely learning
  nonlinear structure and is worth the added complexity.

  If linear probe val acc ≈ majority-class baseline → the embedding space
  contains little usable signal for this task regardless of architecture.

Three L2 strengths are evaluated:
  C=0.01  (strong L2,   λ=100) — heavily regularised
  C=0.1   (moderate L2, λ=10)
  C=1.0   (sklearn default,  λ=1)
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import EMBEDDING_DIM, EMBEDDING_SECTIONS, TRAINING_CONFIG


def run_linear_probe(
    cases: list[dict],
    embeddings_dict: dict,
    val_split: float = TRAINING_CONFIG["val_split"],
    random_seed: int = TRAINING_CONFIG["random_seed"],
) -> dict:
    """
    Fit Logistic Regression baselines on concatenated raw embeddings.

    Uses the same stratified train/val split as prepare_dataset so results are
    directly comparable to the NN val accuracy.

    Args:
        cases: list of case dicts with 'case_id' and 'structured.outcome'
        embeddings_dict: {case_id: {section_name: np.ndarray}}
        val_split: fraction of data held out for validation (default 0.20)
        random_seed: for reproducibility

    Returns:
        dict with keys:
            n_train:               int
            n_val:                 int
            majority_baseline_val: float  — accuracy of always predicting the most common class
            results:               list of {C, train_acc, val_acc}
          OR
            error: str            — if fewer than 10 valid cases found
    """
    # Collect valid cases (have label + embeddings)
    labeled = [
        c for c in cases
        if c["structured"].get("outcome") is not None
        and c["case_id"] in embeddings_dict
    ]
    if len(labeled) < 10:
        return {"error": f"Too few labeled cases for linear probe ({len(labeled)})"}

    # Build feature matrix: concatenate all embedding sections
    X_rows, y = [], []
    for c in labeled:
        embs = embeddings_dict[c["case_id"]]
        row = np.concatenate([
            embs.get(sec, np.zeros(EMBEDDING_DIM, dtype=np.float32)).astype(np.float32)
            for sec in EMBEDDING_SECTIONS
        ])
        X_rows.append(row)
        y.append(int(c["structured"]["outcome"]))

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    # Stratified split — same logic as feature_engineer.prepare_dataset
    rng = np.random.default_rng(random_seed)
    label_to_indices: dict[int, list[int]] = {0: [], 1: [], 2: []}
    for i, lbl in enumerate(y):
        label_to_indices[int(lbl)].append(i)

    train_idx, val_idx = [], []
    for lbl, idxs in label_to_indices.items():
        arr = np.array(idxs)
        rng.shuffle(arr)
        n_val = max(1, int(len(arr) * val_split)) if len(arr) > 1 else 0
        val_idx.extend(arr[:n_val].tolist())
        train_idx.extend(arr[n_val:].tolist())

    X_train, y_train = X[train_idx], y[train_idx]
    X_val,   y_val   = X[val_idx],   y[val_idx]

    # StandardScaler: OpenAI embeddings are unit-norm per vector but
    # different sections may have different scale distributions.
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)

    # Majority-class baseline on val set
    majority_class = int(np.bincount(y_val).argmax())
    majority_baseline = float((y_val == majority_class).mean())

    results = []
    for C in [0.01, 0.1, 1.0]:
        clf = LogisticRegression(
            C=C,
            max_iter=2000,
            solver="lbfgs",
            multi_class="multinomial",
            random_state=random_seed,
            n_jobs=-1,
        )
        clf.fit(X_train_s, y_train)
        results.append({
            "C": C,
            "train_acc": float(clf.score(X_train_s, y_train)),
            "val_acc":   float(clf.score(X_val_s,   y_val)),
        })

    return {
        "n_train": len(train_idx),
        "n_val":   len(val_idx),
        "majority_baseline_val": majority_baseline,
        "results": results,
    }
