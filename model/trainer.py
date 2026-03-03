"""
Model Trainer: Training loop, validation, early stopping, and checkpointing
for the LitigationClassifier neural network.

v4.0 — Cross-Attention Hybrid Architecture (100 K+ cases, no PCA by default).
"""

import json
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    KNN_FILE,
    KNN_THRESHOLD,
    MODEL_CHECKPOINT,
    NUM_CLASSES,
    PCA_FILE,
    SCALER_FILE,
    TRAINING_CONFIG,
    TRAINING_HISTORY_FILE,
    USE_PCA,
)
from model.feature_engineer import (
    EmbeddingPCA,
    FeatureEngineer,
    LitigationDataset,
    collate_fn,
    compute_class_weights,
    prepare_dataset,
)
from model.knn_predictor import KNNLitigationPredictor
from model.neural_net import LitigationClassifier


class EarlyStopping:
    def __init__(self, patience: int = 30, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = -float("inf")
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class LitigationTrainer:
    """
    Manages the full training lifecycle of the LitigationClassifier.
    Supports progress callbacks for Streamlit UI integration.
    """

    def __init__(
        self,
        config: dict = TRAINING_CONFIG,
        device: Optional[str] = None,
        progress_callback: Optional[Callable] = None,
    ):
        self.config = config
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.progress_callback = progress_callback or (lambda **kw: None)

        self.model: Optional[LitigationClassifier] = None
        self.knn: Optional[KNNLitigationPredictor] = None
        self.feature_engineer = FeatureEngineer()
        self.embedding_pca = EmbeddingPCA()
        self.history: dict = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "best_val_acc": 0.0,
            "best_epoch": 0,
            "training_time_sec": 0,
        }

    def _log(self, **kwargs):
        """Send progress update to callback."""
        self.progress_callback(**kwargs)

    def prepare_data(
        self,
        cases: list[dict],
        embeddings_dict: dict,
    ) -> tuple:
        """Prepare datasets for training (optionally fits PCA on embeddings)."""
        pca = None
        if USE_PCA:
            self.embedding_pca.fit(embeddings_dict, cases)
            pca = self.embedding_pca

        train_ds, val_ds, full_ds = prepare_dataset(
            cases,
            embeddings_dict,
            self.feature_engineer,
            val_split=self.config["val_split"],
            random_seed=self.config["random_seed"],
            pca=pca,
        )

        feature_dim = self.feature_engineer.feature_dim
        return train_ds, val_ds, full_ds, feature_dim

    def build_model(self, structured_dim: int) -> LitigationClassifier:
        """Initialize the neural network."""
        model = LitigationClassifier(structured_dim=structured_dim)
        model = model.to(self.device)
        self.model = model
        return model

    def train(
        self,
        cases: list[dict],
        embeddings_dict: dict,
        save_checkpoint: bool = True,
    ) -> dict:
        """
        Full training pipeline.

        Args:
            cases: List of case dicts (labeled)
            embeddings_dict: {case_id: {section: np.ndarray}}
            save_checkpoint: Whether to save best model

        Returns:
            Training history dict
        """
        torch.manual_seed(self.config["random_seed"])
        np.random.seed(self.config["random_seed"])

        # ── Adaptive model selection ──────────────────────────────────────────────
        n_labeled = sum(
            1 for c in cases
            if c["structured"].get("outcome") is not None
        )
        if n_labeled < KNN_THRESHOLD:
            return self._train_knn(cases, save_checkpoint)

        # Neural network path
        self.knn = None

        start_time = time.time()

        # ── Data Preparation ─────────────────────────────────────────────────────
        self._log(phase="preparing", message="Daten werden vorbereitet...")

        train_ds, val_ds, full_ds, feature_dim = self.prepare_data(
            cases, embeddings_dict
        )

        n_with_emb = len(full_ds)

        log_kwargs = {
            "phase": "prepared",
            "train_size": len(train_ds),
            "val_size": len(val_ds),
            "feature_dim": feature_dim,
            "n_with_embeddings": n_with_emb,
        }
        if USE_PCA and self.embedding_pca.is_fitted:
            log_kwargs["pca_variance_retained"] = self.embedding_pca.explained_variance_ratio_sum
        self._log(**log_kwargs)

        if len(train_ds) < 2:
            raise ValueError(
                f"Zu wenige Trainingsdaten ({len(train_ds)} Fälle). "
                "Mindestens 5 Fälle mit Outcome-Label und Embeddings erforderlich."
            )

        batch_size = min(self.config["batch_size"], len(train_ds))

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        ) if len(val_ds) > 0 else None

        # ── Model & Optimizer ────────────────────────────────────────────────────
        model = self.build_model(feature_dim)

        self._log(
            phase="model_built",
            parameters=model.count_parameters(),
            device=str(self.device),
        )

        # Class-weighted CrossEntropy
        class_weights = compute_class_weights(cases).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=self.config["lr_scheduler_factor"],
            patience=self.config["lr_scheduler_patience"],
            verbose=False,
        )

        early_stopping = EarlyStopping(
            patience=self.config["early_stopping_patience"]
        )

        best_val_acc = 0.0
        best_state_dict = None

        # ── Training Loop ────────────────────────────────────────────────────────
        for epoch in range(1, self.config["epochs"] + 1):
            # Train
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0

            for emb_batch, struct_batch, label_batch in train_loader:
                emb_batch = [e.to(self.device) for e in emb_batch]
                struct_batch = struct_batch.to(self.device)
                label_batch = label_batch.to(self.device)

                optimizer.zero_grad()
                logits, _ = model(emb_batch, struct_batch)
                loss = criterion(logits, label_batch)
                loss.backward()

                nn.utils.clip_grad_norm_(
                    model.parameters(), self.config["gradient_clip"]
                )

                optimizer.step()

                train_loss += loss.item() * len(label_batch)
                preds = logits.argmax(dim=-1)
                train_correct += (preds == label_batch).sum().item()
                train_total += len(label_batch)

            avg_train_loss = train_loss / train_total
            train_acc = train_correct / train_total

            # Validate
            val_loss, val_acc, val_pred_dist = self._evaluate(
                model, val_loader, criterion, return_pred_dist=True
            )

            # Track history
            self.history["train_loss"].append(avg_train_loss)
            self.history["val_loss"].append(val_loss if val_loader else avg_train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc if val_loader else train_acc)

            # Monitor val_acc for LR schedule + early stopping
            monitor_metric = val_acc if val_loader else train_acc
            scheduler.step(monitor_metric)

            # Best model tracking
            if monitor_metric > best_val_acc:
                best_val_acc = monitor_metric
                best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
                self.history["best_val_acc"] = best_val_acc
                self.history["best_epoch"] = epoch

            # Progress callback
            if epoch % 5 == 0 or epoch == 1:
                self._log(
                    phase="training",
                    epoch=epoch,
                    total_epochs=self.config["epochs"],
                    train_loss=avg_train_loss,
                    val_loss=val_loss,
                    train_acc=train_acc,
                    val_acc=val_acc,
                    best_val_acc=best_val_acc,
                    lr=optimizer.param_groups[0]["lr"],
                    val_pred_dist=val_pred_dist,
                )

            # Early stopping
            if early_stopping(monitor_metric):
                self._log(
                    phase="early_stop",
                    epoch=epoch,
                    message=f"Early stopping at epoch {epoch}",
                )
                break

        # ── Finalize ─────────────────────────────────────────────────────────────
        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)

        self.history["training_time_sec"] = time.time() - start_time
        self.history["epochs_trained"] = epoch

        # ── Post-training calibration ─────────────────────────────────────────
        self._log(phase="calibrating", message="Kalibrierungs-Diagnostik wird berechnet...")
        try:
            calib = self.compute_calibration_diagnostics(cases, embeddings_dict)
            self.history["calibration"] = {
                "brier_skill_score": calib.get("brier_skill_score", 0.0),
                "brier_score_model": calib.get("brier_score_model"),
                "brier_score_baseline": calib.get("brier_score_baseline"),
                "accuracy_above_baseline": calib.get("accuracy_above_baseline", 0.0),
                "ml_adds_signal": calib.get("ml_adds_signal", False),
                "majority_class_accuracy": calib.get("majority_class_accuracy", 0.5),
            }
        except Exception:
            self.history["calibration"] = {
                "brier_skill_score": 0.0,
                "ml_adds_signal": False,
            }

        if save_checkpoint:
            self.save_checkpoint()

        self._log(
            phase="done",
            best_val_acc=best_val_acc,
            epochs_trained=epoch,
            training_time=self.history["training_time_sec"],
            brier_skill_score=self.history["calibration"]["brier_skill_score"],
        )

        return self.history

    def _train_knn(
        self,
        cases: list[dict],
        save_checkpoint: bool,
    ) -> dict:
        """Train kNN model for small datasets (< KNN_THRESHOLD labeled cases)."""
        start_time = time.time()

        self._log(
            phase="preparing",
            message=f"kNN-Modus (< {KNN_THRESHOLD} Fälle) — Daten werden vorbereitet...",
        )

        self.feature_engineer.fit_transform(cases)
        feature_dim = self.feature_engineer.feature_dim

        labeled = [
            c for c in cases
            if c["structured"].get("outcome") is not None
        ]
        n = len(labeled)

        self._log(
            phase="prepared",
            train_size=n,
            val_size=0,
            feature_dim=feature_dim,
        )

        if n < 1:
            raise ValueError("Keine gültigen Trainingsfälle mit Outcome-Label gefunden.")

        k = min(KNNLitigationPredictor.DEFAULT_K, n)
        self.knn = KNNLitigationPredictor(k=k)
        self.knn.fit(cases, self.feature_engineer)
        self.model = None

        self._log(phase="knn_fitted", n_cases=n, k=k)

        elapsed = time.time() - start_time
        self.history = {
            "model_type": "knn",
            "n_training_cases": n,
            "best_val_acc": 0.0,
            "best_epoch": 0,
            "training_time_sec": elapsed,
            "epochs_trained": 0,
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

        if save_checkpoint:
            self.save_checkpoint()

        self._log(
            phase="done",
            best_val_acc=0.0,
            epochs_trained=0,
            training_time=elapsed,
            model_type="knn",
            n_cases=n,
        )

        return self.history

    def _evaluate(
        self,
        model: LitigationClassifier,
        loader: Optional[DataLoader],
        criterion: nn.Module,
        return_pred_dist: bool = False,
    ) -> tuple[float, float] | tuple[float, float, dict]:
        """Evaluate model on validation set."""
        empty = (0.0, 0.0, {}) if return_pred_dist else (0.0, 0.0)
        if loader is None or len(loader.dataset) == 0:
            return empty

        model.eval()
        total_loss, correct, total = 0.0, 0, 0
        pred_counts = [0] * NUM_CLASSES

        with torch.no_grad():
            for emb_batch, struct_batch, label_batch in loader:
                emb_batch = [e.to(self.device) for e in emb_batch]
                struct_batch = struct_batch.to(self.device)
                label_batch = label_batch.to(self.device)

                logits, _ = model(emb_batch, struct_batch)
                loss = criterion(logits, label_batch)

                total_loss += loss.item() * len(label_batch)
                preds = logits.argmax(dim=-1)
                correct += (preds == label_batch).sum().item()
                total += len(label_batch)

                for cls in range(NUM_CLASSES):
                    pred_counts[cls] += (preds == cls).sum().item()

        if total == 0:
            return empty

        result_loss = total_loss / total
        result_acc = correct / total

        if return_pred_dist:
            dist = {cls: pred_counts[cls] / total for cls in range(NUM_CLASSES)}
            return result_loss, result_acc, dist
        return result_loss, result_acc

    def evaluate_full(self, cases: list[dict], embeddings_dict: dict = None) -> dict:
        """
        Full evaluation on the complete dataset.
        Returns per-class metrics.
        """
        if self.knn is not None:
            return self._evaluate_full_knn(cases)

        if self.model is None:
            raise RuntimeError("Model not trained/loaded.")

        if embeddings_dict is None:
            raise ValueError("embeddings_dict required for neural net evaluation.")

        structured_features = self.feature_engineer.transform(cases)
        pca = self.embedding_pca if (USE_PCA and self.embedding_pca.is_fitted) else None
        full_ds = LitigationDataset(cases, embeddings_dict, structured_features, pca=pca)
        loader = DataLoader(
            full_ds, batch_size=32, shuffle=False, collate_fn=collate_fn
        )

        self.model.eval()
        all_preds, all_labels, all_probs = [], [], []

        with torch.no_grad():
            for emb_batch, struct_batch, label_batch in loader:
                emb_batch = [e.to(self.device) for e in emb_batch]
                struct_batch = struct_batch.to(self.device)

                logits, probs = self.model(emb_batch, struct_batch)
                preds = logits.argmax(dim=-1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(label_batch.numpy())
                all_probs.extend(probs.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        per_class = {}
        for cls in range(NUM_CLASSES):
            tp = ((all_preds == cls) & (all_labels == cls)).sum()
            fp = ((all_preds == cls) & (all_labels != cls)).sum()
            fn = ((all_preds != cls) & (all_labels == cls)).sum()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            per_class[cls] = {"precision": precision, "recall": recall, "f1": f1}

        overall_acc = (all_preds == all_labels).mean()

        return {
            "accuracy": float(overall_acc),
            "per_class": per_class,
            "predictions": all_preds.tolist(),
            "labels": all_labels.tolist(),
            "probabilities": all_probs.tolist(),
        }

    def _evaluate_full_knn(self, cases: list[dict]) -> dict:
        """Evaluate kNN predictor on the full labeled dataset."""
        all_preds, all_labels, all_probs = [], [], []
        for case in cases:
            outcome = case["structured"].get("outcome")
            if outcome is None:
                continue
            result = self.knn.predict_case(case)
            all_preds.append(result["predicted_outcome"])
            all_labels.append(int(outcome))
            all_probs.append([result["p_loss"], result["p_partial"], result["p_win"]])

        if not all_preds:
            return {"accuracy": 0.0, "per_class": {}, "predictions": [], "labels": [], "probabilities": []}

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        per_class = {}
        for cls in range(NUM_CLASSES):
            tp = int(((all_preds == cls) & (all_labels == cls)).sum())
            fp = int(((all_preds == cls) & (all_labels != cls)).sum())
            fn = int(((all_preds != cls) & (all_labels == cls)).sum())
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 else 0.0)
            per_class[cls] = {"precision": precision, "recall": recall, "f1": f1}

        return {
            "accuracy": float((all_preds == all_labels).mean()),
            "per_class": per_class,
            "predictions": all_preds.tolist(),
            "labels": all_labels.tolist(),
            "probabilities": all_probs.tolist(),
        }

    def save_checkpoint(self) -> None:
        """Save model (kNN or NN), feature engineer, and training history."""
        MODEL_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)

        if self.knn is not None:
            torch.save(
                {"model_type": "knn", "history": self.history},
                MODEL_CHECKPOINT,
            )
            self.knn.save(KNN_FILE)
        elif self.model is not None:
            torch.save(
                {
                    "model_type": "neural_net",
                    "model_state_dict": self.model.state_dict(),
                    "model_config": self.model.config,
                    "structured_dim": self.model.structured_dim,
                    "history": self.history,
                },
                MODEL_CHECKPOINT,
            )
        else:
            return

        self.feature_engineer.save(SCALER_FILE)

        if USE_PCA and self.embedding_pca.is_fitted:
            self.embedding_pca.save(PCA_FILE)

        with open(TRAINING_HISTORY_FILE, "w") as f:
            json.dump(self.history, f, indent=2)

    def load_checkpoint(self) -> bool:
        """Load model (kNN or NN) and feature engineer from checkpoint."""
        if not MODEL_CHECKPOINT.exists():
            return False

        checkpoint = torch.load(MODEL_CHECKPOINT, map_location=self.device)
        model_type = checkpoint.get("model_type", "neural_net")

        # ── kNN checkpoint ────────────────────────────────────────────────────
        if model_type == "knn":
            if not KNN_FILE.exists():
                return False
            self.knn = KNNLitigationPredictor()
            self.knn.load(KNN_FILE)
            self.model = None
            self.history = checkpoint.get("history", {})
            if SCALER_FILE.exists():
                self.feature_engineer.load(SCALER_FILE)
            return True

        # ── Neural network checkpoint ─────────────────────────────────────────
        structured_dim = checkpoint.get("structured_dim")
        if structured_dim is None:
            self.model = None
            MODEL_CHECKPOINT.unlink(missing_ok=True)
            return False

        self.model = LitigationClassifier(
            structured_dim=structured_dim,
            config=checkpoint.get("model_config", {}),
        )

        try:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        except RuntimeError:
            self.model = None
            MODEL_CHECKPOINT.unlink(missing_ok=True)
            return False

        self.model = self.model.to(self.device)
        self.model.eval()
        self.knn = None
        self.history = checkpoint.get("history", {})

        if SCALER_FILE.exists():
            self.feature_engineer.load(SCALER_FILE)

        if USE_PCA and PCA_FILE.exists():
            self.embedding_pca.load(PCA_FILE)

        return True

    def compute_calibration_diagnostics(
        self,
        cases: list[dict],
        embeddings_dict: dict = None,
    ) -> dict:
        """Measure whether the ML model adds signal beyond a flat baseline.

        Returns calibration metrics that show if the model's probability
        estimates are useful.  Key metric: **Brier skill score** — positive
        means the model is better-calibrated than always predicting the
        base rate, negative means it's *worse* than guessing the prior.

        Also returns log-loss improvement over the naive baseline.
        """
        eval_result = self.evaluate_full(cases, embeddings_dict)
        if not eval_result.get("probabilities"):
            return {"error": "Keine Vorhersagen verfügbar."}

        from config import map_outcome_label

        probs = np.array(eval_result["probabilities"])   # (N, NUM_CLASSES)
        labels = np.array(eval_result["labels"])          # (N,)  already mapped
        n = len(labels)
        n_cls = NUM_CLASSES

        # --- Base-rate (naive) baseline: always predict class distribution ---
        base_rate = np.zeros(n_cls)
        for c in range(n_cls):
            base_rate[c] = (labels == c).sum() / n

        # --- Brier score (lower is better) ---
        #   BS = mean( sum_c (p_c - y_c)^2 )
        one_hot = np.zeros_like(probs)
        for i in range(n):
            one_hot[i, labels[i]] = 1.0

        brier_model = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
        brier_baseline = float(np.mean(np.sum((base_rate - one_hot) ** 2, axis=1)))

        # Brier skill score: 1 = perfect, 0 = same as baseline, <0 = worse
        brier_skill = 1.0 - brier_model / brier_baseline if brier_baseline > 0 else 0.0

        # --- Log-loss ---
        eps = 1e-15
        probs_clipped = np.clip(probs, eps, 1 - eps)
        base_clipped = np.clip(base_rate, eps, 1 - eps)

        logloss_model = -float(np.mean(
            np.log(probs_clipped[np.arange(n), labels])
        ))
        logloss_baseline = -float(np.mean(
            np.log(base_clipped[labels])
        ))
        logloss_improvement = logloss_baseline - logloss_model

        # --- Per-class accuracy ---
        preds = np.array(eval_result["predictions"])
        per_class_acc = {}
        for c in range(n_cls):
            mask = labels == c
            if mask.sum() > 0:
                per_class_acc[c] = float((preds[mask] == c).mean())

        return {
            "n_samples": n,
            "base_rate": {int(c): float(base_rate[c]) for c in range(n_cls)},
            "overall_accuracy": eval_result["accuracy"],
            "majority_class_accuracy": float(base_rate.max()),
            "accuracy_above_baseline": eval_result["accuracy"] - float(base_rate.max()),
            "brier_score_model": brier_model,
            "brier_score_baseline": brier_baseline,
            "brier_skill_score": brier_skill,
            "logloss_model": logloss_model,
            "logloss_baseline": logloss_baseline,
            "logloss_improvement": logloss_improvement,
            "per_class_accuracy": per_class_acc,
            "ml_adds_signal": brier_skill > 0.0,
            "interpretation": (
                "ML-Modell liefert bessere Wahrscheinlichkeiten als Zufall"
                if brier_skill > 0.0
                else "ML-Modell ist nicht besser als die Basisrate — "
                     "nur juristische Einschätzung verwenden"
            ),
        }

    def load_history(self) -> dict:
        """Load training history from file."""
        if TRAINING_HISTORY_FILE.exists():
            with open(TRAINING_HISTORY_FILE) as f:
                return json.load(f)
        return {}
