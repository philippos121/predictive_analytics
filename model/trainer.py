"""
Model Trainer: Training loop, validation, early stopping, and checkpointing
for the LitigationClassifier neural network.
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
    ENCODER_FILE,
    KNN_FILE,
    KNN_THRESHOLD,
    MODEL_CHECKPOINT,
    NN_CONFIG,
    NN_CONFIG_MEDIUM,
    NN_CONFIG_LARGE,
    NN_SMALL_THRESHOLD,
    NN_MEDIUM_THRESHOLD,
    SCALER_FILE,
    TRAINING_CONFIG,
    TRAINING_CONFIG_MEDIUM,
    TRAINING_CONFIG_LARGE,
    TRAINING_HISTORY_FILE,
)
from model.feature_engineer import (
    FeatureEngineer,
    LitigationDataset,
    collate_fn,
    compute_class_weights,
    prepare_dataset,
)
from model.knn_predictor import KNNLitigationPredictor
from model.neural_net import FocalLoss, LitigationClassifier


class EarlyStopping:
    def __init__(self, patience: int = 30, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
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
        """Prepare datasets for training."""
        train_ds, val_ds, full_ds = prepare_dataset(
            cases,
            embeddings_dict,
            self.feature_engineer,
            val_split=self.config["val_split"],
            random_seed=self.config["random_seed"],
        )

        # We use feature_dim from the feature engineer
        feature_dim = self.feature_engineer.feature_dim

        return train_ds, val_ds, full_ds, feature_dim

    def build_model(
        self,
        structured_dim: int,
        nn_config: Optional[dict] = None,
    ) -> LitigationClassifier:
        """Initialize the neural network with the given (or default) config."""
        kwargs = {"structured_dim": structured_dim}
        if nn_config is not None:
            kwargs["config"] = nn_config
        model = LitigationClassifier(**kwargs)
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
            and c["case_id"] in embeddings_dict
        )
        if n_labeled < KNN_THRESHOLD:
            return self._train_knn(cases, embeddings_dict, save_checkpoint)

        # Neural network path — clear any previous kNN
        self.knn = None

        # ── Tier-adaptive architecture + training config ──────────────────────────
        # Pick NN architecture and training hyper-parameters based on dataset size.
        # User-supplied overrides in self.config (epochs / lr / early_stop from UI)
        # take precedence over tier defaults for those three keys.
        if n_labeled < NN_SMALL_THRESHOLD:
            active_nn_config = NN_CONFIG
            tier_config = TRAINING_CONFIG
            config_tier = "klein"
            # FocalLoss γ=2.0: focus on hard examples in small datasets
            _focal_gamma = 2.0
            _label_smoothing = 0.0
        elif n_labeled < NN_MEDIUM_THRESHOLD:
            active_nn_config = NN_CONFIG_MEDIUM
            tier_config = TRAINING_CONFIG_MEDIUM
            config_tier = "mittel"
            _focal_gamma = 1.5
            _label_smoothing = 0.05
        else:
            active_nn_config = NN_CONFIG_LARGE
            tier_config = TRAINING_CONFIG_LARGE
            config_tier = "groß"
            # CE with label smoothing is more stable than Focal for balanced 3-class
            _focal_gamma = 0.0
            _label_smoothing = 0.10

        # Merge: tier defaults are the base; user UI values override epochs/lr/patience
        USER_KEYS = {"epochs", "learning_rate", "early_stopping_patience"}
        effective_config = {
            **tier_config,
            **{k: v for k, v in self.config.items() if k in USER_KEYS},
            # preserve non-overrideable keys that may differ between tiers
        }

        self._log(
            phase="tier_selected",
            config_tier=config_tier,
            n_labeled=n_labeled,
            dropout_emb=active_nn_config["dropout_embedding"],
            weight_decay=tier_config["weight_decay"],
        )

        start_time = time.time()

        # ── Data Preparation ─────────────────────────────────────────────────────
        self._log(phase="preparing", message="Daten werden vorbereitet...")

        train_ds, val_ds, full_ds, feature_dim = self.prepare_data(cases, embeddings_dict)

        self._log(
            phase="prepared",
            train_size=len(train_ds),
            val_size=len(val_ds),
            feature_dim=feature_dim,
        )

        if len(train_ds) < 2:
            raise ValueError(
                f"Zu wenige Trainingsdaten ({len(train_ds)} Fälle). "
                "Mindestens 5 Fälle mit Outcome-Label erforderlich."
            )

        batch_size = min(effective_config["batch_size"], len(train_ds))

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
        model = self.build_model(feature_dim, nn_config=active_nn_config)

        self._log(
            phase="model_built",
            parameters=model.count_parameters(),
            device=str(self.device),
        )

        # Adaptive loss: Focal for small/medium, label-smoothed CE for large
        class_weights = compute_class_weights(cases).to(self.device)
        if _focal_gamma > 0:
            criterion = FocalLoss(gamma=_focal_gamma, alpha=class_weights)
        else:
            criterion = nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=_label_smoothing,
            )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=effective_config["learning_rate"],
            weight_decay=tier_config["weight_decay"],
        )

        # Medium / Large → CosineAnnealingLR for better loss-landscape exploration.
        # Small → ReduceLROnPlateau (more conservative, reacts to val loss).
        use_cosine = config_tier in ("mittel", "groß")
        if use_cosine:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=effective_config["epochs"],
                eta_min=1e-7,
            )
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=effective_config["lr_scheduler_factor"],
                patience=effective_config["lr_scheduler_patience"],
                verbose=False,
            )

        # Large tier: Stochastic Weight Averaging over the last 20 % of epochs.
        # SWA averages model snapshots to find a flatter, better-generalizing minimum.
        use_swa = config_tier == "groß"
        swa_start = max(1, int(effective_config["epochs"] * 0.80))
        swa_snapshots: list[dict] = []   # state_dicts collected after swa_start

        early_stopping = EarlyStopping(
            patience=effective_config["early_stopping_patience"]
        )

        best_val_acc = 0.0
        best_val_loss = float("inf")
        best_state_dict = None

        # ── Training Loop ────────────────────────────────────────────────────────
        for epoch in range(1, effective_config["epochs"] + 1):
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
                    model.parameters(), effective_config["gradient_clip"]
                )

                optimizer.step()

                train_loss += loss.item() * len(label_batch)
                preds = logits.argmax(dim=-1)
                train_correct += (preds == label_batch).sum().item()
                train_total += len(label_batch)

            avg_train_loss = train_loss / train_total
            train_acc = train_correct / train_total

            # Validate
            val_loss, val_acc = self._evaluate(model, val_loader, criterion)

            # LR schedule
            monitor_loss = val_loss if val_loader else avg_train_loss
            if use_cosine:
                scheduler.step()
            else:
                scheduler.step(monitor_loss)

            # SWA snapshot collection
            if use_swa and epoch >= swa_start:
                swa_snapshots.append(
                    {k: v.clone().cpu() for k, v in model.state_dict().items()}
                )

            # Track history
            self.history["train_loss"].append(avg_train_loss)
            self.history["val_loss"].append(val_loss if val_loader else avg_train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc if val_loader else train_acc)

            # Best model tracking
            metric = val_acc if val_loader else train_acc
            if metric > best_val_acc:
                best_val_acc = metric
                best_val_loss = monitor_loss
                best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
                self.history["best_val_acc"] = best_val_acc
                self.history["best_epoch"] = epoch

            # Progress callback
            if epoch % 5 == 0 or epoch == 1:
                self._log(
                    phase="training",
                    epoch=epoch,
                    total_epochs=effective_config["epochs"],
                    train_loss=avg_train_loss,
                    val_loss=val_loss,
                    train_acc=train_acc,
                    val_acc=val_acc,
                    best_val_acc=best_val_acc,
                    lr=optimizer.param_groups[0]["lr"],
                )

            # Early stopping (only outside the SWA accumulation window)
            if not (use_swa and epoch >= swa_start):
                if early_stopping(monitor_loss):
                    self._log(
                        phase="early_stop",
                        epoch=epoch,
                        message=f"Early stopping at epoch {epoch}",
                    )
                    break

        # ── SWA: average snapshots and re-evaluate ────────────────────────────────
        if use_swa and swa_snapshots:
            avg_state: dict = {}
            for key in swa_snapshots[0]:
                avg_state[key] = torch.stack(
                    [s[key].float() for s in swa_snapshots]
                ).mean(0).to(self.device)
            model.load_state_dict(avg_state)

            swa_val_loss, swa_val_acc = self._evaluate(model, val_loader, criterion)
            self._log(
                phase="swa_done",
                n_snapshots=len(swa_snapshots),
                swa_val_acc=swa_val_acc,
                prev_best_val_acc=best_val_acc,
            )
            if swa_val_acc >= best_val_acc:
                best_val_acc = swa_val_acc
                best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
                self.history["best_val_acc"] = best_val_acc
                self.history["best_epoch"] = epoch

        # ── Finalize ─────────────────────────────────────────────────────────────
        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)

        self.history["training_time_sec"] = time.time() - start_time
        self.history["epochs_trained"] = epoch

        if save_checkpoint:
            self.save_checkpoint()

        self._log(
            phase="done",
            best_val_acc=best_val_acc,
            epochs_trained=epoch,
            training_time=self.history["training_time_sec"],
        )

        return self.history

    def _train_knn(
        self,
        cases: list[dict],
        embeddings_dict: dict,
        save_checkpoint: bool,
    ) -> dict:
        """Train kNN model for small datasets (< KNN_THRESHOLD labeled cases)."""
        import time as _time
        start_time = _time.time()

        self._log(
            phase="preparing",
            message=f"kNN-Modus (< {KNN_THRESHOLD} Fälle) — Daten werden vorbereitet...",
        )

        # Fit scaler on structured features (needed for inference path)
        self.feature_engineer.fit_transform(cases)
        feature_dim = self.feature_engineer.feature_dim

        labeled = [
            c for c in cases
            if c["structured"].get("outcome") is not None
            and c["case_id"] in embeddings_dict
        ]
        n = len(labeled)

        self._log(
            phase="prepared",
            train_size=n,
            val_size=0,
            feature_dim=feature_dim,
        )

        if n < 1:
            raise ValueError("Keine gültigen Trainingsfälle mit Outcome-Label und Embeddings gefunden.")

        k = min(KNNLitigationPredictor.DEFAULT_K, n)
        self.knn = KNNLitigationPredictor(k=k)
        self.knn.fit(cases, embeddings_dict)
        self.model = None

        self._log(phase="knn_fitted", n_cases=n, k=k)

        elapsed = _time.time() - start_time
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
    ) -> tuple[float, float]:
        """Evaluate model on validation set."""
        if loader is None or len(loader.dataset) == 0:
            return 0.0, 0.0

        model.eval()
        total_loss, correct, total = 0.0, 0, 0

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

        return (total_loss / total, correct / total) if total > 0 else (0.0, 0.0)

    def evaluate_full(
        self, cases: list[dict], embeddings_dict: dict
    ) -> dict:
        """
        Full evaluation on the complete dataset.
        Returns per-class metrics.
        """
        if self.knn is not None:
            return self._evaluate_full_knn(cases, embeddings_dict)

        if self.model is None:
            raise RuntimeError("Model not trained/loaded.")

        structured_features = self.feature_engineer.transform(cases)
        full_ds = LitigationDataset(cases, embeddings_dict, structured_features)
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

        # Per-class metrics
        per_class = {}
        for cls in range(3):
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

    def _evaluate_full_knn(self, cases: list[dict], embeddings_dict: dict) -> dict:
        """Evaluate kNN predictor on the full labeled dataset."""
        all_preds, all_labels, all_probs = [], [], []
        for case in cases:
            case_id = case["case_id"]
            outcome = case["structured"].get("outcome")
            if outcome is None or case_id not in embeddings_dict:
                continue
            result = self.knn.predict(embeddings_dict[case_id])
            all_preds.append(result["predicted_outcome"])
            all_labels.append(int(outcome))
            all_probs.append([result["p_loss"], result["p_partial"], result["p_win"]])

        if not all_preds:
            return {"accuracy": 0.0, "per_class": {}, "predictions": [], "labels": [], "probabilities": []}

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        per_class = {}
        for cls in range(3):
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
                    "structured_dim": self.model.structured_encoder.encoder[0].in_features,
                    "history": self.history,
                },
                MODEL_CHECKPOINT,
            )
        else:
            return

        self.feature_engineer.save(SCALER_FILE)

        with open(TRAINING_HISTORY_FILE, "w") as f:
            json.dump(self.history, f, indent=2)

    def load_checkpoint(self) -> bool:
        """Load model (kNN or NN) and feature engineer from checkpoint.

        Returns False (and deletes the checkpoint) if the saved architecture is
        incompatible with the current model configuration, e.g. when the number
        of embedding sections changed (5 → 3).  The user will then be asked to
        retrain the model.
        """
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
        structured_dim = checkpoint["structured_dim"]
        self.model = LitigationClassifier(
            structured_dim=structured_dim,
            config=checkpoint.get("model_config", {}),
        )

        try:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        except RuntimeError:
            # Checkpoint was trained with a different architecture (e.g. different
            # number of embedding sections).  Remove it so the UI shows a clean
            # "no model trained yet" state instead of crashing.
            self.model = None
            MODEL_CHECKPOINT.unlink(missing_ok=True)
            return False

        self.model = self.model.to(self.device)
        self.model.eval()
        self.knn = None
        self.history = checkpoint.get("history", {})

        if SCALER_FILE.exists():
            self.feature_engineer.load(SCALER_FILE)

        return True

    def load_history(self) -> dict:
        """Load training history from file."""
        if TRAINING_HISTORY_FILE.exists():
            with open(TRAINING_HISTORY_FILE) as f:
                return json.load(f)
        return {}
