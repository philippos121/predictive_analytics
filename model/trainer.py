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
    MODEL_CHECKPOINT,
    SCALER_FILE,
    TRAINING_CONFIG,
    TRAINING_HISTORY_FILE,
)
from model.feature_engineer import (
    FeatureEngineer,
    LitigationDataset,
    collate_fn,
    compute_class_weights,
    prepare_dataset,
)
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

        # Class-weighted Focal Loss
        class_weights = compute_class_weights(cases).to(self.device)
        criterion = FocalLoss(gamma=2.0, alpha=class_weights)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.config["lr_scheduler_factor"],
            patience=self.config["lr_scheduler_patience"],
            verbose=False,
        )

        early_stopping = EarlyStopping(
            patience=self.config["early_stopping_patience"]
        )

        best_val_acc = 0.0
        best_val_loss = float("inf")
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

                # Gradient clipping
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
            val_loss, val_acc = self._evaluate(model, val_loader, criterion)

            # LR schedule
            monitor_loss = val_loss if val_loader else avg_train_loss
            scheduler.step(monitor_loss)

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
                    total_epochs=self.config["epochs"],
                    train_loss=avg_train_loss,
                    val_loss=val_loss,
                    train_acc=train_acc,
                    val_acc=val_acc,
                    best_val_acc=best_val_acc,
                    lr=optimizer.param_groups[0]["lr"],
                )

            # Early stopping
            if early_stopping(monitor_loss):
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

        if save_checkpoint:
            self.save_checkpoint()

        self._log(
            phase="done",
            best_val_acc=best_val_acc,
            epochs_trained=epoch,
            training_time=self.history["training_time_sec"],
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

    def save_checkpoint(self) -> None:
        """Save model, feature engineer, and training history."""
        if self.model is None:
            return

        MODEL_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "model_config": self.model.config,
                "structured_dim": self.model.structured_encoder.encoder[0].in_features,
                "history": self.history,
            },
            MODEL_CHECKPOINT,
        )

        self.feature_engineer.save(SCALER_FILE)

        with open(TRAINING_HISTORY_FILE, "w") as f:
            json.dump(self.history, f, indent=2)

    def load_checkpoint(self) -> bool:
        """Load model and feature engineer from checkpoint."""
        if not MODEL_CHECKPOINT.exists():
            return False

        checkpoint = torch.load(MODEL_CHECKPOINT, map_location=self.device)
        structured_dim = checkpoint["structured_dim"]

        self.model = LitigationClassifier(
            structured_dim=structured_dim,
            config=checkpoint.get("model_config", {}),
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

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
