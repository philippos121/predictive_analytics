"""
Predictor: Apply the trained model to new cases for outcome prediction.

Implements the expected value calculation combining:
1. ML model probability (from trained LitigationClassifier)
2. Juristic success estimate (from external AI assessment)
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DEFENSE_LABELS,
    DEFENSE_TYPES,
    EMBEDDING_DIM,
    EMBEDDING_SECTIONS,
    OUTCOME_COLORS,
    OUTCOME_LABELS,
)
from model.feature_engineer import FeatureEngineer
from model.neural_net import LitigationClassifier
from model.trainer import LitigationTrainer


class LitigationPredictor:
    """
    Applies the trained model to new cases and computes expected value.

    Expected Value Formula:
        E[outcome] = w_ml * P_ml(win) + w_jurist * P_jurist(win)

    where:
        P_ml(win) = model's predicted probability of outcome=2 (Obsiegen)
        P_jurist(win) = juristic success estimate (0.0–1.0)
        w_ml, w_jurist = configurable weights (default 0.5 each)
    """

    def __init__(self, trainer: LitigationTrainer):
        self.trainer = trainer
        self.model = trainer.model
        self.feature_engineer = trainer.feature_engineer
        self.device = trainer.device

    @classmethod
    def from_checkpoint(cls) -> Optional["LitigationPredictor"]:
        """Load predictor from saved checkpoint."""
        trainer = LitigationTrainer()
        if trainer.load_checkpoint():
            return cls(trainer)
        return None

    def predict(
        self,
        case_dict: dict,
        embeddings: dict[str, list[float]],
    ) -> dict:
        """
        Predict outcome probabilities for a new case.

        Routes to kNN or neural network depending on which model is loaded.

        Args:
            case_dict: Case metadata dict (same structure as training data)
            embeddings: {section_name: embedding_vector}

        Returns:
            dict with probabilities, predicted class, confidence
        """
        # kNN path — no structured features needed
        if self.trainer.knn is not None:
            return self.trainer.knn.predict(embeddings)

        if self.model is None:
            raise RuntimeError("Kein trainiertes Modell verfügbar.")

        self.model.eval()

        # Encode structured features
        structured = self.feature_engineer.encode_single_transform(case_dict)
        structured_tensor = torch.tensor(
            structured, dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        # Load embeddings
        emb_tensors = []
        for section in EMBEDDING_SECTIONS:
            if section in embeddings and embeddings[section]:
                vec = np.array(embeddings[section], dtype=np.float32)
                if len(vec) != EMBEDDING_DIM:
                    vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)
            else:
                vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)
            emb_tensors.append(
                torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(self.device)
            )

        with torch.no_grad():
            logits, probs = self.model(emb_tensors, structured_tensor)

        probs_np = probs.cpu().numpy()[0]
        predicted_class = int(probs_np.argmax())
        confidence = float(probs_np.max())

        return {
            "predicted_outcome": predicted_class,
            "predicted_label": OUTCOME_LABELS[predicted_class],
            "probabilities": {
                OUTCOME_LABELS[i]: float(probs_np[i]) for i in range(3)
            },
            "confidence": confidence,
            "p_win": float(probs_np[2]),
            "p_partial": float(probs_np[1]),
            "p_loss": float(probs_np[0]),
        }

    def compute_expected_value(
        self,
        ml_result: dict,
        juristic_estimate: float,
        w_ml: float = 0.5,
        w_jurist: float = 0.5,
        streitwert_eur: Optional[float] = None,
        cost_estimate_eur: Optional[float] = None,
    ) -> dict:
        """
        Compute the combined expected value of a case.

        Args:
            ml_result: Output from predict()
            juristic_estimate: AI juristic success probability (0.0–1.0)
            w_ml: Weight for ML model prediction
            w_jurist: Weight for juristic estimate
            streitwert_eur: Case value in EUR (for monetary EV)
            cost_estimate_eur: Estimated litigation costs in EUR

        Returns:
            Expected value analysis dict
        """
        # Normalize weights
        w_total = w_ml + w_jurist
        w_ml_norm = w_ml / w_total
        w_jurist_norm = w_jurist / w_total

        # ML probabilities
        p_win_ml = ml_result["p_win"]
        p_partial_ml = ml_result["p_partial"]

        # Multiplikative Formel: juristic_estimate skaliert beide Anteile.
        # Dadurch gilt: juristic=0 (rechtlich unschlüssig) → Gesamtwahrscheinlichkeit=0,
        # unabhängig vom statistischen Modell. Außerdem ist p_full_success ≤ juristic_estimate.
        #
        #   p_full    = juristic * (w_ml * p_win_ml  + w_jur * 1.0)
        #   p_partial = juristic * (w_ml * p_partial + w_jur * 0.5)
        #
        # Äquivalent zur alten additiven Formel bei juristic=1; konservativer dazwischen.
        p_full_success = juristic_estimate * (w_ml_norm * p_win_ml + w_jurist_norm)
        p_partial_success = juristic_estimate * (w_ml_norm * p_partial_ml + w_jurist_norm * 0.5)
        p_failure = max(0.0, 1.0 - p_full_success - p_partial_success)

        # Weighted win probability
        ev_probability = p_full_success + 0.5 * p_partial_success

        result = {
            "p_full_success_combined": p_full_success,
            "p_partial_success_combined": p_partial_success,
            "p_failure_combined": p_failure,
            "ev_success_probability": ev_probability,
            "ml_weight_applied": w_ml_norm,
            "juristic_weight_applied": w_jurist_norm,
            "juristic_estimate_input": juristic_estimate,
            "recommendation": self._get_recommendation(ev_probability),
        }

        # Monetary expected value
        if streitwert_eur is not None and streitwert_eur > 0:
            ev_gross = (
                p_full_success * streitwert_eur
                + p_partial_success * streitwert_eur * 0.5
                - p_failure * 0.0
            )
            result["streitwert_eur"] = streitwert_eur
            result["ev_gross_eur"] = ev_gross

            if cost_estimate_eur is not None:
                ev_net = ev_gross - cost_estimate_eur
                result["cost_estimate_eur"] = cost_estimate_eur
                result["ev_net_eur"] = ev_net
                result["proceed_recommendation"] = ev_net > 0

        return result

    def _get_recommendation(self, ev_probability: float) -> str:
        if ev_probability >= 0.70:
            return "STARK EMPFOHLEN — Hohe Erfolgschancen"
        elif ev_probability >= 0.55:
            return "EMPFOHLEN — Überwiegende Erfolgschancen"
        elif ev_probability >= 0.45:
            return "NEUTRAL — Ausgeglichene Chancen, Kosten-Nutzen prüfen"
        elif ev_probability >= 0.30:
            return "VORSICHT — Unterdurchschnittliche Erfolgschancen"
        else:
            return "NICHT EMPFOHLEN — Geringe Erfolgschancen"

    def get_feature_importance(self) -> Optional[dict]:
        """
        Approximate feature importance via gradient analysis.
        Only available if neural network is trained (not for kNN mode).
        """
        if self.trainer.knn is not None:
            return None

        if self.model is None:
            return None

        # Feature names from feature engineer
        feature_names = []
        feature_names.append("log_streitwert")
        for ct in __import__(
            "config", fromlist=["CLAIM_TYPES"]
        ).CLAIM_TYPES:
            feature_names.append(f"claim_{ct}")
        feature_names.append("claim_Andere")
        for d in DEFENSE_TYPES:
            feature_names.append(f"defense_{d}")
        feature_names.extend([
            "klaeger_evidence_count",
            "beklagter_evidence_count",
            "legal_basis_count",
        ])
        for inst in ["BG", "LG", "OLG", "OGH"]:
            feature_names.append(f"instanz_{inst}")
        feature_names.append("sachverstaendiger")

        # Get structured encoder weights as proxy
        weights = self.model.structured_encoder.encoder[0].weight.data.abs()
        importance = weights.mean(dim=0).cpu().numpy()

        if len(importance) != len(feature_names):
            return None

        return {
            name: float(imp)
            for name, imp in sorted(
                zip(feature_names, importance),
                key=lambda x: x[1],
                reverse=True,
            )
        }
