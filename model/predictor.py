"""
Predictor: Apply the trained model to new cases for outcome prediction.

v3.0 — Hybrid: Text Embeddings + Structured Data.

For prediction on new cases, embeddings must be computed via OpenAI API
or provided directly. Structured features are encoded from case metadata.

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
    DEFAULT_ML_WEIGHT_NO_CALIBRATION,
    DEFENSE_TYPES,
    EMBEDDING_DIM_USED,
    EMBEDDING_SECTIONS,
    NUM_CLASSES,
    OUTCOME_LABELS,
    PCA_DIM,
    recommended_ml_weight,
)
from model.feature_engineer import prepare_embeddings_for_case
from model.ratg_calculator import RATGKostenrechnung
from model.trainer import LitigationTrainer


class LitigationPredictor:
    """
    Applies the trained model to new cases and computes expected value.

    Expected Value Formula (gewichtetes Mixture zweier Verteilungen):

        Jurist-Verteilung : P_jur  = (win=juristic, partial=0, loss=1-juristic)
        ML-Verteilung     : P_ml   = (win=p_win_ml, partial=p_partial_ml, loss=p_loss_ml)
                            konditioniert auf rechtliche Zulässigkeit (juristic);
                            unbedingtes ML = juristic * P_ml_conditional

        Mixture:
            p_full    = w_ml * juristic * p_win_ml  +  w_jur * juristic
            p_partial = w_ml * juristic * p_partial_ml
            p_failure = 1 - p_full - p_partial

        Garantie: juristic = 0  ->  p_full = p_partial = 0
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

    def get_recommended_ml_weight(self) -> float:
        """Return the recommended ML mixing weight based on stored calibration.

        If Brier Skill Score was computed during training, derives weight from
        it.  Otherwise falls back to a conservative default.
        """
        calib = self.trainer.history.get("calibration", {})
        bss = calib.get("brier_skill_score")
        if bss is not None:
            return recommended_ml_weight(bss)
        return DEFAULT_ML_WEIGHT_NO_CALIBRATION

    def get_calibration_summary(self) -> dict:
        """Return calibration info stored from training, if available."""
        return self.trainer.history.get("calibration", {})

    def predict(
        self,
        case_dict: dict,
        embeddings: Optional[dict[str, np.ndarray]] = None,
    ) -> dict:
        """
        Predict outcome probabilities for a new case.

        Args:
            case_dict: Case dict with 'structured' key
            embeddings: {section: np.ndarray} — pre-computed embedding vectors.
                        Required for neural net. If None, falls back to kNN.

        Returns:
            dict with probabilities, predicted class, confidence
        """
        # kNN path
        if self.trainer.knn is not None:
            return self.trainer.knn.predict_case(case_dict)

        if self.model is None:
            raise RuntimeError("Kein trainiertes Modell verfügbar.")

        self.model.eval()

        # Encode structured features
        structured = self.feature_engineer.encode_single_transform(case_dict)
        structured_tensor = torch.tensor(
            structured, dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        # Prepare embeddings (with PCA if available)
        pca = self.trainer.embedding_pca
        has_pca = pca is not None and pca.is_fitted
        emb_dim = pca.n_components if has_pca else EMBEDDING_DIM_USED

        if embeddings is None:
            # No embeddings available — use zero vectors (reduced accuracy)
            emb_list = [
                np.zeros(emb_dim, dtype=np.float32)
                for _ in EMBEDDING_SECTIONS
            ]
        else:
            emb_list = prepare_embeddings_for_case(
                embeddings, pca=pca if has_pca else None,
            )

        emb_tensors = [
            torch.tensor(e, dtype=torch.float32).unsqueeze(0).to(self.device)
            for e in emb_list
        ]

        with torch.no_grad():
            logits, probs = self.model(emb_tensors, structured_tensor)

        probs_np = probs.cpu().numpy()[0]
        predicted_class = int(probs_np.argmax())
        confidence = float(probs_np.max())

        result = {
            "predicted_outcome": predicted_class,
            "predicted_label": OUTCOME_LABELS[predicted_class],
            "probabilities": {
                OUTCOME_LABELS[i]: float(probs_np[i]) for i in range(NUM_CLASSES)
            },
            "confidence": confidence,
        }

        if NUM_CLASSES == 2:
            result["p_win"] = float(probs_np[1])
            result["p_partial"] = 0.0
            result["p_loss"] = float(probs_np[0])
        else:
            result["p_win"] = float(probs_np[2])
            result["p_partial"] = float(probs_np[1])
            result["p_loss"] = float(probs_np[0])

        return result

    def compute_expected_value(
        self,
        ml_result: dict,
        juristic_estimate: float,
        w_ml: float = 0.5,
        w_jurist: float = 0.5,
        streitwert_eur: Optional[float] = None,
        cost_estimate_eur: Optional[float] = None,
        ratg_kosten: Optional[RATGKostenrechnung] = None,
    ) -> dict:
        """Compute the combined expected value of a case.

        In 2-class mode (NUM_CLASSES == 2), p_partial is always 0 and the
        formula reduces to a simple weighted mixture of ML p(win) and the
        juristic estimate.
        """
        w_total = w_ml + w_jurist
        w_ml_norm = w_ml / w_total
        w_jurist_norm = w_jurist / w_total

        p_win_ml = ml_result["p_win"]
        p_partial_ml = ml_result.get("p_partial", 0.0)

        p_ml_win_adj     = juristic_estimate * p_win_ml
        p_ml_partial_adj = juristic_estimate * p_partial_ml

        p_jur_win = juristic_estimate

        p_full_success    = w_ml_norm * p_ml_win_adj     + w_jurist_norm * p_jur_win
        p_partial_success = w_ml_norm * p_ml_partial_adj
        p_failure = max(0.0, 1.0 - p_full_success - p_partial_success)

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

        if streitwert_eur is not None and streitwert_eur > 0:
            sw = streitwert_eur

            ev_gross = (
                p_full_success    * sw
                + p_partial_success * sw * 0.5
            )
            result["streitwert_eur"] = sw
            result["ev_gross_eur"]   = ev_gross

            if ratg_kosten is not None:
                k_ob  = ratg_kosten.kosten_bei_obsiegen
                k_tob = ratg_kosten.kosten_bei_teilobsiegen
                k_ul  = ratg_kosten.kosten_bei_unterliegen

                ev_net = (
                    p_full_success    * (sw       - k_ob)
                    + p_partial_success * (sw * 0.5 - k_tob)
                    + p_failure         * (         - k_ul)
                )
                result["ratg_kosten"]            = {
                    "instanz":              ratg_kosten.instanz,
                    "komplexitaet":         ratg_kosten.komplexitaet,
                    "ggg":                  ratg_kosten.ggg_pauschalgebuehr,
                    "eigene_anwaltskosten": ratg_kosten.eigene_anwaltskosten,
                    "gegner_anwaltskosten": ratg_kosten.gegner_anwaltskosten,
                    "kosten_obsiegen":      k_ob,
                    "kosten_teilobsiegen":  k_tob,
                    "kosten_unterliegen":   k_ul,
                }
                result["ev_net_eur"]             = ev_net
                result["proceed_recommendation"] = ev_net > 0
                result["cost_model"]             = "RATG"

            elif cost_estimate_eur is not None:
                ev_net = ev_gross - cost_estimate_eur
                result["cost_estimate_eur"]      = cost_estimate_eur
                result["ev_net_eur"]             = ev_net
                result["proceed_recommendation"] = ev_net > 0
                result["cost_model"]             = "manuell"

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
