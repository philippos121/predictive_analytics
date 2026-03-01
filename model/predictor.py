"""
Predictor: Apply the trained model to new cases for outcome prediction.

v2.0 — Structured data only, no embeddings.

Implements the expected value calculation combining:
1. ML model probability (from trained LitigationClassifier)
2. Juristic success estimate (from external AI assessment)
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from model.ratg_calculator import RATGKostenrechnung

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DEFENSE_LABELS,
    DEFENSE_TYPES,
    OUTCOME_COLORS,
    OUTCOME_LABELS,
)
from model.feature_engineer import FeatureEngineer
from model.neural_net import LitigationClassifier
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

    def predict(self, case_dict: dict) -> dict:
        """
        Predict outcome probabilities for a new case.

        Routes to kNN or neural network depending on which model is loaded.

        Args:
            case_dict: Case dict with 'structured' and 'legal_analysis' keys

        Returns:
            dict with probabilities, predicted class, confidence
        """
        # kNN path
        if self.trainer.knn is not None:
            return self.trainer.knn.predict_case(case_dict)

        if self.model is None:
            raise RuntimeError("Kein trainiertes Modell verfügbar.")

        self.model.eval()

        # Encode structured features (includes legal_analysis)
        structured = self.feature_engineer.encode_single_transform(case_dict)
        structured_tensor = torch.tensor(
            structured, dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, probs = self.model(structured_tensor)

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
        ratg_kosten: Optional[RATGKostenrechnung] = None,
    ) -> dict:
        """
        Compute the combined expected value of a case.
        """
        # Normalize weights
        w_total = w_ml + w_jurist
        w_ml_norm = w_ml / w_total
        w_jurist_norm = w_jurist / w_total

        # ML probabilities
        p_win_ml = ml_result["p_win"]
        p_partial_ml = ml_result["p_partial"]

        # Mixture
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

        # Monetary expected value
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

    def get_feature_importance(self) -> Optional[dict]:
        """
        Approximate feature importance via weight analysis.
        Only available if neural network is trained (not for kNN mode).
        """
        if self.trainer.knn is not None:
            return None

        if self.model is None:
            return None

        # Feature names
        from config import CLAIM_TYPES, LEGAL_ANALYSIS_BOOL_FIELDS, LEGAL_ANALYSIS_LIST_FIELDS, RECHTSGEBIET_CATEGORIES

        feature_names = []

        # A) Original structured features
        feature_names.append("log_streitwert")
        for ct in CLAIM_TYPES:
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

        # B) Legal analysis features
        for rg in RECHTSGEBIET_CATEGORIES:
            feature_names.append(f"rechtsgebiet_{rg}")
        for section_key, fields in LEGAL_ANALYSIS_BOOL_FIELDS.items():
            for field in fields:
                feature_names.append(f"la_{section_key}_{field}")
        for section_key, fields in LEGAL_ANALYSIS_LIST_FIELDS.items():
            for field in fields:
                feature_names.append(f"la_{section_key}_{field}_count")

        # Get encoder first-layer weights as proxy for importance
        weights = self.model.encoder[0].weight.data.abs()
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
