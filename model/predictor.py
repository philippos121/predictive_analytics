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
    EMBEDDING_DIM,
    EMBEDDING_SECTIONS,
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

        # Structured features (zeros if scaler not fitted or model has no struct branch)
        struct_tensor = None
        if (self.feature_engineer.is_fitted
                and getattr(self.model, "structured_dim", 0) > 0):
            struct_vec = self.feature_engineer.encode_single_transform(case_dict)
            struct_tensor = (
                torch.tensor(struct_vec, dtype=torch.float32)
                .unsqueeze(0)
                .to(self.device)
            )

        with torch.no_grad():
            logits, probs = self.model(emb_tensors, struct_tensor)

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
        ratg_result: Optional[dict] = None,
    ) -> dict:
        """
        Compute the combined expected value of a case.

        Args:
            ml_result: Output from predict()
            juristic_estimate: AI juristic success probability (0.0–1.0)
            w_ml: Weight for ML model prediction
            w_jurist: Weight for juristic estimate
            streitwert_eur: Case value in EUR (for monetary EV)
            cost_estimate_eur: Manual cost estimate in EUR (used if no ratg_result)
            ratg_result: Output from ratg_calculator.calculate_ratg_costs()
                         (if provided, overrides cost_estimate_eur)

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
        # unabhängig vom statistischen Modell.
        #
        #   p_full    = juristic * (w_ml * p_win_ml + w_jur)
        #   p_partial = juristic *  w_ml * p_partial_ml       ← kein juristic-Anteil für Teilerfolg
        #   p_failure = 1 - p_full - p_partial
        #
        # Das implizite juristische Wahrscheinlichkeits-Tupel lautet damit:
        #   (p_jur_loss = 1 − juristic, p_jur_partial = 0, p_jur_win = juristic)
        # → gültige Verteilung für alle juristic ∈ [0,1].
        # Beweis: p_full + p_partial + p_failure = 1, ev ∈ [0, juristic] ⊆ [0, 1].
        p_full_success = juristic_estimate * (w_ml_norm * p_win_ml + w_jurist_norm)
        p_partial_success = juristic_estimate * w_ml_norm * p_partial_ml
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

        # ── Monetary Expected Value ───────────────────────────────────────────
        if streitwert_eur is not None and streitwert_eur > 0:
            # Brutto-EV: Erfolgsszenario-gewichteter Zuspruch
            ev_gross = (
                p_full_success * streitwert_eur
                + p_partial_success * streitwert_eur * 0.5
            )
            result["streitwert_eur"] = streitwert_eur
            result["ev_gross_eur"] = ev_gross

            # Kosten-EV nach § 41 ZPO:
            #   Sieg:       eigene Kosten vollständig erstattet → 0 Nettobelastung
            #   Teilsieg:   anteilige Kostenbelastung (50 %)
            #   Niederlage: eigene + gegnerische Kosten
            if ratg_result is not None:
                from model.ratg_calculator import compute_cost_risk
                cost_risk = compute_cost_risk(
                    ratg_result,
                    p_win=p_full_success,
                    p_partial=p_partial_success,
                    p_loss=p_failure,
                )
                effective_cost = cost_risk["erwartete_kostenbelastung_eur"]
                result["ratg_result"] = ratg_result
                result["cost_risk_detail"] = cost_risk
                result["cost_estimate_eur"] = effective_cost
                result["cost_source"] = "RATG"
            elif cost_estimate_eur is not None:
                effective_cost = cost_estimate_eur
                result["cost_estimate_eur"] = effective_cost
                result["cost_source"] = "Manuell"
            else:
                effective_cost = None

            if effective_cost is not None:
                ev_net = ev_gross - effective_cost
                result["ev_net_eur"] = ev_net
                result["proceed_recommendation"] = ev_net > 0

        return result

    def predict_with_updated_vorbringen(
        self,
        original_case_dict: dict,
        new_klaeger_text: Optional[str] = None,
        new_beklagter_text: Optional[str] = None,
        api_key: Optional[str] = None,
        original_embeddings: Optional[dict] = None,
    ) -> dict:
        """
        Vorhersage nach neuem Vorbringen / Gegenvorbringen.

        Re-embedded die aktualisierten Texte und berechnet eine neue Vorhersage.
        Unveränderte Sektionen verwenden die original_embeddings, damit die
        Vorhersage nicht durch Null-Vektoren verzerrt wird.

        Args:
            original_case_dict:  Ursprüngliche Falldaten (strukturiert)
            new_klaeger_text:    Aktualisiertes / ergänztes Kläger-Vorbringen
            new_beklagter_text:  Aktualisiertes Beklagten-Vorbringen / Gegenvorbringen
            api_key:             OpenAI API-Key für Embedding
            original_embeddings: Embeddings aus der ursprünglichen Vorhersage
                                  (unveränderte Sektionen werden daraus übernommen)

        Returns:
            Neues predict()-Ergebnis mit aktualisierten Embeddings
        """
        new_sections = {}
        if new_klaeger_text:
            new_sections["klaegervorbringen"] = new_klaeger_text
        if new_beklagter_text:
            new_sections["beklagtenvorbringen"] = new_beklagter_text

        # Re-embed nur die geänderten Sektionen
        new_embeddings = {}
        if new_sections and api_key:
            try:
                from data_extractor.openai_extractor import OpenAIExtractor
                extractor = OpenAIExtractor(api_key)
                new_embeddings = extractor.generate_embeddings(new_sections)
            except Exception as e:
                raise RuntimeError(f"Embedding-Fehler bei neuem Vorbringen: {e}") from e

        # Merge: original_embeddings als Basis, neue Embeddings überschreiben
        # die geänderten Sektionen. So erhalten unveränderte Sektionen ihre
        # ursprünglichen Vektoren statt Null-Vektoren.
        merged_embeddings = {**(original_embeddings or {}), **new_embeddings}

        return self.predict(original_case_dict, merged_embeddings)

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
        """Not available — structured features are encoded via learned MLP."""
        return None
