"""
Data Manager: Handles storage and retrieval of extracted case data.

v2.0 — All data stored in JSON (no HDF5 embeddings).
Each case record contains structured metadata, text sections, and a detailed
structured legal analysis that serves as the primary training signal.
"""

import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATASET_FILE


class DataManager:
    """Manages the case dataset (JSON only, no embeddings)."""

    def __init__(self, dataset_path: Path = DATASET_FILE):
        self.dataset_path = Path(dataset_path)
        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)

    # ─── Dataset Operations ─────────────────────────────────────────────────────

    def load_dataset(self) -> list[dict]:
        """Load all cases from JSON dataset file."""
        if not self.dataset_path.exists():
            return []
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_dataset(self, cases: list[dict]) -> None:
        """Save all cases to JSON dataset file."""
        with open(self.dataset_path, "w", encoding="utf-8") as f:
            json.dump(cases, f, ensure_ascii=False, indent=2)

    def add_case(
        self,
        case_id: str,
        filename: str,
        structured: dict,
        sections: dict[str, str],
        legal_analysis: dict,
    ) -> dict:
        """
        Add a new case to the dataset.
        Returns the complete case record.
        """
        cases = self.load_dataset()

        # Check for duplicate filename
        existing_filenames = {c["filename"] for c in cases}
        if filename in existing_filenames:
            raise ValueError(f"Fall '{filename}' wurde bereits verarbeitet.")

        case_record = {
            "case_id": case_id,
            "filename": filename,
            "processed_at": datetime.now().isoformat(),
            "structured": structured,
            "sections": {k: v for k, v in sections.items()},
            "legal_analysis": legal_analysis,
        }

        cases.append(case_record)
        self.save_dataset(cases)

        return case_record

    def update_case(self, case_id: str, updates: dict) -> bool:
        """Update a case record in the dataset."""
        cases = self.load_dataset()
        for i, case in enumerate(cases):
            if case["case_id"] == case_id:
                # Deep update structured data
                if "structured" in updates:
                    cases[i]["structured"].update(updates["structured"])
                # Deep update legal_analysis
                if "legal_analysis" in updates:
                    if "legal_analysis" not in cases[i]:
                        cases[i]["legal_analysis"] = {}
                    cases[i]["legal_analysis"].update(updates["legal_analysis"])
                # Update other fields
                for k, v in updates.items():
                    if k not in ("structured", "legal_analysis"):
                        cases[i][k] = v
                cases[i]["updated_at"] = datetime.now().isoformat()
                self.save_dataset(cases)
                return True
        return False

    def delete_case(self, case_id: str) -> bool:
        """Remove a case from dataset."""
        cases = self.load_dataset()
        original_len = len(cases)
        cases = [c for c in cases if c["case_id"] != case_id]
        if len(cases) < original_len:
            self.save_dataset(cases)
            return True
        return False

    def get_case(self, case_id: str) -> Optional[dict]:
        """Retrieve a single case by ID."""
        for case in self.load_dataset():
            if case["case_id"] == case_id:
                return case
        return None

    def get_case_by_filename(self, filename: str) -> Optional[dict]:
        """Retrieve a case by source filename."""
        for case in self.load_dataset():
            if case["filename"] == filename:
                return case
        return None

    # ─── Dataset Analysis ────────────────────────────────────────────────────────

    def get_statistics(self) -> dict:
        """Compute dataset statistics."""
        cases = self.load_dataset()
        if not cases:
            return {"total_cases": 0}

        labeled = [c for c in cases if c["structured"].get("outcome") is not None]
        outcomes = [c["structured"]["outcome"] for c in labeled]

        has_analysis = sum(1 for c in cases if c.get("legal_analysis"))

        streitwerte = [
            c["structured"].get("streitwert_eur")
            for c in cases
            if c["structured"].get("streitwert_eur") is not None
        ]

        claim_types = {}
        for c in cases:
            ct = c["structured"].get("anspruchsart", "Unbekannt")
            claim_types[ct] = claim_types.get(ct, 0) + 1

        return {
            "total_cases": len(cases),
            "labeled_cases": len(labeled),
            "unlabeled_cases": len(cases) - len(labeled),
            "cases_with_legal_analysis": has_analysis,
            "outcome_distribution": {
                "unterliegen": outcomes.count(0),
                "teilweise": outcomes.count(1),
                "obsiegen": outcomes.count(2),
            },
            "claim_type_distribution": claim_types,
            "streitwert_stats": {
                "min": float(min(streitwerte)) if streitwerte else None,
                "max": float(max(streitwerte)) if streitwerte else None,
                "mean": float(np.mean(streitwerte)) if streitwerte else None,
                "median": float(np.median(streitwerte)) if streitwerte else None,
            },
        }

    def export_for_training(self) -> list[dict]:
        """
        Prepare dataset for model training.
        Returns only labeled cases with legal analysis data.
        """
        cases = self.load_dataset()

        training_cases = []
        for case in cases:
            # Must have outcome label
            if case["structured"].get("outcome") is None:
                continue
            # Must have legal analysis
            if not case.get("legal_analysis"):
                continue
            training_cases.append(case)

        return training_cases

    def generate_case_id(self) -> str:
        """Generate a unique case ID."""
        return str(uuid.uuid4())[:12].upper()

    def get_processed_filenames(self) -> set[str]:
        """Return set of already-processed filenames."""
        return {c["filename"] for c in self.load_dataset()}
