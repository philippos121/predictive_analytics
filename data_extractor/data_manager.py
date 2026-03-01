"""
Data Manager: Handles storage and retrieval of extracted case data and embeddings.

v3.0 — Hybrid: JSON for structured data + HDF5 for embedding vectors.
"""

import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DATASET_FILE,
    EMBEDDING_DIM,
    EMBEDDING_SECTIONS,
    EMBEDDINGS_FILE,
)


class DataManager:
    """Manages the case dataset (JSON + HDF5 embeddings)."""

    def __init__(
        self,
        dataset_path: Path = DATASET_FILE,
        embeddings_path: Path = EMBEDDINGS_FILE,
    ):
        self.dataset_path = Path(dataset_path)
        self.embeddings_path = Path(embeddings_path)
        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)
        self.embeddings_path.parent.mkdir(parents=True, exist_ok=True)

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
        embeddings: Optional[dict[str, list[float]]] = None,
    ) -> dict:
        """
        Add a new case to the dataset.
        If embeddings are provided, they are saved to HDF5.
        """
        cases = self.load_dataset()

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
            "has_embeddings": embeddings is not None,
        }

        cases.append(case_record)
        self.save_dataset(cases)

        if embeddings:
            self._save_embeddings(case_id, embeddings)

        return case_record

    def update_case(self, case_id: str, updates: dict) -> bool:
        """Update a case record in the dataset."""
        cases = self.load_dataset()
        for i, case in enumerate(cases):
            if case["case_id"] == case_id:
                if "structured" in updates:
                    cases[i]["structured"].update(updates["structured"])
                if "legal_analysis" in updates:
                    if "legal_analysis" not in cases[i]:
                        cases[i]["legal_analysis"] = {}
                    cases[i]["legal_analysis"].update(updates["legal_analysis"])
                for k, v in updates.items():
                    if k not in ("structured", "legal_analysis"):
                        cases[i][k] = v
                cases[i]["updated_at"] = datetime.now().isoformat()
                self.save_dataset(cases)
                return True
        return False

    def delete_case(self, case_id: str) -> bool:
        """Remove a case from dataset and its embeddings."""
        cases = self.load_dataset()
        original_len = len(cases)
        cases = [c for c in cases if c["case_id"] != case_id]
        if len(cases) < original_len:
            self.save_dataset(cases)
            self._delete_embeddings(case_id)
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

    # ─── Embedding Operations ────────────────────────────────────────────────────

    def _save_embeddings(self, case_id: str, embeddings: dict[str, list[float]]) -> None:
        """Save embedding vectors to HDF5 file."""
        with h5py.File(self.embeddings_path, "a") as f:
            if case_id in f:
                del f[case_id]
            grp = f.create_group(case_id)
            for section, vec in embeddings.items():
                vec_array = np.array(vec, dtype=np.float32)
                if len(vec_array) != EMBEDDING_DIM:
                    padded = np.zeros(EMBEDDING_DIM, dtype=np.float32)
                    padded[: min(len(vec_array), EMBEDDING_DIM)] = vec_array[:EMBEDDING_DIM]
                    vec_array = padded
                grp.create_dataset(section, data=vec_array)

    def _delete_embeddings(self, case_id: str) -> None:
        """Remove embeddings for a case from HDF5."""
        if not self.embeddings_path.exists():
            return
        with h5py.File(self.embeddings_path, "a") as f:
            if case_id in f:
                del f[case_id]

    def load_embeddings(self, case_id: str) -> Optional[dict[str, np.ndarray]]:
        """Load embeddings for a single case."""
        if not self.embeddings_path.exists():
            return None
        with h5py.File(self.embeddings_path, "r") as f:
            if case_id not in f:
                return None
            return {
                section: np.array(f[case_id][section])
                for section in f[case_id].keys()
            }

    def load_all_embeddings(self) -> dict[str, dict[str, np.ndarray]]:
        """Load all embeddings from HDF5. Returns {case_id: {section: array}}."""
        if not self.embeddings_path.exists():
            return {}
        result = {}
        with h5py.File(self.embeddings_path, "r") as f:
            for case_id in f.keys():
                result[case_id] = {
                    section: np.array(f[case_id][section])
                    for section in f[case_id].keys()
                }
        return result

    # ─── Dataset Analysis ────────────────────────────────────────────────────────

    def get_statistics(self) -> dict:
        """Compute dataset statistics."""
        cases = self.load_dataset()
        if not cases:
            return {"total_cases": 0}

        labeled = [c for c in cases if c["structured"].get("outcome") is not None]
        outcomes = [c["structured"]["outcome"] for c in labeled]

        has_analysis = sum(1 for c in cases if c.get("legal_analysis"))
        has_embeddings = sum(1 for c in cases if c.get("has_embeddings"))

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
            "cases_with_embeddings": has_embeddings,
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

    def export_for_training(self) -> tuple[list[dict], dict]:
        """
        Prepare dataset for model training.
        Returns (cases, embeddings_dict) — only labeled cases with embeddings.
        """
        cases = self.load_dataset()
        all_embeddings = self.load_all_embeddings()

        training_cases = []
        training_embeddings = {}

        for case in cases:
            cid = case["case_id"]
            if case["structured"].get("outcome") is None:
                continue
            if cid not in all_embeddings:
                continue
            training_cases.append(case)
            training_embeddings[cid] = all_embeddings[cid]

        return training_cases, training_embeddings

    def generate_case_id(self) -> str:
        """Generate a unique case ID."""
        return str(uuid.uuid4())[:12].upper()

    def get_processed_filenames(self) -> set[str]:
        """Return set of already-processed filenames."""
        return {c["filename"] for c in self.load_dataset()}
