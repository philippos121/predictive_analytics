"""
Data Manager: Handles storage and retrieval of extracted case data and embeddings.

Data is stored in two formats:
- JSON: Structured case metadata (human-readable, easy to review/edit)
- HDF5: Embedding vectors (efficient numeric storage)
"""

import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

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

    def load_dataset_safe(self) -> tuple[list[dict], int]:
        """
        Load cases tolerating a truncated file (e.g. extraction still running).
        Walks backwards from the truncation point to find the last complete record.
        Returns (cases, recovered_count) — the original file is never modified.
        """
        if not self.dataset_path.exists():
            return [], 0
        raw = self.dataset_path.read_text(encoding="utf-8")
        try:
            return json.loads(raw), 0
        except json.JSONDecodeError:
            pass

        # File is mid-write: find last '\n  }' that closes a top-level object
        marker = "\n  }"
        pos = len(raw)
        while pos > 0:
            idx = raw.rfind(marker, 0, pos)
            if idx == -1:
                break
            candidate = raw[: idx + len(marker)] + "\n]"
            try:
                cases = json.loads(candidate)
                return cases, len(cases)
            except json.JSONDecodeError:
                pos = idx
        return [], 0

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
        embeddings: dict[str, list[float]],
    ) -> dict:
        """
        Add a new case to the dataset (JSON + HDF5).
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
            "has_embeddings": True,
        }

        cases.append(case_record)
        self.save_dataset(cases)

        # Save embeddings to HDF5
        self._save_embeddings(case_id, embeddings)

        return case_record

    def add_cases_bulk(
        self,
        entries: list[dict],
    ) -> list[dict]:
        """
        Add multiple cases in a single JSON read/write cycle.

        Each entry must contain:
          case_id, filename, structured, sections, embeddings

        Returns the list of created case records.
        Raises ValueError if any filename is already present.
        """
        cases = self.load_dataset()
        existing_filenames = {c["filename"] for c in cases}

        records = []
        for entry in entries:
            filename = entry["filename"]
            if filename in existing_filenames:
                raise ValueError(f"Fall '{filename}' wurde bereits verarbeitet.")
            existing_filenames.add(filename)

            record = {
                "case_id": entry["case_id"],
                "filename": filename,
                "processed_at": datetime.now().isoformat(),
                "structured": entry["structured"],
                "sections": {k: v for k, v in entry["sections"].items()},
                "has_embeddings": True,
            }
            cases.append(record)
            records.append(record)

        # Single JSON write for all cases
        self.save_dataset(cases)

        # Write all embeddings in one HDF5 session
        with h5py.File(self.embeddings_path, "a") as f:
            for entry, record in zip(entries, records):
                case_id = record["case_id"]
                if case_id in f:
                    del f[case_id]
                grp = f.create_group(case_id)
                for section, vec in entry["embeddings"].items():
                    vec_array = np.array(vec, dtype=np.float32)
                    if len(vec_array) != EMBEDDING_DIM:
                        padded = np.zeros(EMBEDDING_DIM, dtype=np.float32)
                        padded[: min(len(vec_array), EMBEDDING_DIM)] = vec_array[: EMBEDDING_DIM]
                        vec_array = padded
                    grp.create_dataset(section, data=vec_array)

        return records

    def update_case(self, case_id: str, updates: dict) -> bool:
        """Update a case record in the dataset."""
        cases = self.load_dataset()
        for i, case in enumerate(cases):
            if case["case_id"] == case_id:
                # Deep update structured data
                if "structured" in updates:
                    cases[i]["structured"].update(updates["structured"])
                # Update other fields
                for k, v in updates.items():
                    if k != "structured":
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
                    # Pad or truncate to expected dimension
                    padded = np.zeros(EMBEDDING_DIM, dtype=np.float32)
                    padded[: min(len(vec_array), EMBEDDING_DIM)] = vec_array[: EMBEDDING_DIM]
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
                for section in EMBEDDING_SECTIONS
                if section in f[case_id]
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
                    for section in EMBEDDING_SECTIONS
                    if section in f[case_id]
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

        return {
            "total_cases": len(cases),
            "labeled_cases": len(labeled),
            "unlabeled_cases": len(cases) - len(labeled),
            "outcome_distribution": {
                "unterliegen": outcomes.count(0),
                "teilweise": outcomes.count(1),
                "obsiegen": outcomes.count(2),
            },
        }

    def export_for_training(self) -> tuple[list[dict], dict[str, dict[str, np.ndarray]]]:
        """
        Prepare dataset and embeddings for model training.
        Returns only labeled cases with complete embeddings.
        Uses the safe loader so training works even while extraction is still running.
        """
        cases, recovered = self.load_dataset_safe()
        if recovered:
            print(f"[DataManager] Datei noch nicht fertig — {recovered} vollständige Fälle geladen.")
        all_embeddings = self.load_all_embeddings()

        training_cases = []
        training_embeddings = {}

        for case in cases:
            cid = case["case_id"]
            # Must have outcome label
            if case["structured"].get("outcome") is None:
                continue
            # Must have embeddings
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
