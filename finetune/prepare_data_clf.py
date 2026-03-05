"""
prepare_data_clf.py — Bereitet Daten für Sequence-Classification vor.

Statt Chat-Format (für Textgenerierung) wird hier ein einfaches
Text → Label-Format erstellt, passend für AutoModelForSequenceClassification.

Ausgabe: Hugging-Face-Dataset mit Spalten: text (str), label (int)
  label 0 = UNTERLIEGEN, label 1 = OBSIEGEN
"""

import json
import argparse
import random
from pathlib import Path

from datasets import Dataset, DatasetDict
from loguru import logger

_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = _SCRIPT_DIR.parent / "data" / "extracted" / "cases_dataset.json"
DEFAULT_OUTPUT = _SCRIPT_DIR.parent / "data" / "prepared_dataset_clf"

# Label-Mapping: text → int
LABEL2ID = {"UNTERLIEGEN": 0, "OBSIEGEN": 1}
ID2LABEL = {0: "UNTERLIEGEN", 1: "OBSIEGEN"}

OUTCOME_INT_TO_LABEL = {0: "UNTERLIEGEN", 1: "UNTERLIEGEN", 2: "OBSIEGEN"}
OUTCOME_STR_TO_LABEL = {
    "unterliegen": "UNTERLIEGEN",
    "teilweise": "UNTERLIEGEN",
    "obsiegen": "OBSIEGEN",
}


def _get_label(outcome) -> str | None:
    if isinstance(outcome, int):
        return OUTCOME_INT_TO_LABEL.get(outcome)
    outcome_lower = str(outcome).strip().lower()
    return OUTCOME_STR_TO_LABEL.get(outcome_lower)


def _normalize_entry(raw: dict) -> dict | None:
    if "sections" in raw and "structured" in raw:
        sections = raw.get("sections", {})
        klaeger = sections.get("klaegervorbringen", "").strip()
        beklagter = sections.get("beklagtenvorbringen", "").strip()
        outcome = raw["structured"].get("outcome")
    else:
        klaeger = raw.get("klaegervorbringen", "").strip()
        beklagter = raw.get("beklagtenvorbringen", "").strip()
        outcome = raw.get("outcome")

    if outcome is None or not klaeger:
        return None

    label = _get_label(outcome)
    if label is None:
        return None

    return {"klaeger": klaeger, "beklagter": beklagter, "label": label}


def build_text(klaeger: str, beklagter: str) -> str:
    """Strukturiertes Format mit expliziten Marker-Tags für Aufmerksamkeit."""
    parts = [f"[KLÄGER]\n{klaeger}"]
    if beklagter:
        parts.append(f"[BEKLAGTER]\n{beklagter}")
    parts.append("[PROGNOSE]")
    return "\n\n".join(parts)


def load_and_validate(json_path: Path) -> list[dict]:
    logger.info(f"Lade Daten aus {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    valid, skipped = [], 0
    for raw in data:
        entry = _normalize_entry(raw)
        if entry is None:
            skipped += 1
        else:
            valid.append(entry)

    logger.info(f"{len(valid)} gültige Einträge geladen ({skipped} übersprungen)")
    return valid


def _balance_entries(entries: list[dict], max_samples: int | None, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_class: dict[str, list[dict]] = {"OBSIEGEN": [], "UNTERLIEGEN": []}
    for e in entries:
        by_class[e["label"]].append(e)

    n_obs = len(by_class["OBSIEGEN"])
    n_unt = len(by_class["UNTERLIEGEN"])
    logger.info(f"Verfügbar: OBSIEGEN={n_obs} | UNTERLIEGEN={n_unt}")

    if max_samples is not None:
        per_class = max_samples // 2
    else:
        per_class = min(n_obs, n_unt)

    per_class = min(per_class, n_obs, n_unt)
    logger.info(f"Balancierte Auswahl: {per_class} pro Klasse ({per_class * 2} gesamt)")

    selected = (
        rng.sample(by_class["OBSIEGEN"], per_class)
        + rng.sample(by_class["UNTERLIEGEN"], per_class)
    )
    rng.shuffle(selected)
    return selected


def create_dataset(
    json_path: Path,
    output_dir: Path,
    val_ratio: float = 0.1,
    seed: int = 42,
    max_samples: int | None = None,
) -> DatasetDict:
    entries = load_and_validate(json_path)
    entries = _balance_entries(entries, max_samples, seed)

    formatted = []
    for e in entries:
        formatted.append({
            "text": build_text(e["klaeger"], e["beklagter"]),
            "label": LABEL2ID[e["label"]],
        })

    ds = Dataset.from_list(formatted)
    split = ds.train_test_split(test_size=val_ratio, seed=seed)
    dd = DatasetDict({"train": split["train"], "validation": split["test"]})

    logger.info(f"Train: {len(dd['train'])} | Validation: {len(dd['validation'])}")
    output_dir.mkdir(parents=True, exist_ok=True)
    dd.save_to_disk(str(output_dir))
    logger.success(f"Dataset gespeichert unter {output_dir}")
    return dd


def main():
    parser = argparse.ArgumentParser(description="Bereite Daten für Classification-Head vor.")
    parser.add_argument("--input", "-i", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", "-o", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", "-n", type=int, default=None)
    args = parser.parse_args()
    create_dataset(args.input, args.output, args.val_ratio, args.seed, args.max_samples)


if __name__ == "__main__":
    main()
