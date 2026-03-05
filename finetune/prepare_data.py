"""
prepare_data.py — Konvertiert das JSON-Dateiformat in ein Trainings-Dataset
für das QLoRA-Finetuning.

Unterstützte Eingabeformate:

1. Extractor-Format (aus data_extractor/data_manager.py):
[
    {
        "case_id": "F566CA14-4F7",
        "filename": "...",
        "structured": { "outcome": 0 },
        "sections": {
            "klaegervorbringen": "...",
            "beklagtenvorbringen": "..."
        },
        ...
    },
    ...
]

2. Flat-Format (aus generate_sample_data.py):
[
    {
        "klaegervorbringen": "...",
        "beklagtenvorbringen": "...",
        "outcome": "obsiegen"
    },
    ...
]

Outcome-Mapping (binär):
  0 / "unterliegen" → UNTERLIEGEN
  1 / "teilweise"   → UNTERLIEGEN   (wird zu UNTERLIEGEN zusammengefasst)
  2 / "obsiegen"    → OBSIEGEN

Ausgabe: Hugging-Face-Dataset im Chat-Format, gespeichert auf Festplatte.
"""

import json
import argparse
import random
from pathlib import Path

from datasets import Dataset, DatasetDict
from loguru import logger

# Default input: ../data/extracted/cases_dataset.json (relative to this script)
_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = _SCRIPT_DIR.parent / "data" / "extracted" / "cases_dataset.json"
DEFAULT_OUTPUT = _SCRIPT_DIR.parent / "data" / "prepared_dataset"

# ---------------------------------------------------------------------------
# Outcome-Mapping (numerisch ↔ Text)
# ---------------------------------------------------------------------------
OUTCOME_INT_TO_LABEL = {0: "UNTERLIEGEN", 1: "UNTERLIEGEN", 2: "OBSIEGEN"}
OUTCOME_STR_TO_LABEL = {
    "unterliegen": "UNTERLIEGEN",
    "teilweise": "UNTERLIEGEN",
    "obsiegen": "OBSIEGEN",
}

# ---------------------------------------------------------------------------
# System-Prompt für das Modell
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "Auf Basis von Klägervorbringen und Beklagtenvorbringen, "
    "prognostiziere OBSIEGEN oder UNTERLIEGEN."
)

# ---------------------------------------------------------------------------
# Prompt-Vorlage
# ---------------------------------------------------------------------------
def build_user_prompt(klaeger: str, beklagter: str) -> str:
    """Erstellt den User-Prompt aus Kläger- und Beklagtenvorbringen."""
    return (
        f"### Klägervorbringen\n{klaeger.strip()}\n\n"
        f"### Beklagtenvorbringen\n{beklagter.strip()}\n\n"
        f"Wie lautet die Prognose für den Verfahrensausgang?"
    )


def build_assistant_response(outcome) -> str:
    """Erstellt die erwartete Modell-Antwort. Akzeptiert int (0/1/2) oder str."""
    if isinstance(outcome, int):
        label = OUTCOME_INT_TO_LABEL.get(outcome)
        if label is None:
            raise ValueError(f"Unbekannter Outcome-Wert: {outcome}. Erlaubt: 0, 1, 2")
        return label

    outcome_lower = str(outcome).strip().lower()
    label = OUTCOME_STR_TO_LABEL.get(outcome_lower)
    if label is None:
        raise ValueError(
            f"Unbekannter Outcome-Wert: '{outcome}'. "
            f"Erlaubt: obsiegen, teilweise, unterliegen (oder 0, 1, 2)"
        )
    return label


def format_as_chat(entry: dict) -> dict:
    """Konvertiert einen Datensatz in das Chat-Messages-Format."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_prompt(
                entry["klaegervorbringen"],
                entry["beklagtenvorbringen"],
            ),
        },
        {
            "role": "assistant",
            "content": build_assistant_response(entry["outcome"]),
        },
    ]
    return {"messages": messages}


def _normalize_entry(raw: dict) -> dict | None:
    """
    Normalisiert einen Eintrag aus entweder dem Extractor-Format oder dem
    Flat-Format in das einheitliche Schema:
        {"klaegervorbringen": str, "beklagtenvorbringen": str, "outcome": int|str}
    Gibt None zurück, wenn Pflichtfelder fehlen.
    """
    # --- Extractor-Format (nested) ---
    if "sections" in raw and "structured" in raw:
        sections = raw.get("sections", {})
        klaeger = sections.get("klaegervorbringen", "").strip()
        beklagter = sections.get("beklagtenvorbringen", "").strip()
        outcome = raw["structured"].get("outcome")
        if outcome is None:
            return None
        if not klaeger:
            return None
        return {
            "klaegervorbringen": klaeger,
            "beklagtenvorbringen": beklagter,
            "outcome": outcome,
        }

    # --- Flat-Format (from generate_sample_data / legacy) ---
    klaeger = raw.get("klaegervorbringen", "").strip()
    beklagter = raw.get("beklagtenvorbringen", "").strip()
    outcome = raw.get("outcome")
    if outcome is None or not klaeger:
        return None
    return {
        "klaegervorbringen": klaeger,
        "beklagtenvorbringen": beklagter,
        "outcome": outcome,
    }


def load_and_validate(json_path: Path) -> list[dict]:
    """Lädt das JSON und prüft die Pflichtfelder. Akzeptiert beide Formate."""
    logger.info(f"Lade Daten aus {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise TypeError("JSON-Datei muss ein Array von Objekten enthalten.")

    valid = []
    skipped = 0
    for i, raw in enumerate(data):
        entry = _normalize_entry(raw)
        if entry is None:
            logger.warning(
                f"Eintrag {i} ({raw.get('filename', raw.get('case_id', '?'))}): "
                f"Fehlende Pflichtfelder — wird übersprungen."
            )
            skipped += 1
        else:
            valid.append(entry)

    logger.info(f"{len(valid)} gültige Einträge geladen ({skipped} übersprungen)")
    return valid


def _balance_entries(
    entries: list[dict],
    max_samples: int | None,
    seed: int,
) -> list[dict]:
    """
    Erstellt ein balanciertes Dataset mit gleich vielen OBSIEGEN und UNTERLIEGEN.

    Wenn max_samples angegeben ist, werden max_samples/2 pro Klasse ausgewählt.
    Wenn eine Klasse weniger Einträge hat, bestimmt diese das Maximum pro Klasse.
    Keine Daten werden gelöscht — es wird nur eine Auswahl getroffen.
    """
    rng = random.Random(seed)

    # Nach Klasse aufteilen
    by_class: dict[str, list[dict]] = {"OBSIEGEN": [], "UNTERLIEGEN": []}
    for e in entries:
        label = build_assistant_response(e["outcome"])
        by_class[label].append(e)

    n_obsiegen = len(by_class["OBSIEGEN"])
    n_unterliegen = len(by_class["UNTERLIEGEN"])
    logger.info(
        f"Verfügbar: OBSIEGEN={n_obsiegen} | UNTERLIEGEN={n_unterliegen} "
        f"(gesamt: {n_obsiegen + n_unterliegen})"
    )

    if max_samples is not None:
        per_class = max_samples // 2
    else:
        # Kein Limit → balance auf die kleinere Klasse
        per_class = min(n_obsiegen, n_unterliegen)

    # Auf tatsächlich verfügbare Anzahl begrenzen
    per_class = min(per_class, n_obsiegen, n_unterliegen)

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
    """Erstellt ein train/validation DatasetDict und speichert es."""
    entries = load_and_validate(json_path)

    # Balancierte Auswahl
    entries = _balance_entries(entries, max_samples, seed)

    # In Chat-Format konvertieren
    formatted = [format_as_chat(e) for e in entries]

    ds = Dataset.from_list(formatted)
    split = ds.train_test_split(test_size=val_ratio, seed=seed)
    dd = DatasetDict({"train": split["train"], "validation": split["test"]})

    # Statistiken — Outcome-Verteilung über alle 3 Klassen
    labels = [build_assistant_response(e["outcome"]) for e in entries]
    counts = {lbl: labels.count(lbl) for lbl in ["OBSIEGEN", "UNTERLIEGEN"]}
    total = len(labels)
    dist_str = " | ".join(
        f"{lbl}={n} ({n/total*100:.1f}%)" for lbl, n in counts.items() if n > 0
    )
    logger.info(f"Outcome-Verteilung: {dist_str}")
    logger.info(f"Train: {len(dd['train'])} | Validation: {len(dd['validation'])}")

    output_dir.mkdir(parents=True, exist_ok=True)
    dd.save_to_disk(str(output_dir))
    logger.success(f"Dataset gespeichert unter {output_dir}")

    return dd


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Bereitet die JSON-Daten für das LLM-Finetuning vor."
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Pfad zur JSON-Eingabedatei (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Ausgabeverzeichnis für das HF-Dataset (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Anteil der Validierungsdaten (default: 0.1 = 10%%)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random-Seed für die Aufteilung (default: 42)",
    )
    parser.add_argument(
        "--max-samples", "-n",
        type=int,
        default=None,
        help=(
            "Maximale Gesamtanzahl der Trainingsbeispiele (balanciert 50/50). "
            "z.B. --max-samples 1000 → 500 OBSIEGEN + 500 UNTERLIEGEN. "
            "Ohne Angabe: Balance auf die kleinere Klasse."
        ),
    )
    args = parser.parse_args()

    create_dataset(args.input, args.output, args.val_ratio, args.seed, args.max_samples)


if __name__ == "__main__":
    main()
