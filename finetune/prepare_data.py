"""
prepare_data.py — Konvertiert das JSON-Dateiformat in ein Trainings-Dataset
für das QLoRA-Finetuning.

Erwartetes JSON-Eingabeformat (Array von Objekten):
[
    {
        "klaegervorbringen": "Der Kläger bringt vor, dass ...",
        "beklagtenvorbringen": "Der Beklagte wendet ein, dass ...",
        "outcome": "obsiegen"   // oder "unterliegen"
    },
    ...
]

Ausgabe: Hugging-Face-Dataset im Chat-Format, gespeichert auf Festplatte.
"""

import json
import argparse
from pathlib import Path

from datasets import Dataset, DatasetDict
from loguru import logger

# ---------------------------------------------------------------------------
# System-Prompt für das Modell
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "Du bist ein juristischer Prognose-Assistent für österreichische Zivilverfahren. "
    "Auf Basis des Klägervorbringens und des Beklagtenvorbringens prognostizierst du "
    "den wahrscheinlichen Verfahrensausgang. Antworte ausschließlich mit "
    "'OBSIEGEN' (Kläger gewinnt) oder 'UNTERLIEGEN' (Kläger verliert), "
    "gefolgt von einer kurzen Begründung in 1–3 Sätzen."
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


def build_assistant_response(outcome: str) -> str:
    """Erstellt die erwartete Modell-Antwort."""
    outcome_lower = outcome.strip().lower()
    if outcome_lower == "obsiegen":
        return "OBSIEGEN"
    elif outcome_lower == "unterliegen":
        return "UNTERLIEGEN"
    else:
        raise ValueError(f"Unbekannter Outcome-Wert: '{outcome}'. Erlaubt: obsiegen, unterliegen")


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


def load_and_validate(json_path: Path) -> list[dict]:
    """Lädt das JSON und prüft die Pflichtfelder."""
    logger.info(f"Lade Daten aus {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise TypeError("JSON-Datei muss ein Array von Objekten enthalten.")

    required_keys = {"klaegervorbringen", "beklagtenvorbringen", "outcome"}
    errors = 0
    for i, entry in enumerate(data):
        missing = required_keys - set(entry.keys())
        if missing:
            logger.warning(f"Eintrag {i}: Fehlende Felder {missing} — wird übersprungen.")
            errors += 1

    valid = [e for e in data if required_keys.issubset(e.keys())]
    logger.info(f"{len(valid)} gültige Einträge geladen ({errors} übersprungen)")
    return valid


def create_dataset(
    json_path: Path,
    output_dir: Path,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> DatasetDict:
    """Erstellt ein train/validation DatasetDict und speichert es."""
    entries = load_and_validate(json_path)

    # In Chat-Format konvertieren
    formatted = [format_as_chat(e) for e in entries]

    ds = Dataset.from_list(formatted)
    split = ds.train_test_split(test_size=val_ratio, seed=seed)
    dd = DatasetDict({"train": split["train"], "validation": split["test"]})

    # Statistiken
    outcomes = [e["outcome"].strip().lower() for e in entries]
    n_obsiegen = outcomes.count("obsiegen")
    n_unterliegen = outcomes.count("unterliegen")
    logger.info(
        f"Outcome-Verteilung: obsiegen={n_obsiegen} ({n_obsiegen/len(outcomes)*100:.1f}%), "
        f"unterliegen={n_unterliegen} ({n_unterliegen/len(outcomes)*100:.1f}%)"
    )
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
        required=True,
        help="Pfad zur JSON-Eingabedatei (Array von Fällen)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/prepared_dataset"),
        help="Ausgabeverzeichnis für das HF-Dataset (default: data/prepared_dataset)",
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
    args = parser.parse_args()

    create_dataset(args.input, args.output, args.val_ratio, args.seed)


if __name__ == "__main__":
    main()
