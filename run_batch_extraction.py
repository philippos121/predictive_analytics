#!/usr/bin/env python3
"""
Batch extraction of legal analysis from dataset.json.

Runs up to 100 concurrent API calls to GPT-5-nano to extract structured
legal analysis for all cases that don't have one yet.

Usage:
    python run_batch_extraction.py                    # uses default dataset path
    python run_batch_extraction.py /path/to/dataset.json
    OPENAI_API_KEY=sk-... python run_batch_extraction.py
"""

import asyncio
import os
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

from config import DATASET_FILE


def main():
    print("=" * 60)
    print("  Batch Legal Analysis Extraction")
    print("  GPT-5-nano · 100 concurrent API calls")
    print("=" * 60)

    # Resolve dataset path
    dataset_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DATASET_FILE
    if not dataset_path.exists():
        print(f"\nFehler: Dataset nicht gefunden: {dataset_path}")
        print("Bitte zuerst Fälle über den Data Extractor (run_extractor.py) anlegen")
        print("oder den Pfad als Argument übergeben.")
        sys.exit(1)

    # Get API key
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("\nFehler: OPENAI_API_KEY Umgebungsvariable nicht gesetzt.")
        print("Setzen Sie die Variable:")
        print("   export OPENAI_API_KEY='sk-...'")
        sys.exit(1)

    print(f"\nDataset: {dataset_path}")
    print(f"Max. parallele Anfragen: 100")
    print(f"Periodisches Speichern: alle 100 Fälle\n")

    from data_extractor.openai_extractor import AsyncBatchExtractor

    def progress(done, total, msg):
        print(f"  {msg}")

    extractor = AsyncBatchExtractor(
        api_key=api_key,
        max_concurrent=100,
        progress_callback=progress,
    )

    n_success, n_errors = asyncio.run(
        extractor.extract_dataset(dataset_path, save_every=100)
    )

    print(f"\n{'=' * 60}")
    print(f"  Ergebnis: {n_success} erfolgreich, {n_errors} Fehler")
    print(f"{'=' * 60}")

    sys.exit(0 if n_errors == 0 else 1)


if __name__ == "__main__":
    main()
