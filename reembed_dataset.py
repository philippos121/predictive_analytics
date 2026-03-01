"""
reembed_dataset.py — Re-generate embeddings for all extracted cases.

Use this after changing OPENAI_EMBEDDING_MODEL or EMBEDDING_DIM in config.py.
Reads section texts from cases_dataset.json, calls the OpenAI embeddings API,
and writes a fresh embeddings.h5.

Usage:
    python reembed_dataset.py [--workers N] [--no-backup]

    --workers N     Parallel API calls (default: 8).
    --no-backup     Skip backing up the existing embeddings.h5.

The old embeddings file is renamed to embeddings_backup.h5 before writing
so nothing is lost if the run is interrupted.  Re-running the script is safe:
cases already written to the new file are skipped automatically.
"""

import argparse
import json
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import openai

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATASET_FILE,
    EMBEDDING_DIM,
    EMBEDDING_SECTIONS,
    EMBEDDINGS_FILE,
    OPENAI_EMBEDDING_MODEL,
)


# ─── OpenAI client (reads OPENAI_API_KEY from environment) ──────────────────

client = openai.OpenAI()

BACKUP_FILE = EMBEDDINGS_FILE.parent / "embeddings_backup.h5"
_h5_lock = threading.Lock()  # h5py is not thread-safe for concurrent writes


# ─── Helpers ────────────────────────────────────────────────────────────────

def _embed(text: str, retries: int = 5) -> np.ndarray:
    """Call OpenAI embeddings API with exponential-backoff retry."""
    if not text or not text.strip():
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    text = text[:32000]  # ~8 000 tokens

    delay = 2.0
    for attempt in range(retries):
        try:
            resp = client.embeddings.create(
                model=OPENAI_EMBEDDING_MODEL,
                input=text,
                dimensions=EMBEDDING_DIM,
            )
            return np.array(resp.data[0].embedding, dtype=np.float32)
        except (openai.RateLimitError, openai.APIConnectionError) as exc:
            if attempt == retries - 1:
                raise
            print(f"  [retry {attempt + 1}/{retries}] {exc} — waiting {delay:.0f}s")
            time.sleep(delay)
            delay = min(delay * 2, 30)


def _load_already_done(path: Path) -> set[str]:
    """Return set of case_ids already written to path."""
    if not path.exists():
        return set()
    with h5py.File(path, "r") as f:
        return set(f.keys())


def _save_case(path: Path, case_id: str, vectors: dict[str, np.ndarray]) -> None:
    """Append one case's embeddings to the HDF5 file (serialised via lock)."""
    with _h5_lock, h5py.File(path, "a") as f:
        if case_id in f:
            del f[case_id]
        grp = f.create_group(case_id)
        for section, vec in vectors.items():
            if len(vec) != EMBEDDING_DIM:
                padded = np.zeros(EMBEDDING_DIM, dtype=np.float32)
                padded[: min(len(vec), EMBEDDING_DIM)] = vec[: EMBEDDING_DIM]
                vec = padded
            grp.create_dataset(section, data=vec)


def _process_case(case: dict) -> tuple[str, dict[str, np.ndarray]] | None:
    """
    Generate embeddings for all EMBEDDING_SECTIONS of one case.
    Returns (case_id, {section: vector}) or None if no sections found.
    """
    case_id = case["case_id"]
    sections = case.get("sections", {})

    vectors: dict[str, np.ndarray] = {}
    for section in EMBEDDING_SECTIONS:
        text = sections.get(section, "")
        vectors[section] = _embed(text)

    if not vectors:
        return None
    return case_id, vectors


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=8, help="Parallel API workers (default: 8)")
    parser.add_argument("--no-backup", action="store_true", help="Skip backup of existing embeddings.h5")
    args = parser.parse_args()

    # ── Load dataset ──────────────────────────────────────────────────────────
    if not DATASET_FILE.exists():
        sys.exit(f"Dataset not found: {DATASET_FILE}")

    with open(DATASET_FILE, encoding="utf-8") as f:
        cases: list[dict] = json.load(f)

    cases_with_sections = [c for c in cases if c.get("sections")]
    print(f"Dataset: {len(cases)} cases total, {len(cases_with_sections)} with sections")
    print(f"Embedding model : {OPENAI_EMBEDDING_MODEL}  ({EMBEDDING_DIM} dims)")
    print(f"Sections        : {EMBEDDING_SECTIONS}")
    print(f"Output file     : {EMBEDDINGS_FILE}")

    if not cases_with_sections:
        sys.exit("No cases with sections found — nothing to embed.")

    # ── Backup / fresh-start logic ────────────────────────────────────────────
    # First run (no backup yet): MOVE the old file out of the way so that
    # _load_already_done finds an empty target and re-embeds everything.
    # Resume run (backup already exists): the current embeddings.h5 already
    # contains only new-model vectors — skip the ones already written.
    if EMBEDDINGS_FILE.exists() and not args.no_backup:
        if not BACKUP_FILE.exists():
            print(f"\nMoving {EMBEDDINGS_FILE.name} → {BACKUP_FILE.name} ...", end=" ", flush=True)
            shutil.move(str(EMBEDDINGS_FILE), BACKUP_FILE)
            print("done.")
        else:
            print(f"\nBackup {BACKUP_FILE.name} already exists — resuming previous run.")

    # ── Skip already-done cases ───────────────────────────────────────────────
    # After a move the target file no longer exists → set is empty → all cases embedded.
    # On resume the target contains only vectors from the new model → safe to skip.
    already_done = _load_already_done(EMBEDDINGS_FILE)
    todo = [c for c in cases_with_sections if c["case_id"] not in already_done]

    print(f"\nAlready embedded : {len(already_done)}")
    print(f"To embed now     : {len(todo)}")

    if not todo:
        print("Nothing to do — all cases already embedded.")
        return

    # Cost estimate: text-embedding-3-small ≈ $0.02 / 1M tokens
    # Rough estimate: avg 1000 tokens per section × 2 sections × N cases
    est_tokens = len(todo) * 2 * 1000
    est_cost = est_tokens / 1_000_000 * 0.02
    print(f"Est. cost        : ~${est_cost:.2f} (@ $0.02/1M tokens, ~1k tokens/section)")
    print()

    # ── Embed in parallel ─────────────────────────────────────────────────────
    done = 0
    errors = 0
    total = len(todo)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_process_case, case): case["case_id"] for case in todo}

        for future in as_completed(futures):
            case_id = futures[future]
            try:
                result = future.result()
                if result is not None:
                    cid, vectors = result
                    _save_case(EMBEDDINGS_FILE, cid, vectors)
                    done += 1
                else:
                    print(f"  [skip] {case_id} — no section text")
            except Exception as exc:
                errors += 1
                print(f"  [error] {case_id}: {exc}")

            pct = (done + errors) / total * 100
            print(f"\r  {done + errors}/{total} ({pct:.0f}%)  ✓ {done}  ✗ {errors}", end="", flush=True)

    print(f"\n\nDone. Embedded {done} cases, {errors} errors.")
    if errors:
        print(f"Tip: re-run the script to retry failed cases (already-done are skipped).")

    if BACKUP_FILE.exists() and not args.no_backup:
        old_size = BACKUP_FILE.stat().st_size / 1024 / 1024
        new_size = EMBEDDINGS_FILE.stat().st_size / 1024 / 1024
        print(f"Old embeddings.h5: {old_size:.1f} MB  →  New: {new_size:.1f} MB")
        print(f"Backup kept at: {BACKUP_FILE}")


if __name__ == "__main__":
    main()
