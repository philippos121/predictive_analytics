"""
Feature Relevance Analysis: Learn which structured features courts rely on.

Uses erstgericht_begruendung (court reasoning) as a META-LEARNING signal:
- For each case, GPT identifies which boolean features the court discussed.
- Aggregates across all cases → feature importance weights.
- Identifies schema gaps (concepts courts discuss that we don't capture).

This is NOT data leakage because:
- Weights are computed across the entire corpus (not per-case at inference).
- Equivalent to a domain expert saying "courts care most about Verjährung."
- At inference time, only the pre-computed weights are used.

Usage:
    python -m analysis.feature_relevance              # Full batch analysis
    python -m analysis.feature_relevance --report-only # Show existing results
"""

import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    DATASET_FILE,
    FEATURE_RELEVANCE_FILE,
    LEGAL_ANALYSIS_BOOL_FIELDS,
)


def _flat_field_names() -> list[str]:
    """All boolean field names from the schema, in config order."""
    names = []
    for fields in LEGAL_ANALYSIS_BOOL_FIELDS.values():
        names.extend(fields)
    return names


def aggregate_relevance(
    relevance_by_case: dict[str, dict],
) -> dict:
    """
    Aggregate per-case feature relevance into corpus-level statistics.

    Returns dict with:
    - feature_counts: {field_name: n_cases_where_court_relied_on_it}
    - feature_importance: {field_name: fraction_of_cases} (0.0–1.0)
    - schema_gaps: [{konzept, beschreibung, count}] sorted by frequency
    - n_cases_analyzed: total cases with erstgericht_begruendung
    """
    all_fields = _flat_field_names()
    field_counter: Counter = Counter()
    gap_counter: Counter = Counter()
    gap_descriptions: dict[str, str] = {}

    n_cases = len(relevance_by_case)

    for case_id, rel in relevance_by_case.items():
        for field_name in rel.get("entscheidungsrelevante_merkmale", []):
            field_counter[field_name] += 1

        for gap in rel.get("nicht_erfasste_konzepte", []):
            konzept = gap.get("konzept", "").strip()
            if konzept:
                gap_counter[konzept] += 1
                if konzept not in gap_descriptions:
                    gap_descriptions[konzept] = gap.get("beschreibung", "")

    # Build importance dict (all fields, including those never mentioned = 0)
    feature_importance = {}
    for field in all_fields:
        count = field_counter.get(field, 0)
        feature_importance[field] = count / n_cases if n_cases > 0 else 0.0

    # Schema gaps sorted by frequency
    schema_gaps = [
        {
            "konzept": k,
            "beschreibung": gap_descriptions.get(k, ""),
            "count": c,
            "frequency": c / n_cases if n_cases > 0 else 0.0,
        }
        for k, c in gap_counter.most_common()
    ]

    return {
        "n_cases_analyzed": n_cases,
        "feature_counts": dict(field_counter.most_common()),
        "feature_importance": feature_importance,
        "schema_gaps": schema_gaps,
    }


def compute_attention_weights(
    feature_importance: dict[str, float],
    alpha: float = 2.0,
    floor: float = 0.5,
) -> dict[str, float]:
    """
    Convert feature importance (0–1) into attention weights for the StructuredEncoder.

    Formula: w = floor + alpha * importance
    - floor=0.5: even features the court never mentions keep 50% weight
    - alpha=2.0: a feature mentioned in 50% of cases gets weight 1.5

    These weights multiply the structured feature vector element-wise
    BEFORE the linear layer, giving the model a court-informed prior
    while still allowing it to learn.
    """
    all_fields = _flat_field_names()
    weights = {}
    for field in all_fields:
        imp = feature_importance.get(field, 0.0)
        weights[field] = floor + alpha * imp
    return weights


def save_results(results: dict, path: Path = FEATURE_RELEVANCE_FILE) -> None:
    """Save aggregated feature relevance results to JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Ergebnisse gespeichert: {path}")


def load_results(path: Path = FEATURE_RELEVANCE_FILE) -> dict:
    """Load previously computed results."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_report(results: dict) -> None:
    """Print a human-readable feature relevance report."""
    n = results["n_cases_analyzed"]
    print(f"\n{'='*70}")
    print(f"  Feature-Relevanz-Analyse ({n} Fälle mit Erstgericht-Begründung)")
    print(f"{'='*70}\n")

    # Top features by importance
    importance = results["feature_importance"]
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    print("TOP FEATURES (vom Gericht am häufigsten als entscheidungsrelevant behandelt):\n")
    for i, (field, imp) in enumerate(sorted_features[:20], 1):
        bar = "#" * int(imp * 40)
        count = results["feature_counts"].get(field, 0)
        print(f"  {i:2d}. {field:<45s} {imp:5.1%}  ({count:3d}/{n})  {bar}")

    # Features never mentioned
    zero_features = [f for f, imp in sorted_features if imp == 0.0]
    if zero_features:
        print(f"\nNIE ERWÄHNT ({len(zero_features)} Merkmale):")
        for f in zero_features:
            print(f"  - {f}")

    # Schema gaps
    gaps = results.get("schema_gaps", [])
    if gaps:
        print(f"\nSCHEMA-LÜCKEN (Konzepte, die das Gericht bespricht, aber nicht erfasst sind):\n")
        for g in gaps[:15]:
            print(f"  - {g['konzept']} ({g['count']}x, {g['frequency']:.0%})")
            if g["beschreibung"]:
                print(f"    → {g['beschreibung']}")

    # Attention weights preview
    weights = results.get("attention_weights", {})
    if weights:
        sorted_w = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        print(f"\nATTENTION-WEIGHTS (top 10 / bottom 5):\n")
        for field, w in sorted_w[:10]:
            print(f"  {field:<45s} {w:.2f}")
        print("  ...")
        for field, w in sorted_w[-5:]:
            print(f"  {field:<45s} {w:.2f}")

    print()


async def run_batch_analysis(
    dataset_path: Path = DATASET_FILE,
    output_path: Path = FEATURE_RELEVANCE_FILE,
    max_concurrent: int = 50,
) -> dict:
    """
    Run full batch feature relevance analysis:
    1. Load dataset
    2. Extract relevance for each case with erstgericht_begruendung
    3. Aggregate results
    4. Compute attention weights
    5. Save and print report
    """
    from data_extractor.data_manager import DataManager
    from data_extractor.openai_extractor import AsyncBatchExtractor

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("FEHLER: OPENAI_API_KEY nicht gesetzt.")
        sys.exit(1)

    dm = DataManager(dataset_path=dataset_path)
    cases = dm.load_dataset()

    n_with_begruendung = sum(
        1 for c in cases
        if c.get("sections", {}).get("erstgericht_begruendung", "").strip()
    )
    print(f"Dataset: {len(cases)} Fälle, davon {n_with_begruendung} mit Erstgericht-Begründung")

    if n_with_begruendung == 0:
        print("Keine Fälle mit Erstgericht-Begründung gefunden.")
        return {}

    def progress(done, total, msg):
        print(f"  {msg}")

    extractor = AsyncBatchExtractor(
        api_key=api_key,
        max_concurrent=max_concurrent,
        progress_callback=progress,
    )

    print(f"Starte Feature-Relevanz-Extraktion ({max_concurrent} parallel)...")
    relevance_by_case, errors = await extractor.extract_feature_relevance_batch(cases)

    if errors:
        print(f"\n{len(errors)} Fehler:")
        for e in errors[:10]:
            print(f"  - {e}")

    # Aggregate
    results = aggregate_relevance(relevance_by_case)
    results["attention_weights"] = compute_attention_weights(
        results["feature_importance"]
    )
    results["per_case_relevance"] = relevance_by_case

    save_results(results, output_path)
    print_report(results)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Feature-Relevanz-Analyse: Lernt aus Erstgericht-Begründungen"
    )
    parser.add_argument(
        "--report-only", action="store_true",
        help="Nur vorhandene Ergebnisse anzeigen (keine API-Aufrufe)"
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=50,
        help="Max parallele API-Aufrufe (default: 50)"
    )
    args = parser.parse_args()

    if args.report_only:
        if not FEATURE_RELEVANCE_FILE.exists():
            print(f"Keine Ergebnisse vorhanden: {FEATURE_RELEVANCE_FILE}")
            sys.exit(1)
        results = load_results()
        print_report(results)
    else:
        asyncio.run(run_batch_analysis(max_concurrent=args.max_concurrent))


if __name__ == "__main__":
    main()
