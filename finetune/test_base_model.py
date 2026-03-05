"""
test_base_model.py — Teste das Basis-Modell (Mistral-7B-Instruct) OHNE Finetuning.

Gibt nur Wahrscheinlichkeiten für die drei Ausgänge aus (keine Textgenerierung).

Verwendung:
  # Interaktiver Modus
  python test_base_model.py

  # Ein synthetisches Beispiel testen
  python test_base_model.py --demo

  # Batch-Test mit Beispieldaten
  python test_base_model.py --batch sample_cases.json --output base_results.json
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from prepare_data import SYSTEM_PROMPT, build_user_prompt

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

OUTCOME_LABELS = ["OBSIEGEN", "UNTERLIEGEN"]


# ---------------------------------------------------------------------------
# bitsandbytes-Diagnose
# ---------------------------------------------------------------------------
def check_bnb_available() -> bool:
    """Prueft ob bitsandbytes CUDA-Quantisierung tatsaechlich funktioniert."""
    try:
        import bitsandbytes as bnb
        # Teste ob die CUDA-Kernels geladen werden koennen
        bnb.functional.get_ptr(None)
        logger.success(f"bitsandbytes {bnb.__version__} — CUDA OK")
        return True
    except Exception as e:
        logger.warning(f"bitsandbytes CUDA nicht verfuegbar: {e}")
        logger.warning(
            "4-bit Quantisierung funktioniert NICHT. "
            "Bekanntes Problem auf Windows + Python 3.13."
        )
        return False


# ---------------------------------------------------------------------------
# Modell laden (OHNE LoRA-Adapter)
# ---------------------------------------------------------------------------
def load_base_model(model_name: str):
    """Laedt das Basis-Modell. Versucht 4-bit, faellt auf GPU+CPU-Split zurueck."""
    logger.info(f"Lade Basis-Modell: {model_name}")

    gc.collect()
    torch.cuda.empty_cache()

    free_mem = torch.cuda.mem_get_info()[0] / 1024**3
    total_mem = torch.cuda.mem_get_info()[1] / 1024**3
    logger.info(f"GPU-Speicher: {free_mem:.1f} GiB frei / {total_mem:.1f} GiB gesamt")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    bnb_works = check_bnb_available()

    if bnb_works:
        # ---- Weg 1: echte 4-bit Quantisierung (~4 GiB VRAM) ----
        logger.info("Lade mit 4-bit Quantisierung (bitsandbytes)...")
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    else:
        # ---- Weg 2: float16 mit GPU+CPU-Split ----
        # Mistral-7B float16 = ~14 GiB. Wir packen so viel wie moeglich
        # auf die GPU und den Rest auf die CPU. Langsamer, aber funktioniert.
        gpu_budget = max(0, int(free_mem) - 2)  # 2 GiB Puffer
        if gpu_budget < 2:
            logger.error(
                f"Nur {free_mem:.1f} GiB frei auf GPU und bitsandbytes "
                f"funktioniert nicht. Bitte andere Prozesse beenden oder "
                f"bitsandbytes reparieren (siehe unten)."
            )
            print("\n  FIX: pip install bitsandbytes>=0.45.0")
            print("  Wenn das nicht hilft: Python 3.11 statt 3.13 verwenden.\n")
            raise SystemExit(1)

        logger.info(
            f"Lade in float16 mit GPU+CPU-Split "
            f"(GPU: {gpu_budget} GiB, Rest: CPU-RAM). "
            f"Langsamer als 4-bit, aber funktioniert."
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            max_memory={0: f"{gpu_budget}GiB", "cpu": "24GiB"},
        )

    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Speicher-Check nach dem Laden
    alloc = torch.cuda.memory_allocated() / 1024**3
    logger.success(f"Modell geladen. GPU-Nutzung: {alloc:.1f} GiB")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Token-IDs für die drei Outcome-Labels ermitteln
# ---------------------------------------------------------------------------
def get_outcome_token_ids(tokenizer) -> dict[str, int]:
    """
    Ermittelt die Token-ID für jedes Outcome-Label.
    Nimmt das erste Token der Tokenisierung (z.B. "OBS" für "OBSIEGEN").
    """
    token_map = {}
    for label in OUTCOME_LABELS:
        ids = tokenizer.encode(label, add_special_tokens=False)
        token_map[label] = ids[0]  # Erstes Token reicht zur Unterscheidung
    logger.info(
        "Outcome-Token-IDs: "
        + ", ".join(f"{lbl}→{tid}" for lbl, tid in token_map.items())
    )
    return token_map


# ---------------------------------------------------------------------------
# Wahrscheinlichkeiten berechnen
# ---------------------------------------------------------------------------
def predict_probabilities(
    model,
    tokenizer,
    outcome_token_ids: dict[str, int],
    klaeger: str,
    beklagter: str,
) -> dict[str, float]:
    """
    Berechnet die Wahrscheinlichkeiten für OBSIEGEN / UNTERLIEGEN
    basierend auf den Logits des nächsten Tokens nach dem Prompt.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(klaeger, beklagter)},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Logits des letzten Tokens = Vorhersage für das nächste Token
    next_token_logits = outputs.logits[0, -1, :]

    # Nur die Logits für unsere drei Outcome-Tokens extrahieren
    target_ids = list(outcome_token_ids.values())
    target_logits = next_token_logits[target_ids]

    # Softmax über die drei Kandidaten → Wahrscheinlichkeiten
    probs = torch.softmax(target_logits.float(), dim=0)

    result = {}
    for i, label in enumerate(outcome_token_ids.keys()):
        result[label] = round(probs[i].item() * 100, 2)

    return result


def format_probabilities(probs: dict[str, float]) -> str:
    """Formatiert die Wahrscheinlichkeiten als lesbaren String."""
    best = max(probs, key=probs.get)
    lines = []
    for label in OUTCOME_LABELS:
        p = probs[label]
        bar = "#" * int(p / 2)
        marker = " <--" if label == best else ""
        lines.append(f"  {label:14s} {p:6.2f}%  {bar}{marker}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Demo-Modus
# ---------------------------------------------------------------------------
DEMO_CASE = {
    "klaegervorbringen": (
        "Der Kläger macht einen Schadenersatzanspruch in Höhe von EUR 25.000 "
        "geltend. Er bringt vor, dass der Beklagte seine vertraglichen Pflichten "
        "aus dem Werkvertrag vom 15.03.2022 schuldhaft verletzt hat, indem er "
        "fehlerhafte Werkleistungen erbracht hat. Der Kläger hat den Beklagten "
        "am 10.06.2023 schriftlich zur Nachbesserung aufgefordert. Die Mängel "
        "wurden durch ein Sachverständigengutachten bestätigt."
    ),
    "beklagtenvorbringen": (
        "Der Beklagte bestreitet das Klagebegehren dem Grunde und der Höhe nach. "
        "Er wendet ein, dass die behauptete Schadenshöhe nicht nachvollziehbar "
        "ist und der Kläger die Mängel erst nach 8 Monaten gerügt hat. "
        "Zudem sei ein Mitverschulden des Klägers von mindestens 40% anzulasten, "
        "da er trotz Kenntnis der Risiken die Werkleistung ohne Prüfung "
        "abgenommen hat."
    ),
}


def run_demo(model, tokenizer, outcome_token_ids):
    """Führt einen Demo-Fall durch."""
    print("\n" + "=" * 60)
    print("  DEMO — Basis-Modell Wahrscheinlichkeiten (ohne Finetuning)")
    print("=" * 60)

    print(f"\nKlaegervorbringen:\n{DEMO_CASE['klaegervorbringen']}\n")
    print(f"Beklagtenvorbringen:\n{DEMO_CASE['beklagtenvorbringen']}\n")
    print("Berechne Wahrscheinlichkeiten...\n")

    probs = predict_probabilities(
        model, tokenizer, outcome_token_ids,
        DEMO_CASE["klaegervorbringen"],
        DEMO_CASE["beklagtenvorbringen"],
    )

    print(f"{'=' * 60}")
    print(f"  PROGNOSE-WAHRSCHEINLICHKEITEN")
    print(f"{'=' * 60}")
    print(format_probabilities(probs))
    print()


# ---------------------------------------------------------------------------
# Batch-Modus
# ---------------------------------------------------------------------------
def batch_predict(model, tokenizer, outcome_token_ids, input_path: Path, output_path: Path):
    """Batch-Prognose — gibt Wahrscheinlichkeiten für jeden Fall aus."""
    logger.info(f"Lade Fälle aus {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    results = []
    for i, case in enumerate(cases):
        logger.info(f"Fall {i + 1}/{len(cases)}")
        probs = predict_probabilities(
            model, tokenizer, outcome_token_ids,
            case["klaegervorbringen"],
            case["beklagtenvorbringen"],
        )
        best = max(probs, key=probs.get)
        results.append({
            **case,
            "probabilities": probs,
            "predicted": best.lower(),
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.success(f"Ergebnisse gespeichert unter {output_path}")

    # Statistik
    if any("outcome" in r for r in results):
        from collections import Counter

        correct = sum(
            1 for r in results
            if r.get("outcome", "").lower() == r["predicted"]
        )
        total = len(results)
        logger.info(f"Basis-Modell Accuracy: {correct}/{total} ({100 * correct / total:.1f}%)")

        predicted = Counter(r["predicted"] for r in results)
        actual = Counter(r.get("outcome", "").lower() for r in results)
        print(f"\n  Verteilung (tatsaechlich):  {dict(actual)}")
        print(f"  Verteilung (vorhergesagt): {dict(predicted)}")

        # Durchschnittliche Konfidenz
        avg_conf = sum(max(r["probabilities"].values()) for r in results) / total
        print(f"  Durchschnittliche Konfidenz: {avg_conf:.1f}%")

    return results


# ---------------------------------------------------------------------------
# Interaktiver Modus
# ---------------------------------------------------------------------------
def interactive_mode(model, tokenizer, outcome_token_ids):
    """Interaktiver Modus — gibt nur Wahrscheinlichkeiten aus."""
    print("\n" + "=" * 60)
    print("  Basis-Modell Test — Nur Wahrscheinlichkeiten")
    print("  (ohne Finetuning — zum Vergleich)")
    print("  Eingabe 'q' zum Beenden")
    print("=" * 60 + "\n")

    while True:
        print("-" * 40)
        klaeger = input("Klaegervorbringen:\n> ").strip()
        if klaeger.lower() == "q":
            break

        beklagter = input("Beklagtenvorbringen:\n> ").strip()
        if beklagter.lower() == "q":
            break

        print("\nBerechne Wahrscheinlichkeiten...")
        probs = predict_probabilities(
            model, tokenizer, outcome_token_ids,
            klaeger, beklagter,
        )

        print(f"\n{'=' * 40}")
        print(format_probabilities(probs))
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Teste das Basis-Modell — gibt nur Wahrscheinlichkeiten aus"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Basis-Modell (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--demo", "-d",
        action="store_true",
        help="Fuehrt einen Demo-Fall durch",
    )
    parser.add_argument(
        "--batch", "-b",
        type=Path,
        default=None,
        help="JSON-Datei fuer Batch-Test (optional)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("base_results.json"),
        help="Ausgabedatei fuer Batch-Ergebnisse (default: base_results.json)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        logger.error("CUDA nicht verfuegbar! GPU wird benoetigt.")
        raise SystemExit(1)

    model, tokenizer = load_base_model(args.model)
    outcome_token_ids = get_outcome_token_ids(tokenizer)

    if args.demo:
        run_demo(model, tokenizer, outcome_token_ids)
    elif args.batch:
        batch_predict(model, tokenizer, outcome_token_ids, args.batch, args.output)
    else:
        interactive_mode(model, tokenizer, outcome_token_ids)


if __name__ == "__main__":
    main()
