"""
test_base_model.py — Teste das Basis-Modell (Mistral-7B-Instruct) OHNE Finetuning.

Damit kannst du sehen, wie gut das Modell "out of the box" funktioniert,
bevor du es mit QLoRA feintunst.

Verwendung:
  # Interaktiver Modus (eigene Fälle eingeben)
  python test_base_model.py

  # Ein synthetisches Beispiel testen
  python test_base_model.py --demo

  # Batch-Test mit Beispieldaten
  python test_base_model.py --batch sample_cases.json --output base_results.json

  # Anderes Modell testen
  python test_base_model.py --model mistralai/Mistral-7B-Instruct-v0.2
"""

import argparse
import json
from pathlib import Path

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from prepare_data import SYSTEM_PROMPT, build_user_prompt

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"


# ---------------------------------------------------------------------------
# Modell laden (OHNE LoRA-Adapter)
# ---------------------------------------------------------------------------
def load_base_model(model_name: str):
    """Lädt das Basis-Modell mit 4-bit Quantisierung (ohne Adapter)."""
    logger.info(f"Lade Basis-Modell: {model_name}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()

    # Pad-Token setzen (Mistral hat standardmäßig keinen)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.success("Basis-Modell geladen und bereit.")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Prognose
# ---------------------------------------------------------------------------
def predict(
    model,
    tokenizer,
    klaeger: str,
    beklagter: str,
    max_new_tokens: int = 256,
    temperature: float = 0.1,
) -> str:
    """Erstellt eine Verfahrensausgang-Prognose mit dem Basis-Modell."""

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
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return response


def extract_outcome(response: str) -> str:
    """Extrahiert das Ergebnis aus der Antwort."""
    upper = response.upper()
    if "TEILWEISE" in upper:
        return "teilweise"
    if "OBSIEGEN" in upper:
        return "obsiegen"
    if "UNTERLIEGEN" in upper:
        return "unterliegen"
    return "unklar"


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


def run_demo(model, tokenizer):
    """Führt einen Demo-Fall durch."""
    print("\n" + "=" * 60)
    print("  DEMO — Basis-Modell Test (ohne Finetuning)")
    print("=" * 60)

    print(f"\n📋 Klägervorbringen:\n{DEMO_CASE['klaegervorbringen']}\n")
    print(f"📋 Beklagtenvorbringen:\n{DEMO_CASE['beklagtenvorbringen']}\n")
    print("Analysiere mit Basis-Modell...\n")

    response = predict(
        model, tokenizer,
        DEMO_CASE["klaegervorbringen"],
        DEMO_CASE["beklagtenvorbringen"],
    )
    outcome = extract_outcome(response)

    print(f"{'=' * 60}")
    print(f"  PROGNOSE: {outcome.upper()}")
    print(f"{'=' * 60}")
    print(f"\nModell-Antwort:\n{response}\n")


# ---------------------------------------------------------------------------
# Batch-Modus
# ---------------------------------------------------------------------------
def batch_predict(model, tokenizer, input_path: Path, output_path: Path):
    """Batch-Prognose zum Vergleich der Basis-Modell-Performance."""
    logger.info(f"Lade Fälle aus {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    results = []
    for i, case in enumerate(cases):
        logger.info(f"Fall {i + 1}/{len(cases)}")
        response = predict(
            model, tokenizer,
            case["klaegervorbringen"],
            case["beklagtenvorbringen"],
        )
        outcome = extract_outcome(response)
        results.append({
            **case,
            "base_prognose": outcome,
            "base_antwort": response,
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.success(f"Ergebnisse gespeichert unter {output_path}")

    # Statistik
    if any("outcome" in r for r in results):
        correct = sum(
            1 for r in results
            if r.get("outcome", "").lower() == r["base_prognose"]
        )
        total = len(results)
        logger.info(f"Basis-Modell Accuracy: {correct}/{total} ({100 * correct / total:.1f}%)")

        # Detaillierte Aufschlüsselung
        from collections import Counter
        predicted = Counter(r["base_prognose"] for r in results)
        actual = Counter(r.get("outcome", "").lower() for r in results)
        print(f"\n  Verteilung (tatsächlich): {dict(actual)}")
        print(f"  Verteilung (vorhergesagt): {dict(predicted)}")
        unclear = predicted.get("unklar", 0)
        if unclear > 0:
            print(f"  ⚠️  {unclear} Fälle konnten nicht klassifiziert werden ('unklar')")

    return results


# ---------------------------------------------------------------------------
# Interaktiver Modus
# ---------------------------------------------------------------------------
def interactive_mode(model, tokenizer):
    """Interaktiver Prognose-Modus mit dem Basis-Modell."""
    print("\n" + "=" * 60)
    print("  Basis-Modell Test — Interaktiver Modus")
    print("  (ohne Finetuning — zum Vergleich)")
    print("  Eingabe 'q' zum Beenden")
    print("=" * 60 + "\n")

    while True:
        print("-" * 40)
        klaeger = input("Klägervorbringen:\n> ").strip()
        if klaeger.lower() == "q":
            break

        beklagter = input("Beklagtenvorbringen:\n> ").strip()
        if beklagter.lower() == "q":
            break

        print("\nAnalysiere mit Basis-Modell...")
        response = predict(model, tokenizer, klaeger, beklagter)
        outcome = extract_outcome(response)

        print(f"\n{'=' * 40}")
        print(f"  PROGNOSE: {outcome.upper()}")
        print(f"{'=' * 40}")
        print(f"  {response}")
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Teste das Basis-Modell (Mistral-7B-Instruct) ohne Finetuning"
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
        help="Führt einen Demo-Fall durch",
    )
    parser.add_argument(
        "--batch", "-b",
        type=Path,
        default=None,
        help="JSON-Datei für Batch-Test (optional)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("base_results.json"),
        help="Ausgabedatei für Batch-Ergebnisse (default: base_results.json)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        logger.error("CUDA nicht verfügbar! GPU wird benötigt.")
        raise SystemExit(1)

    model, tokenizer = load_base_model(args.model)

    if args.demo:
        run_demo(model, tokenizer)
    elif args.batch:
        batch_predict(model, tokenizer, args.batch, args.output)
    else:
        interactive_mode(model, tokenizer)


if __name__ == "__main__":
    main()
