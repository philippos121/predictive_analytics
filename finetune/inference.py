"""
inference.py — Inferenz mit dem feingetunten LoRA-Modell.

Gibt nur Wahrscheinlichkeiten fuer die drei Ausgaenge aus (keine Textgenerierung).

Verwendung:
  # Einzelne Prognose (interaktiv)
  python inference.py --adapter output/legal-lora/final

  # Batch-Prognose aus JSON-Datei
  python inference.py --adapter output/legal-lora/final --batch input.json --output results.json

  # Mit anderem Basis-Modell
  python inference.py --adapter output/legal-lora/final --model meta-llama/Llama-3.1-8B-Instruct
"""

import argparse
import json
from pathlib import Path

import torch
from loguru import logger
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from prepare_data import SYSTEM_PROMPT, build_user_prompt

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_ADAPTER = "output/legal-lora/final"

OUTCOME_LABELS = ["OBSIEGEN", "UNTERLIEGEN"]


# ---------------------------------------------------------------------------
# Modell laden
# ---------------------------------------------------------------------------
def load_model(model_name: str, adapter_path: str):
    """Laedt das Basis-Modell (4-bit) und den LoRA-Adapter."""
    logger.info(f"Lade Basis-Modell: {model_name}")
    logger.info(f"Lade LoRA-Adapter: {adapter_path}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # LoRA-Adapter laden
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.success("Modell geladen und bereit.")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Token-IDs fuer die drei Outcome-Labels ermitteln
# ---------------------------------------------------------------------------
def get_outcome_token_ids(tokenizer) -> dict[str, int]:
    """
    Ermittelt die Token-ID fuer jedes Outcome-Label.
    Nimmt das erste Token der Tokenisierung (z.B. "OBS" fuer "OBSIEGEN").
    """
    token_map = {}
    for label in OUTCOME_LABELS:
        ids = tokenizer.encode(label, add_special_tokens=False)
        token_map[label] = ids[0]
    logger.info(
        "Outcome-Token-IDs: "
        + ", ".join(f"{lbl}->{tid}" for lbl, tid in token_map.items())
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
    Berechnet die Wahrscheinlichkeiten fuer OBSIEGEN / UNTERLIEGEN
    basierend auf den Logits des naechsten Tokens nach dem Prompt.
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

    next_token_logits = outputs.logits[0, -1, :]

    target_ids = list(outcome_token_ids.values())
    target_logits = next_token_logits[target_ids]

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
# Batch-Modus
# ---------------------------------------------------------------------------
def batch_predict(model, tokenizer, outcome_token_ids, input_path: Path, output_path: Path):
    """Batch-Prognose — gibt Wahrscheinlichkeiten fuer jeden Fall aus."""
    logger.info(f"Lade Faelle aus {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    results = []
    for i, case in enumerate(cases):
        logger.info(f"Fall {i+1}/{len(cases)}")
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
        logger.info(f"Accuracy: {correct}/{total} ({100*correct/total:.1f}%)")

        predicted = Counter(r["predicted"] for r in results)
        actual = Counter(r.get("outcome", "").lower() for r in results)
        print(f"\n  Verteilung (tatsaechlich):  {dict(actual)}")
        print(f"  Verteilung (vorhergesagt): {dict(predicted)}")

        avg_conf = sum(max(r["probabilities"].values()) for r in results) / total
        print(f"  Durchschnittliche Konfidenz: {avg_conf:.1f}%")

    return results


# ---------------------------------------------------------------------------
# Interaktiver Modus
# ---------------------------------------------------------------------------
def interactive_mode(model, tokenizer, outcome_token_ids):
    """Interaktiver Prognose-Modus — gibt nur Wahrscheinlichkeiten aus."""
    print("\n" + "=" * 60)
    print("  Verfahrensausgang-Prognose — Wahrscheinlichkeiten")
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

        print(f"\n{'='*40}")
        print(format_probabilities(probs))
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Inferenz mit dem feingetunten Verfahrensausgang-Modell (Wahrscheinlichkeiten)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Basis-Modell (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--adapter", "-a",
        type=str,
        default=DEFAULT_ADAPTER,
        help=f"Pfad zum LoRA-Adapter (default: {DEFAULT_ADAPTER})",
    )
    parser.add_argument(
        "--batch", "-b",
        type=Path,
        default=None,
        help="JSON-Datei fuer Batch-Prognose (optional)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("results.json"),
        help="Ausgabedatei fuer Batch-Ergebnisse (default: results.json)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        logger.error("CUDA nicht verfuegbar!")
        raise SystemExit(1)

    model, tokenizer = load_model(args.model, args.adapter)
    outcome_token_ids = get_outcome_token_ids(tokenizer)

    if args.batch:
        batch_predict(model, tokenizer, outcome_token_ids, args.batch, args.output)
    else:
        interactive_mode(model, tokenizer, outcome_token_ids)


if __name__ == "__main__":
    main()
