"""
inference_clf.py — Inferenz mit dem Classification-Head LoRA-Modell.

Verwendet AutoModelForSequenceClassification + PEFT-Adapter.
Gibt Wahrscheinlichkeiten fuer OBSIEGEN / UNTERLIEGEN aus.

Verwendung:
  # Einzelne Prognose (interaktiv)
  python inference_clf.py --adapter output/legal-lora-clf/final

  # Batch-Prognose aus JSON-Datei
  python inference_clf.py --adapter output/legal-lora-clf/final --batch input.json --output results.json
"""

import argparse
import json
from pathlib import Path

import torch
from loguru import logger
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_ADAPTER = "output/legal-lora-clf/final"
MAX_SEQ_LEN = 2048

ID2LABEL = {0: "UNTERLIEGEN", 1: "OBSIEGEN"}
LABEL2ID = {"UNTERLIEGEN": 0, "OBSIEGEN": 1}
OUTCOME_LABELS = ["OBSIEGEN", "UNTERLIEGEN"]


# ---------------------------------------------------------------------------
# Modell laden
# ---------------------------------------------------------------------------
def load_model(model_name: str, adapter_path: str):
    logger.info(f"Lade Basis-Modell: {model_name}")
    logger.info(f"Lade LoRA-Adapter: {adapter_path}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    logger.success("Modell geladen und bereit.")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Text aufbauen (gleich wie in prepare_data_clf.py)
# ---------------------------------------------------------------------------
def build_text(klaeger: str, beklagter: str) -> str:
    parts = [f"Klägervorbringen: {klaeger}"]
    if beklagter:
        parts.append(f"Beklagtenvorbringen: {beklagter}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Vorhersage
# ---------------------------------------------------------------------------
def predict_probabilities(
    model,
    tokenizer,
    klaeger: str,
    beklagter: str,
) -> dict[str, float]:
    text = build_text(klaeger, beklagter)
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding="max_length",
    ).to(model.device)

    with torch.no_grad():
        logits = model(**inputs).logits[0]

    probs = torch.softmax(logits.float(), dim=0)

    return {
        "UNTERLIEGEN": round(probs[0].item() * 100, 2),
        "OBSIEGEN": round(probs[1].item() * 100, 2),
    }


def format_probabilities(probs: dict[str, float]) -> str:
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
def batch_predict(model, tokenizer, input_path: Path, output_path: Path):
    logger.info(f"Lade Faelle aus {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    results = []
    for i, case in enumerate(cases):
        logger.info(f"Fall {i+1}/{len(cases)}")
        probs = predict_probabilities(
            model, tokenizer,
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
def interactive_mode(model, tokenizer):
    print("\n" + "=" * 60)
    print("  Verfahrensausgang-Prognose (Classification Head)")
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
        probs = predict_probabilities(model, tokenizer, klaeger, beklagter)
        print(f"\n{'='*40}")
        print(format_probabilities(probs))
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Inferenz mit Classification-Head LoRA-Modell"
    )
    parser.add_argument("--model", "-m", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--adapter", "-a", type=str, default=DEFAULT_ADAPTER)
    parser.add_argument("--batch", "-b", type=Path, default=None)
    parser.add_argument("--output", "-o", type=Path, default=Path("results.json"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        logger.error("CUDA nicht verfuegbar!")
        raise SystemExit(1)

    model, tokenizer = load_model(args.model, args.adapter)

    if args.batch:
        batch_predict(model, tokenizer, args.batch, args.output)
    else:
        interactive_mode(model, tokenizer)


if __name__ == "__main__":
    main()
