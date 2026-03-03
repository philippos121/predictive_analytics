"""
inference.py — Inferenz mit dem feingetunten LoRA-Modell.

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


# ---------------------------------------------------------------------------
# Modell laden
# ---------------------------------------------------------------------------
def load_model(model_name: str, adapter_path: str):
    """Lädt das Basis-Modell (4-bit) und den LoRA-Adapter."""
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

    logger.success("Modell geladen und bereit.")
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
    """Erstellt eine Verfahrensausgang-Prognose."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(klaeger, beklagter)},
    ]

    # Chat-Template anwenden
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

    # Nur die generierten Tokens dekodieren
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return response


def extract_outcome(response: str) -> str:
    """Extrahiert das Ergebnis (OBSIEGEN/UNTERLIEGEN) aus der Antwort."""
    upper = response.upper()
    if "OBSIEGEN" in upper:
        return "obsiegen"
    elif "UNTERLIEGEN" in upper:
        return "unterliegen"
    return "unklar"


# ---------------------------------------------------------------------------
# Batch-Modus
# ---------------------------------------------------------------------------
def batch_predict(model, tokenizer, input_path: Path, output_path: Path):
    """Batch-Prognose für eine JSON-Datei."""
    logger.info(f"Lade Fälle aus {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    results = []
    for i, case in enumerate(cases):
        logger.info(f"Fall {i+1}/{len(cases)}")
        response = predict(
            model, tokenizer,
            case["klaegervorbringen"],
            case["beklagtenvorbringen"],
        )
        outcome = extract_outcome(response)
        results.append({
            **case,
            "prognose": outcome,
            "modell_antwort": response,
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.success(f"Ergebnisse gespeichert unter {output_path}")

    # Statistik
    if any("outcome" in r for r in results):
        correct = sum(
            1 for r in results
            if r.get("outcome", "").lower() == r["prognose"]
        )
        logger.info(f"Accuracy: {correct}/{len(results)} ({100*correct/len(results):.1f}%)")

    return results


# ---------------------------------------------------------------------------
# Interaktiver Modus
# ---------------------------------------------------------------------------
def interactive_mode(model, tokenizer):
    """Interaktiver Prognose-Modus."""
    print("\n" + "=" * 60)
    print("  Verfahrensausgang-Prognose — Interaktiver Modus")
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

        print("\nAnalysiere...")
        response = predict(model, tokenizer, klaeger, beklagter)
        outcome = extract_outcome(response)

        print(f"\n{'='*40}")
        print(f"  PROGNOSE: {outcome.upper()}")
        print(f"{'='*40}")
        print(f"  {response}")
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Inferenz mit dem feingetunten Verfahrensausgang-Modell"
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
        help="JSON-Datei für Batch-Prognose (optional)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("results.json"),
        help="Ausgabedatei für Batch-Ergebnisse (default: results.json)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        logger.error("CUDA nicht verfügbar!")
        raise SystemExit(1)

    model, tokenizer = load_model(args.model, args.adapter)

    if args.batch:
        batch_predict(model, tokenizer, args.batch, args.output)
    else:
        interactive_mode(model, tokenizer)


if __name__ == "__main__":
    main()
