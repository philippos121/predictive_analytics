"""Side-by-side comparison of base Mistral vs fine-tuned LoRA model."""

import argparse
from pathlib import Path

import torch
from loguru import logger
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MAX_SEQ_LEN = 1024

_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = "LeoLM/leo-mistral-hessianai-7b-chat"
DEFAULT_ADAPTER = str(_SCRIPT_DIR.parent / "output" / "legal-lora" / "final")

SYSTEM_PROMPT = (
    "Du bist ein juristischer Prognose-Assistent für österreichische Zivilverfahren. "
    "Auf Basis des Klägervorbringens und des Beklagtenvorbringens prognostizierst du "
    "den wahrscheinlichen Verfahrensausgang. Antworte ausschließlich mit "
    "'OBSIEGEN' (Kläger gewinnt) oder 'UNTERLIEGEN' (Kläger verliert), "
    "gefolgt von einer kurzen Begründung in 1–3 Sätzen."
)


def load_models(model_name: str, adapter_path: str):
    """Load base model and fine-tuned model."""
    logger.info(f"Loading tokenizer from {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    logger.info(f"Loading base model: {model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    logger.info(f"Loading fine-tuned model with adapter: {adapter_path}")
    ft_model = PeftModel.from_pretrained(base_model, adapter_path)
    ft_model.eval()

    return tokenizer, base_model, ft_model


def generate_response(model, tokenizer, messages, max_new_tokens=100):
    """Generate a response from a model given chat messages."""
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        input_text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description="Interactive base vs fine-tuned comparison")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    args = parser.parse_args()

    tokenizer, base_model, ft_model = load_models(args.model, args.adapter)

    print("\n" + "=" * 70)
    print("  BASE vs FINE-TUNED  —  Interactive Comparison")
    print("  Type 'quit' to exit")
    print("=" * 70)

    while True:
        print("\n--- Klägervorbringen (plaintiff's argument) ---")
        print("(Enter text, then press Enter twice to finish)")
        klaeger_lines = []
        while True:
            line = input()
            if line == "":
                break
            klaeger_lines.append(line)
        klaeger = "\n".join(klaeger_lines)

        if klaeger.strip().lower() == "quit":
            break

        print("\n--- Beklagtenvorbringen (defendant's argument) ---")
        print("(Enter text, then press Enter twice to finish)")
        beklagter_lines = []
        while True:
            line = input()
            if line == "":
                break
            beklagter_lines.append(line)
        beklagter = "\n".join(beklagter_lines)

        if beklagter.strip().lower() == "quit":
            break

        user_prompt = (
            f"### Klägervorbringen\n{klaeger.strip()}\n\n"
            f"### Beklagtenvorbringen\n{beklagter.strip()}\n\n"
            f"Wie lautet die Prognose für den Verfahrensausgang?"
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        print("\nGenerating responses...")

        # Base model (disable adapter)
        ft_model.disable_adapter_layers()
        base_response = generate_response(ft_model, tokenizer, messages)

        # Fine-tuned model (enable adapter)
        ft_model.enable_adapter_layers()
        ft_response = generate_response(ft_model, tokenizer, messages)

        print("\n" + "=" * 70)
        print("BASE MODEL:")
        print("-" * 35)
        print(base_response)
        print("\nFINE-TUNED MODEL:")
        print("-" * 35)
        print(ft_response)
        print("=" * 70)


if __name__ == "__main__":
    main()
