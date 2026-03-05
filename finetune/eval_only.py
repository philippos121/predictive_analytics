"""Standalone logit-based classification evaluation on a saved LoRA adapter.

Instead of generating free text and parsing labels, this script compares the
model's next-token probabilities for OBSIEGEN vs UNTERLIEGEN directly.
This eliminates garbled/unparseable outputs entirely.
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import load_from_disk
from loguru import logger
from peft import PeftModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MAX_SEQ_LEN = 2048
LABEL_CLASSES = ["OBSIEGEN", "UNTERLIEGEN"]
# Map legacy 3-class labels to binary (TEILWEISE → UNTERLIEGEN)
LABEL_ALIAS = {"TEILWEISE": "UNTERLIEGEN"}

_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_ADAPTER = str(_SCRIPT_DIR.parent / "output" / "legal-lora" / "final")
DEFAULT_DATASET = str(_SCRIPT_DIR.parent / "data" / "prepared_dataset")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output", default=None, help="Dir for results JSON (default: adapter dir)")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit validation samples (default: all)")
    args = parser.parse_args()

    output_dir = args.output or args.adapter

    # Load tokenizer
    logger.info(f"Lade Tokenizer von {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model in 4-bit
    logger.info(f"Lade Base-Model in 4-bit: {args.model}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load LoRA adapter
    logger.info(f"Lade LoRA-Adapter von {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    # Determine token IDs for each label's first token
    label_token_ids = {}
    for label in LABEL_CLASSES:
        ids = tokenizer.encode(label, add_special_tokens=False)
        label_token_ids[label] = ids[0]
    logger.info(f"Label-Token-IDs: { {l: t for l, t in label_token_ids.items()} }")

    # Load validation split
    logger.info(f"Lade Dataset von {args.dataset}")
    ds = load_from_disk(args.dataset)
    val_ds = ds["validation"]
    if args.max_samples:
        val_ds = val_ds.select(range(min(args.max_samples, len(val_ds))))
    logger.info(f"Validation-Samples: {len(val_ds)}")

    # Run logit-based classification
    y_true, y_pred, y_probs = [], [], []

    for i, sample in enumerate(val_ds):
        messages = sample["messages"]
        true_label_raw = messages[-1]["content"].strip()
        true_label = LABEL_ALIAS.get(true_label_raw, true_label_raw)
        y_true.append(true_label)

        input_messages = messages[:-1]
        input_text = tokenizer.apply_chat_template(
            input_messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True,
                           max_length=MAX_SEQ_LEN).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)
            last_logits = outputs.logits[0, -1, :]

        # Softmax over only the two label tokens
        label_ids = torch.tensor([label_token_ids[l] for l in LABEL_CLASSES],
                                 device=last_logits.device)
        label_logits = last_logits[label_ids]
        probs = torch.softmax(label_logits, dim=0)

        pred_idx = probs.argmax().item()
        pred_label = LABEL_CLASSES[pred_idx]
        prob_dict = {l: round(p.item(), 4) for l, p in zip(LABEL_CLASSES, probs)}

        y_pred.append(pred_label)
        y_probs.append(prob_dict)

        if i < 10:
            logger.info(f"Sample {i}: true={true_label} | pred={pred_label} | probs={prob_dict}")

        if (i + 1) % 50 == 0:
            logger.info(f"  {i + 1}/{len(val_ds)} Samples ausgewertet...")

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    logger.info(f"Accuracy: {acc:.4f} ({sum(1 for a, b in zip(y_true, y_pred) if a == b)}/{len(y_true)})")

    report = classification_report(y_true, y_pred, labels=LABEL_CLASSES,
                                   target_names=LABEL_CLASSES, zero_division=0)
    logger.info(f"\nClassification Report:\n{report}")

    cm = confusion_matrix(y_true, y_pred, labels=LABEL_CLASSES)
    logger.info(f"Confusion Matrix (Zeilen=True, Spalten=Predicted):")
    logger.info(f"             {LABEL_CLASSES}")
    for label, row in zip(LABEL_CLASSES, cm):
        logger.info(f"  {label:12s} {row.tolist()}")

    # Save
    report_dict = classification_report(y_true, y_pred, labels=LABEL_CLASSES,
                                        target_names=LABEL_CLASSES, zero_division=0,
                                        output_dict=True)
    eval_results = {
        "accuracy": acc,
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "labels": LABEL_CLASSES,
        "num_samples": len(y_true),
        "method": "logit-based",
    }

    results_path = Path(output_dir) / "eval_classification.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)
    logger.success(f"Ergebnisse gespeichert unter {results_path}")


if __name__ == "__main__":
    main()
