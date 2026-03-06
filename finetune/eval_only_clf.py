"""Standalone evaluation for the SequenceClassification LoRA model.

Usage:
  python eval_only_clf.py
  python eval_only_clf.py --adapter D:/legal-lora-clf/final --dataset data/prepared_dataset_clf
  python eval_only_clf.py --max_samples 200
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from loguru import logger
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_ADAPTER = "D:/legal-lora-clf/final"
DEFAULT_DATASET = str(_SCRIPT_DIR.parent / "data" / "prepared_dataset_clf")
MAX_SEQ_LEN = 4096
NUM_LABELS = 2
ID2LABEL = {0: "UNTERLIEGEN", 1: "OBSIEGEN"}
LABEL2ID = {"UNTERLIEGEN": 0, "OBSIEGEN": 1}


def main():
    parser = argparse.ArgumentParser(description="Standalone eval for CLF LoRA model")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Base model name")
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER, help="Path to saved LoRA adapter")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Path to prepared dataset")
    parser.add_argument("--output", default=None, help="Dir for results JSON (default: adapter dir)")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit validation samples")
    args = parser.parse_args()

    output_dir = args.output or args.adapter

    if not torch.cuda.is_available():
        logger.error("CUDA nicht verfügbar!")
        raise SystemExit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    # Load tokenizer
    logger.info(f"Lade Tokenizer von {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # Load base model in 4-bit
    logger.info(f"Lade Base-Model in 4-bit: {args.model}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "eager"

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=attn_impl,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # Load LoRA adapter
    logger.info(f"Lade LoRA-Adapter von {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    trainable, total = model.get_nb_trainable_parameters()
    logger.info(f"Parameter: {trainable:,} trainierbar / {total:,} gesamt")

    # Load dataset
    logger.info(f"Lade Dataset von {args.dataset}")
    ds = load_from_disk(args.dataset)
    val_ds = ds["validation"]
    if args.max_samples:
        val_ds = val_ds.select(range(min(args.max_samples, len(val_ds))))
    logger.info(f"Validation-Samples: {len(val_ds)}")

    # Tokenize
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_SEQ_LEN,
        )

    val_ds = val_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Use Trainer.predict for batched evaluation
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro"),
            "f1_binary": f1_score(labels, preds, average="binary"),
        }

    eval_args = TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=1,
        bf16=True,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=True,
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    logger.info("=== Evaluation startet ===")
    eval_output = trainer.predict(val_ds)
    eval_metrics = eval_output.metrics
    preds = np.argmax(eval_output.predictions, axis=-1)
    labels = eval_output.label_ids

    acc = eval_metrics.get('test_accuracy', eval_metrics.get('eval_accuracy'))
    f1 = eval_metrics.get('test_f1_macro', eval_metrics.get('eval_f1_macro'))
    f1_bin = eval_metrics.get('test_f1_binary', eval_metrics.get('eval_f1_binary'))
    logger.info(f"Accuracy: {acc:.4f} | F1-macro: {f1:.4f} | F1-binary: {f1_bin:.4f}")

    cm = confusion_matrix(labels, preds)
    report = classification_report(
        labels, preds, target_names=["UNTERLIEGEN", "OBSIEGEN"], digits=4
    )
    logger.info(f"\nConfusion Matrix:\n{cm}")
    logger.info(f"\nClassification Report:\n{report}")

    # Save results
    report_dict = classification_report(
        labels, preds, target_names=["UNTERLIEGEN", "OBSIEGEN"],
        digits=4, output_dict=True
    )
    eval_results = {
        "accuracy": float(acc),
        "f1_macro": float(f1),
        "f1_binary": float(f1_bin),
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "labels": ["UNTERLIEGEN", "OBSIEGEN"],
        "num_samples": len(labels),
        "method": "sequence_classification",
        "adapter": args.adapter,
    }

    results_path = Path(output_dir) / "eval_classification_clf.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)
    logger.success(f"Ergebnisse gespeichert unter {results_path}")


if __name__ == "__main__":
    main()
