"""
finetune_clf.py — QLoRA-Finetuning mit Classification-Head.

Verwendet AutoModelForSequenceClassification statt CausalLM.
Das Modell bekommt einen linearen 2-Klassen-Kopf auf den letzten Hidden-State.
Deutlich dateneffizienter als der generative Ansatz für binäre Klassifikation.

Optimiert für 16 GB VRAM.

Ablauf:
  1. python prepare_data_clf.py -i cases.json
  2. python finetune_clf.py
  3. python inference_clf.py --adapter output/legal-lora-clf/final
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    TrainingArguments,
    Trainer,
)

_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_OUTPUT = str(_SCRIPT_DIR.parent / "output" / "legal-lora-clf")
DEFAULT_DATASET = str(_SCRIPT_DIR.parent / "data" / "prepared_dataset_clf")
MAX_SEQ_LEN = 4096
NUM_LABELS = 2
ID2LABEL = {0: "UNTERLIEGEN", 1: "OBSIEGEN"}
LABEL2ID = {"UNTERLIEGEN": 0, "OBSIEGEN": 1}


def get_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def get_lora_config() -> LoraConfig:
    return LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )


def load_model_and_tokenizer(model_name: str):
    logger.info(f"Lade Modell: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Left-padding is critical for causal models used for classification.
    # The classification head reads the LAST non-pad token's hidden state.
    # With right-padding (default), the last token is always a pad token,
    # so the model would classify based on padding — not the actual text.
    # Left-padding ensures the real content ends at the rightmost position.
    tokenizer.padding_side = "left"

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        quantization_config=get_bnb_config(),
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    model.config.pad_token_id = tokenizer.pad_token_id

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False

    # Classification head: train in full precision with small init
    # Default random init produces extreme logits → loss explodes (>>0.69).
    # Small init keeps initial predictions near 50/50 → stable start.
    for name, param in model.named_parameters():
        if "score" in name:
            param.requires_grad = True
            param.data = param.data.float()
            if "weight" in name:
                torch.nn.init.normal_(param.data, mean=0.0, std=0.01)

    lora_config = get_lora_config()
    model = get_peft_model(model, lora_config)

    trainable, total = model.get_nb_trainable_parameters()
    logger.info(f"Trainierbare Parameter: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    return model, tokenizer


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="binary"),
        "precision": precision_score(labels, preds, average="binary", zero_division=0),
        "recall": recall_score(labels, preds, average="binary", zero_division=0),
    }


def train(model_name: str, dataset_path: str, output_dir: str, max_samples: int | None = None):
    logger.info(f"Lade Dataset aus {dataset_path}")
    dataset = load_from_disk(dataset_path)
    train_ds = dataset["train"]
    val_ds = dataset["validation"]

    if max_samples is not None:
        train_limit = min(max_samples, len(train_ds))
        val_limit = min(max_samples // 5, len(val_ds))
        train_ds = train_ds.select(range(train_limit))
        val_ds = val_ds.select(range(max(val_limit, 1)))
        logger.info(f"Test-Modus: {train_limit} Train / {val_limit} Val")

    logger.info(f"Train: {len(train_ds)} | Validation: {len(val_ds)}")

    model, tokenizer = load_model_and_tokenizer(model_name)

    # Tokenize — only truncate, NO padding here.
    # DataCollatorWithPadding will pad dynamically per batch to the longest
    # sample in that batch. This avoids wasting compute on 2048-token padding
    # when most texts are much shorter.
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_SEQ_LEN,
        )

    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    val_ds = val_ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Training config — lighter LoRA allows batch_size=2 with 4096 seq len
    num_epochs = 3
    batch_size = 2
    grad_accum = 8

    # Auto-reduce grad_accum for small datasets so training doesn't stall.
    # Each accumulated batch processes long sequences through a 7B model with
    # gradient checkpointing, so fewer accum steps = faster visible progress.
    num_batches = max(len(train_ds) // batch_size, 1)
    if num_batches <= grad_accum:
        grad_accum = 1
        logger.info(f"grad_accumulation_steps auf {grad_accum} reduziert (kleines Dataset)")

    steps_per_epoch = num_batches // grad_accum

    logger.info(
        f"Training: max {num_epochs} Epochen (Early Stopping), "
        f"eff. Batch={batch_size * grad_accum}, "
        f"~{steps_per_epoch} Schritte/Epoche"
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=num_epochs,
        warmup_ratio=0.1,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=max(steps_per_epoch // 2, 1),
        save_strategy="steps",
        save_steps=max(steps_per_epoch // 2, 1),
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        seed=42,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    logger.info("=== Training startet ===")
    train_result = trainer.train()

    metrics = train_result.metrics
    logger.info(f"Training abgeschlossen. Loss: {metrics.get('train_loss', '?'):.4f}")

    # Save adapter
    final_dir = Path(output_dir) / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger.success(f"LoRA-Adapter + Classification-Head gespeichert unter {final_dir}")

    # Final evaluation with confusion matrix
    eval_output = trainer.predict(val_ds)
    eval_metrics = eval_output.metrics
    preds = np.argmax(eval_output.predictions, axis=-1)
    labels = eval_output.label_ids

    logger.info(
        f"Eval — Accuracy: {eval_metrics.get('test_accuracy', '?'):.4f} | "
        f"F1: {eval_metrics.get('test_f1', '?'):.4f}"
    )

    cm = confusion_matrix(labels, preds)
    report = classification_report(
        labels, preds, target_names=["UNTERLIEGEN", "OBSIEGEN"], digits=4
    )
    logger.info(f"\nConfusion Matrix:\n{cm}")
    logger.info(f"\nClassification Report:\n{report}")

    # Save metrics
    metrics_path = Path(output_dir) / "training_metrics.json"
    all_metrics = {
        **metrics,
        **eval_metrics,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    logger.info(f"Metriken gespeichert unter {metrics_path}")

    return trainer


def main():
    parser = argparse.ArgumentParser(
        description="QLoRA-Finetuning mit Classification-Head"
    )
    parser.add_argument("--model", "-m", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--dataset", "-d", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--output", "-o", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        logger.error("CUDA nicht verfügbar!")
        raise SystemExit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    train(args.model, args.dataset, args.output, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
