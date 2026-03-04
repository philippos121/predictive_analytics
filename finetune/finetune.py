"""
finetune.py — QLoRA-Finetuning eines 7B-LLM für Verfahrensausgang-Prognose.

Optimiert für 16 GB VRAM (z.B. RTX 4080, RTX 3090, RTX A4000).
Verwendet 4-bit-Quantisierung (NF4) + LoRA-Adapter über die PEFT-Bibliothek.

Ablauf:
  1. python prepare_data.py -i cases.json -o data/prepared_dataset
  2. python finetune.py --dataset data/prepared_dataset
  3. python inference.py --adapter output/legal-lora --prompt "..."

Basis-Modell: mistralai/Mistral-7B-Instruct-v0.3
(austauschbar über --model, z.B. meta-llama/Llama-3.1-8B-Instruct)
"""

import argparse
import json
import os
import re
from pathlib import Path

import torch
from datasets import load_from_disk
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_OUTPUT = str(_SCRIPT_DIR.parent / "output" / "legal-lora")
DEFAULT_DATASET = str(_SCRIPT_DIR.parent / "data" / "prepared_dataset")
MAX_SEQ_LEN = 1024  # Token-Limit pro Sample (spart VRAM)
LABEL_CLASSES = ["OBSIEGEN", "UNTERLIEGEN"]


# ---------------------------------------------------------------------------
# 4-bit-Quantisierungskonfiguration
# ---------------------------------------------------------------------------
def get_bnb_config() -> BitsAndBytesConfig:
    """Erstellt die BitsAndBytes-Konfiguration für 4-bit QLoRA."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,  # Nested quantization — spart ~0.4 GB
    )


# ---------------------------------------------------------------------------
# LoRA-Konfiguration
# ---------------------------------------------------------------------------
def get_lora_config() -> LoraConfig:
    """LoRA-Adapter-Konfiguration (rank 64 ist ein guter Kompromiss)."""
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,                       # LoRA-Rang
        lora_alpha=128,             # Skalierung (alpha = 2*r ist Faustregel)
        lora_dropout=0.05,
        target_modules=[            # Standard-Module für Mistral / Llama
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
    )


# ---------------------------------------------------------------------------
# Modell laden
# ---------------------------------------------------------------------------
def load_model_and_tokenizer(model_name: str):
    """Lädt Modell (4-bit quantisiert) und Tokenizer."""
    logger.info(f"Lade Modell: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    # Pad-Token setzen (viele Modelle haben keines)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=get_bnb_config(),
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",  # "flash_attention_2" falls installiert
    )

    # Gradient-Checkpointing vorbereiten
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False  # Muss für Training deaktiviert werden

    # LoRA-Adapter anwenden
    lora_config = get_lora_config()
    model = get_peft_model(model, lora_config)

    trainable, total = model.get_nb_trainable_parameters()
    logger.info(
        f"Trainierbare Parameter: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.2f}%)"
    )

    return model, tokenizer


# ---------------------------------------------------------------------------
# Trainings-Konfiguration
# ---------------------------------------------------------------------------
def get_training_args(output_dir: str, num_train_samples: int) -> SFTConfig:
    """Erstellt die Trainings-Argumente, optimiert für 16 GB VRAM."""

    # Epochen und Schritte
    num_epochs = 3
    batch_size = 2
    grad_accum = 8  # Effektive Batch-Größe: 2 * 8 = 16
    steps_per_epoch = num_train_samples // (batch_size * grad_accum)
    total_steps = steps_per_epoch * num_epochs

    logger.info(
        f"Training: {num_epochs} Epochen, effektive Batch-Größe={batch_size * grad_accum}, "
        f"~{total_steps} Schritte"
    )

    return SFTConfig(
        output_dir=output_dir,

        # --- Batching ---
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,

        # --- Scheduler ---
        num_train_epochs=num_epochs,
        warmup_ratio=0.05,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        weight_decay=0.01,

        # --- Speicheroptimierung ---
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,                              # BFloat16 (Ampere+)
        optim="paged_adamw_8bit",               # 8-bit-Optimizer (spart ~2 GB)
        max_grad_norm=0.3,

        # --- Logging ---
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=max(steps_per_epoch // 2, 1),
        save_strategy="steps",
        save_steps=max(steps_per_epoch, 1),
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # --- SFT-spezifisch ---
        max_length=MAX_SEQ_LEN,
        packing=False,                          # Kein Packing (Textlänge variiert stark)
        dataset_text_field=None,                # Wir nutzen das chat-Format

        # --- Sonstiges ---
        report_to="none",                       # Kein W&B / MLflow nötig
        seed=42,
        dataloader_pin_memory=True,
        remove_unused_columns=True,
    )


# ---------------------------------------------------------------------------
# Klassifikations-Evaluation (Accuracy, F1, Confusion Matrix)
# ---------------------------------------------------------------------------
def evaluate_classification(model, tokenizer, val_ds, output_dir: str):
    """Generiert Vorhersagen auf dem Validierungs-Set und berechnet Metriken."""
    logger.info("=== Klassifikations-Evaluation ===")

    model.eval()
    y_true = []
    y_pred = []

    for i, sample in enumerate(val_ds):
        messages = sample["messages"]

        # Erwartetes Label aus der Assistant-Antwort
        true_label = messages[-1]["content"].strip()
        y_true.append(true_label)

        # Nur System + User als Input (ohne Assistant-Antwort)
        input_messages = messages[:-1]
        input_text = tokenizer.apply_chat_template(
            input_messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True,
                           max_length=MAX_SEQ_LEN).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                temperature=1.0,
            )

        # Nur die generierten Tokens dekodieren
        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # Label aus der Antwort extrahieren (erstes Wort matchen)
        pred_label = "UNBEKANNT"
        for label in LABEL_CLASSES:
            if label in prediction.upper():
                pred_label = label
                break
        y_pred.append(pred_label)

        if (i + 1) % 50 == 0:
            logger.info(f"  {i + 1}/{len(val_ds)} Samples ausgewertet...")

    # Metriken berechnen
    acc = accuracy_score(y_true, y_pred)
    logger.info(f"Accuracy: {acc:.4f} ({sum(1 for a, b in zip(y_true, y_pred) if a == b)}/{len(y_true)})")

    # Classification Report (Precision, Recall, F1 pro Klasse)
    report = classification_report(
        y_true, y_pred,
        labels=LABEL_CLASSES,
        target_names=LABEL_CLASSES,
        zero_division=0,
    )
    logger.info(f"\nClassification Report:\n{report}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=LABEL_CLASSES)
    logger.info(f"Confusion Matrix (Zeilen=True, Spalten=Predicted):")
    logger.info(f"             {LABEL_CLASSES}")
    for label, row in zip(LABEL_CLASSES, cm):
        logger.info(f"  {label:12s} {row.tolist()}")

    # Metriken als JSON speichern
    report_dict = classification_report(
        y_true, y_pred,
        labels=LABEL_CLASSES,
        target_names=LABEL_CLASSES,
        zero_division=0,
        output_dict=True,
    )
    eval_results = {
        "accuracy": acc,
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "labels": LABEL_CLASSES,
        "num_samples": len(y_true),
        "num_unknown": y_pred.count("UNBEKANNT"),
    }

    results_path = Path(output_dir) / "eval_classification.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)
    logger.success(f"Klassifikations-Ergebnisse gespeichert unter {results_path}")

    return eval_results


# ---------------------------------------------------------------------------
# Hauptfunktion
# ---------------------------------------------------------------------------
def train(
    model_name: str,
    dataset_path: str,
    output_dir: str,
):
    """Führt das vollständige QLoRA-Finetuning durch."""

    # 1. Dataset laden
    logger.info(f"Lade Dataset aus {dataset_path}")
    dataset = load_from_disk(dataset_path)
    train_ds = dataset["train"]
    val_ds = dataset["validation"]
    logger.info(f"Train: {len(train_ds)} Samples | Validation: {len(val_ds)} Samples")

    # 2. Modell + Tokenizer laden
    model, tokenizer = load_model_and_tokenizer(model_name)

    # 3. Training konfigurieren
    training_args = get_training_args(output_dir, len(train_ds))

    # 4. Trainer erstellen
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    # 5. Training starten
    logger.info("=== Training startet ===")
    train_result = trainer.train()

    # 6. Metriken loggen
    metrics = train_result.metrics
    logger.info(f"Training abgeschlossen. Loss: {metrics.get('train_loss', '?'):.4f}")

    # 7. Bestes Modell (nur LoRA-Adapter) speichern
    final_dir = Path(output_dir) / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger.success(f"LoRA-Adapter gespeichert unter {final_dir}")

    # 8. Evaluation — Loss
    logger.info("=== Evaluation auf Validierungs-Set ===")
    eval_metrics = trainer.evaluate()
    logger.info(f"Eval-Loss: {eval_metrics.get('eval_loss', '?'):.4f}")

    # 9. Klassifikations-Metriken (Accuracy, F1, Confusion Matrix)
    eval_classification(model, tokenizer, val_ds, output_dir)

    # Trainings-Metriken speichern
    metrics_path = Path(output_dir) / "training_metrics.json"
    all_metrics = {**metrics, **eval_metrics}
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"Trainings-Metriken gespeichert unter {metrics_path}")

    return trainer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="QLoRA-Finetuning für österreichische Zivilverfahrens-Prognose"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Hugging-Face-Modell-ID (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default=DEFAULT_DATASET,
        help=f"Pfad zum vorbereiteten Dataset (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Ausgabeverzeichnis für Adapter + Checkpoints (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Pfad zu einem Checkpoint zum Fortsetzen des Trainings",
    )
    args = parser.parse_args()

    # CUDA-Check
    if not torch.cuda.is_available():
        logger.error("CUDA nicht verfügbar! QLoRA benötigt eine NVIDIA-GPU.")
        raise SystemExit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    if gpu_mem < 14:
        logger.warning(
            f"Nur {gpu_mem:.1f} GB VRAM erkannt. Mindestens 16 GB empfohlen. "
            "Training könnte fehlschlagen oder sehr langsam sein."
        )

    trainer = train(args.model, args.dataset, args.output)

    if args.resume:
        trainer.train(resume_from_checkpoint=args.resume)


if __name__ == "__main__":
    main()
