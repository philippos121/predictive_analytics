"""
Shared configuration for the Predictive Litigation Analytics system.
Österreichisches Zivilrecht (ABGB/ZPO) — Eingabe: OGH-Urteile (TXT).
Extrahiert werden: Erstgericht-Vorbringen der Parteien + Erstgericht-Entscheidung.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # lädt .env aus dem Projektverzeichnis

# ─── Project Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
EXTRACTED_DIR = DATA_DIR / "extracted"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
MODELS_DIR = DATA_DIR / "models"

for _d in [DATA_DIR, EXTRACTED_DIR, EMBEDDINGS_DIR, MODELS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

DATASET_FILE = EXTRACTED_DIR / "cases_dataset.json"
EMBEDDINGS_FILE = EMBEDDINGS_DIR / "embeddings.h5"
MODEL_CHECKPOINT = MODELS_DIR / "litigation_model.pt"
SCALER_FILE = MODELS_DIR / "feature_scaler.pkl"
ENCODER_FILE = MODELS_DIR / "label_encoders.pkl"
TRAINING_HISTORY_FILE = MODELS_DIR / "training_history.json"
KNN_FILE = MODELS_DIR / "litigation_knn.pkl"

# ─── Adaptive Model Selection ────────────────────────────────────────────────────
# Below KNN_THRESHOLD labeled cases → k-Nearest-Neighbour (cosine similarity)
# At or above                       → LitigationClassifier neural network
KNN_THRESHOLD = 50

# ─── OpenAI Configuration ───────────────────────────────────────────────────────
# gpt-5-mini: does not support temperature parameter.
OPENAI_EXTRACTION_MODEL = "gpt-5-mini"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536  # Dimension of text-embedding-3-small

# Juristische Analyse: GPT 5.2 mit Reasoning und Web-Suche (ris.bka.gv.at).
LEGAL_ANALYSIS_MODEL = "gpt-5.2"

# API rate limiting (per worker thread)
OPENAI_REQUEST_DELAY_SEC = 0.3       # Delay between sequential API calls within one file
OPENAI_MAX_RETRIES = 5
OPENAI_RETRY_DELAY_SEC = 2.0

# ─── Parallelization ────────────────────────────────────────────────────────────
# Number of parallel extraction workers (each worker = 1 OGH-Urteil gleichzeitig).
# Embeddings within each file are always parallelized (2 calls gleichzeitig).
# Empfehlung: 5 Workers bei Tier-1 OpenAI-Account (10k RPM).
PARALLEL_WORKERS = 500

# ─── Outcome Labels ─────────────────────────────────────────────────────────────
OUTCOME_LABELS = {
    0: "Unterliegen",
    1: "Teilweises Obsiegen/Unterliegen",
    2: "Obsiegen",
}

OUTCOME_COLORS = {
    0: "#E74C3C",   # Red
    1: "#F39C12",   # Orange
    2: "#27AE60",   # Green
}

OUTCOME_ICONS = {
    0: "❌",
    1: "⚖️",
    2: "✅",
}

# ─── Legal Claim Types (Anspruchsarten) ─────────────────────────────────────────
# Typische Klagegegenstände im österreichischen Zivilrecht (ABGB/ZPO)
CLAIM_TYPES = [
    "Kaufpreisforderung",
    "Werkentgelt",
    "Schadenersatz (Vertrag)",
    "Schadenersatz (Delikt)",
    "Gewährleistung",
    "Irrtum / Anfechtung",
    "Ungerechtfertigte Bereicherung",
    "Darlehensrückzahlung",
    "Miete / Pachtzins",
    "Herausgabe / Eigentümerklage",
    "Unterlassung",
    "Feststellungsklage",
    "Wechsel / Scheck",
    "Unterhaltsklage",
    "Sonstige Forderung",
]

# ─── Defense Types (Einwendungen des Beklagten) ──────────────────────────────────
DEFENSE_TYPES = [
    "mangel",               # Sachmängel / Gewährleistung (§ 922 ABGB)
    "irrtum",               # Irrtum (§ 871 ABGB)
    "nichterfuellung",      # Nichterfüllung / Zug-um-Zug (§ 1052 ABGB)
    "verjaehrung",          # Verjährung (§ 1478 ABGB)
    "aufrechnung",          # Gegenforderung / Aufrechnung (§ 1438 ABGB)
    "listige_irrefuehrung", # Arglistige Irreführung (§ 870 ABGB)
    "unmoeglichkeit",       # Unmöglichkeit der Leistung (§ 878 ABGB)
    "unzustaendigkeit",     # Unzuständigkeit des Gerichts (JN)
    "fehlende_aktivlegitimation",  # Fehlende Aktivlegitimation
    "keine_passivlegitimation",    # Fehlende Passivlegitimation
    "zahlung_erfolgt",      # Zahlung bereits geleistet (§ 1412 ABGB)
    "andere",               # Sonstige Einwendungen
]

DEFENSE_LABELS = {
    "mangel": "Sachmangel / Gewährleistung (§ 922 ABGB)",
    "irrtum": "Irrtum (§ 871 ABGB)",
    "nichterfuellung": "Nichterfüllung (§ 1052 ABGB)",
    "verjaehrung": "Verjährung (§ 1478 ABGB)",
    "aufrechnung": "Aufrechnung / Gegenforderung (§ 1438 ABGB)",
    "listige_irrefuehrung": "Arglistige Täuschung (§ 870 ABGB)",
    "unmoeglichkeit": "Unmöglichkeit der Leistung (§ 878 ABGB)",
    "unzustaendigkeit": "Unzuständigkeit (JN)",
    "fehlende_aktivlegitimation": "Fehlende Aktivlegitimation",
    "keine_passivlegitimation": "Fehlende Passivlegitimation",
    "zahlung_erfolgt": "Zahlung bereits erfolgt (§ 1412 ABGB)",
    "andere": "Sonstige Einwendungen",
}

# ─── Verfahrensarten (Prozessart beim Erstgericht) ──────────────────────────────
VERFAHRENSARTEN = [
    "Mahnverfahren",          # §§ 244 ff ZPO — häufig Forderungen
    "Ordentliches Verfahren", # Regelfall
    "Urkundenverfahren",      # § 448 ZPO — urkundengestützte Forderungen
    "Wechselverfahren",       # § 555 ZPO — Wechsel/Scheck
]

# ─── Beweismittel-Typen ──────────────────────────────────────────────────────────
# Typen aufgenommener oder angebotener Beweismittel beim Erstgericht
BEWEISMITTEL_TYPEN = [
    "urkunden",           # Urkunden, Verträge, Rechnungen, Schriftverkehr
    "zeugen",             # Zeugenvernehmungen
    "sachverstaendige",   # Sachverständigengutachten (angeboten, nicht zwingend bestellt)
    "parteienvernehmung", # Parteienvernehmung / Parteiaussage
]

# ─── Text Sections for Embeddings ───────────────────────────────────────────────
# Genau zwei Abschnitte fließen als Embeddings ins Modell ein:
# - Kläger-Vorbringen: Was begehrt der Kläger?
# - Beklagten-Vorbringen: Welche Einwendungen macht der Beklagte?
#
# Beweiswürdigung, Feststellungen und rechtliche Beurteilung sind KEIN Input —
# sie sind Teil des Outputs bzw. der richterlichen Entscheidungsfindung.
EMBEDDING_SECTIONS = [
    "klaegervorbringen",
    "beklagtenvorbringen",
]

EMBEDDING_SECTION_LABELS = {
    "klaegervorbringen": "Kläger-Vorbringen",
    "beklagtenvorbringen": "Beklagten-Vorbringen",
}

# ─── Adaptive Neural Network Configurations ──────────────────────────────────────
# Automatically selected in trainer.py based on n_training_cases.
#
# Root cause of train≫val gap (overfitting):
#   freeze_encoders=False with a 2-layer encoder creates 1.64M learnable params
#   in the encoders alone.  With few training cases these memorise the training
#   set instead of generalising → train 80 % / val 50 %.
#
# Rule of thumb (n_train = 80 % of labeled cases):
#   n_train < 1 500  → SMALL  : frozen encoders, tiny fusion         (~  40k params)
#   1 500 – 5 000    → MEDIUM : 1-layer learned encoder, med. fusion  (~ 820k params)
#   ≥ 5 000          → LARGE  : 2-layer learned encoder, full fusion  (~1.84M params)
#
# Parameter budgets (text-embedding-3-small, 1536-dim input):
#   SMALL  — 2 enc (frozen 1536→128): 394k (no grad) + fusion [128,64]: ~36k trainable
#   MEDIUM — 2 enc (learned 1536→128): 394k + fusion [128,64]: ~430k trainable
#   LARGE  — 2 enc (learned 1536→256→128): 919k + fusion [256,128]: ~1.05M trainable
#
# vs. text-embedding-3-large (3072-dim): encoder params halved → less overfitting risk.

# ── SMALL: < 1 500 training cases ────────────────────────────────────────────────
# Frozen encoders prevent the 1536→128 projection from memorising.
# Only the tiny attention + fusion head (~36k params) trains → near-zero overfit risk.
NN_CONFIG_SMALL = {
    "embedding_hidden_dim": 0,           # single linear layer: 1536 → 128 directly
    "embedding_output_dim": 128,
    "embedding_noise_std": 0.05,         # stronger noise for small data
    "freeze_encoders": True,             # FROZEN — prevents memorisation
    "use_section_attention": False,      # disabled: adds params + high-dim noise in small data
    "use_interaction_features": False,   # disabled: diff/prod create memorisable patterns in high-dim space
    "structured_dim": 0,
    "fusion_dims": [128, 64],            # fusion input: 2×128 = 256 → 128 → 64
    "dropout_embedding": 0.40,
    "dropout_fusion": 0.50,
    "num_classes": 3,
}

# ── MEDIUM: 1 500 – 5 000 training cases ─────────────────────────────────────────
# Single-layer learned projection (1536→128) — half the params of the 2-layer encoder.
# Higher dropout + weight decay compensate for the reduced dataset size.
NN_CONFIG_MEDIUM = {
    "embedding_hidden_dim": 0,           # single linear layer: 1536 → 128 directly
    "embedding_output_dim": 128,
    "embedding_noise_std": 0.03,
    "freeze_encoders": False,            # LEARNED single projection
    "use_section_attention": False,      # disabled: adds memorisable signal without clear gain
    "use_interaction_features": False,   # disabled: element-wise ops on high-dim vectors overfit
    "structured_dim": 0,
    "fusion_dims": [128, 64],            # fusion input: 2×128 = 256 → 128 → 64
    "dropout_embedding": 0.30,
    "dropout_fusion": 0.45,
    "num_classes": 3,
}

# ── LARGE: ≥ 5 000 training cases ────────────────────────────────────────────────
# With ~8 000 training examples the model can LEARN the 1536→128 projection
# instead of relying on a frozen one.  A learned projection finds the
# class-relevant directions in the embedding space.
#
# Fusion input breakdown (no interaction features, no attention):
#   kl_enc(128) + bk_enc(128) = 256
#
# Regularisation raised for 8k dataset (6 400 training samples, ~160 params/example
# with 1536-dim input — down from ~287 with 3072-dim → inherently less overfit risk).
NN_CONFIG = {
    "embedding_hidden_dim": 256,         # two-layer encoder: 1536 → 256 → 128
    "embedding_output_dim": 128,         # learned — finds class-relevant directions
    "embedding_noise_std": 0.02,         # raised 0.01→0.02: stronger input noise regularisation
    "freeze_encoders": False,            # LEARNED projection (needs 5 000+ training cases)
    "use_section_attention": False,      # disabled: linear baseline first; add back only if needed
    "use_interaction_features": False,   # disabled: diff/prod in 1536-dim space create memorisable noise
    "structured_dim": 0,                 # embeddings only
    "fusion_dims": [256, 128],           # fusion input: 2×128 = 256 → 256 → 128
    "dropout_embedding": 0.40,           # raised 0.20→0.40: prevents encoder memorising noise
    "dropout_fusion": 0.45,              # raised 0.35→0.45: prevents fusion layers memorising
    "num_classes": 3,
}

# ─── Training Configuration ──────────────────────────────────────────────────────
# Base config — trainer.py overrides epochs/weight_decay/batch_size adaptively.
# User-settable UI params (epochs, learning_rate, early_stopping_patience)
# are applied on top of these defaults when changed in the sidebar.

TRAINING_CONFIG = {
    "epochs": 300,
    "batch_size": 64,                 # larger batch for larger dataset
    "learning_rate": 2e-4,            # slightly lower LR for larger learned encoder
    "weight_decay": 1e-3,             # L2 regularisation
    "lr_scheduler_patience": 30,
    "lr_scheduler_factor": 0.5,
    "early_stopping_patience": 50,
    "use_swa": True,
    "val_split": 0.20,                # 20 % val → 8 000 training examples at 10k
    "random_seed": 42,
    "gradient_clip": 1.0,
    "label_smoothing": 0.1,
    "mixup_alpha": 0.3,
}

# ─── UI Configuration ───────────────────────────────────────────────────────────
APP_TITLE_EXTRACTOR = "Litigation Data Extractor (OGH-Urteile, ö. Zivilrecht)"
APP_TITLE_MODEL = "Predictive Litigation Analytics (OGH-Datenbasis)"
APP_VERSION = "2.1.0"
APP_AUTHOR = "Predictive Litigation Analytics System"
