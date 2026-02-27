"""
Shared configuration for the Predictive Litigation Analytics system.
Österreichisches Zivilrecht (ABGB/ZPO) — Eingabe: OGH-Urteile (TXT).
Extrahiert werden: Erstgericht-Vorbringen der Parteien + Erstgericht-Entscheidung.
"""

import os
from pathlib import Path

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
OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 3072  # Dimension of text-embedding-3-large

# Juristische Analyse: GPT 5.2 mit Reasoning und Web-Suche (ris.bka.gv.at).
LEGAL_ANALYSIS_MODEL = "gpt-5.2"

# API rate limiting (per worker thread)
OPENAI_REQUEST_DELAY_SEC = 0.3       # Delay between sequential API calls within one file
OPENAI_MAX_RETRIES = 5
OPENAI_RETRY_DELAY_SEC = 2.0

# ─── Parallelization ────────────────────────────────────────────────────────────
# Number of parallel extraction workers (each worker = 1 OGH-Urteil gleichzeitig).
# Embeddings within each file are always parallelized (3 calls gleichzeitig).
# Empfehlung: 5 Workers bei Tier-1 OpenAI-Account (10k RPM).
PARALLEL_WORKERS = 5

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

# ─── Text Sections for Embeddings ───────────────────────────────────────────────
# Nur diese drei Abschnitte fließen als Embeddings ins Modell ein:
# - Kläger-Vorbringen: Was begehrt der Kläger?
# - Beklagten-Vorbringen: Welche Einwendungen macht der Beklagte?
# - Aufgenommene Beweise: Welche Beweise wurden vom Gericht aufgenommen?
#   (faktische Beschreibung ohne Bewertung, wird per GPT aus Beweiswürdigung
#    und Feststellungen generiert)
#
# Beweiswürdigung, Feststellungen und rechtliche Beurteilung sind KEIN Input —
# sie sind Teil des Outputs bzw. der richterlichen Entscheidungsfindung.
EMBEDDING_SECTIONS = [
    "klaegervorbringen",
    "beklagtenvorbringen",
    "aufgenommene_beweise",
]

EMBEDDING_SECTION_LABELS = {
    "klaegervorbringen": "Kläger-Vorbringen",
    "beklagtenvorbringen": "Beklagten-Vorbringen",
    "aufgenommene_beweise": "Aufgenommene Beweise",
}

# ─── Adaptive Model-Size Thresholds ─────────────────────────────────────────────
# trainer.py selects one of three configs automatically based on n_labeled cases.
#   < NN_SMALL_THRESHOLD  → small  (kNN or tiny NN; heavy overfit risk)
#   < NN_MEDIUM_THRESHOLD → medium (moderate capacity + regularization)
#   ≥ NN_MEDIUM_THRESHOLD → large  (full capacity + strong regularization)
NN_SMALL_THRESHOLD = 150
NN_MEDIUM_THRESHOLD = 600

# ─── Neural Network Configurations ──────────────────────────────────────────────
# text-embedding-3-large outputs 3072-dim vectors that are already highly semantic.
# The first projection layer (3072 → hidden_dim) dominates parameter count, so
# we keep hidden_dim fixed across tiers and instead scale depth + regularization.
#
# Small  (50–149 cases)   — ~2.56M params; mild dropout
# Medium (150–599 cases)  — ~2.57M params; deeper fusion, more dropout
# Large  (≥ 600 cases)    — ~2.82M params; deepest fusion, heavy dropout + L2

NN_CONFIG = {                        # Small / backward-compatible default
    "embedding_hidden_dim": 256,
    "embedding_output_dim": 128,
    "structured_hidden_dim": 64,
    "fusion_dims": [128],
    "dropout_embedding": 0.30,
    "dropout_fusion": 0.30,
    "num_classes": 3,
}

NN_CONFIG_MEDIUM = {                 # Medium: reduced hidden, deeper fusion, moderate dropout
    "embedding_hidden_dim": 128,     # Reduced from 256: 3072×128 vs 3072×256 → 50 % fewer params
    "embedding_output_dim": 128,
    "embedding_noise_std": 0.01,     # Light Gaussian noise on embeddings
    "structured_hidden_dim": 128,
    "fusion_dims": [256, 128],
    "dropout_embedding": 0.40,
    "dropout_fusion": 0.40,
    "num_classes": 3,
}

NN_CONFIG_LARGE = {                  # Large: 2-layer encoder, moderate dropout + noise
    "embedding_hidden_dim": 256,     # Restore nonlinear projection: 3072→256→256
    "embedding_output_dim": 256,     # Larger per-section encoding (3×256=768 fusion input)
    "embedding_noise_std": 0.02,     # Gaussian noise (regularization)
    "structured_hidden_dim": 128,
    "fusion_dims": [512, 256],       # 768→512→256
    "dropout_embedding": 0.35,       # Reduced: 50% caused underfitting at ~57% val-acc
    "dropout_fusion": 0.35,
    "num_classes": 3,
}

# ─── Training Configurations ─────────────────────────────────────────────────────
# Base (small): current defaults kept for backward compatibility.
# Medium / Large: lower LR, larger batch, higher weight-decay, more epochs.
# User-settable UI params (epochs, learning_rate, early_stopping_patience)
# are applied on top of these when explicitly changed in the sidebar.

TRAINING_CONFIG = {                  # Small / backward-compatible default
    "epochs": 200,
    "batch_size": 16,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "lr_scheduler_patience": 15,
    "lr_scheduler_factor": 0.5,
    "early_stopping_patience": 30,
    "val_split": 0.2,
    "random_seed": 42,
    "gradient_clip": 1.0,
}

TRAINING_CONFIG_MEDIUM = {
    "epochs": 300,
    "batch_size": 32,
    "learning_rate": 5e-4,
    "weight_decay": 5e-4,
    "lr_scheduler_patience": 20,
    "lr_scheduler_factor": 0.5,
    "early_stopping_patience": 45,
    "val_split": 0.2,
    "random_seed": 42,
    "gradient_clip": 1.0,
}

TRAINING_CONFIG_LARGE = {
    "epochs": 400,
    "batch_size": 32,
    "learning_rate": 3e-4,
    "weight_decay": 5e-4,            # Reduced from 1e-3: less regularization to match lower dropout
    "lr_scheduler_patience": 25,
    "lr_scheduler_factor": 0.5,
    "early_stopping_patience": 60,
    "val_split": 0.2,
    "random_seed": 42,
    "gradient_clip": 1.0,
}

# ─── UI Configuration ───────────────────────────────────────────────────────────
APP_TITLE_EXTRACTOR = "Litigation Data Extractor (OGH-Urteile, ö. Zivilrecht)"
APP_TITLE_MODEL = "Predictive Litigation Analytics (OGH-Datenbasis)"
APP_VERSION = "2.1.0"
APP_AUTHOR = "Predictive Litigation Analytics System"
