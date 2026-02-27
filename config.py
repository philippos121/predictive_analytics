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
# gpt-4.1-mini: released April 2025, cost-efficient with 1M context window.
# Pricing: $0.40/1M input tokens, $1.60/1M output tokens.
# Update OPENAI_EXTRACTION_MODEL when newer models are released.
OPENAI_EXTRACTION_MODEL = "gpt-4.1-mini"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 3072  # Dimension of text-embedding-3-large

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

# ─── Neural Network Configuration ───────────────────────────────────────────────
# Dimensionen sind auf kleine Datensätze (50–300 Fälle) ausgelegt.
# Die Embeddings (text-embedding-3-large) sind bereits hochwertige Repräsentationen,
# sodass einfache Projektionen ausreichen. Ein zu tiefes Netz würde bei wenigen
# Fällen overfittten.
#
# Parameteranzahl ca.:
#   3 × EmbeddingEncoder (3072→256→128):  ~2,5 Mio.
#   StructuredEncoder (37→64→64):         ~6,5 K
#   Fusion (448→128→3):                   ~58 K
#   Gesamt: ~2,56 Mio.  (früher: ~5,7 Mio.)
#
# Für >500 Fälle können embedding_hidden_dim=512, embedding_output_dim=256
# und fusion_dims=[256, 128] gesetzt werden.
NN_CONFIG = {
    "embedding_hidden_dim": 256,     # Intermediate dim per embedding encoder
    "embedding_output_dim": 128,     # Output dim per embedding encoder
    "structured_hidden_dim": 64,     # Structured feature encoder hidden dim
    "fusion_dims": [128],            # Single hidden fusion layer (448 → 128 → 3)
    "dropout_embedding": 0.3,
    "dropout_fusion": 0.3,
    "num_classes": 3,                # win / partial / loss
}

# ─── Training Configuration ─────────────────────────────────────────────────────
TRAINING_CONFIG = {
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

# ─── UI Configuration ───────────────────────────────────────────────────────────
APP_TITLE_EXTRACTOR = "Litigation Data Extractor (OGH-Urteile, ö. Zivilrecht)"
APP_TITLE_MODEL = "Predictive Litigation Analytics (OGH-Datenbasis)"
APP_VERSION = "2.1.0"
APP_AUTHOR = "Predictive Litigation Analytics System"
