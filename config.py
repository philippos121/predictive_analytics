"""
Shared configuration for the Predictive Litigation Analytics system.
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
# NOTE: "GPT 5.2 mini" does not exist as an OpenAI model (as of 2026).
# Using gpt-4o-mini as the state-of-the-art cost-efficient model.
# Update OPENAI_EXTRACTION_MODEL when newer models are available.
OPENAI_EXTRACTION_MODEL = "gpt-4o-mini"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 3072  # Dimension of text-embedding-3-large

# API rate limiting
OPENAI_REQUEST_DELAY_SEC = 0.5       # Delay between API requests
OPENAI_MAX_RETRIES = 5
OPENAI_RETRY_DELAY_SEC = 2.0

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
    "mangel",               # Sachmängel / Gewährleistung
    "irrtum",               # Irrtum (§ 871 ABGB)
    "nichterfuellung",      # Nichterfüllung / Einrede des nicht erfüllten Vertrags
    "verjaehrung",          # Verjährung
    "aufrechnung",          # Gegenforderung / Aufrechnung
    "listige_irrefuehrung", # Arglistige Täuschung
    "unmoeglichkeit",       # Unmöglichkeit der Leistung
    "unzustaendigkeit",     # Unzuständigkeit des Gerichts
    "fehlende_aktivlegitimation",  # Fehlende Aktivlegitimation
    "keine_passivlegitimation",    # Fehlende Passivlegitimation
    "zahlung_erfolgt",      # Zahlung bereits geleistet
    "andere",               # Sonstige Einwendungen
]

DEFENSE_LABELS = {
    "mangel": "Sachmangel / Gewährleistung",
    "irrtum": "Irrtum (§ 871 ABGB)",
    "nichterfuellung": "Nichterfüllung (§ 1052 ABGB)",
    "verjaehrung": "Verjährung",
    "aufrechnung": "Aufrechnung / Gegenforderung",
    "listige_irrefuehrung": "Arglistige Täuschung (§ 870 ABGB)",
    "unmoeglichkeit": "Unmöglichkeit der Leistung",
    "unzustaendigkeit": "Unzuständigkeit",
    "fehlende_aktivlegitimation": "Fehlende Aktivlegitimation",
    "keine_passivlegitimation": "Fehlende Passivlegitimation",
    "zahlung_erfolgt": "Zahlung bereits erfolgt",
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
APP_TITLE_EXTRACTOR = "⚖️ Litigation Data Extractor"
APP_TITLE_MODEL = "🔮 Predictive Litigation Analytics"
APP_VERSION = "1.0.0"
APP_AUTHOR = "Predictive Litigation Analytics System"
