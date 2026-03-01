"""
Shared configuration for the Predictive Litigation Analytics system.

v3.0 — Hybrid: Text Embeddings + Structured Data.
Uses pre-computed text-embedding-3-large vectors (kläger/beklagten vorbringen)
combined with structured metadata features for classification.
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
# Below KNN_THRESHOLD labeled cases → k-Nearest-Neighbour (structured features)
# At or above                       → LitigationClassifier neural network
KNN_THRESHOLD = 50

# ─── OpenAI Configuration ───────────────────────────────────────────────────────
OPENAI_EXTRACTION_MODEL = "gpt-5-nano-2025-08-07"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 3072  # Full dimension of text-embedding-3-large
EMBEDDING_DIM_USED = 1024  # Truncated dimension for training (first N dims)

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

# ─── Embedding Sections ──────────────────────────────────────────────────────────
# Text sections whose embeddings are used for training.
# Only kläger + beklagten vorbringen — these contain the actual legal arguments.
EMBEDDING_SECTIONS = [
    "klaegervorbringen",
    "beklagtenvorbringen",
]

EMBEDDING_SECTION_LABELS = {
    "klaegervorbringen": "Kläger-Vorbringen",
    "beklagtenvorbringen": "Beklagten-Vorbringen",
}

# ─── Structured Legal Analysis Schema ────────────────────────────────────────────
# Detailed extraction schema for Austrian civil proceedings.
# Extracted by GPT from each ruling. Used as auxiliary structured features.

RECHTSGEBIET_CATEGORIES = [
    "Schuldrecht_Vertrag",
    "Schuldrecht_Gesetzlich",
    "Sachenrecht",
    "Familienrecht",
    "Erbrecht",
    "Immaterialgueterrecht",
    "Gesellschaftsrecht",
    "Sonstiges",
]

# Boolean fields from legal_analysis used as features for training.
# Grouped by schema section for reference.
LEGAL_ANALYSIS_BOOL_FIELDS = {
    "fall_metadaten": [
        "verbrauchergeschaeft_kschg",
        "streitwert_bekannt",
    ],
    "klaegervorbringen_anspruchsgrundlagen": [
        "vertraglich_erfuellung",
        "vertraglich_gewaehrleistung",
        "vertraglich_poenale",
        "quasi_vertraglich",
        "schadenersatz_ex_contractu",
        "schadenersatz_ex_delicto",
        "bereicherung",
        "dinglich_eigentum",
        "dinglich_besitz",
        "unterlassung_beseitigung",
        "sonstige_anspruchsgrundlage",
    ],
    "beklagtenvorbringen_prozessual": [
        "unzuständigkeit",
        "streitanhaengigkeit_rechtskraft",
        "mangelnde_partei_prozessfaehigkeit",
        "sonstiges_prozesshindernis",
    ],
    "beklagtenvorbringen_materiell_rechtshindernd": [
        "mangelnde_geschaeftsfaehigkeit",
        "dissens_scherz_scheinvertrag",
        "sittenwidrigkeit_gesetzwidrigkeit",
        "formmangel",
        "irrtum_list_drohung",
        "laesio_enormis_wucher",
        "sonstige_rechtshindernde_einwendung",
    ],
    "beklagtenvorbringen_materiell_rechtsvernichtend": [
        "erfuellung_zahlung",
        "aufrechnung_kompensation",
        "ruecktritt_kuendigung",
        "unmoeglichkeit",
        "verzicht_erlass",
        "sonstige_rechtsvernichtende_einwendung",
    ],
    "beklagtenvorbringen_materiell_rechtshemmend": [
        "verjaehrung_praeklusion",
        "zug_um_zug_einrede",
        "zurueckbehaltungsrecht",
        "mangelnde_faelligkeit_stundung",
        "sonstige_rechtshemmende_einrede",
    ],
    "beklagtenvorbringen_allgemein": [
        "mangelnde_aktiv_passivlegitimation",
        "mitverschulden_schadensminderung",
        "bestreitet_tatbestand_komplett",
        "bestreitet_nur_rechtliche_wertung",
        "bestreitet_hoehe",
        "bestreitet_verschulden",
        "sonstige_allgemeine_bestreitung",
    ],
}

# List fields where we use the count as a feature
LEGAL_ANALYSIS_LIST_FIELDS = {
    "klaegervorbringen_anspruchsgrundlagen": ["zitierte_normen_klaeger"],
    "beklagtenvorbringen_allgemein": ["zitierte_normen_beklagter"],
}

# ─── Text Sections (for reference/display, not for training) ─────────────────────
SECTION_LABELS = {
    "klaegervorbringen": "Kläger-Vorbringen",
    "beklagtenvorbringen": "Beklagten-Vorbringen",
    "feststellungen": "Feststellungen",
    "beweisw_rdigung": "Beweiswürdigung",
    "aufgenommene_beweise": "Aufgenommene Beweise",
}

# ─── Neural Network Configuration ───────────────────────────────────────────────
# Hybrid: per-section embedding encoders + structured feature encoder → fusion → 3-class.
#
# Each embedding section (1024-dim) is compressed to 64-dim, then all are
# concatenated with encoded structured features and fused.
#
# Parameteranzahl ca.:
#   2 × EmbeddingEncoder (1024→128→64): ~140 K
#   StructuredEncoder (struct→32):       ~1 K
#   Fusion (160→64→3):                   ~12 K
#   Gesamt: ~155 K
NN_CONFIG = {
    "embedding_hidden_dim": 128,     # Intermediate dim per embedding encoder
    "embedding_output_dim": 64,      # Output dim per embedding encoder
    "structured_hidden_dim": 32,     # Structured feature encoder hidden dim
    "fusion_dims": [64],             # Fusion layer dimensions (compact)
    "dropout_embedding": 0.5,        # Dropout for embedding encoders
    "dropout_fusion": 0.5,           # Dropout for fusion layers
    "num_classes": 3,                # win / partial / loss
}

# ─── Training Configuration ─────────────────────────────────────────────────────
TRAINING_CONFIG = {
    "epochs": 200,
    "batch_size": 64,
    "learning_rate": 3e-4,
    "weight_decay": 5e-2,
    "lr_scheduler_patience": 7,
    "lr_scheduler_factor": 0.5,
    "early_stopping_patience": 12,
    "val_split": 0.2,
    "random_seed": 42,
    "gradient_clip": 1.0,
}

# ─── UI Configuration ───────────────────────────────────────────────────────────
APP_TITLE_EXTRACTOR = "Litigation Data Extractor"
APP_TITLE_MODEL = "Predictive Litigation Analytics"
APP_VERSION = "3.0.0"
APP_AUTHOR = "Predictive Litigation Analytics System"
