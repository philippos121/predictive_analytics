"""
Shared configuration for the Predictive Litigation Analytics system.

v2.0 — Structured Data Training (no embeddings).
Uses GPT-5-nano to extract a detailed legal analysis schema from each ruling,
then trains a neural network purely on structured features.
"""

import os
from pathlib import Path

# ─── Project Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
EXTRACTED_DIR = DATA_DIR / "extracted"
MODELS_DIR = DATA_DIR / "models"

for _d in [DATA_DIR, EXTRACTED_DIR, MODELS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

DATASET_FILE = EXTRACTED_DIR / "cases_dataset.json"
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
# Using gpt-5-nano for structured data extraction from rulings.
# No embeddings — the model trains on structured legal analysis features only.
OPENAI_EXTRACTION_MODEL = "gpt-5-nano-2025-08-07"

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

# ─── Structured Legal Analysis Schema ────────────────────────────────────────────
# Detailed extraction schema for Austrian civil proceedings.
# Extracted by GPT from each ruling and used as primary training features.
# Replaces the previous embedding-based approach.

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
# Rein strukturierte Daten — kein Embedding-Encoder.
# Durch die detaillierte Legal-Analysis (~77 Features) ist das Netz kompakter
# und schneller zu trainieren als die bisherige Embedding-Architektur.
#
# Parameteranzahl ca.:
#   StructuredEncoder (features→64→32):  ~4 K
#   Fusion (32→16→3):                    ~0.6 K
#   Gesamt: ~5 K
NN_CONFIG = {
    "hidden_dims": [64, 32],         # Smaller encoder — reduce overfitting on sparse features
    "fusion_dims": [16],             # Compact fusion layer
    "dropout": 0.4,                  # Higher dropout for noisy GPT-extracted boolean features
    "num_classes": 3,                # win / partial / loss
}

# ─── Training Configuration ─────────────────────────────────────────────────────
TRAINING_CONFIG = {
    "epochs": 300,
    "batch_size": 64,
    "learning_rate": 3e-4,
    "weight_decay": 1e-2,
    "lr_scheduler_patience": 20,
    "lr_scheduler_factor": 0.5,
    "early_stopping_patience": 40,
    "val_split": 0.2,
    "random_seed": 42,
    "gradient_clip": 1.0,
}

# ─── UI Configuration ───────────────────────────────────────────────────────────
APP_TITLE_EXTRACTOR = "Litigation Data Extractor"
APP_TITLE_MODEL = "Predictive Litigation Analytics"
APP_VERSION = "2.0.0"
APP_AUTHOR = "Predictive Litigation Analytics System"
