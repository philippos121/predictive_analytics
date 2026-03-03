"""
Shared configuration for the Predictive Litigation Analytics system.

v4.0 — Cross-Attention Hybrid Architecture.
Uses pre-computed text-embedding-3-large vectors (kläger/beklagten vorbringen)
with bidirectional cross-attention, combined with structured metadata features.
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
PCA_FILE = MODELS_DIR / "embedding_pca.pkl"
ENCODER_FILE = MODELS_DIR / "label_encoders.pkl"
TRAINING_HISTORY_FILE = MODELS_DIR / "training_history.json"
KNN_FILE = MODELS_DIR / "litigation_knn.pkl"
FEATURE_RELEVANCE_FILE = MODELS_DIR / "feature_relevance.json"

# ─── Adaptive Model Selection ────────────────────────────────────────────────────
# Below KNN_THRESHOLD labeled cases → k-Nearest-Neighbour (structured features)
# At or above                       → LitigationClassifier neural network
KNN_THRESHOLD = 50

# ─── Classification Mode ────────────────────────────────────────────────────────
# NUM_CLASSES = 2: Binary (Obsiegen vs Nicht-Obsiegen).
#   Merges "Teilweises Obsiegen" into "Nicht-Obsiegen" (class 0).
#   Rationale: "Teilweise" outcomes depend on evidence quality and sub-claim
#   granularity that cannot be predicted from pleadings alone.
# NUM_CLASSES = 3: Ternary (Obsiegen / Teilweise / Unterliegen) — original mode.
NUM_CLASSES = 2

# ─── OpenAI Configuration ───────────────────────────────────────────────────────
OPENAI_EXTRACTION_MODEL = "gpt-5-nano-2025-08-07"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 3072  # Full dimension of text-embedding-3-large
EMBEDDING_DIM_USED = 1024  # Truncated dimension for training (first N dims)

# PCA: disabled by default.  With 100 K+ cases the curse of dimensionality
# is not a problem (100 K / (2 × 1024) ≈ 50 samples per dimension — fine).
# Set USE_PCA = True + PCA_DIM for small-dataset experiments.
USE_PCA = False
PCA_DIM = 128  # Only used when USE_PCA = True

# API rate limiting
OPENAI_REQUEST_DELAY_SEC = 0.5       # Delay between API requests
OPENAI_MAX_RETRIES = 5
OPENAI_RETRY_DELAY_SEC = 2.0

# ─── Outcome Labels ─────────────────────────────────────────────────────────────
if NUM_CLASSES == 2:
    OUTCOME_LABELS = {
        0: "Nicht-Obsiegen",
        1: "Obsiegen",
    }
    OUTCOME_COLORS = {
        0: "#E74C3C",   # Red
        1: "#27AE60",   # Green
    }
    OUTCOME_ICONS = {
        0: "❌",
        1: "✅",
    }
else:
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


def map_outcome_label(raw_outcome: int) -> int:
    """Map 3-class raw outcome label to the configured NUM_CLASSES scheme.

    Raw labels from extraction: 0=loss, 1=partial, 2=win.
    When NUM_CLASSES == 2: 0,1 → 0 (Nicht-Obsiegen), 2 → 1 (Obsiegen).
    When NUM_CLASSES == 3: identity mapping.
    """
    if NUM_CLASSES == 2:
        return 1 if raw_outcome == 2 else 0
    return raw_outcome

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
# Text sections whose embeddings are computed and used as NN training INPUTS.
# Only kläger + beklagten vorbringen — available at prediction time.
#
# NOTE: erstgericht_begruendung is extracted and stored but NOT used as input.
# It would be data leakage — at prediction time you don't have the court's
# reasoning yet. It is available for:
#   - LLM fine-tuning as a training TARGET (chain-of-thought → outcome)
#   - Post-hoc analysis / explainability
#   - Future reward-model training
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
    "erstgericht_begruendung": "Erstgericht-Begründung",
    "aufgenommene_beweise": "Aufgenommene Beweise",
}

# ─── Neural Network Configuration ───────────────────────────────────────────────
# v4.0 — Cross-Attention Hybrid Architecture (100 K+ cases, no PCA).
#
# Full 1024-dim embeddings → shared encoder → multi-head cross-attention → fusion.
# The cross-attention models the INTERACTION between Kläger and Beklagter —
# the key insight that legal outcomes depend on claims IN CONTEXT OF defenses.
#
# With 100K cases we can use wider layers and lower dropout than v3.
#
# Parameteranzahl ca.:
#   SharedEmbeddingEncoder (1024→256→256):   ~330 K  (one encoder, shared weights)
#   MultiHead CrossAttention (256, 4 heads): ~265 K  (Q/K/V + output + FFN)
#   StructuredEncoder (77→64):               ~5 K
#   Fusion (576→256→128→2):                  ~180 K
#   Gesamt: ~780 K
NN_CONFIG = {
    "embedding_hidden_dim": 256,     # Intermediate dim per embedding encoder
    "embedding_output_dim": 256,     # Output dim per embedding encoder
    "structured_hidden_dim": 64,     # Structured feature encoder hidden dim
    "fusion_dims": [256, 128],       # Deeper fusion (two layers)
    "dropout_embedding": 0.3,        # Lower dropout — 100K cases regularize better
    "dropout_fusion": 0.3,           # Lower dropout for fusion
    "n_attention_heads": 4,          # Multi-head cross-attention
    "num_classes": NUM_CLASSES,
}

# ─── Training Configuration ─────────────────────────────────────────────────────
TRAINING_CONFIG = {
    "epochs": 100,              # 100K cases — converges faster
    "batch_size": 256,           # Larger batches with 100K cases
    "learning_rate": 1e-3,       # Higher LR works with larger batches
    "weight_decay": 1e-2,        # Lighter regularization — data regularizes
    "lr_scheduler_patience": 5,
    "lr_scheduler_factor": 0.5,
    "early_stopping_patience": 10,
    "val_split": 0.1,            # 10% val = 10K cases — plenty
    "random_seed": 42,
    "gradient_clip": 1.0,
}

# ─── UI Configuration ───────────────────────────────────────────────────────────
# ─── Calibration-Aware ML Weighting ──────────────────────────────────────────
# Derives how much weight the ML model should get in the expected-value
# mixture, based on its Brier Skill Score (BSS).
#
# BSS <= 0    → model adds no signal → w_ml = 0 (jurist only)
# BSS  0..0.25 → modest signal      → w_ml ramps linearly 0 → 0.5
# BSS >= 0.25  → strong signal      → w_ml capped at 0.5
#
# The jurist always gets at least 50 % weight — the ML nudges, never overrides.

DEFAULT_ML_WEIGHT_NO_CALIBRATION = 0.25  # Fallback when no BSS is available


def recommended_ml_weight(brier_skill_score: float) -> float:
    """Derive ML mixing weight from Brier Skill Score.

    Returns a value in [0.0, 0.5].  A 63 % accuracy model with BSS ~ 0.10
    gets w_ml ~ 0.20: enough to nudge the expected-value estimate, not enough
    to dominate it.
    """
    if brier_skill_score <= 0.0:
        return 0.0
    # Linear ramp: BSS 0 → 0 weight, BSS 0.25 → 0.5 weight
    return min(0.5, brier_skill_score * 2.0)


APP_TITLE_EXTRACTOR = "Litigation Data Extractor"
APP_TITLE_MODEL = "Predictive Litigation Analytics"
APP_VERSION = "3.0.0"
APP_AUTHOR = "Predictive Litigation Analytics System"
