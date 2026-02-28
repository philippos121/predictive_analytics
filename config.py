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
]

EMBEDDING_SECTION_LABELS = {
    "klaegervorbringen": "Kläger-Vorbringen",
    "beklagtenvorbringen": "Beklagten-Vorbringen",
    "aufgenommene_beweise": "Aufgenommene Beweise",
}

# ─── Neural Network Configuration ───────────────────────────────────────────────
# text-embedding-3-large outputs 3072-dim vectors that are already highly semantic.
# A SINGLE linear projection per section (embedding_hidden_dim = 0) keeps the
# dominant parameter cost (2 × 3072 × emb_output_dim) small.
# No structured features — embeddings only (klaeger + beklagter).
#
# Encoders are FROZEN (random projection) — only attention + fusion are trained.
# A learned 3072→k projection has enough freedom to memorise every training
# example regardless of dropout / weight-decay strength.  Freezing removes this
# source of memorisation entirely; the fixed random projection still preserves
# geometric structure (Johnson-Lindenstrauss) so classification remains possible.
#
# Learnable parameter accounting (emb_out=32, freeze_encoders=True, fusion=[64]):
#   2 encoders:   FROZEN  (196 672 params, not trained)
#   attention:    32×1 + 1                      =       33   ← learnable
#   fusion:       (2+1)×32×64 + 64+64+64       =    6 336   ← learnable
#   classifier:   64×3 + 3                      =      195   ← learnable
#   ──────────────────────────────────────────────────────────
#   Total learnable                              ≈    6 564   (~8.8 params/example @ 750 training)

NN_CONFIG = {
    "embedding_hidden_dim": 0,       # Single projection 3072 → 32
    "embedding_output_dim": 32,      # larger: frozen projection needs more dims
    "embedding_noise_std": 0.0,      # no noise needed on frozen encoders
    "freeze_encoders": True,         # KEY: fixed random projection, only fusion trained
    "use_section_attention": True,   # Attention over the 2 text sections
    "structured_dim": 0,             # No structured features
    "fusion_dims": [64],             # fusion_input = (2+1)×32 = 96 → 64 → 3
    "dropout_embedding": 0.0,        # frozen encoders cannot overfit
    "dropout_fusion": 0.50,
    "num_classes": 3,
}

# ─── Training Configuration ──────────────────────────────────────────────────────
# User-settable UI params (epochs, learning_rate, early_stopping_patience)
# are applied on top of these defaults when changed in the sidebar.

TRAINING_CONFIG = {
    "epochs": 200,
    "batch_size": 32,
    "learning_rate": 3e-4,
    "weight_decay": 5e-4,            # light: only ~6.5 k params are trainable
    "lr_scheduler_patience": 25,
    "lr_scheduler_factor": 0.5,
    "early_stopping_patience": 40,   # stop 40 epochs after val acc peaks
    "use_swa": False,                 # disabled: tiny head converges fast, SWA averages overfit snapshots
    "val_split": 0.25,
    "random_seed": 42,
    "gradient_clip": 1.0,
    "label_smoothing": 0.1,
    "mixup_alpha": 0.2,
}

# ─── UI Configuration ───────────────────────────────────────────────────────────
APP_TITLE_EXTRACTOR = "Litigation Data Extractor (OGH-Urteile, ö. Zivilrecht)"
APP_TITLE_MODEL = "Predictive Litigation Analytics (OGH-Datenbasis)"
APP_VERSION = "2.1.0"
APP_AUTHOR = "Predictive Litigation Analytics System"
