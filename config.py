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

# ─── Neural Network Architecture for 10 000-ruling dataset ──────────────────────
# With ~8 000 training examples the model can LEARN the 3072→128 projection
# instead of relying on a frozen random one.  A learned projection finds the
# class-relevant directions in the embedding space; a random one does not
# (we saw train 50 % / val 45 % with freeze_encoders=True, confirming this).
#
# Interaction features (use_interaction_features=True):
#   diff = kl_enc − bk_enc  →  net directional advantage of Kläger over Beklagter
#   prod = kl_enc * bk_enc  →  element-wise resonance (where both parties agree)
#   Both are computed from the encoded vectors — zero extra parameters — yet
#   directly capture the adversarial dynamics that plain concatenation misses.
#   The scalar SectionAttention alone only weights globally; diff/prod let the
#   fusion head see the *relative* semantic positions per case.
#
# Parameter budget  (emb_hidden=256, emb_out=128, interaction=True, fusion=[256,128]):
#   2 encoders:   2 × (3072×256+256 + 256×128+128)  = 1 639 168  ← learned (2-layer)
#                   first step  3072→256 : 12× compression (was 24×)
#                   second step  256→128 : nonlinear selective refinement
#   interaction:  0  (diff + prod computed, not learned)
#   attention:    128×1 + 1                          =      129   ← learned
#   fusion:       640×256+256 + LN(256)              =  165 120   ← learned
#                 256×128+128 + LN(128)              =   33 152   ← learned
#   classifier:   128×3 + 3                          =      387   ← learned
#   ──────────────────────────────────────────────────────────────
#   Total                                             ≈ 1 837 956  (~230 params/example @ 8 000 training)
#
# Fusion input breakdown:
#   kl_enc(128) + bk_enc(128) + diff(128) + prod(128) + attended(128) = 640
#
# Regularisation mix:
#   • dropout_embedding 0.20  — encoder regularisation
#   • dropout_fusion    0.35  — fusion regularisation
#   • Gaussian noise std 0.01 — stochastic input perturbation
#   • weight_decay 1e-3, label_smoothing 0.1, mixup 0.3, SWA (last 40 % epochs)

NN_CONFIG = {
    "embedding_hidden_dim": 256,         # two-layer encoder: 3072 → 256 → 128
    "embedding_output_dim": 128,         # learned — finds class-relevant directions
    "embedding_noise_std": 0.01,         # light Gaussian noise on raw embeddings
    "freeze_encoders": False,            # LEARNED projection (needs 7 500+ training cases)
    "use_section_attention": True,       # attention over klaeger + beklagter
    "use_interaction_features": True,    # diff + prod of encoded sections (zero extra params)
    "structured_dim": 0,                 # embeddings only
    "fusion_dims": [256, 128],           # fusion head appropriate for 10k dataset
    "dropout_embedding": 0.20,
    "dropout_fusion": 0.35,
    "num_classes": 3,
}

# ─── Training Configuration ──────────────────────────────────────────────────────
# Tuned for ~10 000 labeled rulings (8 000 train / 2 000 val with val_split=0.20).
# User-settable UI params (epochs, learning_rate, early_stopping_patience)
# are applied on top of these defaults when changed in the sidebar.

TRAINING_CONFIG = {
    "epochs": 300,
    "batch_size": 64,                 # larger batch for larger dataset
    "learning_rate": 2e-4,            # slightly lower LR for larger learned encoder
    "weight_decay": 1e-3,             # stronger L2 to regularise ~920k params
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
