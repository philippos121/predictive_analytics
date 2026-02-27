"""
Predictive Litigation Analytics — Model Training & Prediction UI

Streamlit interface for:
1. Training the neural network on extracted case data
2. Evaluating model performance
3. Predicting outcomes for new cases
4. Computing expected value of litigation
"""

import json
import os
import sys
import time
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    APP_VERSION,
    CLAIM_TYPES,
    DEFENSE_LABELS,
    DEFENSE_TYPES,
    EMBEDDING_DIM,
    EMBEDDING_SECTION_LABELS,
    KNN_THRESHOLD,
    LEGAL_ANALYSIS_MODEL,
    MODEL_CHECKPOINT,
    NN_MEDIUM_THRESHOLD,
    NN_SMALL_THRESHOLD,
    OUTCOME_COLORS,
    OUTCOME_ICONS,
    OUTCOME_LABELS,
    TRAINING_CONFIG,
    TRAINING_CONFIG_MEDIUM,
    TRAINING_CONFIG_LARGE,
    TRAINING_HISTORY_FILE,
)
from data_extractor.data_manager import DataManager
from model.legal_analyzer import LegalAnalyzer
from model.predictor import LitigationPredictor
from model.ratg_calculator import calculate_ratg_costs
from model.trainer import LitigationTrainer

# ─── Page Config ─────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Predictive Litigation Analytics",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    :root {
        --col-bg:       #ffffff;
        --col-surface:  #f5f5f5;
        --col-border:   #cccccc;
        --col-text:     #1a1a1a;
        --col-muted:    #666666;
        --col-primary:  #1c3a5e;
        --col-accent:   #2a6496;
        --col-success:  #2c6e49;
        --col-warning:  #7d5a00;
        --col-danger:   #8b1a1a;
        --font-mono:    'IBM Plex Mono', 'Courier New', monospace;
        --font-sans:    'IBM Plex Sans', 'Helvetica Neue', sans-serif;
    }

    html, body, [class*="css"] { font-family: var(--font-sans); }

    .main .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* ── Header ── */
    .app-header {
        border-left: 4px solid var(--col-primary);
        padding: 0.8rem 1.2rem;
        margin-bottom: 1.5rem;
        background: var(--col-surface);
        border-top: 1px solid var(--col-border);
        border-right: 1px solid var(--col-border);
        border-bottom: 1px solid var(--col-border);
    }
    .app-header h1 {
        color: var(--col-primary);
        margin: 0;
        font-size: 1.4rem;
        font-weight: 600;
        letter-spacing: 0.01em;
        font-family: var(--font-sans);
    }
    .app-header .subtitle {
        color: var(--col-muted);
        font-size: 0.82rem;
        margin: 0.2rem 0 0 0;
        font-family: var(--font-mono);
        letter-spacing: 0.02em;
    }

    /* ── Section title ── */
    .section-title {
        font-size: 1.0rem;
        font-weight: 600;
        color: var(--col-primary);
        border-bottom: 1px solid var(--col-border);
        padding-bottom: 0.3rem;
        margin-bottom: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* ── Probability bars ── */
    .prob-row {
        margin: 5px 0;
    }
    .prob-header {
        display: flex;
        justify-content: space-between;
        font-size: 0.82rem;
        font-family: var(--font-mono);
        color: var(--col-text);
        margin-bottom: 2px;
    }
    .prob-track {
        background: #e0e0e0;
        height: 14px;
        width: 100%;
    }
    .prob-fill {
        height: 100%;
    }
    .prob-win     { background: #2c6e49; }
    .prob-partial { background: #7d5a00; }
    .prob-loss    { background: #8b1a1a; }

    /* ── Outcome result card ── */
    .result-card {
        border: 1px solid var(--col-border);
        border-left: 4px solid var(--col-primary);
        background: var(--col-surface);
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
    }
    .result-card.outcome-0 { border-left-color: var(--col-danger); }
    .result-card.outcome-1 { border-left-color: var(--col-warning); }
    .result-card.outcome-2 { border-left-color: var(--col-success); }

    .result-label {
        font-size: 1.3rem;
        font-weight: 600;
        font-family: var(--font-sans);
        letter-spacing: 0.01em;
    }
    .result-label.outcome-0 { color: var(--col-danger); }
    .result-label.outcome-1 { color: var(--col-warning); }
    .result-label.outcome-2 { color: var(--col-success); }

    .confidence-tag {
        display: inline-block;
        font-family: var(--font-mono);
        font-size: 0.78rem;
        border: 1px solid var(--col-border);
        padding: 0.15rem 0.5rem;
        color: var(--col-muted);
        background: white;
        margin-top: 0.3rem;
    }

    /* ── EV cards ── */
    .ev-card {
        border: 1px solid var(--col-border);
        padding: 1rem;
        text-align: center;
        background: var(--col-surface);
    }
    .ev-card.positive { border-top: 3px solid var(--col-success); }
    .ev-card.negative { border-top: 3px solid var(--col-danger); }
    .ev-card.neutral  { border-top: 3px solid var(--col-warning); }

    .ev-value {
        font-family: var(--font-mono);
        font-size: 2rem;
        font-weight: 600;
        line-height: 1.1;
    }
    .ev-label {
        font-size: 0.78rem;
        color: var(--col-muted);
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-top: 0.3rem;
        font-family: var(--font-sans);
    }
    .ev-sublabel {
        font-size: 0.72rem;
        color: var(--col-muted);
        font-family: var(--font-mono);
        margin-top: 6px;
    }

    /* ── Training log ── */
    .training-log {
        background: #1a1a1a;
        color: #d4d4d4;
        border: 1px solid #333333;
        padding: 0.8rem 1rem;
        font-family: var(--font-mono);
        font-size: 0.80rem;
        height: 280px;
        overflow-y: auto;
        line-height: 1.6;
    }
    .log-line   { margin: 1px 0; }
    .log-ok     { color: #6db88c; }
    .log-warn   { color: #c8a84b; }
    .log-error  { color: #c86060; }
    .log-info   { color: #6a9fca; }

    /* ── Note box ── */
    .note-box {
        border: 1px solid var(--col-border);
        border-left: 3px solid var(--col-accent);
        background: var(--col-surface);
        padding: 0.7rem 1rem;
        font-size: 0.85rem;
        margin: 0.6rem 0;
    }

    /* ── Metrics ── */
    div[data-testid="stMetricValue"] {
        font-family: var(--font-mono);
        font-size: 1.6rem;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: var(--col-muted);
    }
</style>
""", unsafe_allow_html=True)


# ─── Session State ────────────────────────────────────────────────────────────────

def init_session():
    defaults = {
        "trainer": None,
        "predictor": None,
        "training_running": False,
        "training_log": [],
        "training_history": {},
        "openai_api_key": os.environ.get("OPENAI_API_KEY", ""),
        "prediction_result": None,
        "prediction_case_dict": None,   # Falldaten der letzten Vorhersage
        "prediction_sections": {},      # Textsektionen (für Re-Embedding)
        "prediction_embeddings": {},    # Embedding-Vektoren der letzten Vorhersage
        "juristic_analysis": None,      # Ergebnis der juristischen Analyse
        "ratg_result": None,            # RATG-Kostenberechnung
        "ev_result": None,
        "updated_prediction": None,     # Vorhersage nach Vorbringen-Update
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

dm = DataManager()

# Try to load existing model
if st.session_state.predictor is None and MODEL_CHECKPOINT.exists():
    predictor = LitigationPredictor.from_checkpoint()
    if predictor:
        st.session_state.predictor = predictor
        st.session_state.trainer = predictor.trainer


# ─── Header ──────────────────────────────────────────────────────────────────────

st.markdown(f"""
<div class="app-header">
    <h1>Predictive Litigation Analytics</h1>
    <div class="subtitle">Machine Learning Modell für österreichische Zivilprozesse &nbsp;·&nbsp; v{APP_VERSION} &nbsp;·&nbsp; kNN (< {KNN_THRESHOLD} Fälle) · LitigationClassifier (≥ {KNN_THRESHOLD} Fälle)</div>
</div>
""", unsafe_allow_html=True)


# ─── Sidebar ─────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("**Modell-Status**")

    model_exists = MODEL_CHECKPOINT.exists()
    if model_exists:
        st.success("Trainiertes Modell vorhanden")
        mtime = MODEL_CHECKPOINT.stat().st_mtime
        from datetime import datetime
        st.caption(f"Zuletzt trainiert: {datetime.fromtimestamp(mtime).strftime('%d.%m.%Y %H:%M')}")
    else:
        st.warning("Noch kein Modell trainiert")

    stats = dm.get_statistics()
    st.metric("Dataset-Größe", stats.get("total_cases", 0))
    st.metric("Beschriftete Fälle", stats.get("labeled_cases", 0))

    if stats.get("labeled_cases", 0) > 0:
        oc = stats.get("outcome_distribution", {})
        st.markdown("**Outcome-Verteilung:**")
        st.caption(
            f"Obsiegen: {oc.get('obsiegen', 0)}  |  "
            f"Teilw.: {oc.get('teilweise', 0)}  |  "
            f"Unterl.: {oc.get('unterliegen', 0)}"
        )

    st.divider()
    # Pick tier-appropriate defaults based on current dataset size
    _all_cases = dm.load_dataset()
    _n_labeled = sum(
        1 for c in _all_cases
        if c.get("structured", {}).get("outcome") is not None
    ) if _all_cases else 0
    if _n_labeled >= NN_MEDIUM_THRESHOLD:
        _default_cfg = TRAINING_CONFIG_LARGE
        _tier_hint = f"Tier: Groß (≥{NN_MEDIUM_THRESHOLD} Fälle)"
    elif _n_labeled >= NN_SMALL_THRESHOLD:
        _default_cfg = TRAINING_CONFIG_MEDIUM
        _tier_hint = f"Tier: Mittel (≥{NN_SMALL_THRESHOLD} Fälle)"
    else:
        _default_cfg = TRAINING_CONFIG
        _tier_hint = f"Tier: Klein (<{NN_SMALL_THRESHOLD} Fälle)"
    st.markdown(f"**Training-Parameter** — {_tier_hint}")
    epochs = st.slider("Max. Epochen", 50, 600, _default_cfg["epochs"], 50)
    lr = st.select_slider(
        "Lernrate",
        [1e-4, 5e-4, 1e-3, 5e-3],
        value=_default_cfg["learning_rate"],
        format_func=lambda x: f"{x:.0e}",
    )
    early_stop = st.slider(
        "Early Stopping (Epochen)",
        10, 80, _default_cfg["early_stopping_patience"], 5,
    )

    st.divider()
    api_key = st.text_input(
        "OpenAI API Key (für neue Fälle)",
        value=st.session_state.openai_api_key,
        type="password",
    )
    st.session_state.openai_api_key = api_key

    st.divider()
    if st.button("App neu laden", use_container_width=True):
        st.rerun()


# ─── Main Tabs ────────────────────────────────────────────────────────────────────

tab_train, tab_eval, tab_predict, tab_jur, tab_ev = st.tabs([
    "Training",
    "Evaluation",
    "Vorhersage",
    "Juristische Analyse",
    "Erwartungswert",
])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1: TRAINING
# ════════════════════════════════════════════════════════════════════════════════

with tab_train:
    st.markdown('<div class="section-title">Modell trainieren</div>', unsafe_allow_html=True)

    cases, embeddings_dict = dm.export_for_training()
    n_cases = len(cases)

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    col_s1.metric("Trainingsdaten", n_cases)
    col_s2.metric("kNN-Modus (< Fälle)", KNN_THRESHOLD)
    col_s3.metric("Neuronales Netz (≥)", KNN_THRESHOLD)
    col_s4.metric("Ideal", "100+")

    if n_cases < 1:
        st.error(
            "Keine beschrifteten Fälle mit Embeddings vorhanden. "
            "Bitte zuerst Urteile mit dem Data Extractor verarbeiten."
        )
    elif n_cases < KNN_THRESHOLD:
        st.info(
            f"{n_cases} Fälle — **kNN-Modus** (Ähnlichkeitssuche) wird verwendet. "
            f"Ab {KNN_THRESHOLD} beschrifteten Fällen schaltet das System automatisch "
            f"auf das neuronale Netz um."
        )
    else:
        st.success(f"{n_cases} Fälle — neuronales Netz wird trainiert.")

    st.divider()

    # ── Architecture Info ────────────────────────────────────────────────────────
    with st.expander("Modell-Architektur", expanded=False):
        if n_cases < KNN_THRESHOLD:
            st.markdown(f"""
            **Aktiver Modus: k-Nearest-Neighbour** (< {KNN_THRESHOLD} Fälle)
            - **Methode**: Kosinus-Ähnlichkeit auf verketteten Text-Embeddings (3 × 3072 = 9.216 dim)
            - **k**: min(5, Trainingsgröße) nächste Nachbarn, gewichtetes Soft-Voting
            - **Parameter**: 0 — kein Gradientenverfahren, kein Training
            - **Vorteil**: sofort einsatzbereit, kein Overfitting-Risiko

            Ab **{KNN_THRESHOLD} beschrifteten Fällen** schaltet das System automatisch auf das
            neuronale Netz um.
            """)
        else:
            st.markdown(f"""
            **Aktiver Modus: Neuronales Netz** (≥ {KNN_THRESHOLD} Fälle)
            - **Embedding Encoder** (3×): Linear(3072 → 256) + LayerNorm + GELU + Dropout → Linear(256 → 128)
            - **Fusion Network**: Linear(384 → 128) + LayerNorm + GELU + Dropout → Linear(128 → 3)
            - **Loss**: Focal Loss (γ=2) mit Klassen-Gewichtung
            - **Optimizer**: AdamW mit ReduceLROnPlateau
            - **Regularisierung**: LayerNorm, Dropout, Gradient Clipping, Early Stopping
            - **Parameter gesamt**: ~1,19 Mio.

            **Input-Embeddings (3 Abschnitte):**
            - **Kläger-Vorbringen**: Was begehrt der Kläger?
            - **Beklagten-Vorbringen**: Welche Einwendungen macht der Beklagte?
            - **Aufgenommene Beweise**: Faktische Beschreibung der aufgenommenen Beweise
              (Art, Anzahl, welche Partei — ohne Bewertung)
            - 3 × 128 = **384 dim Fusion-Input**

            **Nicht im Input**: Feststellungen, Beweiswürdigung, Rechtliche Beurteilung,
            strukturierte Merkmale (Streitwert, Anspruchsart, Einwendungen) — rein embedding-basiert.

            **Output:** 3 Klassen (Unterliegen / Teilweise / Obsiegen)
            """)

    # ── Training Controls ────────────────────────────────────────────────────────
    col_btn1, col_btn2 = st.columns([2, 1])

    with col_btn1:
        train_btn = st.button(
            f"Training starten  ({n_cases} Fälle)",
            type="primary",
            disabled=n_cases < 1 or st.session_state.training_running,
            use_container_width=True,
        )

    with col_btn2:
        if MODEL_CHECKPOINT.exists():
            if st.button("Modell neu trainieren", use_container_width=True):
                MODEL_CHECKPOINT.unlink(missing_ok=True)
                st.session_state.predictor = None
                st.session_state.trainer = None
                st.rerun()

    # ── Training Progress ────────────────────────────────────────────────────────
    if train_btn and not st.session_state.training_running and n_cases >= 1:
        st.session_state.training_running = True
        st.session_state.training_log = []

        custom_config = {
            **TRAINING_CONFIG,
            "epochs": epochs,
            "learning_rate": lr,
            "early_stopping_patience": early_stop,
        }

        log_container = st.empty()
        chart_container = st.empty()
        metrics_container = st.container()

        progress_bar = st.progress(0)
        status_text = st.empty()

        train_losses, val_losses, train_accs, val_accs = [], [], [], []

        def progress_cb(**kwargs):
            phase = kwargs.get("phase", "")

            if phase == "tier_selected":
                _tier_labels = {
                    "klein": "Klein (<150 Fälle)",
                    "mittel": "Mittel (150–599 Fälle)",
                    "groß": "Groß (≥600 Fälle)",
                }
                tier = kwargs.get("config_tier", "?")
                st.session_state.training_log.append(
                    f'[INFO] Datensatz-Tier: {_tier_labels.get(tier, tier)} | '
                    f'{kwargs["n_labeled"]} gelabelte Fälle | '
                    f'Dropout: {kwargs["dropout_emb"]:.0%} | '
                    f'Weight-Decay: {kwargs["weight_decay"]:.0e}'
                )

            elif phase == "preparing":
                log_container.markdown(
                    f'<div class="training-log"><div class="log-line log-info">'
                    f'[INFO] {kwargs.get("message", "")}</div></div>',
                    unsafe_allow_html=True,
                )

            elif phase == "prepared":
                log_line = f'[OK]  Train: {kwargs["train_size"]} | Val: {kwargs["val_size"]}'
                if "feature_dim" in kwargs:
                    log_line += f' | Features: {kwargs["feature_dim"]}'
                st.session_state.training_log.append(log_line)

            elif phase == "model_built":
                st.session_state.training_log.append(
                    f'[OK]  Modell gebaut: {kwargs["parameters"]:,} Parameter | '
                    f'Device: {kwargs["device"]}'
                )

            elif phase == "training":
                epoch = kwargs["epoch"]
                total = kwargs["total_epochs"]
                train_losses.append(kwargs.get("train_loss", 0))
                val_losses.append(kwargs.get("val_loss", 0))
                train_accs.append(kwargs.get("train_acc", 0))
                val_accs.append(kwargs.get("val_acc", 0))

                progress_bar.progress(epoch / total)
                status_text.markdown(
                    f"**Epoche {epoch}/{total}** &nbsp;|&nbsp; "
                    f"Train Loss: `{kwargs.get('train_loss', 0):.4f}` &nbsp;|&nbsp; "
                    f"Val Acc: `{kwargs.get('val_acc', 0):.1%}` &nbsp;|&nbsp; "
                    f"Beste Val Acc: `{kwargs.get('best_val_acc', 0):.1%}` &nbsp;|&nbsp; "
                    f"LR: `{kwargs.get('lr', 0):.2e}`"
                )

                if len(train_losses) > 1:
                    fig = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=("Loss", "Accuracy"),
                    )
                    fig.add_trace(
                        go.Scatter(y=train_losses, name="Train Loss",
                                   line=dict(color="#1c3a5e", width=1.5)),
                        row=1, col=1,
                    )
                    fig.add_trace(
                        go.Scatter(y=val_losses, name="Val Loss",
                                   line=dict(color="#8b1a1a", width=1.5, dash="dash")),
                        row=1, col=1,
                    )
                    fig.add_trace(
                        go.Scatter(y=train_accs, name="Train Acc",
                                   line=dict(color="#2c6e49", width=1.5)),
                        row=1, col=2,
                    )
                    fig.add_trace(
                        go.Scatter(y=val_accs, name="Val Acc",
                                   line=dict(color="#7d5a00", width=1.5, dash="dash")),
                        row=1, col=2,
                    )
                    fig.update_layout(
                        height=280,
                        margin=dict(t=30, b=10),
                        showlegend=True,
                        paper_bgcolor="white",
                        plot_bgcolor="#f5f5f5",
                        font=dict(family="IBM Plex Sans, sans-serif", size=11),
                    )
                    fig.update_xaxes(showgrid=True, gridcolor="#dddddd", gridwidth=1)
                    fig.update_yaxes(showgrid=True, gridcolor="#dddddd", gridwidth=1)
                    chart_container.plotly_chart(fig, use_container_width=True)

            elif phase == "knn_fitted":
                st.session_state.training_log.append(
                    f'[OK]  kNN-Index erstellt: {kwargs["n_cases"]} Fälle | k={kwargs["k"]}'
                )

            elif phase == "early_stop":
                st.session_state.training_log.append(
                    f'[WARN] {kwargs.get("message", "Early stopping")}'
                )

            elif phase == "swa_done":
                prev = kwargs.get("prev_best_val_acc", 0)
                swa = kwargs.get("swa_val_acc", 0)
                tag = " ✓ besser als Einzelmodell" if swa >= prev else ""
                st.session_state.training_log.append(
                    f'[OK]  SWA ({kwargs["n_snapshots"]} Snapshots): '
                    f'Val-Acc={swa:.1%}{tag}'
                )

            elif phase == "done":
                if kwargs.get("model_type") == "knn":
                    st.session_state.training_log.append(
                        f'[OK]  FERTIG — kNN-Modell ({kwargs.get("n_cases", 0)} Fälle) | '
                        f'Zeit: {kwargs["training_time"]:.1f}s'
                    )
                else:
                    st.session_state.training_log.append(
                        f'[OK]  FERTIG — Beste Val-Accuracy: {kwargs["best_val_acc"]:.1%} | '
                        f'Epochen: {kwargs["epochs_trained"]} | '
                        f'Zeit: {kwargs["training_time"]:.1f}s'
                    )

        trainer = LitigationTrainer(config=custom_config, progress_callback=progress_cb)

        try:
            history = trainer.train(cases, embeddings_dict, save_checkpoint=True)

            st.session_state.trainer = trainer
            st.session_state.predictor = LitigationPredictor(trainer)
            st.session_state.training_history = history
            st.session_state.training_running = False

            progress_bar.empty()
            status_text.empty()

            if history.get("model_type") == "knn":
                n_knn = history.get("n_training_cases", 0)
                st.success(
                    f"kNN-Modell bereit. {n_knn} Fälle indexiert. "
                    f"Für neuronales Netz: ≥ {KNN_THRESHOLD} beschriftete Fälle erforderlich."
                )
            else:
                st.success(
                    f"Training abgeschlossen. "
                    f"Beste Validierungs-Accuracy: **{history['best_val_acc']:.1%}** "
                    f"(Epoche {history['best_epoch']})"
                )
            st.markdown(
                '<div class="note-box">Wechseln Sie zum Tab "Evaluation" für detaillierte Auswertung.</div>',
                unsafe_allow_html=True,
            )

        except Exception as e:
            st.session_state.training_running = False
            progress_bar.empty()
            st.error(f"Trainingsfehler: {e}")

    # ── Show Training Log ────────────────────────────────────────────────────────
    if st.session_state.training_log:
        st.markdown("**Training-Protokoll:**")
        log_html = '<div class="training-log">'
        for line in st.session_state.training_log:
            cls = "log-ok" if "[OK]" in line else "log-warn" if "[WARN]" in line else "log-info"
            log_html += f'<div class="log-line {cls}">{line}</div>'
        log_html += "</div>"
        st.markdown(log_html, unsafe_allow_html=True)

    # ── Existing Training History ────────────────────────────────────────────────
    if TRAINING_HISTORY_FILE.exists() and not st.session_state.training_running:
        with open(TRAINING_HISTORY_FILE) as f:
            hist = json.load(f)

        if hist.get("model_type") == "knn":
            st.markdown("**Aktives Modell: k-Nearest-Neighbour**")
            col_h1, col_h2 = st.columns(2)
            col_h1.metric("Indexierte Fälle", hist.get("n_training_cases", "—"))
            col_h2.metric("Trainingszeit", f"{hist.get('training_time_sec', 0):.1f}s")
            st.info(
                f"kNN-Modus aktiv (< {KNN_THRESHOLD} Trainingsfälle). "
                f"Sobald ≥ {KNN_THRESHOLD} beschriftete Fälle vorhanden sind, "
                f"wird automatisch das neuronale Netz verwendet."
            )

        elif hist.get("train_loss"):
            st.markdown("**Letztes Training**")
            col_h1, col_h2, col_h3, col_h4 = st.columns(4)
            col_h1.metric("Beste Val-Accuracy", f"{hist.get('best_val_acc', 0):.1%}")
            col_h2.metric("Beste Epoche", hist.get("best_epoch", "—"))
            col_h3.metric("Trainierte Epochen", hist.get("epochs_trained", "—"))
            col_h4.metric("Trainingszeit", f"{hist.get('training_time_sec', 0):.0f}s")

            tl = hist["train_loss"]
            vl = hist["val_loss"]
            ta = hist["train_acc"]
            va = hist["val_acc"]

            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Verlust (Loss)", "Genauigkeit (Accuracy)"),
            )
            fig.add_trace(go.Scatter(y=tl, name="Train Loss", line=dict(color="#1c3a5e", width=1.5)), row=1, col=1)
            fig.add_trace(go.Scatter(y=vl, name="Val Loss",   line=dict(color="#8b1a1a", width=1.5, dash="dash")), row=1, col=1)
            fig.add_trace(go.Scatter(y=ta, name="Train Acc",  line=dict(color="#2c6e49", width=1.5)), row=1, col=2)
            fig.add_trace(go.Scatter(y=va, name="Val Acc",    line=dict(color="#7d5a00", width=1.5, dash="dash")), row=1, col=2)
            fig.update_layout(
                height=320,
                margin=dict(t=40, b=20),
                paper_bgcolor="white",
                plot_bgcolor="#f5f5f5",
                font=dict(family="IBM Plex Sans, sans-serif", size=11),
            )
            fig.update_xaxes(showgrid=True, gridcolor="#dddddd", gridwidth=1)
            fig.update_yaxes(showgrid=True, gridcolor="#dddddd", gridwidth=1)
            st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2: EVALUATION
# ════════════════════════════════════════════════════════════════════════════════

with tab_eval:
    st.markdown('<div class="section-title">Modell-Evaluation</div>', unsafe_allow_html=True)

    if st.session_state.predictor is None:
        st.info("Kein Modell vorhanden. Bitte zuerst das Modell im Tab 'Training' trainieren.")
    else:
        predictor = st.session_state.predictor
        trainer = st.session_state.trainer or predictor.trainer

        if st.button("Vollständige Evaluation berechnen", type="primary"):
            with st.spinner("Evaluiere Modell auf gesamtem Dataset..."):
                try:
                    cases, emb_dict = dm.export_for_training()
                    eval_result = trainer.evaluate_full(cases, emb_dict)
                    st.session_state.eval_result = eval_result
                except Exception as e:
                    st.error(f"Fehler bei Evaluation: {e}")

        if hasattr(st.session_state, "eval_result") and st.session_state.eval_result:
            result = st.session_state.eval_result

            # ── Overall Metrics ──────────────────────────────────────────────────
            st.markdown("**Gesamt-Metriken**")
            col_e1, col_e2, col_e3 = st.columns(3)
            col_e1.metric("Overall Accuracy", f"{result['accuracy']:.1%}")

            per_class = result.get("per_class", {})
            macro_f1 = np.mean([per_class[c]["f1"] for c in per_class]) if per_class else 0
            col_e2.metric("Macro F1-Score", f"{macro_f1:.3f}")
            col_e3.metric("Ausgewertete Fälle", len(result.get("labels", [])))

            st.markdown("**Per-Klasse Metriken**")
            class_df = pd.DataFrame([
                {
                    "Klasse": OUTCOME_LABELS[cls],
                    "Precision": f"{metrics['precision']:.3f}",
                    "Recall": f"{metrics['recall']:.3f}",
                    "F1-Score": f"{metrics['f1']:.3f}",
                }
                for cls, metrics in per_class.items()
            ])
            st.dataframe(class_df, use_container_width=True, hide_index=True)

            # ── Confusion Matrix ─────────────────────────────────────────────────
            st.markdown("**Konfusionsmatrix**")
            labels = result["labels"]
            preds = result["predictions"]

            cm = np.zeros((3, 3), dtype=int)
            for true, pred in zip(labels, preds):
                cm[int(true)][int(pred)] += 1

            class_names = [OUTCOME_LABELS[i] for i in range(3)]
            fig_cm = px.imshow(
                cm,
                labels=dict(x="Vorhergesagt", y="Tatsächlich", color="Anzahl"),
                x=class_names,
                y=class_names,
                color_continuous_scale=[
                    [0.0, "#ffffff"],
                    [0.5, "#7aadcc"],
                    [1.0, "#1c3a5e"],
                ],
                text_auto=True,
            )
            fig_cm.update_layout(
                height=380,
                paper_bgcolor="white",
                font=dict(family="IBM Plex Sans, sans-serif", size=12),
            )
            st.plotly_chart(fig_cm, use_container_width=True)

            # ── Probability Distribution ─────────────────────────────────────────
            st.markdown("**Vorhersage-Konfidenz nach wahrer Klasse**")
            if result.get("probabilities"):
                probs = np.array(result["probabilities"])
                labels_arr = np.array(labels)

                fig_probs = go.Figure()
                colors = ["#8b1a1a", "#7d5a00", "#2c6e49"]
                for cls in range(3):
                    mask = labels_arr == cls
                    if mask.sum() > 0:
                        fig_probs.add_trace(go.Box(
                            y=probs[mask, cls],
                            name=f"{OUTCOME_LABELS[cls]} (true)",
                            marker_color=colors[cls],
                            line_color=colors[cls],
                            boxpoints="all",
                            jitter=0.3,
                            pointpos=-1.8,
                        ))
                fig_probs.update_layout(
                    yaxis_title="Predicted Probability",
                    height=320,
                    paper_bgcolor="white",
                    plot_bgcolor="#f5f5f5",
                    font=dict(family="IBM Plex Sans, sans-serif", size=11),
                )
                fig_probs.update_yaxes(showgrid=True, gridcolor="#dddddd")
                st.plotly_chart(fig_probs, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3: PREDICTION
# ════════════════════════════════════════════════════════════════════════════════

with tab_predict:
    st.markdown('<div class="section-title">Neuen Fall vorhersagen</div>', unsafe_allow_html=True)

    if st.session_state.predictor is None:
        st.warning("Kein trainiertes Modell. Bitte zuerst trainieren.")
    else:
        st.caption(
            "Geben Sie die Eckdaten eines neuen Falles ein. "
            "Das Modell berechnet die Outcome-Wahrscheinlichkeiten."
        )

        col_input1, col_input2 = st.columns([3, 2])

        with col_input1:
            with st.form("predict_form"):
                p_klaeger_text = st.text_area(
                    "Kläger-Vorbringen",
                    placeholder="Vorbringen des Klägers…",
                    height=140,
                )
                p_beklagter_text = st.text_area(
                    "Beklagten-Vorbringen",
                    placeholder="Einwendungen des Beklagten…",
                    height=140,
                )

                predict_btn = st.form_submit_button(
                    "Vorhersage berechnen", type="primary", use_container_width=True
                )

        if predict_btn:
            case_dict = {
                "structured": {
                    "streitwert_eur": None,
                    "instanz": "BG",
                    "anspruchsart": "Andere",
                    "anspruchsgruende": [],
                    "einwendungen": {d: False for d in DEFENSE_TYPES},
                    "klaeger_beweismittel": [],
                    "beklagter_beweismittel": [],
                    "sachverstaendiger_bestellt": False,
                }
            }

            sections = {
                "klaegervorbringen": p_klaeger_text,
                "beklagtenvorbringen": p_beklagter_text,
            }

            embeddings = {}
            if st.session_state.openai_api_key and (p_klaeger_text or p_beklagter_text):
                with st.spinner("Generiere Embeddings via OpenAI..."):
                    try:
                        from data_extractor.openai_extractor import OpenAIExtractor
                        extractor = OpenAIExtractor(st.session_state.openai_api_key)
                        embeddings = extractor.generate_embeddings(sections)
                    except Exception as e:
                        st.warning(f"Embedding-Fehler: {e} — Null-Vektoren werden verwendet.")
            elif not st.session_state.openai_api_key and (p_klaeger_text or p_beklagter_text):
                st.warning(
                    "Kein OpenAI API-Key gesetzt — Textvorbringen wird **nicht** als Embedding "
                    "in die Vorhersage einbezogen. Bitte API-Key in der Sidebar eintragen."
                )

            with st.spinner("Berechne Vorhersage..."):
                result = st.session_state.predictor.predict(case_dict, embeddings)
                st.session_state.prediction_result = result
                st.session_state.prediction_case_dict = case_dict
                st.session_state.prediction_sections = sections
                st.session_state.prediction_embeddings = embeddings
                # Reset downstream results when new prediction is made
                st.session_state.juristic_analysis = None
                st.session_state.ratg_result = None
                st.session_state.ev_result = None
                st.session_state.updated_prediction = None

        # ── Display Results ──────────────────────────────────────────────────────
        with col_input2:
            if st.session_state.prediction_result:
                result = st.session_state.prediction_result
                pred_cls = result["predicted_outcome"]

                st.markdown("**Ergebnis**")

                st.markdown(
                    f'<div class="result-card outcome-{pred_cls}">'
                    f'<div class="result-label outcome-{pred_cls}">{OUTCOME_LABELS[pred_cls]}</div>'
                    f'<div class="confidence-tag">Konfidenz: {result["confidence"]:.1%}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

                st.markdown("**Wahrscheinlichkeiten:**")

                probs = [
                    ("Obsiegen",   result["p_win"],     "prob-win"),
                    ("Teilweise",  result["p_partial"],  "prob-partial"),
                    ("Unterliegen",result["p_loss"],     "prob-loss"),
                ]

                for label, prob, css_class in probs:
                    pct = int(prob * 100)
                    st.markdown(
                        f'<div class="prob-row">'
                        f'<div class="prob-header"><span>{label}</span>'
                        f'<span>{prob:.1%}</span></div>'
                        f'<div class="prob-track">'
                        f'<div class="prob-fill {css_class}" style="width:{pct}%"></div>'
                        f"</div></div>",
                        unsafe_allow_html=True,
                    )

                # Gauge chart — scientific style
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=result["p_win"] * 100,
                    number={"suffix": "%", "font": {"family": "IBM Plex Mono", "size": 28}},
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "P(Obsiegen)", "font": {"family": "IBM Plex Sans", "size": 13}},
                    gauge={
                        "axis": {"range": [0, 100], "tickfont": {"family": "IBM Plex Mono", "size": 10}},
                        "bar": {"color": "#1c3a5e", "thickness": 0.25},
                        "bgcolor": "white",
                        "borderwidth": 1,
                        "bordercolor": "#cccccc",
                        "steps": [
                            {"range": [0,  30], "color": "#f5e0e0"},
                            {"range": [30, 55], "color": "#f5f0e0"},
                            {"range": [55, 100],"color": "#e0f0e8"},
                        ],
                        "threshold": {
                            "line": {"color": "#2c6e49", "width": 2},
                            "thickness": 0.75,
                            "value": 55,
                        },
                    },
                ))
                fig_gauge.update_layout(
                    height=230,
                    margin=dict(t=40, b=0, l=20, r=20),
                    paper_bgcolor="white",
                    font=dict(family="IBM Plex Sans, sans-serif"),
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

                result_json = json.dumps(result, ensure_ascii=False, indent=2)
                st.download_button(
                    "Ergebnis als JSON exportieren",
                    data=result_json.encode(),
                    file_name="vorhersage_ergebnis.json",
                    mime="application/json",
                )

        # ── Juristische KI-Schnellanalyse (volle Breite) ─────────────────────────
        if st.session_state.prediction_result:
            st.divider()
            st.markdown("**Juristische KI-Einschätzung**")
            st.caption(
                f"Nutzt {LEGAL_ANALYSIS_MODEL} mit Web-Suche auf ris.bka.gv.at / ogh.gv.at. "
                "Vollständige Analyse mit Quellen und Judikatur → Tab **Juristische Analyse**."
            )

            if not st.session_state.openai_api_key:
                st.info("OpenAI API-Key in der Sidebar eintragen, um die KI-Rechtseinschätzung zu aktivieren.")
            else:
                jur_quick = st.session_state.juristic_analysis
                if not jur_quick or jur_quick.get("error"):
                    if st.button(
                        "KI-Rechtslageeinschätzung berechnen",
                        key="jur_quick_btn",
                        type="secondary",
                    ):
                        _analysis_input = {
                            **(st.session_state.prediction_case_dict or {}),
                            "sections": st.session_state.prediction_sections or {},
                        }
                        with st.spinner(f"Analysiert österreichisches Recht ({LEGAL_ANALYSIS_MODEL})…"):
                            try:
                                _analyzer = LegalAnalyzer(api_key=st.session_state.openai_api_key)
                                _jur = _analyzer.analyze(_analysis_input)
                                st.session_state.juristic_analysis = _jur
                                st.rerun()
                            except Exception as _e:
                                st.error(f"KI-Analysefehler: {_e}")
                else:
                    # Show inline combined result
                    _jur_prob = jur_quick.get("erfolgseinschaetzung", 0.5)
                    _ml_prob  = st.session_state.prediction_result["p_win"]
                    _combined = 0.5 * _ml_prob + 0.5 * _jur_prob

                    _c1, _c2, _c3, _c4 = st.columns(4)
                    _c1.metric("ML-Modell P(Obsiegen)", f"{_ml_prob:.1%}")
                    _c2.metric("KI-Recht P(Obsiegen)", f"{_jur_prob:.1%}",
                               help="Juristische Einschätzung durch " + LEGAL_ANALYSIS_MODEL)
                    _c3.metric(
                        "Kombiniert (50/50)",
                        f"{_combined:.1%}",
                        delta=f"{(_combined - _ml_prob):+.1%} vs. ML",
                    )
                    _konfidenz = jur_quick.get("konfidenz", "?")
                    _c4.metric("Jurist. Konfidenz", _konfidenz)

                    if jur_quick.get("einschaetzung_begruendung"):
                        with st.expander("KI-Begründung"):
                            st.write(jur_quick["einschaetzung_begruendung"])

                    st.caption(
                        "Gewichtung ML / Juristik im Tab **Erwartungswert** frei einstellbar."
                    )

                    if st.button("Analyse zurücksetzen", key="jur_quick_reset"):
                        st.session_state.juristic_analysis = None
                        st.rerun()


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4: JURISTISCHE ANALYSE
# ════════════════════════════════════════════════════════════════════════════════

with tab_jur:
    st.markdown('<div class="section-title">Juristische Analyse (KI + Rechtsweb-Suche)</div>', unsafe_allow_html=True)

    st.caption(
        f"Nutzt **{LEGAL_ANALYSIS_MODEL}** mit Web-Suche, bevorzugt auf "
        "[ris.bka.gv.at](https://ris.bka.gv.at) und [ogh.gv.at](https://ogh.gv.at). "
        "Liefert eine österreichische Rechtslageeinschätzung als Komplement zur ML-Vorhersage."
    )

    if st.session_state.prediction_result is None:
        st.info("Bitte zuerst eine Vorhersage im Tab **'Vorhersage'** berechnen.")
    elif not st.session_state.openai_api_key:
        st.warning("OpenAI API-Key erforderlich (in der Sidebar eintragen).")
    else:
        ml_result = st.session_state.prediction_result
        case_dict_raw = st.session_state.prediction_case_dict or {}
        sections = st.session_state.prediction_sections or {}

        # Zeige ML-Kurzzusammenfassung
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("ML: P(Obsiegen)", f"{ml_result['p_win']:.1%}")
        with c2:
            st.metric("ML: P(Teilweise)", f"{ml_result['p_partial']:.1%}")
        with c3:
            st.metric("ML: P(Unterliegen)", f"{ml_result['p_loss']:.1%}")

        st.markdown("")

        # ── Analyse-Button ────────────────────────────────────────────────────
        col_jur_btn, col_jur_info = st.columns([1, 2])
        with col_jur_btn:
            jur_btn = st.button(
                "Juristische Analyse starten",
                type="primary",
                use_container_width=True,
                help="Startet die KI-gestützte Rechtsprüfung mit Web-Suche (~20–60 Sek.)",
            )

        if jur_btn:
            # Kombiniere case_dict und sections für die Analyse
            analysis_input = {}
            if isinstance(case_dict_raw.get("structured"), dict):
                analysis_input.update(case_dict_raw["structured"])
            elif isinstance(case_dict_raw, dict):
                analysis_input.update(case_dict_raw)
            analysis_input.update(sections)

            with st.spinner(
                f"Juristische Analyse läuft ({LEGAL_ANALYSIS_MODEL}, Web-Suche)…"
            ):
                try:
                    analyzer = LegalAnalyzer(api_key=st.session_state.openai_api_key)
                    jur_result = analyzer.analyze(analysis_input)
                    st.session_state.juristic_analysis = jur_result
                except Exception as e:
                    st.error(f"Analyse fehlgeschlagen: {e}")
                    jur_result = None

        # ── Ergebnis anzeigen ─────────────────────────────────────────────────
        jur = st.session_state.juristic_analysis
        if jur:
            if jur.get("error"):
                st.error(f"Fehler bei der Analyse: {jur['error']}")

            # Erfolgseinschätzung + Konfidenz
            p_jur = jur["erfolgseinschaetzung"]
            color_jur = "#2c6e49" if p_jur >= 0.55 else "#7d5a00" if p_jur >= 0.4 else "#8b1a1a"
            konf = jur.get("konfidenz", "mittel")

            col_j1, col_j2, col_j3 = st.columns(3)
            with col_j1:
                st.markdown(
                    f'<div class="ev-card">'
                    f'<div class="ev-value" style="color:{color_jur}">{p_jur:.0%}</div>'
                    f'<div class="ev-label">Juristische Erfolgseinschätzung</div>'
                    f'<div class="ev-sublabel">Konfidenz: <strong>{konf}</strong></div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with col_j2:
                st.markdown(
                    f'<div class="ev-card">'
                    f'<div class="ev-value" style="color:#1c3a5e">{ml_result["p_win"]:.0%}</div>'
                    f'<div class="ev-label">ML-Erfolgswahrscheinlichkeit</div>'
                    f'<div class="ev-sublabel">Statistisches Modell</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with col_j3:
                p_kombi = (p_jur + ml_result["p_win"]) / 2
                color_k = "#2c6e49" if p_kombi >= 0.55 else "#7d5a00" if p_kombi >= 0.4 else "#8b1a1a"
                st.markdown(
                    f'<div class="ev-card">'
                    f'<div class="ev-value" style="color:{color_k}">{p_kombi:.0%}</div>'
                    f'<div class="ev-label">Schnellkombination (Ø, gleiche Gewichtung)</div>'
                    f'<div class="ev-sublabel">Für genauere Gewichtung → Tab Erwartungswert</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

            if jur.get("einschaetzung_begruendung"):
                st.info(f"**Begründung:** {jur['einschaetzung_begruendung']}")

            st.markdown("---")

            # Stärken / Schwächen
            col_sw1, col_sw2 = st.columns(2)
            with col_sw1:
                st.markdown("**Stärken (Kläger)**")
                for s in jur.get("staerken_klaeger", []):
                    st.markdown(f"- {s}")
            with col_sw2:
                st.markdown("**Schwächen / Risiken (Kläger)**")
                for s in jur.get("schwaechen_klaeger", []):
                    st.markdown(f"- {s}")

            # Normen + Judikatur
            col_n1, col_n2 = st.columns(2)
            with col_n1:
                norms = jur.get("relevante_normen", [])
                if norms:
                    st.markdown("**Relevante Rechtsnormen**")
                    for n in norms:
                        st.markdown(f"- `{n}`")
            with col_n2:
                jud = jur.get("relevante_judikatur", [])
                if jud:
                    st.markdown("**Einschlägige OGH-Judikatur**")
                    for j in jud:
                        st.markdown(f"- {j}")

            # Volltext-Analyse
            if jur.get("analyse"):
                with st.expander("Vollständige juristische Analyse"):
                    st.markdown(jur["analyse"])

            # Quellen
            sources = jur.get("sources", [])
            if sources:
                with st.expander(f"Web-Quellen ({len(sources)})"):
                    for s in sources:
                        url = s.get("url", "")
                        title = s.get("title", url)
                        st.markdown(f"- [{title}]({url})")

            # Export
            st.download_button(
                "Juristische Analyse als JSON exportieren",
                data=json.dumps(jur, ensure_ascii=False, indent=2).encode(),
                file_name="juristische_analyse.json",
                mime="application/json",
            )


# ════════════════════════════════════════════════════════════════════════════════
# TAB 5: EXPECTED VALUE
# ════════════════════════════════════════════════════════════════════════════════

with tab_ev:
    st.markdown('<div class="section-title">Erwartungswert-Kalkulation</div>', unsafe_allow_html=True)

    st.caption(
        "Kombiniert ML-Vorhersage + juristische Einschätzung zur Berechnung des monetären "
        "Erwartungswerts inkl. RATG-Anwaltskosten und GGG-Gerichtsgebühren."
    )

    if st.session_state.prediction_result is None:
        st.info("Bitte zuerst eine Vorhersage im Tab **'Vorhersage'** berechnen.")
    else:
        ml_result = st.session_state.prediction_result

        # ── Schritt 1: Parameter ─────────────────────────────────────────────────
        st.markdown("#### 1. Parameter")
        ev_c1, ev_c2, ev_c3 = st.columns(3)

        with ev_c1:
            # Juristische Einschätzung: aus Tab 4 oder manuell
            jur = st.session_state.juristic_analysis
            default_jur = jur["erfolgseinschaetzung"] if jur and not jur.get("error") else 0.6
            juristic_estimate = st.slider(
                "Juristische Erfolgseinschätzung",
                0.0, 1.0, float(default_jur), 0.05,
                format="%.0f%%",
                help="Aus Tab 'Juristische Analyse' übernommen oder manuell. 0 = keine Chance, 1 = sicher.",
            )
            if jur and not jur.get("error"):
                st.caption(
                    f"KI-Analyse: {jur['erfolgseinschaetzung']:.0%} "
                    f"(Konfidenz: {jur.get('konfidenz','?')})"
                )
            w_ml = st.slider(
                "Gewichtung ML-Modell",
                0.0, 1.0, 0.5, 0.1,
                help="0.5 = gleichgewichtig; höher = mehr Vertrauen in ML-Modell.",
            )
            w_jurist = 1.0 - w_ml

        with ev_c2:
            streitwert = st.number_input(
                "Streitwert (EUR)",
                min_value=0.0, max_value=10_000_000.0,
                value=float(
                    (st.session_state.prediction_case_dict or {})
                    .get("structured", {})
                    .get("streitwert_eur") or 10_000.0
                ),
                step=500.0,
            )
            use_ratg = st.checkbox(
                "RATG + GGG Kosten automatisch berechnen",
                value=True,
                help=(
                    "Berechnet Anwaltskosten nach RATG Anlage 1 + GGG TP 1. "
                    "Deaktivieren für manuelle Kosteneingabe."
                ),
            )

        with ev_c3:
            if use_ratg:
                st.markdown("**RATG-Kostenparameter**")
                n_verh = st.number_input("Anzahl Verhandlungen", 1, 10, 2)
                h_verh = st.number_input("Ø Stunden pro Verhandlung", 0.5, 8.0, 2.0, 0.5)
                n_schrift_kl = st.number_input("Vorb. Schriftsätze (Kläger)", 0, 5, 1)
                n_schrift_bk = st.number_input("Vorb. Schriftsätze (Beklagter)", 0, 5, 1)
            else:
                manual_costs = st.number_input(
                    "Manuelle Kostenabschätzung (EUR)",
                    0.0, 500_000.0, 3_000.0, 500.0,
                    help="Anwaltskosten beider Seiten + Gerichtsgebühren (falls Verlust).",
                )

        ev_btn = st.button("Erwartungswert berechnen", type="primary", use_container_width=True)

        if ev_btn and st.session_state.predictor is not None:
            predictor = st.session_state.predictor
            ratg_res = None

            if use_ratg and streitwert > 0:
                try:
                    ratg_res = calculate_ratg_costs(
                        streitwert_eur=streitwert,
                        klage=True,
                        klagebeantwortung=True,
                        vorbereitende_schriftsaetze_klaeger=n_schrift_kl,
                        vorbereitende_schriftsaetze_beklagter=n_schrift_bk,
                        anzahl_verhandlungen=n_verh,
                        stunden_pro_verhandlung=h_verh,
                        include_ust=True,
                    )
                    st.session_state.ratg_result = ratg_res
                except Exception as e:
                    st.warning(f"RATG-Berechnung fehlgeschlagen: {e}")

            ev_result = predictor.compute_expected_value(
                ml_result=ml_result,
                juristic_estimate=juristic_estimate,
                w_ml=w_ml,
                w_jurist=w_jurist,
                streitwert_eur=streitwert if streitwert > 0 else None,
                cost_estimate_eur=None if use_ratg else (manual_costs if not use_ratg else None),
                ratg_result=ratg_res,
            )
            st.session_state.ev_result = ev_result

        # ── Schritt 2: RATG-Kostenaufstellung ────────────────────────────────────
        ratg = st.session_state.ratg_result
        if ratg:
            st.markdown("---")
            st.markdown("#### 2. Kostenaufstellung (RATG + GGG)")

            st.caption(ratg["disclaimer"])

            ratg_c1, ratg_c2, ratg_c3 = st.columns(3)
            with ratg_c1:
                st.markdown("**Kläger (RATG + USt)**")
                for key, pos in ratg["klaeger_positionen"].items():
                    st.markdown(f"- {pos['bezeichnung']}: **EUR {pos['netto']:,.2f}** netto")
                st.markdown(f"**Summe Kläger: EUR {ratg['klaeger_brutto_eur']:,.2f}** (inkl. 20 % USt)")

            with ratg_c2:
                st.markdown("**Beklagter (RATG + USt)**")
                for key, pos in ratg["beklagter_positionen"].items():
                    st.markdown(f"- {pos['bezeichnung']}: **EUR {pos['netto']:,.2f}** netto")
                st.markdown(f"**Summe Beklagter: EUR {ratg['beklagter_brutto_eur']:,.2f}** (inkl. 20 % USt)")

            with ratg_c3:
                st.markdown("**Gerichtsgebühren (GGG)**")
                ggg_data = ratg.get("ggg", {})
                st.markdown(
                    f"- {ggg_data.get('ggg_tp1_bezeichnung','TP 1 GGG')}: "
                    f"**EUR {ratg['ggg_tp1_eur']:,.2f}**"
                )
                st.caption(ggg_data.get("ggg_hinweis_praet_vergleich", ""))
                st.markdown(
                    f"**Gesamtbelastung (bei Niederlage):** "
                    f"EUR {ratg['gesamt_bei_niederlage_eur']:,.2f}"
                )
                st.caption(
                    f"Einheitssatz: {ratg['einheitssatz_pct']} — "
                    + ratg["klagebeantwortung_es_hinweis"]
                )

        # ── Schritt 3: Erwartungswert-Ergebnis ───────────────────────────────────
        if st.session_state.ev_result:
            ev = st.session_state.ev_result

            st.markdown("---")
            st.markdown("#### 3. Erwartungswert-Ergebnis")

            col_ev_r1, col_ev_r2, col_ev_r3 = st.columns(3)
            with col_ev_r1:
                p = ev["p_full_success_combined"]
                st.markdown(
                    f'<div class="ev-card {"positive" if p > 0.5 else "neutral"}">'
                    f'<div class="ev-value" style="color:#2c6e49">{p:.0%}</div>'
                    f'<div class="ev-label">Vollständiges Obsiegen</div></div>',
                    unsafe_allow_html=True,
                )
            with col_ev_r2:
                p2 = ev["p_partial_success_combined"]
                st.markdown(
                    f'<div class="ev-card neutral">'
                    f'<div class="ev-value" style="color:#7d5a00">{p2:.0%}</div>'
                    f'<div class="ev-label">Teilweises Obsiegen</div></div>',
                    unsafe_allow_html=True,
                )
            with col_ev_r3:
                p3 = ev["p_failure_combined"]
                st.markdown(
                    f'<div class="ev-card {"negative" if p3 > 0.5 else "neutral"}">'
                    f'<div class="ev-value" style="color:#8b1a1a">{p3:.0%}</div>'
                    f'<div class="ev-label">Unterliegen</div></div>',
                    unsafe_allow_html=True,
                )

            st.markdown("")
            ev_prob = ev["ev_success_probability"]
            color_ev = "#2c6e49" if ev_prob > 0.55 else "#7d5a00" if ev_prob > 0.4 else "#8b1a1a"

            col_ev_m1, col_ev_m2 = st.columns(2)
            with col_ev_m1:
                st.markdown(
                    f'<div class="ev-card">'
                    f'<div class="ev-value" style="color:{color_ev}">{ev_prob:.0%}</div>'
                    f'<div class="ev-label">Kombinierte Erfolgswahrscheinlichkeit</div>'
                    f'<div class="ev-sublabel">'
                    f'ML: {ev["ml_weight_applied"]:.0%} | Juristisch: {ev["juristic_weight_applied"]:.0%}'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )
                st.info(f"**Empfehlung:** {ev['recommendation']}")

                # Kostenaufschlüsselung
                if "cost_risk_detail" in ev:
                    cr = ev["cost_risk_detail"]
                    with st.expander("Kostendetail (§ 41 ZPO)"):
                        st.markdown(
                            f"- Bei Sieg: **EUR {cr['kosten_bei_sieg_eur']:,.0f}** (vollständig erstattet)\n"
                            f"- Bei Teilsieg: **EUR {cr['kosten_bei_teilsieg_eur']:,.0f}**\n"
                            f"- Bei Niederlage: **EUR {cr['kosten_bei_niederlage_eur']:,.0f}** "
                            f"(RATG + GGG beider Seiten)\n"
                            f"- **Erwartete Kostenbelastung: EUR {cr['erwartete_kostenbelastung_eur']:,.0f}**"
                        )

            with col_ev_m2:
                if "ev_gross_eur" in ev:
                    ev_gross = ev["ev_gross_eur"]
                    ev_net = ev.get("ev_net_eur", ev_gross)
                    proceed = ev.get("proceed_recommendation", ev_net > 0)
                    net_color = "#2c6e49" if proceed else "#8b1a1a"
                    verdict = "Klagbetreibung empfohlen" if proceed else "Klagbetreibung nicht empfohlen"
                    cost_label = ev.get("cost_source", "Kosten")

                    st.markdown(
                        f'<div class="ev-card {"positive" if proceed else "negative"}">'
                        f'<div class="ev-sublabel">Erwartungswert (brutto, ohne Kosten)</div>'
                        f'<div style="font-family:IBM Plex Mono,monospace;font-size:1.4rem;'
                        f'font-weight:600;color:#1c3a5e">EUR {ev_gross:,.0f}</div>'
                        f'<div class="ev-sublabel" style="margin-top:8px">'
                        f'Erwartete Kostenbelastung ({cost_label})</div>'
                        f'<div style="font-family:IBM Plex Mono,monospace;font-size:1.1rem;'
                        f'color:#8b1a1a">− EUR {ev.get("cost_estimate_eur",0):,.0f}</div>'
                        f'<div class="ev-sublabel" style="margin-top:8px">Netto-Erwartungswert</div>'
                        f'<div style="font-family:IBM Plex Mono,monospace;font-size:1.8rem;'
                        f'font-weight:600;color:{net_color}">EUR {ev_net:,.0f}</div>'
                        f'<div style="margin-top:8px;font-size:0.82rem;color:{net_color};'
                        f'font-weight:600">{verdict}</div>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            # ── Waterfall Chart ──────────────────────────────────────────────────
            if "ev_gross_eur" in ev:
                sw_val = ev["streitwert_eur"]
                costs_val = ev.get("cost_estimate_eur", 0) or 0
                ev_g = ev["ev_gross_eur"]
                ev_n = ev.get("ev_net_eur", ev_g)

                fig_wf = go.Figure(go.Waterfall(
                    name="EV",
                    orientation="v",
                    measure=["absolute", "relative", "relative", "total"],
                    x=["Streitwert", "Erfolgsfaktor", "Verfahrenskosten (erwartet)", "Netto-EV"],
                    y=[sw_val, ev_g - sw_val, -costs_val, 0],
                    text=[
                        f"EUR {sw_val:,.0f}",
                        f"EUR {ev_g - sw_val:,.0f}",
                        f"−EUR {costs_val:,.0f}",
                        f"EUR {ev_n:,.0f}",
                    ],
                    textposition="outside",
                    connector={"line": {"color": "#aaaaaa", "width": 1}},
                    increasing={"marker": {"color": "#2c6e49"}},
                    decreasing={"marker": {"color": "#8b1a1a"}},
                    totals={"marker": {"color": "#1c3a5e"}},
                ))
                fig_wf.update_layout(
                    height=320, showlegend=False,
                    paper_bgcolor="white", plot_bgcolor="#f5f5f5",
                    font=dict(family="IBM Plex Sans, sans-serif", size=11),
                )
                fig_wf.update_yaxes(showgrid=True, gridcolor="#dddddd")
                st.plotly_chart(fig_wf, use_container_width=True)

        # ── Schritt 4: Vorbringen / Gegenvorbringen aktualisieren ────────────────
        st.markdown("---")
        st.markdown("#### 4. Aktualisierung nach neuem Vorbringen / Gegenvorbringen")
        st.caption(
            "Nach Einbringung weiterer Schriftsätze: Aktualisiertes Vorbringen "
            "re-embedden und neue Vorhersage berechnen."
        )

        if st.session_state.prediction_case_dict is None:
            st.info("Bitte zuerst eine Vorhersage berechnen.")
        elif not st.session_state.openai_api_key:
            st.warning("OpenAI API-Key für Re-Embedding erforderlich.")
        else:
            with st.form("vorbringen_update_form"):
                vb_c1, vb_c2 = st.columns(2)
                with vb_c1:
                    new_klaeger = st.text_area(
                        "Aktualisiertes Kläger-Vorbringen",
                        placeholder="Neues / ergänztes Vorbringen des Klägers…",
                        height=100,
                    )
                with vb_c2:
                    new_beklagter = st.text_area(
                        "Aktualisiertes Gegenvorbringen (Beklagter)",
                        placeholder="Neues Gegenvorbringen, Einwendungen, Beweise…",
                        height=100,
                    )
                new_beweise = st.text_area(
                    "Aktualisierte Beweislage",
                    placeholder="Neue Beweismittel, Sachverständigenaussagen…",
                    height=70,
                )
                update_btn = st.form_submit_button(
                    "Vorhersage mit aktualisierten Texten berechnen",
                    type="primary",
                )

            if update_btn and (new_klaeger or new_beklagter or new_beweise):
                with st.spinner("Re-Embedding + neue Vorhersage läuft…"):
                    try:
                        predictor = st.session_state.predictor
                        updated = predictor.predict_with_updated_vorbringen(
                            original_case_dict=st.session_state.prediction_case_dict,
                            new_klaeger_text=new_klaeger or None,
                            new_beklagter_text=new_beklagter or None,
                            new_beweise_text=new_beweise or None,
                            api_key=st.session_state.openai_api_key,
                            original_embeddings=st.session_state.prediction_embeddings,
                        )
                        st.session_state.updated_prediction = updated
                    except Exception as e:
                        st.error(f"Fehler: {e}")

            upd = st.session_state.updated_prediction
            if upd:
                orig = st.session_state.prediction_result
                st.markdown("**Vergleich: Original vs. nach Vorbringen-Update**")

                upd_cols = st.columns(3)
                deltas = {
                    "P(Obsiegen)":   (orig["p_win"],     upd["p_win"]),
                    "P(Teilweise)":  (orig["p_partial"],  upd["p_partial"]),
                    "P(Unterliegen)":(orig["p_loss"],     upd["p_loss"]),
                }
                for col, (label, (v_orig, v_upd)) in zip(upd_cols, deltas.items()):
                    delta = v_upd - v_orig
                    col.metric(label, f"{v_upd:.1%}", f"{delta:+.1%}")

                # EV nach Update (mit bestehenden RATG-Kosten)
                if st.session_state.ev_result and "juristic_estimate_input" in st.session_state.ev_result:
                    old_ev = st.session_state.ev_result
                    jur_inp = old_ev["juristic_estimate_input"]
                    w_ml_inp = old_ev["ml_weight_applied"]
                    w_j_inp = old_ev["juristic_weight_applied"]
                    predictor = st.session_state.predictor
                    new_ev = predictor.compute_expected_value(
                        ml_result=upd,
                        juristic_estimate=jur_inp,
                        w_ml=w_ml_inp,
                        w_jurist=w_j_inp,
                        streitwert_eur=old_ev.get("streitwert_eur"),
                        ratg_result=st.session_state.ratg_result,
                        cost_estimate_eur=old_ev.get("cost_estimate_eur") if not st.session_state.ratg_result else None,
                    )
                    ev_new_prob = new_ev["ev_success_probability"]
                    ev_old_prob = old_ev["ev_success_probability"]
                    delta_ev = ev_new_prob - ev_old_prob
                    color_new = "#2c6e49" if ev_new_prob > 0.55 else "#7d5a00" if ev_new_prob > 0.4 else "#8b1a1a"
                    st.markdown(
                        f'<div class="ev-card">'
                        f'<div class="ev-sublabel">Neue kombinierte Erfolgswahrscheinlichkeit</div>'
                        f'<div class="ev-value" style="color:{color_new}">{ev_new_prob:.0%}</div>'
                        f'<div class="ev-sublabel">Veränderung: {delta_ev:+.1%} gegenüber Original</div>'
                        f'<div style="margin-top:4px;font-size:0.82rem">{new_ev["recommendation"]}</div>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )

        # ── Export ───────────────────────────────────────────────────────────────
        if st.session_state.ev_result:
            st.markdown("---")
            combined_export = {
                "ml_prediction": st.session_state.prediction_result,
                "juristische_analyse": st.session_state.juristic_analysis,
                "ratg_kosten": st.session_state.ratg_result,
                "erwartungswert": st.session_state.ev_result,
                "updated_prediction": st.session_state.updated_prediction,
            }
            st.download_button(
                "Vollständigen Bericht als JSON exportieren",
                data=json.dumps(combined_export, ensure_ascii=False, indent=2).encode(),
                file_name="litigation_analyse_komplett.json",
                mime="application/json",
            )


# ─── Footer ───────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f"<div style='text-align:center;color:#888888;font-size:0.75rem;"
    f"font-family:IBM Plex Mono,monospace;letter-spacing:0.04em'>"
    f"Predictive Litigation Analytics &nbsp;·&nbsp; v{APP_VERSION} &nbsp;·&nbsp; "
    f"kNN (< {KNN_THRESHOLD} Fälle) · LitigationClassifier (≥ {KNN_THRESHOLD} Fälle) &nbsp;·&nbsp; "
    f"Device: {'CUDA' if __import__('torch').cuda.is_available() else 'CPU'}"
    f"</div>",
    unsafe_allow_html=True,
)
