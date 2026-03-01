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
    KNN_THRESHOLD,
    MODEL_CHECKPOINT,
    OUTCOME_COLORS,
    OUTCOME_ICONS,
    OUTCOME_LABELS,
    TRAINING_CONFIG,
    TRAINING_HISTORY_FILE,
)
from data_extractor.data_manager import DataManager
from model.feature_engineer import FeatureEngineer
from model.predictor import LitigationPredictor
from model.ratg_calculator import RATGKostenrechnung, berechne_ratg_kosten
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
        "ev_result": None,
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
    st.markdown("**Training-Parameter**")
    epochs = st.slider("Max. Epochen", 50, 500, TRAINING_CONFIG["epochs"], 50)
    lr = st.select_slider(
        "Lernrate",
        [1e-4, 3e-4, 5e-4, 1e-3, 5e-3],
        value=TRAINING_CONFIG["learning_rate"],
        format_func=lambda x: f"{x:.0e}",
    )
    early_stop = st.slider(
        "Early Stopping (Epochen)",
        10, 60, TRAINING_CONFIG["early_stopping_patience"], 5,
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

tab_train, tab_eval, tab_predict, tab_ev = st.tabs([
    "Training",
    "Evaluation",
    "Vorhersage",
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
            "Keine beschrifteten Fälle mit Legal-Analyse vorhanden. "
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
        fe = FeatureEngineer()
        if n_cases < KNN_THRESHOLD:
            st.markdown(f"""
            **Aktiver Modus: k-Nearest-Neighbour** (< {KNN_THRESHOLD} Fälle)
            - **Methode**: Kosinus-Ähnlichkeit auf strukturierten Features (~{fe.feature_dim} dim)
            - **k**: min(5, Trainingsgröße) nächste Nachbarn, gewichtetes Soft-Voting
            - **Parameter**: 0 — kein Gradientenverfahren, kein Training
            - **Vorteil**: sofort einsatzbereit, kein Overfitting-Risiko

            Ab **{KNN_THRESHOLD} beschrifteten Fällen** schaltet das System automatisch auf das
            neuronale Netz um.
            """)
        else:
            from config import EMBEDDING_DIM_USED, NN_CONFIG as _nn_cfg, PCA_DIM
            st.markdown(f"""
            **Aktiver Modus: Hybrid-Neuronales Netz** (≥ {KNN_THRESHOLD} Fälle)
            - **Embedding-Input**: 2 × {EMBEDDING_DIM_USED}-dim → PCA → 2 × {PCA_DIM}-dim (Kläger + Beklagter)
            - **Embedding-Encoder**: {PCA_DIM} → {_nn_cfg['embedding_hidden_dim']} → {_nn_cfg['embedding_output_dim']} (pro Sektion)
            - **Structured Encoder**: {fe.feature_dim} → {_nn_cfg['structured_hidden_dim']} (Metadaten)
            - **Fusion**: Concat → {_nn_cfg['fusion_dims']} → 3 Klassen
            - **Loss**: CrossEntropyLoss mit Klassen-Gewichtung
            - **Optimizer**: AdamW mit ReduceLROnPlateau
            - **Regularisierung**: PCA, LayerNorm, Dropout ({_nn_cfg['dropout_fusion']}), Gradient Clipping, Early Stopping
            - **Parameter gesamt**: ~62 K

            **Input:**
            - Text-Embeddings (text-embedding-3-large, {EMBEDDING_DIM_USED}-dim → PCA {PCA_DIM}-dim)
            - Strukturierte Metadaten (Streitwert, Instanz, Anspruchsart, Einwendungen)

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

            if phase == "preparing":
                log_container.markdown(
                    f'<div class="training-log"><div class="log-line log-info">'
                    f'[INFO] {kwargs.get("message", "")}</div></div>',
                    unsafe_allow_html=True,
                )

            elif phase == "prepared":
                n_emb = kwargs.get("n_with_embeddings", "?")
                pca_var = kwargs.get("pca_variance_retained")
                pca_str = f" | PCA Varianz: {pca_var:.1%}" if pca_var else ""
                msg = (
                    f'[OK]  Train: {kwargs["train_size"]} | Val: {kwargs["val_size"]} | '
                    f'Structured: {kwargs["feature_dim"]} | '
                    f'Mit Embeddings: {n_emb}{pca_str}'
                )
                st.session_state.training_log.append(msg)

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
                # Show prediction distribution to diagnose degenerate models
                pred_dist = kwargs.get("val_pred_dist", {})
                dist_str = ""
                if pred_dist:
                    dist_str = (
                        f" &nbsp;|&nbsp; Pred: "
                        f"L={pred_dist.get(0, 0):.0%} "
                        f"T={pred_dist.get(1, 0):.0%} "
                        f"W={pred_dist.get(2, 0):.0%}"
                    )
                status_text.markdown(
                    f"**Epoche {epoch}/{total}** &nbsp;|&nbsp; "
                    f"Train Loss: `{kwargs.get('train_loss', 0):.4f}` &nbsp;|&nbsp; "
                    f"Val Acc: `{kwargs.get('val_acc', 0):.1%}` &nbsp;|&nbsp; "
                    f"Beste Val Acc: `{kwargs.get('best_val_acc', 0):.1%}` &nbsp;|&nbsp; "
                    f"LR: `{kwargs.get('lr', 0):.2e}`"
                    f"{dist_str}"
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
                st.markdown("**Falldaten**")
                fc1, fc2, fc3 = st.columns(3)

                with fc1:
                    p_streitwert = st.number_input(
                        "Streitwert (EUR)", min_value=0.0, step=500.0, value=10000.0
                    )
                    p_instanz = st.selectbox("Instanz", ["BG", "LG", "OLG", "OGH"])

                with fc2:
                    p_claim_type = st.selectbox("Anspruchsart", CLAIM_TYPES + ["Andere"])
                    p_anspruchsgruende = st.number_input(
                        "Anzahl Anspruchsgrundlagen", 1, 10, 2
                    )

                with fc3:
                    p_klaeger_beweismittel = st.number_input(
                        "Kläger-Beweismittel (Anzahl)", 0, 20, 3
                    )
                    p_beklagter_beweismittel = st.number_input(
                        "Beklagten-Beweismittel (Anzahl)", 0, 20, 2
                    )
                    p_sv = st.checkbox("Sachverständiger bestellt")

                st.markdown("**Einwendungen des Beklagten**")
                ew_c = st.columns(4)
                p_einwendungen = {}
                for i, d in enumerate(DEFENSE_TYPES):
                    with ew_c[i % 4]:
                        p_einwendungen[d] = st.checkbox(
                            DEFENSE_LABELS.get(d, d), key=f"pred_ew_{d}"
                        )

                st.markdown("**Textvorbringen & Beweise**")
                p_klaeger_text = st.text_area(
                    "Kläger-Vorbringen",
                    placeholder="Beschreiben Sie das Vorbringen des Klägers...",
                    height=100,
                )
                p_beklagter_text = st.text_area(
                    "Beklagten-Vorbringen",
                    placeholder="Einwendungen und Vorbringen des Beklagten...",
                    height=80,
                )
                p_aufgenommene_beweise = st.text_area(
                    "Aufgenommene Beweise",
                    placeholder=(
                        "Faktische Beschreibung der tatsächlich aufgenommenen Beweise — "
                        "ohne eigene Bewertung. Beispiel: 'Drei Zeugen bestätigten "
                        "übereinstimmend das klägerische Vorbringen; zwei Urkunden "
                        "(Rechnungen) sprechen dagegen; ein bautechnisches "
                        "Sachverständigengutachten liegt vor.'"
                    ),
                    height=100,
                    help=(
                        "Beschreiben Sie Art und Anzahl der Beweismittel (Zeugen, "
                        "Urkunden, Sachverständige), welche Partei sie beigebracht hat "
                        "und ob sie das jeweilige Vorbringen stützen oder widerlegen. "
                        "Keine rechtliche Bewertung."
                    ),
                )

                predict_btn = st.form_submit_button(
                    "Vorhersage berechnen", type="primary"
                )

        if predict_btn:
            case_dict = {
                "structured": {
                    "streitwert_eur": p_streitwert if p_streitwert > 0 else None,
                    "instanz": p_instanz,
                    "anspruchsart": p_claim_type,
                    "anspruchsgruende": [""] * p_anspruchsgruende,
                    "einwendungen": p_einwendungen,
                    "klaeger_beweismittel": [""] * p_klaeger_beweismittel,
                    "beklagter_beweismittel": [""] * p_beklagter_beweismittel,
                    "sachverstaendiger_bestellt": p_sv,
                },
                "sections": {
                    "klaegervorbringen": p_klaeger_text,
                    "beklagtenvorbringen": p_beklagter_text,
                    "aufgenommene_beweise": p_aufgenommene_beweise,
                },
            }

            with st.spinner("Embeddings berechnen & Vorhersage..."):
                # Generate embeddings for the text sections via OpenAI
                pred_embeddings = None
                if p_klaeger_text.strip() or p_beklagter_text.strip():
                    try:
                        import openai
                        from config import EMBEDDING_DIM, OPENAI_EMBEDDING_MODEL
                        client = openai.OpenAI(api_key=st.session_state.openai_api_key)
                        pred_embeddings = {}
                        for section_key, text in [
                            ("klaegervorbringen", p_klaeger_text),
                            ("beklagtenvorbringen", p_beklagter_text),
                        ]:
                            if text.strip():
                                resp = client.embeddings.create(
                                    model=OPENAI_EMBEDDING_MODEL,
                                    input=text.strip(),
                                    dimensions=EMBEDDING_DIM,
                                )
                                pred_embeddings[section_key] = np.array(
                                    resp.data[0].embedding, dtype=np.float32
                                )
                    except Exception as e:
                        st.warning(
                            f"Embedding-Berechnung fehlgeschlagen: {e}. "
                            "Vorhersage basiert nur auf strukturierten Daten."
                        )
                        pred_embeddings = None

                result = st.session_state.predictor.predict(
                    case_dict, embeddings=pred_embeddings
                )
                st.session_state.prediction_result = result

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


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4: EXPECTED VALUE
# ════════════════════════════════════════════════════════════════════════════════

with tab_ev:
    st.markdown('<div class="section-title">Erwartungswert-Kalkulation</div>', unsafe_allow_html=True)

    st.markdown("""
    Kombiniert die ML-Modell-Vorhersage mit der juristischen Erfolgseinschätzung
    zur Berechnung des Gesamterwartungswerts:

    > **E[outcome] = w_ML × P_ML(Obsiegen) + w_Jur × P_Juristisch(Obsiegen)**
    """)

    if st.session_state.prediction_result is None:
        st.info("Bitte zuerst eine Vorhersage im Tab 'Vorhersage' berechnen.")
    else:
        ml_result = st.session_state.prediction_result

        col_ev1, col_ev2 = st.columns([2, 1])

        with col_ev1:
            st.markdown("**Parameter**")

            ev_c1, ev_c2 = st.columns(2)

            with ev_c1:
                juristic_estimate = st.slider(
                    "Juristische Erfolgseinschätzung",
                    0.0, 1.0, 0.6, 0.05,
                    format="%.2f",
                    help="Einschätzung des juristischen KI-Assistenten (0 = keine Chance, 1 = sicher)",
                )
                w_ml = st.slider(
                    "Gewichtung ML-Modell",
                    0.0, 1.0, 0.5, 0.1,
                    help="Höher = mehr Vertrauen in das ML-Modell",
                )
                w_jurist = 1.0 - w_ml

            with ev_c2:
                streitwert = st.number_input(
                    "Streitwert (EUR)", 0.0, 10_000_000.0, 10000.0, 500.0
                )
                instanz_ev = st.selectbox(
                    "Instanz",
                    ["BG", "LG", "OLG", "OGH"],
                    index=1,
                    help="Zuständiges Gericht (beeinflusst RATG-Tarifposten und Einheitssatz)",
                )
                komplexitaet_ev = st.selectbox(
                    "Verfahrenskomplexität",
                    ["einfach", "mittel", "komplex"],
                    index=1,
                    help="Einfach: 1–2 Verhandlungstage · Mittel: 2–3 · Komplex: 4–5+",
                )

            # ── RATG-Vorschau ────────────────────────────────────────────────
            if streitwert > 0:
                _prev = berechne_ratg_kosten(streitwert, instanz_ev, komplexitaet_ev)
                with st.expander("RATG/GGG-Kostenrechnung (Vorschau)", expanded=False):
                    st.caption(
                        "Beträge nach RATG (BGBl. I Nr. 195/2013 i.d.F.) und GGG. "
                        "Angaben ohne Gewähr — bitte gegen aktuelles Amtsblatt prüfen."
                    )
                    _c1, _c2, _c3 = st.columns(3)
                    with _c1:
                        st.metric("GGG-Pauschalgebühr", f"EUR {_prev.ggg_pauschalgebuehr:,.0f}")
                        st.metric("Eigene Anwaltskosten (RATG)", f"EUR {_prev.eigene_anwaltskosten:,.0f}")
                    with _c2:
                        st.metric(
                            "Kosten bei Obsiegen (§ 41 ZPO)",
                            f"EUR {_prev.kosten_bei_obsiegen:,.0f}",
                            help="Gegner ersetzt alle Kosten vollständig.",
                        )
                        st.metric(
                            "Kosten bei Teilerfolg (§ 43 ZPO)",
                            f"EUR {_prev.kosten_bei_teilobsiegen:,.0f}",
                            help="Eigene Anwaltskosten + 50 % GGG; jede Partei trägt ihre Kosten.",
                        )
                    with _c3:
                        st.metric(
                            "Kosten bei Unterliegen (§ 41 ZPO)",
                            f"EUR {_prev.kosten_bei_unterliegen:,.0f}",
                            help="Eigene Anwaltskosten + GGG + gegnerische RATG-Kosten.",
                            delta=f"-EUR {_prev.kosten_bei_unterliegen:,.0f}",
                            delta_color="inverse",
                        )

                    # TP-Aufschlüsselung
                    st.markdown("**Tarifposten-Aufschlüsselung (eigene Seite)**")
                    _rows = []
                    for _tp, _d in _prev.tp_positionen.items():
                        _rows.append({
                            "Tarifpost": _tp,
                            "Anzahl": _d["anzahl"],
                            "Einzelbetrag (EUR)": f"{_d['einzel_eur']:,.2f}",
                            "Gesamt (EUR)": f"{_d['gesamt_eur']:,.2f}",
                        })
                    _rows.append({
                        "Tarifpost": "Einheitssatz ("
                            + f"{int(_prev.einheitssatz_betrag / _prev.tp_summe_basis * 100)} %)",
                        "Anzahl": "—",
                        "Einzelbetrag (EUR)": "—",
                        "Gesamt (EUR)": f"{_prev.einheitssatz_betrag:,.2f}",
                    })
                    _rows.append({
                        "Tarifpost": "**Summe Anwaltskosten**",
                        "Anzahl": "—",
                        "Einzelbetrag (EUR)": "—",
                        "Gesamt (EUR)": f"**{_prev.eigene_anwaltskosten:,.2f}**",
                    })
                    st.table(_rows)

            ev_btn = st.button(
                "Erwartungswert berechnen",
                type="primary",
                use_container_width=True,
            )

        if ev_btn:
            predictor = st.session_state.predictor
            ratg_kosten = (
                berechne_ratg_kosten(streitwert, instanz_ev, komplexitaet_ev)
                if streitwert > 0
                else None
            )
            ev_result = predictor.compute_expected_value(
                ml_result=ml_result,
                juristic_estimate=juristic_estimate,
                w_ml=w_ml,
                w_jurist=w_jurist,
                streitwert_eur=streitwert if streitwert > 0 else None,
                ratg_kosten=ratg_kosten,
            )
            st.session_state.ev_result = ev_result

        if st.session_state.ev_result:
            ev = st.session_state.ev_result

            st.markdown("---")
            st.markdown("**Erwartungswert-Ergebnis**")

            # ── Probability Summary ──────────────────────────────────────────────
            col_ev_r1, col_ev_r2, col_ev_r3 = st.columns(3)

            with col_ev_r1:
                p = ev["p_full_success_combined"]
                css = "positive" if p > 0.5 else "neutral"
                st.markdown(
                    f'<div class="ev-card {css}">'
                    f'<div class="ev-value" style="color:#2c6e49">{p:.0%}</div>'
                    f'<div class="ev-label">Vollständiges Obsiegen</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

            with col_ev_r2:
                p2 = ev["p_partial_success_combined"]
                st.markdown(
                    f'<div class="ev-card neutral">'
                    f'<div class="ev-value" style="color:#7d5a00">{p2:.0%}</div>'
                    f'<div class="ev-label">Teilweises Obsiegen</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

            with col_ev_r3:
                p3 = ev["p_failure_combined"]
                css3 = "negative" if p3 > 0.5 else "neutral"
                st.markdown(
                    f'<div class="ev-card {css3}">'
                    f'<div class="ev-value" style="color:#8b1a1a">{p3:.0%}</div>'
                    f'<div class="ev-label">Unterliegen</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

            st.markdown("")

            # ── Monetary EV ─────────────────────────────────────────────────────
            ev_prob = ev["ev_success_probability"]
            col_ev_m1, col_ev_m2 = st.columns(2)

            with col_ev_m1:
                color = "#2c6e49" if ev_prob > 0.55 else "#7d5a00" if ev_prob > 0.4 else "#8b1a1a"
                st.markdown(
                    f'<div class="ev-card">'
                    f'<div class="ev-value" style="color:{color}">{ev_prob:.0%}</div>'
                    f'<div class="ev-label">Kombinierte Erfolgswahrscheinlichkeit</div>'
                    f'<div class="ev-sublabel">'
                    f'ML: {ev["ml_weight_applied"]:.0%} &nbsp;|&nbsp; '
                    f'Juristisch: {ev["juristic_weight_applied"]:.0%}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

                st.info(f"**Empfehlung:** {ev['recommendation']}")

            with col_ev_m2:
                if "ev_gross_eur" in ev:
                    ev_gross = ev["ev_gross_eur"]
                    ev_net   = ev.get("ev_net_eur", ev_gross)
                    proceed  = ev.get("proceed_recommendation", ev_net > 0)
                    css_card  = "positive" if proceed else "negative"
                    net_color = "#2c6e49" if proceed else "#8b1a1a"
                    verdict   = "Klagbetreibung empfohlen" if proceed else "Klagbetreibung nicht empfohlen"
                    cost_model = ev.get("cost_model", "")
                    cost_badge = " · RATG" if cost_model == "RATG" else (" · manuell" if cost_model else "")

                    st.markdown(
                        f'<div class="ev-card {css_card}">'
                        f'<div class="ev-sublabel">Erwartungswert (brutto, ohne Kosten)</div>'
                        f'<div style="font-family:IBM Plex Mono,monospace;font-size:1.5rem;'
                        f'font-weight:600;color:#1c3a5e">EUR {ev_gross:,.0f}</div>'
                        f'<div class="ev-sublabel" style="margin-top:10px">'
                        f'Erwartungswert (netto, asymm. ZPO-Kosten{cost_badge})</div>'
                        f'<div style="font-family:IBM Plex Mono,monospace;font-size:1.8rem;'
                        f'font-weight:600;color:{net_color}">EUR {ev_net:,.0f}</div>'
                        f'<div style="margin-top:10px;font-size:0.82rem;font-family:IBM Plex Sans,sans-serif;'
                        f'color:{net_color};font-weight:600">{verdict}</div>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            # ── RATG-Kostenszenarien ─────────────────────────────────────────
            if "ratg_kosten" in ev:
                rk = ev["ratg_kosten"]
                sw_val = ev["streitwert_eur"]
                p_ob  = ev["p_full_success_combined"]
                p_tob = ev["p_partial_success_combined"]
                p_ul  = ev["p_failure_combined"]

                with st.expander("Kostenszenarien nach RATG/ZPO", expanded=True):
                    _sc1, _sc2, _sc3 = st.columns(3)
                    with _sc1:
                        netto_ob = sw_val - rk["kosten_obsiegen"]
                        st.markdown(
                            f'<div class="ev-card positive" style="text-align:center">'
                            f'<div style="font-weight:600;color:#2c6e49">Obsiegen ({p_ob:.0%})</div>'
                            f'<div style="font-size:0.8rem;color:#555;margin:4px 0">§ 41 ZPO – Gegner ersetzt alle Kosten</div>'
                            f'<div style="font-family:monospace;font-size:1.1rem;color:#2c6e49">+EUR {netto_ob:,.0f}</div>'
                            f'<div style="font-size:0.75rem;color:#888">Kosten: EUR {rk["kosten_obsiegen"]:,.0f}</div>'
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                    with _sc2:
                        netto_tob = sw_val * 0.5 - rk["kosten_teilobsiegen"]
                        col = "#2c6e49" if netto_tob >= 0 else "#8b1a1a"
                        sign = "+" if netto_tob >= 0 else ""
                        st.markdown(
                            f'<div class="ev-card neutral" style="text-align:center">'
                            f'<div style="font-weight:600;color:#7d5a00">Teilerfolg ({p_tob:.0%})</div>'
                            f'<div style="font-size:0.8rem;color:#555;margin:4px 0">§ 43 ZPO – eigene Anwaltskosten + ½ GGG</div>'
                            f'<div style="font-family:monospace;font-size:1.1rem;color:{col}">'
                            f'{sign}EUR {netto_tob:,.0f}</div>'
                            f'<div style="font-size:0.75rem;color:#888">Kosten: EUR {rk["kosten_teilobsiegen"]:,.0f}</div>'
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                    with _sc3:
                        netto_ul = -rk["kosten_unterliegen"]
                        st.markdown(
                            f'<div class="ev-card negative" style="text-align:center">'
                            f'<div style="font-weight:600;color:#8b1a1a">Unterliegen ({p_ul:.0%})</div>'
                            f'<div style="font-size:0.8rem;color:#555;margin:4px 0">§ 41 ZPO – eigene + GGG + Gegner-RATG</div>'
                            f'<div style="font-family:monospace;font-size:1.1rem;color:#8b1a1a">'
                            f'EUR {netto_ul:,.0f}</div>'
                            f'<div style="font-size:0.75rem;color:#888">'
                            f'davon Gegnerkosten: EUR {rk["gegner_anwaltskosten"]:,.0f}</div>'
                            f"</div>",
                            unsafe_allow_html=True,
                        )

            # ── Waterfall Chart ──────────────────────────────────────────────────
            if "ev_gross_eur" in ev:
                sw_val = ev["streitwert_eur"]
                ev_g   = ev["ev_gross_eur"]
                ev_n   = ev.get("ev_net_eur", ev_g)

                if "ratg_kosten" in ev:
                    rk     = ev["ratg_kosten"]
                    p_ob   = ev["p_full_success_combined"]
                    p_tob  = ev["p_partial_success_combined"]
                    p_ul   = ev["p_failure_combined"]
                    # Erwartete Kostenbelastung (probabilistisch)
                    ek_ob  = p_ob  * rk["kosten_obsiegen"]        # = 0
                    ek_tob = p_tob * rk["kosten_teilobsiegen"]
                    ek_ul  = p_ul  * rk["kosten_unterliegen"]
                    ek_ges = ek_ob + ek_tob + ek_ul

                    fig_wf = go.Figure(go.Waterfall(
                        name="EV",
                        orientation="v",
                        measure=["absolute", "relative", "relative", "relative", "relative", "total"],
                        x=[
                            "Streitwert",
                            "Erfolgsfaktor",
                            "E[Gegnerkosten bei Unterliegen]",
                            "E[Eigene Kosten bei Teilerfolg]",
                            "E[GGG-Anteil]",
                            "Netto-EV",
                        ],
                        y=[
                            sw_val,
                            ev_g - sw_val,
                            -(p_ul * rk["gegner_anwaltskosten"]),
                            -(ek_tob),
                            -(p_ul * rk["ggg"] + p_tob * rk["ggg"] * 0.5),
                            0,
                        ],
                        text=[
                            f"EUR {sw_val:,.0f}",
                            f"EUR {ev_g - sw_val:,.0f}",
                            f"-EUR {p_ul * rk['gegner_anwaltskosten']:,.0f}",
                            f"-EUR {ek_tob:,.0f}",
                            f"-EUR {p_ul * rk['ggg'] + p_tob * rk['ggg'] * 0.5:,.0f}",
                            f"EUR {ev_n:,.0f}",
                        ],
                        textposition="outside",
                        connector={"line": {"color": "#aaaaaa", "width": 1}},
                        increasing={"marker": {"color": "#2c6e49"}},
                        decreasing={"marker": {"color": "#8b1a1a"}},
                        totals={"marker": {"color": "#1c3a5e"}},
                    ))
                else:
                    costs = ev.get("cost_estimate_eur", 0)
                    fig_wf = go.Figure(go.Waterfall(
                        name="EV",
                        orientation="v",
                        measure=["absolute", "relative", "relative", "total"],
                        x=["Streitwert", "Erfolgsfaktor", "Verfahrenskosten", "Netto-EV"],
                        y=[sw_val, ev_g - sw_val, -costs, 0],
                        text=[
                            f"EUR {sw_val:,.0f}",
                            f"EUR {ev_g - sw_val:,.0f}",
                            f"-EUR {costs:,.0f}",
                            f"EUR {ev_n:,.0f}",
                        ],
                        textposition="outside",
                        connector={"line": {"color": "#aaaaaa", "width": 1}},
                        increasing={"marker": {"color": "#2c6e49"}},
                        decreasing={"marker": {"color": "#8b1a1a"}},
                        totals={"marker": {"color": "#1c3a5e"}},
                    ))

                fig_wf.update_layout(
                    title=None,
                    height=340,
                    showlegend=False,
                    paper_bgcolor="white",
                    plot_bgcolor="#f5f5f5",
                    font=dict(family="IBM Plex Sans, sans-serif", size=11),
                )
                fig_wf.update_yaxes(showgrid=True, gridcolor="#dddddd")
                st.plotly_chart(fig_wf, use_container_width=True)

            # ── Export ───────────────────────────────────────────────────────────
            combined = {
                "ml_prediction": ml_result,
                "erwartungswert": ev,
            }
            st.download_button(
                "Vollständigen Bericht als JSON exportieren",
                data=json.dumps(combined, ensure_ascii=False, indent=2).encode(),
                file_name="erwartungswert_analyse.json",
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
