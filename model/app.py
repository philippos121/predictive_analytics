"""
Predictive Litigation Analytics — Model Training & Prediction UI

Sophisticated Streamlit interface for:
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
from model.trainer import LitigationTrainer

# ─── Page Config ─────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Predictive Litigation Analytics",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    :root {
        --primary: #1a365d;
        --secondary: #2d6a9f;
        --accent: #c9a227;
        --success: #27ae60;
        --warning: #f39c12;
        --danger: #e74c3c;
    }
    .main .block-container { padding-top: 1rem; padding-bottom: 2rem; }

    .app-header {
        background: linear-gradient(135deg, #1a365d 0%, #4a235a 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(26, 54, 93, 0.4);
    }
    .app-header h1 { color: white; margin: 0; font-size: 1.8rem; }
    .app-header p { color: #d4b4e0; margin: 0.3rem 0 0 0; }

    .prob-bar-container { background: #eee; border-radius: 20px; height: 24px; overflow: hidden; margin: 4px 0; }
    .prob-bar { height: 100%; border-radius: 20px; display: flex; align-items: center; padding-left: 8px; color: white; font-weight: bold; font-size: 0.85rem; transition: width 0.5s ease; }
    .prob-win { background: linear-gradient(90deg, #27ae60, #2ecc71); }
    .prob-partial { background: linear-gradient(90deg, #e67e22, #f39c12); }
    .prob-loss { background: linear-gradient(90deg, #c0392b, #e74c3c); }

    .ev-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border: 2px solid transparent;
        text-align: center;
    }
    .ev-card.positive { border-color: #27ae60; }
    .ev-card.negative { border-color: #e74c3c; }
    .ev-card.neutral { border-color: #f39c12; }

    .metric-large { font-size: 2.5rem; font-weight: bold; }
    .metric-label { font-size: 0.9rem; color: #666; margin-top: 4px; }

    .training-log {
        background: #1e1e1e;
        color: #dcdcdc;
        border-radius: 8px;
        padding: 1rem;
        font-family: monospace;
        font-size: 0.85rem;
        height: 300px;
        overflow-y: auto;
    }
    .log-line { margin: 2px 0; }
    .log-success { color: #6bcb77; }
    .log-warning { color: #f4d35e; }
    .log-error { color: #e74c3c; }
    .log-info { color: #4cc9f0; }

    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1a365d;
        border-bottom: 3px solid #c9a227;
        padding-bottom: 0.3rem;
        margin-bottom: 1rem;
    }

    .outcome-gauge {
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        margin: 0.5rem 0;
    }
    .outcome-0 { background: linear-gradient(135deg, #fdecea, #ffe0e0); border: 2px solid #e74c3c; }
    .outcome-1 { background: linear-gradient(135deg, #fff3e0, #ffe8c0); border: 2px solid #f39c12; }
    .outcome-2 { background: linear-gradient(135deg, #e8f5e9, #d0f0da); border: 2px solid #27ae60; }

    .confidence-badge {
        display: inline-block;
        background: #e8f0fe;
        color: #1a365d;
        border-radius: 20px;
        padding: 0.2rem 0.8rem;
        font-size: 0.85rem;
        font-weight: 600;
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
    <h1>🔮 Predictive Litigation Analytics</h1>
    <p>Machine Learning Modell für österreichische Zivilprozesse · v{APP_VERSION}</p>
</div>
""", unsafe_allow_html=True)


# ─── Sidebar ─────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 📊 Model Status")

    model_exists = MODEL_CHECKPOINT.exists()
    if model_exists:
        st.success("✅ Trainiertes Modell vorhanden")
        mtime = MODEL_CHECKPOINT.stat().st_mtime
        from datetime import datetime
        st.caption(f"Zuletzt trainiert: {datetime.fromtimestamp(mtime).strftime('%d.%m.%Y %H:%M')}")
    else:
        st.warning("⚠️ Noch kein Modell trainiert")

    stats = dm.get_statistics()
    st.metric("Dataset-Größe", stats.get("total_cases", 0))
    st.metric("Beschriftete Fälle", stats.get("labeled_cases", 0))

    if stats.get("labeled_cases", 0) > 0:
        oc = stats.get("outcome_distribution", {})
        st.markdown("**Outcome-Verteilung:**")
        st.markdown(
            f"✅ Obsiegen: {oc.get('obsiegen', 0)} | "
            f"⚖️ Teilw.: {oc.get('teilweise', 0)} | "
            f"❌ Unterl.: {oc.get('unterliegen', 0)}"
        )

    st.divider()
    st.markdown("### ⚙️ Training-Config")
    epochs = st.slider("Max. Epochen", 50, 500, TRAINING_CONFIG["epochs"], 50)
    lr = st.select_slider(
        "Lernrate",
        [1e-4, 5e-4, 1e-3, 5e-3],
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
    if st.button("🔄 App neu laden", use_container_width=True):
        st.rerun()


# ─── Main Tabs ────────────────────────────────────────────────────────────────────

tab_train, tab_eval, tab_predict, tab_ev = st.tabs([
    "🏋️ Training",
    "📈 Evaluation",
    "🔮 Vorhersage",
    "💰 Erwartungswert",
])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1: TRAINING
# ════════════════════════════════════════════════════════════════════════════════

with tab_train:
    st.markdown('<div class="section-header">🏋️ Modell trainieren</div>', unsafe_allow_html=True)

    cases, embeddings_dict = dm.export_for_training()
    n_cases = len(cases)

    # Status
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    col_s1.metric("Trainingsdaten", n_cases)
    col_s2.metric("Min. empfohlen", "20")
    col_s3.metric("Gut (>50)", "50+")
    col_s4.metric("Ideal (>100)", "100+")

    if n_cases < 5:
        st.error(
            f"⚠️ Nicht genug Daten für Training! "
            f"{n_cases}/5 beschriftete Fälle mit Embeddings. "
            "Bitte zuerst mehr Urteile mit dem Data Extractor verarbeiten."
        )
    elif n_cases < 20:
        st.warning(
            f"⚠️ Nur {n_cases} Fälle — Modell wird trainiert, "
            "aber mit mehr Daten (>50) wird die Genauigkeit erheblich besser."
        )
    else:
        st.success(f"✅ {n_cases} Fälle für das Training verfügbar.")

    st.divider()

    # ── Architecture Info ────────────────────────────────────────────────────────
    with st.expander("🧠 Modell-Architektur", expanded=False):
        fe = FeatureEngineer()
        st.markdown(f"""
        **Netzwerk-Architektur:**
        - **Embedding Encoder** (5×): Linear(3072 → 512) + LayerNorm + GELU + Dropout → Linear(512 → 256)
        - **Structured Encoder**: Linear({fe.feature_dim} → 64) + LayerNorm + GELU
        - **Fusion Network**: Linear(1344 → 512) → Linear(512 → 256) → Linear(256 → 128) → Linear(128 → 3)
        - **Loss**: Focal Loss (γ=2) mit Klassen-Gewichtung
        - **Optimizer**: AdamW mit ReduceLROnPlateau
        - **Regulierung**: LayerNorm, Dropout, Gradient Clipping, Early Stopping

        **Input-Dimensionen:**
        - 5 Textabschnitt-Embeddings: 5 × 3072 = 15.360 dim
        - Strukturierte Features: {fe.feature_dim} dim (Streitwert, Anspruchsart, Einwendungen, etc.)

        **Output:** 3 Klassen (Unterliegen / Teilweise / Obsiegen)
        """)

    # ── Training Controls ────────────────────────────────────────────────────────
    col_btn1, col_btn2 = st.columns([2, 1])

    with col_btn1:
        train_btn = st.button(
            f"▶ Training starten ({n_cases} Fälle)",
            type="primary",
            disabled=n_cases < 3 or st.session_state.training_running,
            use_container_width=True,
        )

    with col_btn2:
        if MODEL_CHECKPOINT.exists():
            if st.button("🔄 Modell neu trainieren", use_container_width=True):
                MODEL_CHECKPOINT.unlink(missing_ok=True)
                st.session_state.predictor = None
                st.session_state.trainer = None
                st.rerun()

    # ── Training Progress ────────────────────────────────────────────────────────
    if train_btn and not st.session_state.training_running and n_cases >= 3:
        st.session_state.training_running = True
        st.session_state.training_log = []

        # Setup trainer
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
                st.session_state.training_log.append(
                    f'[✓] Train: {kwargs["train_size"]} | Val: {kwargs["val_size"]} | '
                    f'Features: {kwargs["feature_dim"]}'
                )

            elif phase == "model_built":
                st.session_state.training_log.append(
                    f'[✓] Modell gebaut: {kwargs["parameters"]:,} Parameter | '
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
                    f"**Epoche {epoch}/{total}** | "
                    f"Train Loss: `{kwargs.get('train_loss', 0):.4f}` | "
                    f"Val Acc: `{kwargs.get('val_acc', 0):.1%}` | "
                    f"Beste Val Acc: `{kwargs.get('best_val_acc', 0):.1%}` | "
                    f"LR: `{kwargs.get('lr', 0):.2e}`"
                )

                # Update training chart
                if len(train_losses) > 1:
                    epochs_range = list(range(1, len(train_losses) + 1, max(1, len(train_losses) // 100)))
                    if len(train_losses) - 1 not in epochs_range:
                        epochs_range.append(len(train_losses))

                    fig = make_subplots(rows=1, cols=2, subplot_titles=("Loss", "Accuracy"))
                    fig.add_trace(
                        go.Scatter(y=train_losses, name="Train Loss", line=dict(color="#2d6a9f")),
                        row=1, col=1,
                    )
                    fig.add_trace(
                        go.Scatter(y=val_losses, name="Val Loss", line=dict(color="#e74c3c", dash="dash")),
                        row=1, col=1,
                    )
                    fig.add_trace(
                        go.Scatter(y=train_accs, name="Train Acc", line=dict(color="#27ae60")),
                        row=1, col=2,
                    )
                    fig.add_trace(
                        go.Scatter(y=val_accs, name="Val Acc", line=dict(color="#f39c12", dash="dash")),
                        row=1, col=2,
                    )
                    fig.update_layout(height=300, margin=dict(t=30, b=10), showlegend=True)
                    chart_container.plotly_chart(fig, use_container_width=True)

            elif phase == "early_stop":
                st.session_state.training_log.append(
                    f'[⚠] {kwargs.get("message", "Early stopping")}'
                )

            elif phase == "done":
                st.session_state.training_log.append(
                    f'[✅] FERTIG! Beste Val-Accuracy: {kwargs["best_val_acc"]:.1%} | '
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

            st.success(
                f"✅ Training abgeschlossen! "
                f"Beste Validierungs-Accuracy: **{history['best_val_acc']:.1%}** "
                f"(Epoche {history['best_epoch']})"
            )
            st.info("💡 Wechseln Sie zum Tab 'Evaluation' für detaillierte Auswertung.")

        except Exception as e:
            st.session_state.training_running = False
            progress_bar.empty()
            st.error(f"❌ Trainingsfehler: {e}")

    # ── Show Training Log ────────────────────────────────────────────────────────
    if st.session_state.training_log:
        st.markdown("**Training-Protokoll:**")
        log_html = '<div class="training-log">'
        for line in st.session_state.training_log:
            cls = "log-success" if "✅" in line else "log-warning" if "⚠" in line else "log-info"
            log_html += f'<div class="log-line {cls}">{line}</div>'
        log_html += "</div>"
        st.markdown(log_html, unsafe_allow_html=True)

    # ── Existing Training History ────────────────────────────────────────────────
    if TRAINING_HISTORY_FILE.exists() and not st.session_state.training_running:
        with open(TRAINING_HISTORY_FILE) as f:
            hist = json.load(f)

        if hist.get("train_loss"):
            st.markdown("### 📈 Letztes Training")
            col_h1, col_h2, col_h3, col_h4 = st.columns(4)
            col_h1.metric("Beste Val-Accuracy", f"{hist.get('best_val_acc', 0):.1%}")
            col_h2.metric("Beste Epoche", hist.get("best_epoch", "—"))
            col_h3.metric("Trainierte Epochen", hist.get("epochs_trained", "—"))
            col_h4.metric("Trainingszeit", f"{hist.get('training_time_sec', 0):.0f}s")

            tl = hist["train_loss"]
            vl = hist["val_loss"]
            ta = hist["train_acc"]
            va = hist["val_acc"]

            fig = make_subplots(rows=1, cols=2, subplot_titles=("Verlust (Loss)", "Genauigkeit (Accuracy)"))
            fig.add_trace(go.Scatter(y=tl, name="Train Loss", line=dict(color="#2d6a9f")), row=1, col=1)
            fig.add_trace(go.Scatter(y=vl, name="Val Loss", line=dict(color="#e74c3c", dash="dash")), row=1, col=1)
            fig.add_trace(go.Scatter(y=ta, name="Train Acc", line=dict(color="#27ae60")), row=1, col=2)
            fig.add_trace(go.Scatter(y=va, name="Val Acc", line=dict(color="#f39c12", dash="dash")), row=1, col=2)
            fig.update_layout(height=350, margin=dict(t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2: EVALUATION
# ════════════════════════════════════════════════════════════════════════════════

with tab_eval:
    st.markdown('<div class="section-header">📈 Modell-Evaluation</div>', unsafe_allow_html=True)

    if st.session_state.predictor is None:
        st.info("Kein Modell vorhanden. Bitte zuerst das Modell im Tab 'Training' trainieren.")
    else:
        predictor = st.session_state.predictor
        trainer = st.session_state.trainer or predictor.trainer

        if st.button("🔄 Vollständige Evaluation berechnen", type="primary"):
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
            st.markdown("### Gesamt-Metriken")
            col_e1, col_e2, col_e3 = st.columns(3)
            col_e1.metric("Overall Accuracy", f"{result['accuracy']:.1%}")

            per_class = result.get("per_class", {})
            macro_f1 = np.mean([per_class[c]["f1"] for c in per_class]) if per_class else 0
            col_e2.metric("Macro F1-Score", f"{macro_f1:.3f}")
            col_e3.metric("Ausgewertete Fälle", len(result.get("labels", [])))

            st.markdown("### Per-Klasse Metriken")
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
            st.markdown("### Konfusionsmatrix")
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
                color_continuous_scale="Blues",
                text_auto=True,
            )
            fig_cm.update_layout(height=400)
            st.plotly_chart(fig_cm, use_container_width=True)

            # ── Probability Distribution ─────────────────────────────────────────
            st.markdown("### Probability-Verteilung")
            if result.get("probabilities"):
                probs = np.array(result["probabilities"])
                labels_arr = np.array(labels)

                fig_probs = go.Figure()
                colors = ["#e74c3c", "#f39c12", "#27ae60"]
                for cls in range(3):
                    mask = labels_arr == cls
                    if mask.sum() > 0:
                        fig_probs.add_trace(go.Box(
                            y=probs[mask, cls],
                            name=f"{OUTCOME_LABELS[cls]} (true)",
                            marker_color=colors[cls],
                            boxpoints="all",
                        ))
                fig_probs.update_layout(
                    title="Vorhersage-Konfidenz nach wahrer Klasse",
                    yaxis_title="Predicted Probability",
                    height=350,
                )
                st.plotly_chart(fig_probs, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3: PREDICTION
# ════════════════════════════════════════════════════════════════════════════════

with tab_predict:
    st.markdown('<div class="section-header">🔮 Neuen Fall vorhersagen</div>', unsafe_allow_html=True)

    if st.session_state.predictor is None:
        st.warning("⚠️ Kein trainiertes Modell. Bitte zuerst trainieren.")
    else:
        st.markdown(
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
                        "Streitwert (€)", min_value=0.0, step=500.0, value=10000.0
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

                st.markdown("**Textvorbringen** (für Embeddings)")
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
                p_rechtl = st.text_area(
                    "Relevante Rechtsfragen (optional)",
                    height=60,
                )

                predict_btn = st.form_submit_button(
                    "🔮 Vorhersage berechnen", type="primary"
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
                }
            }

            sections = {
                "klaegervorbringen": p_klaeger_text,
                "beklagtenvorbringen": p_beklagter_text,
                "feststellungen": "",
                "beweisw_rdigung": "",
                "rechtliche_beurteilung": p_rechtl,
            }

            # Generate embeddings if API key provided
            embeddings = {}
            if st.session_state.openai_api_key and (p_klaeger_text or p_beklagter_text):
                with st.spinner("Generiere Embeddings via OpenAI..."):
                    try:
                        from data_extractor.openai_extractor import OpenAIExtractor
                        extractor = OpenAIExtractor(st.session_state.openai_api_key)
                        embeddings = extractor.generate_embeddings(sections)
                    except Exception as e:
                        st.warning(f"Embedding-Fehler: {e} — Verwende Null-Vektoren.")

            with st.spinner("Berechne Vorhersage..."):
                result = st.session_state.predictor.predict(case_dict, embeddings)
                st.session_state.prediction_result = result

        # ── Display Results ──────────────────────────────────────────────────────
        with col_input2:
            if st.session_state.prediction_result:
                result = st.session_state.prediction_result
                pred_cls = result["predicted_outcome"]

                st.markdown("## 📊 Ergebnis")

                # Predicted outcome gauge
                outcome_css = f"outcome-{pred_cls}"
                st.markdown(
                    f'<div class="outcome-gauge {outcome_css}">'
                    f'<div style="font-size:2.5rem">{OUTCOME_ICONS[pred_cls]}</div>'
                    f'<div style="font-size:1.4rem;font-weight:bold">{OUTCOME_LABELS[pred_cls]}</div>'
                    f'<span class="confidence-badge">Konfidenz: {result["confidence"]:.0%}</span>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

                st.markdown("**Wahrscheinlichkeiten:**")

                # Probability bars
                probs = [
                    ("Obsiegen", result["p_win"], "prob-win", "#27ae60"),
                    ("Teilweise", result["p_partial"], "prob-partial", "#f39c12"),
                    ("Unterliegen", result["p_loss"], "prob-loss", "#e74c3c"),
                ]

                for label, prob, css_class, color in probs:
                    pct = int(prob * 100)
                    st.markdown(
                        f'<div style="margin: 6px 0">'
                        f'<div style="display:flex;justify-content:space-between;font-size:0.85rem">'
                        f"<span>{label}</span><span><b>{prob:.1%}</b></span></div>"
                        f'<div class="prob-bar-container">'
                        f'<div class="prob-bar {css_class}" style="width:{pct}%">'
                        f"</div></div></div>",
                        unsafe_allow_html=True,
                    )

                # Plotly gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=result["p_win"] * 100,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Obsiegen-Wahrsch. (%)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#2d6a9f"},
                        "steps": [
                            {"range": [0, 30], "color": "#fdecea"},
                            {"range": [30, 55], "color": "#fff3e0"},
                            {"range": [55, 100], "color": "#e8f5e9"},
                        ],
                        "threshold": {
                            "line": {"color": "#27ae60", "width": 3},
                            "thickness": 0.75,
                            "value": 55,
                        },
                    },
                ))
                fig_gauge.update_layout(height=250, margin=dict(t=40, b=0, l=20, r=20))
                st.plotly_chart(fig_gauge, use_container_width=True)

                # Download result
                result_json = json.dumps(result, ensure_ascii=False, indent=2)
                st.download_button(
                    "⬇ Ergebnis als JSON",
                    data=result_json.encode(),
                    file_name="vorhersage_ergebnis.json",
                    mime="application/json",
                )


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4: EXPECTED VALUE
# ════════════════════════════════════════════════════════════════════════════════

with tab_ev:
    st.markdown('<div class="section-header">💰 Erwartungswert-Kalkulation</div>', unsafe_allow_html=True)

    st.markdown("""
    Kombiniert die **ML-Modell-Vorhersage** mit der **juristischen Erfolgseinschätzung**
    des KI-Systems zur Berechnung des Gesamterwartungswerts:

    > **E[outcome] = w_ML × P_ML(Obsiegen) + w_Jur × P_Juristisch(Obsiegen)**
    """)

    if st.session_state.prediction_result is None:
        st.info(
            "Bitte zuerst eine Vorhersage im Tab 'Vorhersage' berechnen."
        )
    else:
        ml_result = st.session_state.prediction_result

        col_ev1, col_ev2 = st.columns([2, 1])

        with col_ev1:
            st.markdown("### Parameter")

            ev_c1, ev_c2 = st.columns(2)

            with ev_c1:
                juristic_estimate = st.slider(
                    "Juristische Erfolgseinschätzung (KI-System)",
                    0.0, 1.0, 0.6, 0.05,
                    format="%.0f%%",
                    help="Einschätzung des juristischen KI-Assistenten (0=keine Chance, 1=sicher)",
                )
                # Convert to 0-1
                w_ml = st.slider(
                    "Gewichtung ML-Modell",
                    0.0, 1.0, 0.5, 0.1,
                    help="Höher = mehr Vertrauen in das ML-Modell",
                )
                w_jurist = 1.0 - w_ml

            with ev_c2:
                streitwert = st.number_input(
                    "Streitwert (€)", 0.0, 10_000_000.0, 10000.0, 500.0
                )
                cost_estimate = st.number_input(
                    "Geschätzte Verfahrenskosten (€)",
                    0.0, 500_000.0, 3000.0, 500.0,
                    help="Anwalts- und Gerichtskosten (beider Parteien falls Verlust)",
                )

            ev_btn = st.button(
                "💰 Erwartungswert berechnen",
                type="primary",
                use_container_width=True,
            )

        if ev_btn:
            predictor = st.session_state.predictor
            ev_result = predictor.compute_expected_value(
                ml_result=ml_result,
                juristic_estimate=juristic_estimate,
                w_ml=w_ml,
                w_jurist=w_jurist,
                streitwert_eur=streitwert if streitwert > 0 else None,
                cost_estimate_eur=cost_estimate if cost_estimate > 0 else None,
            )
            st.session_state.ev_result = ev_result

        if st.session_state.ev_result:
            ev = st.session_state.ev_result

            st.markdown("---")
            st.markdown("### 📊 Erwartungswert-Ergebnis")

            # ── Probability Summary ──────────────────────────────────────────────
            col_ev_r1, col_ev_r2, col_ev_r3 = st.columns(3)

            with col_ev_r1:
                p = ev["p_full_success_combined"]
                st.markdown(
                    f'<div class="ev-card {"positive" if p > 0.5 else "neutral"}">'
                    f'<div class="metric-large" style="color:#27ae60">{p:.0%}</div>'
                    f'<div class="metric-label">Vollständiges Obsiegen</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

            with col_ev_r2:
                p2 = ev["p_partial_success_combined"]
                st.markdown(
                    f'<div class="ev-card neutral">'
                    f'<div class="metric-large" style="color:#f39c12">{p2:.0%}</div>'
                    f'<div class="metric-label">Teilweises Obsiegen</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

            with col_ev_r3:
                p3 = ev["p_failure_combined"]
                st.markdown(
                    f'<div class="ev-card {"negative" if p3 > 0.5 else "neutral"}">'
                    f'<div class="metric-large" style="color:#e74c3c">{p3:.0%}</div>'
                    f'<div class="metric-label">Unterliegen</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

            st.markdown("")

            # ── Monetary EV ─────────────────────────────────────────────────────
            ev_prob = ev["ev_success_probability"]
            col_ev_m1, col_ev_m2 = st.columns(2)

            with col_ev_m1:
                color = "#27ae60" if ev_prob > 0.55 else "#f39c12" if ev_prob > 0.4 else "#e74c3c"
                st.markdown(
                    f'<div class="ev-card">'
                    f'<div class="metric-large" style="color:{color}">{ev_prob:.0%}</div>'
                    f'<div class="metric-label">Kombinierte Erfolgwahrscheinlichkeit</div>'
                    f'<div style="margin-top:8px;font-size:0.8rem;color:#666">'
                    f'ML: {ev["ml_weight_applied"]:.0%} | '
                    f'Juristisch: {ev["juristic_weight_applied"]:.0%}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

                st.info(f"**Empfehlung:** {ev['recommendation']}")

            with col_ev_m2:
                if "ev_gross_eur" in ev:
                    ev_gross = ev["ev_gross_eur"]
                    ev_net = ev.get("ev_net_eur", ev_gross)
                    proceed = ev.get("proceed_recommendation", ev_net > 0)

                    net_color = "#27ae60" if proceed else "#e74c3c"
                    net_icon = "✅" if proceed else "❌"

                    st.markdown(
                        f'<div class="ev-card {"positive" if proceed else "negative"}">'
                        f'<div style="font-size:0.9rem;color:#666">Erwartungswert (brutto)</div>'
                        f'<div style="font-size:1.8rem;font-weight:bold;color:#2d6a9f">€ {ev_gross:,.0f}</div>'
                        f'<div style="font-size:0.9rem;color:#666;margin-top:8px">Erwartungswert (netto, nach Kosten)</div>'
                        f'<div style="font-size:2rem;font-weight:bold;color:{net_color}">€ {ev_net:,.0f}</div>'
                        f'<div style="margin-top:10px">{net_icon} {"Klagbetreibung empfohlen" if proceed else "Klagbetreibung nicht empfohlen"}</div>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            # ── Waterfall Chart ──────────────────────────────────────────────────
            if "ev_gross_eur" in ev:
                sw = ev["streitwert_eur"]
                costs = ev.get("cost_estimate_eur", 0)
                ev_g = ev["ev_gross_eur"]
                ev_n = ev.get("ev_net_eur", ev_g)

                fig_wf = go.Figure(go.Waterfall(
                    name="EV Berechnung",
                    orientation="v",
                    measure=["absolute", "relative", "relative", "total"],
                    x=["Streitwert", "Erfolgsfaktor", "Verfahrenskosten", "Netto-EV"],
                    y=[sw, ev_g - sw, -costs, 0],
                    text=[f"€{sw:,.0f}", f"€{ev_g-sw:,.0f}", f"-€{costs:,.0f}", f"€{ev_n:,.0f}"],
                    textposition="outside",
                    connector={"line": {"color": "rgb(63, 63, 63)"}},
                    increasing={"marker": {"color": "#27ae60"}},
                    decreasing={"marker": {"color": "#e74c3c"}},
                    totals={"marker": {"color": "#2d6a9f"}},
                ))
                fig_wf.update_layout(
                    title="Erwartungswert-Berechnung",
                    height=350,
                    showlegend=False,
                )
                st.plotly_chart(fig_wf, use_container_width=True)

            # ── Export ───────────────────────────────────────────────────────────
            combined = {
                "ml_prediction": ml_result,
                "erwartungswert": ev,
            }
            st.download_button(
                "⬇ Vollständigen Bericht als JSON",
                data=json.dumps(combined, ensure_ascii=False, indent=2).encode(),
                file_name="erwartungswert_analyse.json",
                mime="application/json",
            )


# ─── Footer ───────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f"<div style='text-align:center;color:#6c757d;font-size:0.8rem'>"
    f"Predictive Litigation Analytics · v{APP_VERSION} · "
    f"Modell: LitigationClassifier (PyTorch) · "
    f"Device: {'CUDA' if __import__('torch').cuda.is_available() else 'CPU'}"
    f"</div>",
    unsafe_allow_html=True,
)
