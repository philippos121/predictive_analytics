"""
Litigation Data Extractor — Streamlit UI

Sophisticated UI for extracting structured legal data from Austrian civil
judgment PDFs using OpenAI GPT-4o-mini and text-embedding-3-large.
"""

import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_extras.metric_cards import style_metric_cards

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    APP_TITLE_EXTRACTOR,
    APP_VERSION,
    CLAIM_TYPES,
    DEFENSE_LABELS,
    DEFENSE_TYPES,
    EMBEDDING_SECTION_LABELS,
    OUTCOME_COLORS,
    OUTCOME_LABELS,
)
from data_extractor.data_manager import DataManager
from data_extractor.openai_extractor import OpenAIExtractor
from data_extractor.pdf_processor import PDFProcessingError, extract_text_from_pdf

# ─── Page Config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Litigation Data Extractor",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Main theme */
    :root {
        --primary: #1a365d;
        --secondary: #2d6a9f;
        --accent: #c9a227;
        --success: #27ae60;
        --warning: #f39c12;
        --danger: #e74c3c;
        --bg-card: #f8f9fa;
        --text-muted: #6c757d;
    }

    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }

    /* Header */
    .app-header {
        background: linear-gradient(135deg, #1a365d 0%, #2d6a9f 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(26, 54, 93, 0.3);
    }
    .app-header h1 { color: white; margin: 0; font-size: 1.8rem; }
    .app-header p { color: #b8d4f0; margin: 0.3rem 0 0 0; font-size: 0.95rem; }

    /* Cards */
    .stat-card {
        background: white;
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid var(--secondary);
    }
    .stat-card.success { border-left-color: var(--success); }
    .stat-card.warning { border-left-color: var(--warning); }
    .stat-card.danger { border-left-color: var(--danger); }

    /* PDF Item */
    .pdf-item {
        background: #f0f4f8;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.4rem 0;
        display: flex;
        align-items: center;
        gap: 0.8rem;
    }
    .pdf-item.processed { background: #e8f5e9; border-left: 3px solid #27ae60; }
    .pdf-item.error { background: #fdecea; border-left: 3px solid #e74c3c; }
    .pdf-item.pending { background: #f0f4f8; border-left: 3px solid #90a4ae; }

    /* Progress */
    .progress-container {
        background: #f0f4f8;
        border-radius: 10px;
        padding: 1.2rem;
        margin: 1rem 0;
    }

    /* Section badge */
    .section-badge {
        display: inline-block;
        background: #e8f0fe;
        color: #1a73e8;
        border-radius: 20px;
        padding: 0.2rem 0.7rem;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.2rem;
    }

    /* Outcome badge */
    .outcome-0 { background: #fdecea; color: #c62828; }
    .outcome-1 { background: #fff3e0; color: #e65100; }
    .outcome-2 { background: #e8f5e9; color: #1b5e20; }

    /* Defense tag */
    .defense-tag {
        display: inline-block;
        background: #e8f0fe;
        color: #3c4043;
        border-radius: 4px;
        padding: 0.15rem 0.5rem;
        font-size: 0.78rem;
        margin: 0.15rem;
    }
    .defense-tag.active {
        background: #ff6d00;
        color: white;
    }

    /* Sidebar */
    .sidebar-section {
        background: #f0f4f8;
        border-radius: 8px;
        padding: 0.8rem;
        margin: 0.5rem 0;
    }

    /* Table */
    .dataframe { font-size: 0.85rem; }

    /* Highlight box */
    .highlight-box {
        background: linear-gradient(135deg, #e8f0fe, #f3e5f5);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }

    /* Step indicator */
    .step-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 28px;
        height: 28px;
        border-radius: 50%;
        background: var(--secondary);
        color: white;
        font-weight: bold;
        font-size: 0.85rem;
        margin-right: 0.5rem;
    }
    .step-badge.done { background: var(--success); }
</style>
""", unsafe_allow_html=True)


# ─── Session State Initialization ────────────────────────────────────────────────

def init_session_state():
    defaults = {
        "api_key": os.environ.get("OPENAI_API_KEY", ""),
        "pdf_folder": "",
        "processing": False,
        "process_log": [],
        "current_file": "",
        "progress_pct": 0.0,
        "data_manager": None,
        "active_tab": "extractor",
        "selected_case_id": None,
        "edit_mode": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if st.session_state.data_manager is None:
        st.session_state.data_manager = DataManager()


init_session_state()
dm: DataManager = st.session_state.data_manager


# ─── Header ─────────────────────────────────────────────────────────────────────

st.markdown(f"""
<div class="app-header">
    <h1>⚖️ Litigation Data Extractor</h1>
    <p>Strukturierte Erfassung österreichischer Zivilurteile für Predictive Analytics · v{APP_VERSION}</p>
</div>
""", unsafe_allow_html=True)


# ─── Sidebar ─────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Konfiguration")

    api_key = st.text_input(
        "OpenAI API Key",
        value=st.session_state.api_key,
        type="password",
        help="Ihr OpenAI API Key. Wird nur lokal gespeichert.",
        key="api_key_input",
    )
    st.session_state.api_key = api_key

    if api_key:
        st.success("✓ API Key gesetzt")
    else:
        st.warning("⚠ Kein API Key")

    st.divider()
    st.markdown("### 📁 Urteilsordner")
    pdf_folder = st.text_input(
        "Pfad zum Urteilsordner",
        value=st.session_state.pdf_folder,
        placeholder="/pfad/zu/urteilen",
        help="Ordner mit österreichischen Zivilurteilen als PDF",
    )
    st.session_state.pdf_folder = pdf_folder

    if pdf_folder:
        folder = Path(pdf_folder)
        if folder.exists() and folder.is_dir():
            pdf_files = list(folder.glob("*.pdf")) + list(folder.glob("*.PDF"))
            processed = dm.get_processed_filenames()
            pending = [f for f in pdf_files if f.name not in processed]
            st.success(f"✓ {len(pdf_files)} PDFs gefunden")
            st.info(f"📋 {len(pending)} noch nicht verarbeitet")
        else:
            st.error("Ordner nicht gefunden")

    st.divider()
    st.markdown("### 📊 Dataset")
    stats = dm.get_statistics()
    st.metric("Gesamt Fälle", stats.get("total_cases", 0))
    st.metric("Beschriftet", stats.get("labeled_cases", 0))
    st.metric("Unvollständig", stats.get("unlabeled_cases", 0))

    st.divider()
    st.markdown("### 🔗 Navigation")
    if st.button("🔄 Daten neu laden", use_container_width=True):
        st.rerun()


# ─── Main Tabs ───────────────────────────────────────────────────────────────────

tab_extract, tab_dataset, tab_review, tab_manual = st.tabs([
    "📥 Extraktion",
    "📊 Dataset",
    "🔍 Fälle überprüfen",
    "✏️ Manuelle Eingabe",
])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1: EXTRACTION
# ════════════════════════════════════════════════════════════════════════════════

with tab_extract:
    st.markdown("## 📥 Automatische Datenextraktion aus PDF-Urteilen")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
        <div class="highlight-box">
        <b>So funktioniert die Extraktion:</b><br>
        <span class="step-badge">1</span> PDF-Text wird extrahiert (PyMuPDF)<br>
        <span class="step-badge">2</span> GPT-4o-mini extrahiert strukturierte Rechtsdaten<br>
        <span class="step-badge">3</span> text-embedding-3-large erstellt Embedding-Vektoren<br>
        <span class="step-badge">4</span> Daten werden im Dataset gespeichert
        </div>
        """, unsafe_allow_html=True)

    with col2:
        if st.session_state.pdf_folder and Path(st.session_state.pdf_folder).exists():
            folder = Path(st.session_state.pdf_folder)
            pdf_files = list(folder.glob("*.pdf")) + list(folder.glob("*.PDF"))
            processed_fnames = dm.get_processed_filenames()
            pending_files = [f for f in pdf_files if f.name not in processed_fnames]

            st.metric("PDFs bereit zur Verarbeitung", len(pending_files))
            st.metric("Bereits verarbeitet", len(processed_fnames))

    st.divider()

    # ── File Selection ───────────────────────────────────────────────────────────
    if st.session_state.pdf_folder:
        folder = Path(st.session_state.pdf_folder)
        if folder.exists():
            pdf_files = sorted(
                list(folder.glob("*.pdf")) + list(folder.glob("*.PDF"))
            )
            processed_fnames = dm.get_processed_filenames()
            pending_files = [f for f in pdf_files if f.name not in processed_fnames]

            if pending_files:
                st.markdown("### 📂 Ausstehende PDFs")
                process_mode = st.radio(
                    "Verarbeitungsmodus",
                    ["Alle ausstehenden PDFs", "Ausgewählte PDFs"],
                    horizontal=True,
                )

                if process_mode == "Ausgewählte PDFs":
                    selected_files = st.multiselect(
                        "PDFs auswählen",
                        [f.name for f in pending_files],
                        default=[f.name for f in pending_files[:3]],
                    )
                    files_to_process = [f for f in pending_files if f.name in selected_files]
                else:
                    files_to_process = pending_files

                if files_to_process:
                    # Processing controls
                    col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])

                    with col_btn1:
                        start_btn = st.button(
                            f"▶ Extraktion starten ({len(files_to_process)} Dateien)",
                            type="primary",
                            disabled=not st.session_state.api_key or st.session_state.processing,
                            use_container_width=True,
                        )

                    if not st.session_state.api_key:
                        st.warning("⚠️ Bitte OpenAI API Key in der Seitenleiste eingeben.")

                    # ── Processing Logic ─────────────────────────────────────────
                    if start_btn and not st.session_state.processing:
                        st.session_state.processing = True
                        st.session_state.process_log = []

                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        log_container = st.container()

                        results = {"success": 0, "error": 0, "errors": []}

                        def progress_cb(msg, pct):
                            pass  # Will be updated in loop

                        extractor = OpenAIExtractor(st.session_state.api_key)

                        for file_idx, pdf_path in enumerate(files_to_process):
                            file_pct_base = file_idx / len(files_to_process)
                            file_pct_step = 1.0 / len(files_to_process)

                            status_text.markdown(
                                f"**Verarbeite:** `{pdf_path.name}` "
                                f"({file_idx + 1}/{len(files_to_process)})"
                            )

                            try:
                                # Step 1: Extract PDF text
                                progress_bar.progress(
                                    file_pct_base + file_pct_step * 0.1
                                )
                                text = extract_text_from_pdf(pdf_path)

                                if len(text) < 200:
                                    raise ValueError(f"Text zu kurz ({len(text)} Zeichen) — möglicherweise gescanntes PDF")

                                # Step 2-4: OpenAI extraction + embeddings
                                progress_holder = st.empty()

                                def progress_cb(msg, pct):
                                    progress_bar.progress(
                                        file_pct_base + file_pct_step * (0.1 + pct * 0.85)
                                    )
                                    progress_holder.markdown(f"_⟳ {msg}_")

                                extractor.progress_callback = progress_cb
                                extracted = extractor.process_judgment(text)

                                # Step 5: Save to dataset
                                case_id = dm.generate_case_id()
                                dm.add_case(
                                    case_id=case_id,
                                    filename=pdf_path.name,
                                    structured=extracted["structured"],
                                    sections=extracted["sections"],
                                    embeddings=extracted["embeddings"],
                                )

                                progress_holder.empty()
                                results["success"] += 1
                                st.session_state.process_log.append({
                                    "file": pdf_path.name,
                                    "status": "success",
                                    "case_id": case_id,
                                    "outcome": extracted["structured"].get("outcome"),
                                    "streitwert": extracted["structured"].get("streitwert_eur"),
                                })

                                log_container.success(f"✅ **{pdf_path.name}** — Fall-ID: `{case_id}`")

                            except Exception as e:
                                results["error"] += 1
                                error_msg = str(e)
                                st.session_state.process_log.append({
                                    "file": pdf_path.name,
                                    "status": "error",
                                    "error": error_msg,
                                })
                                log_container.error(f"❌ **{pdf_path.name}** — {error_msg}")

                            progress_bar.progress(
                                (file_idx + 1) / len(files_to_process)
                            )

                        # Done
                        st.session_state.processing = False
                        status_text.empty()
                        progress_bar.empty()

                        st.markdown("---")
                        col_r1, col_r2 = st.columns(2)
                        col_r1.metric("✅ Erfolgreich", results["success"])
                        col_r2.metric("❌ Fehler", results["error"])

                        if results["success"] > 0:
                            st.success(
                                f"Extraktion abgeschlossen! {results['success']} Urteile"
                                " wurden dem Dataset hinzugefügt."
                            )
                            st.info(
                                "💡 **Nächste Schritte:** Wechseln Sie zum Tab "
                                "'Fälle überprüfen', um die extrahierten Daten "
                                "zu kontrollieren und ggf. zu korrigieren."
                            )

            else:
                st.success("✅ Alle PDFs in diesem Ordner wurden bereits verarbeitet!")
        else:
            st.info("Bitte geben Sie einen gültigen Ordnerpfad in der Seitenleiste ein.")
    else:
        st.info(
            "👆 Geben Sie in der Seitenleiste den Pfad zu Ihrem Urteilsordner ein, "
            "um die Extraktion zu starten."
        )

    # ── Previously Processed (log) ───────────────────────────────────────────────
    if st.session_state.process_log:
        st.divider()
        st.markdown("### 📋 Verarbeitungsprotokoll dieser Sitzung")
        log_df = pd.DataFrame(st.session_state.process_log)
        st.dataframe(log_df, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2: DATASET OVERVIEW
# ════════════════════════════════════════════════════════════════════════════════

with tab_dataset:
    st.markdown("## 📊 Dataset-Übersicht")

    cases = dm.load_dataset()
    stats = dm.get_statistics()

    if not cases:
        st.info("📭 Dataset ist leer. Starten Sie die Extraktion im Tab 'Extraktion'.")
    else:
        # ── Key Metrics ──────────────────────────────────────────────────────────
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📁 Gesamt Fälle", stats["total_cases"])
        col2.metric("✅ Beschriftet", stats["labeled_cases"])
        col3.metric(
            "⚖️ Obsiegen",
            stats["outcome_distribution"]["obsiegen"],
        )
        col4.metric(
            "❌ Unterliegen",
            stats["outcome_distribution"]["unterliegen"],
        )
        style_metric_cards()

        st.divider()

        col_chart1, col_chart2 = st.columns(2)

        # ── Outcome Distribution ──────────────────────────────────────────────────
        with col_chart1:
            st.markdown("#### Urteilsergebnisse")
            outcome_data = stats["outcome_distribution"]
            if sum(outcome_data.values()) > 0:
                fig = go.Figure(data=[go.Pie(
                    labels=["Unterliegen", "Teilweise", "Obsiegen"],
                    values=[
                        outcome_data["unterliegen"],
                        outcome_data["teilweise"],
                        outcome_data["obsiegen"],
                    ],
                    hole=0.4,
                    marker_colors=["#E74C3C", "#F39C12", "#27AE60"],
                    textinfo="label+percent",
                )])
                fig.update_layout(
                    showlegend=False,
                    height=300,
                    margin=dict(t=10, b=10, l=10, r=10),
                )
                st.plotly_chart(fig, use_container_width=True)

        # ── Claim Type Distribution ───────────────────────────────────────────────
        with col_chart2:
            st.markdown("#### Anspruchsarten")
            ct_data = stats.get("claim_type_distribution", {})
            if ct_data:
                ct_df = pd.DataFrame(
                    list(ct_data.items()), columns=["Anspruchsart", "Anzahl"]
                ).sort_values("Anzahl", ascending=True)
                fig2 = px.bar(
                    ct_df,
                    x="Anzahl",
                    y="Anspruchsart",
                    orientation="h",
                    color="Anzahl",
                    color_continuous_scale="Blues",
                )
                fig2.update_layout(
                    height=300,
                    margin=dict(t=10, b=10, l=10, r=10),
                    showlegend=False,
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig2, use_container_width=True)

        # ── Streitwert Distribution ───────────────────────────────────────────────
        streitwerte = [
            c["structured"].get("streitwert_eur")
            for c in cases
            if c["structured"].get("streitwert_eur") is not None
        ]

        if streitwerte:
            st.markdown("#### Streitwert-Verteilung")
            sw_stats = stats["streitwert_stats"]
            sw_col1, sw_col2, sw_col3, sw_col4 = st.columns(4)
            sw_col1.metric("Min", f"€ {sw_stats['min']:,.0f}" if sw_stats["min"] else "—")
            sw_col2.metric("Max", f"€ {sw_stats['max']:,.0f}" if sw_stats["max"] else "—")
            sw_col3.metric("Mittel", f"€ {sw_stats['mean']:,.0f}" if sw_stats["mean"] else "—")
            sw_col4.metric("Median", f"€ {sw_stats['median']:,.0f}" if sw_stats["median"] else "—")

            sw_df = pd.DataFrame({"Streitwert (€)": streitwerte})
            fig3 = px.histogram(
                sw_df, x="Streitwert (€)",
                nbins=20,
                color_discrete_sequence=["#2d6a9f"],
            )
            fig3.update_layout(height=250, margin=dict(t=10, b=30, l=10, r=10))
            st.plotly_chart(fig3, use_container_width=True)

        st.divider()

        # ── Cases Table ──────────────────────────────────────────────────────────
        st.markdown("#### 📋 Alle Fälle")

        table_data = []
        for c in cases:
            s = c["structured"]
            outcome = s.get("outcome")
            outcome_label = OUTCOME_LABELS.get(outcome, "—") if outcome is not None else "⚠️ Nicht beschriftet"
            table_data.append({
                "Fall-ID": c["case_id"],
                "Datei": c["filename"],
                "Datum": s.get("datum", "—"),
                "Gericht": s.get("instanz", "—"),
                "Anspruchsart": s.get("anspruchsart", "—"),
                "Streitwert €": s.get("streitwert_eur"),
                "Ergebnis": outcome_label,
                "Verarbeitet": c.get("processed_at", "")[:10],
            })

        if table_data:
            df = pd.DataFrame(table_data)
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Streitwert €": st.column_config.NumberColumn(
                        "Streitwert €", format="€ %.0f"
                    ),
                },
            )

        # ── Export ───────────────────────────────────────────────────────────────
        st.divider()
        st.markdown("#### 📤 Dataset exportieren")
        col_exp1, col_exp2 = st.columns(2)

        with col_exp1:
            if st.button("⬇ JSON herunterladen", use_container_width=True):
                json_str = json.dumps(
                    [
                        {k: v for k, v in c.items() if k != "embeddings"}
                        for c in cases
                    ],
                    ensure_ascii=False,
                    indent=2,
                )
                st.download_button(
                    "📥 Download JSON",
                    data=json_str.encode("utf-8"),
                    file_name="litigation_dataset.json",
                    mime="application/json",
                )

        with col_exp2:
            if st.button("⬇ CSV herunterladen", use_container_width=True):
                csv_data = pd.DataFrame(table_data).to_csv(index=False)
                st.download_button(
                    "📥 Download CSV",
                    data=csv_data.encode("utf-8"),
                    file_name="litigation_dataset.csv",
                    mime="text/csv",
                )


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3: CASE REVIEW
# ════════════════════════════════════════════════════════════════════════════════

with tab_review:
    st.markdown("## 🔍 Fälle überprüfen und bearbeiten")
    st.markdown(
        "Kontrollieren Sie die extrahierten Daten und korrigieren Sie Fehler "
        "der automatischen Extraktion."
    )

    cases = dm.load_dataset()

    if not cases:
        st.info("Keine Fälle im Dataset.")
    else:
        # Case selector
        case_options = {
            f"{c['filename']} [{c['case_id']}]": c["case_id"]
            for c in cases
        }
        selected_label = st.selectbox(
            "Fall auswählen",
            list(case_options.keys()),
        )
        selected_case_id = case_options[selected_label]
        case = dm.get_case(selected_case_id)

        if case:
            st.markdown("---")
            s = case["structured"]

            # ── Case Header ──────────────────────────────────────────────────────
            col_info1, col_info2, col_info3 = st.columns([2, 1, 1])

            with col_info1:
                st.markdown(f"**Datei:** `{case['filename']}`")
                st.markdown(f"**Fall-ID:** `{case['case_id']}`")
                st.markdown(f"**Verarbeitet:** {case.get('processed_at', '')[:10]}")

            with col_info2:
                outcome = s.get("outcome")
                if outcome is not None:
                    color = OUTCOME_COLORS[outcome]
                    label = OUTCOME_LABELS[outcome]
                    st.markdown(
                        f"**Ergebnis:**<br>"
                        f"<span style='color:{color};font-size:1.1em;font-weight:bold'>"
                        f"{label}</span>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.warning("⚠️ Ergebnis nicht extrahiert")

            with col_info3:
                sw = s.get("streitwert_eur")
                if sw:
                    st.metric("Streitwert", f"€ {sw:,.2f}")
                date = s.get("datum")
                if date:
                    st.markdown(f"**Datum:** {date}")

            st.markdown("---")

            # ── Edit Form ────────────────────────────────────────────────────────
            with st.expander("✏️ Daten bearbeiten / korrigieren", expanded=False):
                with st.form(f"edit_form_{selected_case_id}"):
                    st.markdown("**Grunddaten**")

                    edit_col1, edit_col2, edit_col3 = st.columns(3)

                    with edit_col1:
                        new_datum = st.text_input(
                            "Datum (YYYY-MM-DD)",
                            value=s.get("datum", "") or "",
                        )
                        new_gericht = st.text_input(
                            "Gericht",
                            value=s.get("gericht", "") or "",
                        )
                        new_instanz = st.selectbox(
                            "Instanz",
                            ["", "BG", "LG", "OLG", "OGH"],
                            index=["", "BG", "LG", "OLG", "OGH"].index(
                                s.get("instanz", "") or ""
                            ) if s.get("instanz", "") in ["", "BG", "LG", "OLG", "OGH"] else 0,
                        )

                    with edit_col2:
                        new_streitwert = st.number_input(
                            "Streitwert (€)",
                            min_value=0.0,
                            value=float(s.get("streitwert_eur") or 0.0),
                            step=100.0,
                        )
                        new_claim_type = st.selectbox(
                            "Anspruchsart",
                            CLAIM_TYPES + ["Andere"],
                            index=(
                                CLAIM_TYPES.index(s.get("anspruchsart", "Andere"))
                                if s.get("anspruchsart") in CLAIM_TYPES
                                else len(CLAIM_TYPES)
                            ),
                        )

                    with edit_col3:
                        new_outcome = st.selectbox(
                            "Urteilsergebnis (Outcome) *",
                            [0, 1, 2],
                            format_func=lambda x: f"{x} – {OUTCOME_LABELS[x]}",
                            index=int(s.get("outcome") or 0),
                        )
                        new_zugesprochener_anteil = st.slider(
                            "Zugesprochener Anteil (%)",
                            0, 100,
                            value=int(s.get("zugesprochener_anteil_prozent") or 0),
                        )

                    st.markdown("**Einwendungen des Beklagten**")
                    einwendungen = s.get("einwendungen", {})
                    new_einwendungen = {}
                    ew_cols = st.columns(4)
                    for i, defense in enumerate(DEFENSE_TYPES):
                        with ew_cols[i % 4]:
                            new_einwendungen[defense] = st.checkbox(
                                DEFENSE_LABELS.get(defense, defense),
                                value=einwendungen.get(defense, False),
                                key=f"ew_{selected_case_id}_{defense}",
                            )

                    if st.form_submit_button("💾 Änderungen speichern", type="primary"):
                        updates = {
                            "structured": {
                                "datum": new_datum or None,
                                "gericht": new_gericht or None,
                                "instanz": new_instanz or None,
                                "streitwert_eur": new_streitwert if new_streitwert > 0 else None,
                                "anspruchsart": new_claim_type,
                                "outcome": new_outcome,
                                "zugesprochener_anteil_prozent": new_zugesprochener_anteil,
                                "einwendungen": new_einwendungen,
                            }
                        }
                        if dm.update_case(selected_case_id, updates):
                            st.success("✅ Änderungen gespeichert!")
                            st.rerun()

            # ── Display Structured Data ──────────────────────────────────────────
            st.markdown("### 📋 Extrahierte Daten")

            col_d1, col_d2 = st.columns(2)

            with col_d1:
                st.markdown("**Anspruchsgrundlagen:**")
                for ag in s.get("anspruchsgruende", []):
                    st.markdown(f"  • {ag}")

                st.markdown("**Kläger-Vorbringen (Zusammenfassung):**")
                st.markdown(
                    f"> {s.get('klaeger_anspruch_zusammenfassung', '—')}"
                )

                st.markdown("**Kläger-Beweismittel:**")
                for bm in s.get("klaeger_beweismittel", []):
                    st.markdown(f"  • {bm}")

            with col_d2:
                st.markdown("**Aktive Einwendungen:**")
                einwendungen = s.get("einwendungen", {})
                active_defenses = [
                    DEFENSE_LABELS.get(d, d)
                    for d in DEFENSE_TYPES
                    if einwendungen.get(d)
                ]
                if active_defenses:
                    for ad in active_defenses:
                        st.markdown(
                            f"  🔸 {ad}",
                        )
                else:
                    st.markdown("  _Keine spezifischen Einwendungen_")

                st.markdown("**Beklagten-Beweismittel:**")
                for bm in s.get("beklagter_beweismittel", []):
                    st.markdown(f"  • {bm}")

                if s.get("sachverstaendiger_bestellt"):
                    fachgebiet = s.get("sachverstaendigen_fachgebiet", "")
                    st.info(f"🔬 Sachverständiger: {fachgebiet or 'Ja'}")

            # ── Extracted Sections ───────────────────────────────────────────────
            st.markdown("### 📄 Extrahierte Textabschnitte")
            sections = case.get("sections", {})

            for sec_key, sec_label in EMBEDDING_SECTION_LABELS.items():
                text = sections.get(sec_key, "")
                if text:
                    with st.expander(f"📖 {sec_label} ({len(text)} Zeichen)"):
                        st.text_area(
                            sec_label,
                            value=text,
                            height=200,
                            key=f"sec_{selected_case_id}_{sec_key}",
                            disabled=True,
                        )

            # ── Delete ───────────────────────────────────────────────────────────
            st.divider()
            with st.expander("⚠️ Gefahrenzone"):
                if st.button(
                    "🗑 Diesen Fall löschen",
                    type="secondary",
                    key=f"delete_{selected_case_id}",
                ):
                    confirm = st.checkbox(
                        "Ich bestätige die Löschung dieses Falls",
                        key=f"confirm_delete_{selected_case_id}",
                    )
                    if confirm:
                        dm.delete_case(selected_case_id)
                        st.success("Fall gelöscht.")
                        st.rerun()


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4: MANUAL ENTRY
# ════════════════════════════════════════════════════════════════════════════════

with tab_manual:
    st.markdown("## ✏️ Manuelle Fallerfassung")
    st.markdown(
        "Fügen Sie Fälle manuell hinzu, falls kein PDF vorliegt oder "
        "die automatische Extraktion unvollständig war."
    )

    if not st.session_state.api_key:
        st.warning("⚠️ OpenAI API Key erforderlich für Embedding-Generierung.")

    with st.form("manual_entry_form"):
        st.markdown("### 📋 Grunddaten")
        m_col1, m_col2, m_col3 = st.columns(3)

        with m_col1:
            m_datum = st.text_input("Datum (YYYY-MM-DD)")
            m_instanz = st.selectbox("Instanz", ["BG", "LG", "OLG", "OGH"])
            m_streitwert = st.number_input("Streitwert (€)", min_value=0.0, step=100.0)

        with m_col2:
            m_claim_type = st.selectbox("Anspruchsart", CLAIM_TYPES + ["Andere"])
            m_anspruchsgruende = st.text_area(
                "Anspruchsgrundlagen (eine pro Zeile)",
                placeholder="§ 1295 ABGB\n§ 922 ABGB",
                height=100,
            )
            m_outcome = st.selectbox(
                "Urteilsergebnis *",
                [0, 1, 2],
                format_func=lambda x: f"{x} – {OUTCOME_LABELS[x]}",
            )

        with m_col3:
            m_zugesprochener_anteil = st.slider("Zugesprochener Anteil (%)", 0, 100, 0)
            m_kostenentscheidung = st.selectbox(
                "Kostenentscheidung",
                ["", "Kläger", "Beklagter", "Geteilt"],
            )
            m_sv_bestellt = st.checkbox("Sachverständiger bestellt")
            if m_sv_bestellt:
                m_sv_fachgebiet = st.text_input("Sachverständigen-Fachgebiet")
            else:
                m_sv_fachgebiet = None

        st.markdown("### 📝 Textvorbringen")
        t_col1, t_col2 = st.columns(2)

        with t_col1:
            m_klaeger_vorbringen = st.text_area(
                "Kläger-Vorbringen",
                placeholder="Beschreiben Sie das Vorbringen des Klägers...",
                height=150,
            )
            m_feststellungen = st.text_area(
                "Feststellungen",
                placeholder="Sachverhaltsfeststellungen des Gerichts...",
                height=150,
            )
            m_rechtliche_beurteilung = st.text_area(
                "Rechtliche Beurteilung",
                placeholder="Rechtliche Beurteilung des Gerichts...",
                height=150,
            )

        with t_col2:
            m_beklagter_vorbringen = st.text_area(
                "Beklagten-Vorbringen",
                placeholder="Beschreiben Sie das Vorbringen des Beklagten...",
                height=150,
            )
            m_beweisw = st.text_area(
                "Beweiswürdigung",
                placeholder="Beweiswürdigung des Gerichts...",
                height=150,
            )

        st.markdown("### ⚖️ Einwendungen")
        ew_cols = st.columns(4)
        m_einwendungen = {}
        for i, defense in enumerate(DEFENSE_TYPES):
            with ew_cols[i % 4]:
                m_einwendungen[defense] = st.checkbox(
                    DEFENSE_LABELS.get(defense, defense),
                    key=f"manual_ew_{defense}",
                )

        st.markdown("### 🔎 Beweismittel")
        bm_col1, bm_col2 = st.columns(2)
        with bm_col1:
            m_klaeger_beweismittel = st.text_area(
                "Kläger-Beweismittel (eine pro Zeile)",
                placeholder="Urkunden\nZeugenaussage\nSachverständigengutachten",
                height=80,
            )
        with bm_col2:
            m_beklagter_beweismittel = st.text_area(
                "Beklagten-Beweismittel (eine pro Zeile)",
                placeholder="Gegenbeweise...",
                height=80,
            )

        submitted = st.form_submit_button(
            "💾 Fall speichern und Embeddings generieren",
            type="primary",
        )

    if submitted:
        if not m_klaeger_vorbringen and not m_beklagter_vorbringen:
            st.error("Bitte mindestens Kläger- oder Beklagten-Vorbringen eingeben.")
        else:
            sections = {
                "klaegervorbringen": m_klaeger_vorbringen,
                "beklagtenvorbringen": m_beklagter_vorbringen,
                "feststellungen": m_feststellungen,
                "beweisw_rdigung": m_beweisw,
                "rechtliche_beurteilung": m_rechtliche_beurteilung,
            }

            structured = {
                "datum": m_datum or None,
                "instanz": m_instanz,
                "gericht": None,
                "streitwert_eur": m_streitwert if m_streitwert > 0 else None,
                "streitwert_unbekannt": m_streitwert == 0,
                "anspruchsart": m_claim_type,
                "anspruchsgruende": [
                    x.strip() for x in m_anspruchsgruende.splitlines() if x.strip()
                ],
                "klaeger_anspruch_zusammenfassung": m_klaeger_vorbringen[:200],
                "beklagter_vorbringen_zusammenfassung": m_beklagter_vorbringen[:200],
                "einwendungen": m_einwendungen,
                "klaeger_beweismittel": [
                    x.strip() for x in m_klaeger_beweismittel.splitlines() if x.strip()
                ],
                "beklagter_beweismittel": [
                    x.strip() for x in m_beklagter_beweismittel.splitlines() if x.strip()
                ],
                "sachverstaendiger_bestellt": m_sv_bestellt,
                "sachverstaendigen_fachgebiet": m_sv_fachgebiet,
                "outcome": m_outcome,
                "zugesprochener_anteil_prozent": m_zugesprochener_anteil,
                "kostenentscheidung": m_kostenentscheidung or None,
                "besonderheiten": [],
            }

            with st.spinner("Generiere Embeddings..."):
                try:
                    extractor = OpenAIExtractor(st.session_state.api_key)
                    embeddings = extractor.generate_embeddings(sections)
                except Exception as e:
                    st.error(f"Fehler bei Embedding-Generierung: {e}")
                    embeddings = {
                        sec: [0.0] * EMBEDDING_DIM
                        for sec in sections
                    }

            case_id = dm.generate_case_id()
            filename = f"manuell_{case_id}.manual"

            dm.add_case(
                case_id=case_id,
                filename=filename,
                structured=structured,
                sections=sections,
                embeddings=embeddings,
            )

            st.success(
                f"✅ Fall `{case_id}` erfolgreich gespeichert! "
                "Wechseln Sie zum Tab 'Dataset' für eine Übersicht."
            )


# ─── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f"<div style='text-align:center;color:#6c757d;font-size:0.8rem'>"
    f"Predictive Litigation Analytics · v{APP_VERSION} · "
    f"Modell: {OPENAI_EXTRACTION_MODEL} · Embeddings: text-embedding-3-large"
    f"</div>",
    unsafe_allow_html=True,
)
