"""
Litigation Data Extractor — Streamlit UI

Strukturierte Erfassung österreichischer OGH-Zivilurteile aus TXT-Dateien.
Extrahiert werden: Erstgericht-Vorbringen der Parteien + Erstgericht-Entscheidung.
OGH-Strafurteile werden automatisch herausgefiltert (nur Zivilsenat).
"""

import json
import os
import sys
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
    OPENAI_EXTRACTION_MODEL,
    OUTCOME_COLORS,
    OUTCOME_LABELS,
    PARALLEL_WORKERS,
)
from concurrent.futures import ThreadPoolExecutor, as_completed

from data_extractor.data_manager import DataManager
from data_extractor.openai_extractor import OpenAIExtractor
from data_extractor.txt_processor import StrafurteilError, TxtProcessingError, read_txt_file

# ─── Page Config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Litigation Data Extractor",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Scientific / instrument-panel style */
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

    html, body, [class*="css"] {
        font-family: var(--font-sans);
    }

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

    /* ── Section headings ── */
    .section-title {
        font-size: 1.0rem;
        font-weight: 600;
        color: var(--col-primary);
        border-bottom: 1px solid var(--col-border);
        padding-bottom: 0.3rem;
        margin-bottom: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-family: var(--font-sans);
    }

    /* ── Pipeline description box ── */
    .pipeline-box {
        background: var(--col-surface);
        border: 1px solid var(--col-border);
        padding: 0.9rem 1.1rem;
        font-size: 0.85rem;
        line-height: 1.7;
        font-family: var(--font-mono);
        color: var(--col-text);
    }
    .pipeline-box .step {
        display: inline-block;
        background: var(--col-primary);
        color: white;
        width: 20px;
        height: 20px;
        text-align: center;
        line-height: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.5rem;
        vertical-align: middle;
    }

    /* ── PDF item list ── */
    .pdf-item {
        padding: 0.5rem 0.8rem;
        margin: 0.25rem 0;
        font-size: 0.83rem;
        font-family: var(--font-mono);
        border-left: 3px solid var(--col-border);
        background: var(--col-surface);
    }
    .pdf-item.processed { border-left-color: var(--col-success); }
    .pdf-item.error     { border-left-color: var(--col-danger); }
    .pdf-item.pending   { border-left-color: var(--col-muted); }

    /* ── Sidebar ── */
    .sidebar-section {
        background: var(--col-surface);
        border: 1px solid var(--col-border);
        padding: 0.7rem;
        margin: 0.4rem 0;
        font-size: 0.85rem;
    }

    /* ── Data / code elements ── */
    code, .mono {
        font-family: var(--font-mono);
        font-size: 0.82rem;
        background: var(--col-surface);
        padding: 0.1rem 0.3rem;
        border: 1px solid var(--col-border);
    }

    /* ── Defense tags ── */
    .defense-tag {
        display: inline-block;
        border: 1px solid var(--col-border);
        color: var(--col-muted);
        padding: 0.1rem 0.45rem;
        font-size: 0.75rem;
        font-family: var(--font-mono);
        margin: 0.1rem;
        background: var(--col-surface);
    }
    .defense-tag.active {
        border-color: var(--col-primary);
        color: var(--col-primary);
        background: #e8f0fa;
        font-weight: 600;
    }

    /* ── Outcome label ── */
    .outcome-label {
        display: inline-block;
        font-family: var(--font-mono);
        font-size: 0.82rem;
        font-weight: 600;
        padding: 0.2rem 0.6rem;
        border: 1px solid currentColor;
    }
    .outcome-0 { color: var(--col-danger);  background: #fdf5f5; }
    .outcome-1 { color: var(--col-warning); background: #fdf8ed; }
    .outcome-2 { color: var(--col-success); background: #f3f9f5; }

    /* ── Table ── */
    .dataframe { font-size: 0.82rem; font-family: var(--font-mono); }

    /* ── Status/note box ── */
    .note-box {
        border: 1px solid var(--col-border);
        border-left: 3px solid var(--col-accent);
        background: var(--col-surface);
        padding: 0.7rem 1rem;
        font-size: 0.85rem;
        margin: 0.6rem 0;
        color: var(--col-text);
    }

    /* ── Streamlit element overrides ── */
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


# ─── Session State Initialization ────────────────────────────────────────────────

def init_session_state():
    defaults = {
        "api_key": os.environ.get("OPENAI_API_KEY", ""),
        "txt_folder": "",
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
    <h1>Litigation Data Extractor &mdash; OGH-Urteile (ö. Zivilrecht)</h1>
    <div class="subtitle">Strukturierte Erfassung aus OGH-Urteilen (TXT) &mdash; Erstgericht-Vorbringen &amp; Ersturteil &mdash; v{APP_VERSION} &mdash; {OPENAI_EXTRACTION_MODEL}</div>
</div>
""", unsafe_allow_html=True)


# ─── Sidebar ─────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("**Konfiguration**")

    api_key = st.text_input(
        "OpenAI API Key",
        value=st.session_state.api_key,
        type="password",
        help="Ihr OpenAI API Key. Wird nur lokal gespeichert.",
        key="api_key_input",
    )
    st.session_state.api_key = api_key

    if api_key:
        st.success("API Key gesetzt")
    else:
        st.warning("Kein API Key")

    st.divider()
    st.markdown("**Urteilsordner**")
    txt_folder = st.text_input(
        "Pfad zum OGH-Urteilsordner",
        value=st.session_state.txt_folder,
        placeholder="/pfad/zu/ogh-urteilen",
        help="Ordner mit OGH-Zivilurteilen als TXT-Dateien. OGH-Strafurteile werden automatisch herausgefiltert.",
    )
    st.session_state.txt_folder = txt_folder

    if txt_folder:
        folder = Path(txt_folder)
        if folder.exists() and folder.is_dir():
            txt_files = list(folder.glob("*.txt")) + list(folder.glob("*.TXT"))
            processed = dm.get_processed_filenames()
            pending = [f for f in txt_files if f.name not in processed]
            st.success(f"{len(txt_files)} TXT-Dateien gefunden")
            st.info(f"{len(pending)} noch nicht verarbeitet")
        else:
            st.error("Ordner nicht gefunden")

    st.divider()
    st.markdown("**Dataset**")
    stats = dm.get_statistics()
    st.metric("Gesamt Fälle", stats.get("total_cases", 0))
    st.metric("Beschriftet", stats.get("labeled_cases", 0))
    st.metric("Unvollständig", stats.get("unlabeled_cases", 0))

    st.divider()
    st.markdown("**Navigation**")
    if st.button("Daten neu laden", use_container_width=True):
        st.rerun()


# ─── Main Tabs ───────────────────────────────────────────────────────────────────

tab_extract, tab_dataset, tab_review, tab_manual = st.tabs([
    "Extraktion",
    "Dataset",
    "Fälle überprüfen",
    "Manuelle Eingabe",
])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1: EXTRACTION
# ════════════════════════════════════════════════════════════════════════════════

with tab_extract:
    st.markdown('<div class="section-title">Automatische Datenextraktion aus OGH-Zivilurteilen (TXT)</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
        <div class="pipeline-box">
            <b>Verarbeitungspipeline (Österr. Zivilrecht / OGH):</b><br>
            <span class="step">1</span> TXT-Datei lesen &amp; OGH-Strafurteil-Erkennung (Keyword-Filter)<br>
            <span class="step">2</span> Extraktion: Kläger-/Beklagten-Vorbringen + Ersturteil (GPT-4o-mini, ABGB/ZPO)<br>
            <span class="step">3</span> Embedding-Vektoren (text-embedding-3-large, 3072 dim)<br>
            <span class="step">4</span> Persistierung im JSON-Dataset (Trainings-Label = Erstgericht-Outcome)
        </div>
        """, unsafe_allow_html=True)

    with col2:
        if st.session_state.txt_folder and Path(st.session_state.txt_folder).exists():
            folder = Path(st.session_state.txt_folder)
            txt_files = list(folder.glob("*.txt")) + list(folder.glob("*.TXT"))
            processed_fnames = dm.get_processed_filenames()
            pending_files = [f for f in txt_files if f.name not in processed_fnames]

            st.metric("TXT-Dateien bereit", len(pending_files))
            st.metric("Bereits verarbeitet", len(processed_fnames))

    st.divider()

    # ── File Selection ───────────────────────────────────────────────────────────
    if st.session_state.txt_folder:
        folder = Path(st.session_state.txt_folder)
        if folder.exists():
            txt_files = sorted(
                list(folder.glob("*.txt")) + list(folder.glob("*.TXT"))
            )
            processed_fnames = dm.get_processed_filenames()
            pending_files = [f for f in txt_files if f.name not in processed_fnames]

            if pending_files:
                st.markdown("**Ausstehende TXT-Dateien**")
                process_mode = st.radio(
                    "Verarbeitungsmodus",
                    ["Alle ausstehenden TXT-Dateien", "Ausgewählte TXT-Dateien"],
                    horizontal=True,
                )

                if process_mode == "Ausgewählte TXT-Dateien":
                    selected_files = st.multiselect(
                        "TXT-Dateien auswählen",
                        [f.name for f in pending_files],
                        default=[f.name for f in pending_files[:3]],
                    )
                    files_to_process = [f for f in pending_files if f.name in selected_files]
                else:
                    files_to_process = pending_files

                if files_to_process:
                    col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])

                    with col_btn1:
                        start_btn = st.button(
                            f"Extraktion starten  ({len(files_to_process)} Dateien)",
                            type="primary",
                            disabled=not st.session_state.api_key or st.session_state.processing,
                            use_container_width=True,
                        )
                    with col_btn2:
                        n_workers = st.number_input(
                            "Parallele Workers",
                            min_value=1,
                            max_value=20,
                            value=PARALLEL_WORKERS,
                            step=1,
                            help="Anzahl gleichzeitig verarbeiteter OGH-Urteile. "
                                 "Empfehlung: 5 bei Tier-1 OpenAI-Account.",
                        )

                    if not st.session_state.api_key:
                        st.warning("Bitte OpenAI API Key in der Seitenleiste eingeben.")

                    # ── Processing Logic (parallel) ──────────────────────────────
                    if start_btn and not st.session_state.processing:
                        st.session_state.processing = True
                        st.session_state.process_log = []

                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        log_container = st.container()

                        api_key = st.session_state.api_key
                        n_files = len(files_to_process)

                        # ── Worker function (runs in thread pool) ────────────────
                        def _extract_worker(txt_path: Path) -> dict:
                            """
                            Verarbeitet ein OGH-Urteil komplett (TXT lesen →
                            GPT-Extraktion → Embeddings parallel).
                            Jeder Worker bekommt seinen eigenen OpenAI-Client.
                            """
                            try:
                                text = read_txt_file(txt_path)
                                extractor = OpenAIExtractor(api_key)
                                extracted = extractor.process_judgment(text)
                                return {
                                    "status": "success",
                                    "file": txt_path.name,
                                    "extracted": extracted,
                                }
                            except StrafurteilError as e:
                                return {
                                    "status": "gefiltert",
                                    "file": txt_path.name,
                                    "error": str(e),
                                }
                            except Exception as e:
                                return {
                                    "status": "error",
                                    "file": txt_path.name,
                                    "error": str(e),
                                }

                        # ── Phase 1: Parallel extraction ─────────────────────────
                        status_text.markdown(
                            f"**Extrahiere parallel** ({n_workers} Workers) ..."
                        )
                        raw_results: list[dict] = []

                        with ThreadPoolExecutor(max_workers=n_workers) as pool:
                            futures = {
                                pool.submit(_extract_worker, f): f
                                for f in files_to_process
                            }
                            for completed_idx, future in enumerate(
                                as_completed(futures), start=1
                            ):
                                res = future.result()
                                raw_results.append(res)
                                progress_bar.progress(completed_idx / n_files * 0.9)

                                if res["status"] == "success":
                                    log_container.success(f"OK  {res['file']}")
                                elif res["status"] == "gefiltert":
                                    log_container.warning(
                                        f"OGH-STRAFURTEIL gefiltert  {res['file']}"
                                    )
                                else:
                                    log_container.error(
                                        f"FEHLER  {res['file']}  →  {res['error']}"
                                    )

                        # ── Phase 2: Sequential save to DataManager ───────────────
                        status_text.markdown("**Speichere in Dataset...**")
                        results = {"success": 0, "error": 0, "gefiltert": 0}

                        for res in raw_results:
                            if res["status"] == "success":
                                extracted = res["extracted"]
                                case_id = dm.generate_case_id()
                                dm.add_case(
                                    case_id=case_id,
                                    filename=res["file"],
                                    structured=extracted["structured"],
                                    sections=extracted["sections"],
                                    embeddings=extracted["embeddings"],
                                )
                                results["success"] += 1
                                st.session_state.process_log.append({
                                    "file": res["file"],
                                    "status": "success",
                                    "case_id": case_id,
                                    "outcome": extracted["structured"].get("outcome"),
                                    "streitwert": extracted["structured"].get("streitwert_eur"),
                                })
                            elif res["status"] == "gefiltert":
                                results["gefiltert"] += 1
                                st.session_state.process_log.append({
                                    "file": res["file"],
                                    "status": "gefiltert (OGH-Strafurteil)",
                                    "error": res.get("error", ""),
                                })
                            else:
                                results["error"] += 1
                                st.session_state.process_log.append({
                                    "file": res["file"],
                                    "status": "error",
                                    "error": res.get("error", ""),
                                })

                        # ── Done ─────────────────────────────────────────────────
                        st.session_state.processing = False
                        status_text.empty()
                        progress_bar.progress(1.0)

                        st.markdown("---")
                        col_r1, col_r2, col_r3 = st.columns(3)
                        col_r1.metric("Erfolgreich", results["success"])
                        col_r2.metric("OGH-Strafurteile gefiltert", results["gefiltert"])
                        col_r3.metric("Fehler", results["error"])

                        if results["success"] > 0:
                            st.success(
                                f"Extraktion abgeschlossen. {results['success']} Zivilurteile"
                                " wurden dem Dataset hinzugefügt."
                            )
                            st.markdown(
                                '<div class="note-box">Wechseln Sie zum Tab '
                                '"Fälle überprüfen", um die extrahierten Daten '
                                'zu kontrollieren und ggf. zu korrigieren.</div>',
                                unsafe_allow_html=True,
                            )

            else:
                st.success("Alle TXT-Dateien in diesem Ordner wurden bereits verarbeitet.")
        else:
            st.info("Bitte geben Sie einen gültigen Ordnerpfad in der Seitenleiste ein.")
    else:
        st.info(
            "Geben Sie in der Seitenleiste den Pfad zu Ihrem Urteilsordner ein, "
            "um die Extraktion zu starten."
        )

    # ── Previously Processed (log) ───────────────────────────────────────────────
    if st.session_state.process_log:
        st.divider()
        st.markdown("**Verarbeitungsprotokoll dieser Sitzung**")
        log_df = pd.DataFrame(st.session_state.process_log)
        st.dataframe(log_df, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2: DATASET OVERVIEW
# ════════════════════════════════════════════════════════════════════════════════

with tab_dataset:
    st.markdown('<div class="section-title">Dataset-Übersicht</div>', unsafe_allow_html=True)

    cases = dm.load_dataset()
    stats = dm.get_statistics()

    if not cases:
        st.info("Dataset ist leer. Starten Sie die Extraktion im Tab 'Extraktion'.")
    else:
        # ── Key Metrics ──────────────────────────────────────────────────────────
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Gesamt Fälle", stats["total_cases"])
        col2.metric("Beschriftet", stats["labeled_cases"])
        col3.metric("Obsiegen", stats["outcome_distribution"]["obsiegen"])
        col4.metric("Unterliegen", stats["outcome_distribution"]["unterliegen"])
        style_metric_cards()

        st.divider()

        col_chart1, col_chart2 = st.columns(2)

        # ── Outcome Distribution ──────────────────────────────────────────────────
        with col_chart1:
            st.markdown("**Urteilsergebnisse**")
            outcome_data = stats["outcome_distribution"]
            if sum(outcome_data.values()) > 0:
                fig = go.Figure(data=[go.Pie(
                    labels=["Unterliegen", "Teilweise", "Obsiegen"],
                    values=[
                        outcome_data["unterliegen"],
                        outcome_data["teilweise"],
                        outcome_data["obsiegen"],
                    ],
                    hole=0.35,
                    marker_colors=["#8b1a1a", "#7d5a00", "#2c6e49"],
                    textinfo="label+percent",
                    textfont=dict(family="IBM Plex Mono, monospace", size=11),
                )])
                fig.update_layout(
                    showlegend=False,
                    height=280,
                    margin=dict(t=10, b=10, l=10, r=10),
                    paper_bgcolor="white",
                    plot_bgcolor="white",
                    font=dict(family="IBM Plex Sans, sans-serif"),
                )
                st.plotly_chart(fig, use_container_width=True)

        # ── Claim Type Distribution ───────────────────────────────────────────────
        with col_chart2:
            st.markdown("**Anspruchsarten**")
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
                    color_discrete_sequence=["#1c3a5e"],
                )
                fig2.update_layout(
                    height=280,
                    margin=dict(t=10, b=10, l=10, r=10),
                    showlegend=False,
                    paper_bgcolor="white",
                    plot_bgcolor="#f5f5f5",
                    font=dict(family="IBM Plex Sans, sans-serif", size=11),
                )
                fig2.update_xaxes(showgrid=True, gridcolor="#dddddd", gridwidth=1)
                fig2.update_yaxes(showgrid=False)
                st.plotly_chart(fig2, use_container_width=True)

        # ── Streitwert Distribution ───────────────────────────────────────────────
        streitwerte = [
            c["structured"].get("streitwert_eur")
            for c in cases
            if c["structured"].get("streitwert_eur") is not None
        ]

        if streitwerte:
            st.markdown("**Streitwert-Verteilung**")
            sw_stats = stats["streitwert_stats"]
            sw_col1, sw_col2, sw_col3, sw_col4 = st.columns(4)
            sw_col1.metric("Min", f"EUR {sw_stats['min']:,.0f}" if sw_stats["min"] else "—")
            sw_col2.metric("Max", f"EUR {sw_stats['max']:,.0f}" if sw_stats["max"] else "—")
            sw_col3.metric("Mittelwert", f"EUR {sw_stats['mean']:,.0f}" if sw_stats["mean"] else "—")
            sw_col4.metric("Median", f"EUR {sw_stats['median']:,.0f}" if sw_stats["median"] else "—")

            sw_df = pd.DataFrame({"Streitwert (EUR)": streitwerte})
            fig3 = px.histogram(
                sw_df, x="Streitwert (EUR)",
                nbins=20,
                color_discrete_sequence=["#2a6496"],
            )
            fig3.update_layout(
                height=220,
                margin=dict(t=10, b=30, l=10, r=10),
                paper_bgcolor="white",
                plot_bgcolor="#f5f5f5",
                font=dict(family="IBM Plex Sans, sans-serif", size=11),
            )
            fig3.update_xaxes(showgrid=True, gridcolor="#dddddd")
            fig3.update_yaxes(showgrid=True, gridcolor="#dddddd")
            st.plotly_chart(fig3, use_container_width=True)

        st.divider()

        # ── Cases Table ──────────────────────────────────────────────────────────
        st.markdown("**Alle Fälle**")

        table_data = []
        for c in cases:
            s = c["structured"]
            outcome = s.get("outcome")
            outcome_label = OUTCOME_LABELS.get(outcome, "—") if outcome is not None else "[nicht beschriftet]"
            table_data.append({
                "Fall-ID": c["case_id"],
                "Datei": c["filename"],
                "Datum": s.get("datum", "—"),
                "Gericht": s.get("instanz", "—"),
                "Anspruchsart": s.get("anspruchsart", "—"),
                "Streitwert EUR": s.get("streitwert_eur"),
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
                    "Streitwert EUR": st.column_config.NumberColumn(
                        "Streitwert EUR", format="EUR %.0f"
                    ),
                },
            )

        # ── Export ───────────────────────────────────────────────────────────────
        st.divider()
        st.markdown("**Dataset exportieren**")
        col_exp1, col_exp2 = st.columns(2)

        with col_exp1:
            if st.button("JSON exportieren", use_container_width=True):
                json_str = json.dumps(
                    [
                        {k: v for k, v in c.items() if k != "embeddings"}
                        for c in cases
                    ],
                    ensure_ascii=False,
                    indent=2,
                )
                st.download_button(
                    "Download JSON",
                    data=json_str.encode("utf-8"),
                    file_name="litigation_dataset.json",
                    mime="application/json",
                )

        with col_exp2:
            if st.button("CSV exportieren", use_container_width=True):
                csv_data = pd.DataFrame(table_data).to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    data=csv_data.encode("utf-8"),
                    file_name="litigation_dataset.csv",
                    mime="text/csv",
                )


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3: CASE REVIEW
# ════════════════════════════════════════════════════════════════════════════════

with tab_review:
    st.markdown('<div class="section-title">Fälle überprüfen und bearbeiten</div>', unsafe_allow_html=True)
    st.caption(
        "Kontrollieren Sie die extrahierten Daten und korrigieren Sie Fehler "
        "der automatischen Extraktion."
    )

    cases = dm.load_dataset()

    if not cases:
        st.info("Keine Fälle im Dataset.")
    else:
        # Case selector
        case_options = {
            f"{c['filename']}  [{c['case_id']}]": c["case_id"]
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
                        f'<span class="outcome-label outcome-{outcome}">'
                        f"{label}</span>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.warning("Ergebnis nicht extrahiert")

            with col_info3:
                sw = s.get("streitwert_eur")
                if sw:
                    st.metric("Streitwert", f"EUR {sw:,.2f}")
                date = s.get("datum")
                if date:
                    st.markdown(f"**Datum:** {date}")

            st.markdown("---")

            # ── Edit Form ────────────────────────────────────────────────────────
            with st.expander("Daten bearbeiten / korrigieren", expanded=False):
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
                        _instanz_opts = ["", "BG", "LG", "OLG", "OGH"]
                        new_instanz = st.selectbox(
                            "Instanz",
                            _instanz_opts,
                            index=_instanz_opts.index(s.get("instanz", "") or "")
                            if s.get("instanz", "") in _instanz_opts else 0,
                        )

                    with edit_col2:
                        new_streitwert = st.number_input(
                            "Streitwert (EUR)",
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
                            format_func=lambda x: f"{x} — {OUTCOME_LABELS[x]}",
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

                    if st.form_submit_button("Änderungen speichern", type="primary"):
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
                            st.success("Änderungen gespeichert.")
                            st.rerun()

            # ── Display Structured Data ──────────────────────────────────────────
            st.markdown("**Extrahierte Daten**")

            col_d1, col_d2 = st.columns(2)

            with col_d1:
                st.markdown("**Anspruchsgrundlagen:**")
                for ag in s.get("anspruchsgruende", []):
                    st.markdown(f"  - {ag}")

                st.markdown("**Kläger-Vorbringen (Zusammenfassung):**")
                st.markdown(
                    f"> {s.get('klaeger_anspruch_zusammenfassung', '—')}"
                )

                st.markdown("**Kläger-Beweismittel:**")
                for bm in s.get("klaeger_beweismittel", []):
                    st.markdown(f"  - {bm}")

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
                        st.markdown(f"  - {ad}")
                else:
                    st.markdown("  _Keine spezifischen Einwendungen_")

                st.markdown("**Beklagten-Beweismittel:**")
                for bm in s.get("beklagter_beweismittel", []):
                    st.markdown(f"  - {bm}")

                if s.get("sachverstaendiger_bestellt"):
                    fachgebiet = s.get("sachverstaendigen_fachgebiet", "")
                    st.info(f"Sachverständiger: {fachgebiet or 'Ja'}")

            # ── Extracted Sections ───────────────────────────────────────────────
            st.markdown("**Extrahierte Textabschnitte**")
            sections = case.get("sections", {})

            for sec_key, sec_label in EMBEDDING_SECTION_LABELS.items():
                text = sections.get(sec_key, "")
                if text:
                    with st.expander(f"{sec_label}  ({len(text)} Zeichen)"):
                        st.text_area(
                            sec_label,
                            value=text,
                            height=200,
                            key=f"sec_{selected_case_id}_{sec_key}",
                            disabled=True,
                        )

            # ── Delete ───────────────────────────────────────────────────────────
            st.divider()
            with st.expander("Fall löschen"):
                if st.button(
                    "Diesen Fall löschen",
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
    st.markdown('<div class="section-title">Manuelle Fallerfassung</div>', unsafe_allow_html=True)
    st.caption(
        "Fügen Sie Fälle manuell hinzu, falls kein PDF vorliegt oder "
        "die automatische Extraktion unvollständig war."
    )

    if not st.session_state.api_key:
        st.warning("OpenAI API Key erforderlich für Embedding-Generierung.")

    with st.form("manual_entry_form"):
        st.markdown("**Grunddaten**")
        m_col1, m_col2, m_col3 = st.columns(3)

        with m_col1:
            m_datum = st.text_input("Datum Ersturteil (YYYY-MM-DD)")
            m_instanz = st.selectbox("Instanz des Erstgerichts", ["BG", "LG", "OLG", "OGH"])
            m_streitwert = st.number_input("Streitwert (EUR)", min_value=0.0, step=100.0)

        with m_col2:
            m_claim_type = st.selectbox("Anspruchsart", CLAIM_TYPES + ["Andere"])
            m_anspruchsgruende = st.text_area(
                "Anspruchsgrundlagen (eine pro Zeile)",
                placeholder="§ 1295 ABGB\n§ 922 ABGB\n§ 879 ABGB",
                height=100,
            )
            m_outcome = st.selectbox(
                "Urteilsergebnis *",
                [0, 1, 2],
                format_func=lambda x: f"{x} — {OUTCOME_LABELS[x]}",
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

        st.markdown("**Textvorbringen**")
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

        st.markdown("**Einwendungen**")
        ew_cols = st.columns(4)
        m_einwendungen = {}
        for i, defense in enumerate(DEFENSE_TYPES):
            with ew_cols[i % 4]:
                m_einwendungen[defense] = st.checkbox(
                    DEFENSE_LABELS.get(defense, defense),
                    key=f"manual_ew_{defense}",
                )

        st.markdown("**Beweismittel**")
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
            "Fall speichern und Embeddings generieren",
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
                f"Fall `{case_id}` erfolgreich gespeichert. "
                "Wechseln Sie zum Tab 'Dataset' für eine Übersicht."
            )


# ─── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f"<div style='text-align:center;color:#888888;font-size:0.75rem;"
    f"font-family:IBM Plex Mono,monospace;letter-spacing:0.04em'>"
    f"Predictive Litigation Analytics &nbsp;·&nbsp; Österr. Zivilrecht (ABGB/ZPO) &nbsp;·&nbsp; OGH-Datenbasis &nbsp;·&nbsp; v{APP_VERSION} &nbsp;·&nbsp; "
    f"{OPENAI_EXTRACTION_MODEL} &nbsp;·&nbsp; text-embedding-3-large"
    f"</div>",
    unsafe_allow_html=True,
)
