"""
Documentation Generator: Creates a comprehensive PDF documentation
for the Predictive Litigation Analytics system using ReportLab.
"""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents

from config import (
    APP_VERSION,
    CLAIM_TYPES,
    DEFENSE_LABELS,
    DEFENSE_TYPES,
    NN_CONFIG,
    OUTCOME_LABELS,
    SECTION_LABELS,
    TRAINING_CONFIG,
)

# ─── Colors ──────────────────────────────────────────────────────────────────────

NAVY = colors.HexColor("#1a365d")
STEEL_BLUE = colors.HexColor("#2d6a9f")
GOLD = colors.HexColor("#c9a227")
LIGHT_GRAY = colors.HexColor("#f5f5f5")
MID_GRAY = colors.HexColor("#888888")
SUCCESS_GREEN = colors.HexColor("#27ae60")
WARNING_ORANGE = colors.HexColor("#f39c12")
DANGER_RED = colors.HexColor("#e74c3c")
WHITE = colors.white
BLACK = colors.black


# ─── Styles ───────────────────────────────────────────────────────────────────────

def build_styles():
    base = getSampleStyleSheet()

    styles = {
        "title_page_title": ParagraphStyle(
            "title_page_title",
            fontName="Helvetica-Bold",
            fontSize=28,
            textColor=WHITE,
            alignment=TA_CENTER,
            spaceAfter=12,
        ),
        "title_page_subtitle": ParagraphStyle(
            "title_page_subtitle",
            fontName="Helvetica",
            fontSize=14,
            textColor=colors.HexColor("#b8d4f0"),
            alignment=TA_CENTER,
            spaceAfter=6,
        ),
        "h1": ParagraphStyle(
            "h1",
            fontName="Helvetica-Bold",
            fontSize=18,
            textColor=NAVY,
            spaceBefore=20,
            spaceAfter=8,
            borderPad=4,
        ),
        "h2": ParagraphStyle(
            "h2",
            fontName="Helvetica-Bold",
            fontSize=14,
            textColor=STEEL_BLUE,
            spaceBefore=14,
            spaceAfter=6,
        ),
        "h3": ParagraphStyle(
            "h3",
            fontName="Helvetica-Bold",
            fontSize=11,
            textColor=NAVY,
            spaceBefore=10,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "body",
            fontName="Helvetica",
            fontSize=10,
            textColor=BLACK,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
            leading=16,
        ),
        "body_center": ParagraphStyle(
            "body_center",
            fontName="Helvetica",
            fontSize=10,
            textColor=BLACK,
            alignment=TA_CENTER,
            spaceAfter=6,
        ),
        "code": ParagraphStyle(
            "code",
            fontName="Courier",
            fontSize=9,
            textColor=colors.HexColor("#333333"),
            backColor=LIGHT_GRAY,
            spaceAfter=4,
            leading=14,
            leftIndent=10,
            rightIndent=10,
            borderPad=6,
        ),
        "bullet": ParagraphStyle(
            "bullet",
            fontName="Helvetica",
            fontSize=10,
            leftIndent=20,
            bulletIndent=10,
            spaceAfter=3,
            leading=14,
        ),
        "caption": ParagraphStyle(
            "caption",
            fontName="Helvetica-Oblique",
            fontSize=8,
            textColor=MID_GRAY,
            alignment=TA_CENTER,
            spaceAfter=8,
        ),
        "note": ParagraphStyle(
            "note",
            fontName="Helvetica-Oblique",
            fontSize=9,
            textColor=colors.HexColor("#555555"),
            backColor=colors.HexColor("#fff9e6"),
            spaceAfter=8,
            leftIndent=10,
            rightIndent=10,
            borderPad=8,
        ),
        "toc_heading": ParagraphStyle(
            "toc_heading",
            fontName="Helvetica-Bold",
            fontSize=12,
            textColor=NAVY,
            spaceAfter=8,
            spaceBefore=16,
        ),
        "toc_entry1": ParagraphStyle(
            "toc_entry1",
            fontName="Helvetica",
            fontSize=10,
            leftIndent=0,
            spaceAfter=2,
        ),
        "toc_entry2": ParagraphStyle(
            "toc_entry2",
            fontName="Helvetica",
            fontSize=9,
            leftIndent=20,
            textColor=MID_GRAY,
            spaceAfter=2,
        ),
    }
    return styles


S = build_styles()


# ─── Table Styles ────────────────────────────────────────────────────────────────

def table_style(header_color=NAVY, alternate=True):
    base = [
        ("BACKGROUND", (0, 0), (-1, 0), header_color),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 9),
        ("ROWBACKGROUND", (0, 1), (-1, -1), [WHITE, LIGHT_GRAY]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]
    return TableStyle(base)


# ─── Page Template ────────────────────────────────────────────────────────────────

class DocTemplate(SimpleDocTemplate):
    def __init__(self, *args, **kwargs):
        self.doc_title = kwargs.pop("doc_title", "")
        super().__init__(*args, **kwargs)

    def afterPage(self):
        pass

    def handle_pageBegin(self):
        pass


def header_footer(canvas, doc):
    canvas.saveState()
    page_num = canvas.getPageNumber()

    if page_num > 1:
        # Header
        canvas.setFillColor(NAVY)
        canvas.rect(2 * cm, A4[1] - 1.8 * cm, A4[0] - 4 * cm, 0.03 * cm, fill=1)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(MID_GRAY)
        canvas.drawString(2 * cm, A4[1] - 1.5 * cm, "Predictive Litigation Analytics")
        canvas.drawRightString(
            A4[0] - 2 * cm, A4[1] - 1.5 * cm, "Technische Dokumentation"
        )

        # Footer
        canvas.setFillColor(NAVY)
        canvas.rect(2 * cm, 1.5 * cm, A4[0] - 4 * cm, 0.03 * cm, fill=1)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(MID_GRAY)
        canvas.drawString(2 * cm, 1.0 * cm, f"© {datetime.now().year} · v{APP_VERSION}")
        canvas.drawRightString(A4[0] - 2 * cm, 1.0 * cm, f"Seite {page_num}")

    canvas.restoreState()


# ─── Content Builders ─────────────────────────────────────────────────────────────

def spacer(n: float = 0.5) -> Spacer:
    return Spacer(1, n * cm)


def hr(color=GOLD) -> HRFlowable:
    return HRFlowable(width="100%", thickness=2, color=color, spaceAfter=8)


def bold_p(text: str, style_key="body") -> Paragraph:
    return Paragraph(f"<b>{text}</b>", S[style_key])


def bullet_list(items: list[str]) -> list:
    return [Paragraph(f"• {item}", S["bullet"]) for item in items]


def info_box(text: str) -> list:
    return [
        Paragraph(f"ℹ️  {text}", S["note"]),
        spacer(0.2),
    ]


def build_title_page() -> list:
    """Build the title/cover page."""
    story = []

    # Cover background (simulated with colored table)
    cover_data = [
        [Paragraph("", S["body"])],
        [Paragraph("⚖️", ParagraphStyle("icon", fontName="Helvetica", fontSize=48, textColor=GOLD, alignment=TA_CENTER))],
        [Paragraph("Predictive Litigation Analytics", S["title_page_title"])],
        [Paragraph("Machine Learning System für österreichische Zivilprozesse", S["title_page_subtitle"])],
        [Paragraph("", S["body"])],
        [Paragraph(f"Version {APP_VERSION}", S["title_page_subtitle"])],
        [Paragraph(f"Dokumentation · {datetime.now().strftime('%B %Y')}", S["title_page_subtitle"])],
        [Paragraph("", S["body"])],
        [Paragraph("", S["body"])],
    ]

    cover_table = Table(cover_data, colWidths=[A4[0] - 4 * cm])
    cover_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), NAVY),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("ROUNDEDCORNERS", [10, 10, 10, 10]),
    ]))

    story.append(spacer(3))
    story.append(cover_table)
    story.append(spacer(2))

    # Subtitle info
    info_data = [
        ["System", "Predictive Litigation Analytics"],
        ["Version", APP_VERSION],
        ["Datum", datetime.now().strftime("%d. %B %Y")],
        ["Sprache", "Python 3.10+"],
        ["ML Framework", "PyTorch"],
        ["KI-Modell", "OpenAI gpt-5-nano-2025-08-07 (Structured Data)"],
        ["Klassifikation", "3-Klassen (Obsiegen / Teilweise / Unterliegen)"],
    ]

    info_table = Table(info_data, colWidths=[5 * cm, 10 * cm])
    info_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (0, -1), STEEL_BLUE),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
        ("ROWBACKGROUND", (0, 0), (-1, -1), [WHITE, LIGHT_GRAY]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))

    story.append(info_table)
    story.append(PageBreak())

    return story


def build_toc() -> list:
    story = []
    story.append(Paragraph("Inhaltsverzeichnis", S["h1"]))
    story.append(hr())

    toc_entries = [
        ("1", "Systemübersicht", ""),
        ("1.1", "Ziel und Anwendungsbereich", "  "),
        ("1.2", "Systemkomponenten", "  "),
        ("1.3", "Technologie-Stack", "  "),
        ("2", "Installation und Setup", ""),
        ("2.1", "Voraussetzungen", "  "),
        ("2.2", "Installationsanleitung", "  "),
        ("2.3", "Konfiguration", "  "),
        ("3", "Datenextraktion (Programm 1)", ""),
        ("3.1", "Übersicht", "  "),
        ("3.2", "PDF-Verarbeitung", "  "),
        ("3.3", "OpenAI-Extraktion", "  "),
        ("3.4", "Strukturierte Legal-Analyse", "  "),
        ("3.5", "Datenspeicherung", "  "),
        ("3.6", "Benutzeroberfläche", "  "),
        ("4", "Machine Learning Modell (Programm 2)", ""),
        ("4.1", "Architektur-Übersicht", "  "),
        ("4.2", "Netzwerk-Design", "  "),
        ("4.3", "Feature Engineering", "  "),
        ("4.4", "Training", "  "),
        ("4.5", "Evaluation", "  "),
        ("4.6", "Vorhersage neuer Fälle", "  "),
        ("5", "Erwartungswert-Kalkulation", ""),
        ("6", "Datenschutz und Sicherheit", ""),
        ("7", "Limitierungen und Ausblick", ""),
        ("8", "Technische Referenz", ""),
    ]

    for num, title, indent in toc_entries:
        level = 1 if not indent else 2
        style = "toc_entry1" if level == 1 else "toc_entry2"
        prefix = f"<b>{num}</b>" if level == 1 else num
        story.append(Paragraph(f"{indent}{prefix}  {title}", S[style]))

    story.append(PageBreak())
    return story


def build_section_1() -> list:
    """1. System Overview"""
    story = []

    story.append(Paragraph("1. Systemübersicht", S["h1"]))
    story.append(hr())

    story.append(Paragraph("1.1 Ziel und Anwendungsbereich", S["h2"]))
    story.append(Paragraph(
        "Das Predictive Litigation Analytics System ist ein Machine-Learning-basiertes Werkzeug "
        "für Rechtsanwaltskanzleien, das auf Basis historischer Zivilurteile die Erfolgschancen "
        "neuer Mandate prognostiziert. Es richtet sich an österreichische Zivilrechtspraktiker "
        "und verarbeitet Urteile erster Instanz (Bezirks- und Landesgerichte).",
        S["body"],
    ))

    story.append(Paragraph(
        "Das System unterstützt die anwaltliche Entscheidungsfindung durch:",
        S["body"],
    ))
    story.extend(bullet_list([
        "Automatisierte Datenextraktion aus PDF-Urteilen mittels KI",
        "Training eines neuronalen Netzes auf den kanzleispezifischen Falldaten",
        "Vorhersage des Verfahrensausgangs (Obsiegen / Teilweise / Unterliegen)",
        "Berechnung des kombinierten Erwartungswerts (ML + juristische KI-Einschätzung)",
        "Vollständige Datenhaltung lokal in der Kanzlei (Datenschutz)",
    ]))

    story.append(spacer(0.4))
    story.extend(info_box(
        "Das System ersetzt keine anwaltliche Beratung. Es ist als Entscheidungsunterstützung "
        "zu verstehen, das die Einschätzung eines Rechtsanwalts quantitativ ergänzt."
    ))

    story.append(Paragraph("1.2 Systemkomponenten", S["h2"]))

    comp_data = [
        ["Komponente", "Beschreibung", "Technologie"],
        ["Programm 1:\nData Extractor", "Extrahiert strukturierte Rechtsdaten\nund Legal-Analyse aus PDF-Urteilen", "Streamlit, OpenAI API,\nPyMuPDF, JSON"],
        ["Programm 2:\nML Model Trainer\n& Predictor", "Trainiert das neuronale Netz und\nberechnet Vorhersagen für neue Fälle", "PyTorch, Streamlit,\nPlotly"],
        ["Datenspeicher", "JSON-Dataset\n(lokal, kein Cloud-Speicher)", "JSON"],
        ["Dokumentation", "Dieses PDF-Dokument", "ReportLab"],
    ]

    t = Table(comp_data, colWidths=[4 * cm, 8 * cm, 5 * cm])
    t.setStyle(table_style(NAVY))
    story.append(t)

    story.append(spacer(0.4))
    story.append(Paragraph("1.3 Technologie-Stack", S["h2"]))

    tech_data = [
        ["Kategorie", "Technologie", "Zweck"],
        ["Programmiersprache", "Python 3.10+", "Gesamtes System"],
        ["UI Framework", "Streamlit 1.36+", "Weboberfläche (lokal)"],
        ["PDF-Verarbeitung", "PyMuPDF (fitz)", "Text-Extraktion aus PDFs"],
        ["KI-Extraktion", "OpenAI gpt-5-nano", "Strukturierte Datenerkennung"],
        ["Legal-Analyse", "GPT-5-nano (Structured)", "Detaillierte Rechtliche Analyse"],
        ["Deep Learning", "PyTorch 2.3+", "Neuronales Netz"],
        ["Visualisierung", "Plotly, Altair", "Diagramme und Charts"],
        ["Datenspeicher", "JSON", "Persistente Datenhaltung"],
        ["Normalisierung", "scikit-learn StandardScaler", "Feature-Normalisierung"],
        ["Dokumentation", "ReportLab", "PDF-Generierung"],
    ]

    t2 = Table(tech_data, colWidths=[5 * cm, 6 * cm, 6 * cm])
    t2.setStyle(table_style(STEEL_BLUE))
    story.append(t2)

    story.append(PageBreak())
    return story


def build_section_2() -> list:
    """2. Installation"""
    story = []

    story.append(Paragraph("2. Installation und Setup", S["h1"]))
    story.append(hr())

    story.append(Paragraph("2.1 Voraussetzungen", S["h2"]))
    story.extend(bullet_list([
        "Python 3.10 oder höher (empfohlen: 3.11)",
        "pip (Python-Paketmanager)",
        "OpenAI API Key (für Extraktion und Legal-Analyse)",
        "Mindestens 4 GB RAM (8 GB empfohlen)",
        "Festplattenspeicher: ca. 50 MB pro 1.000 Urteile (JSON)",
        "GPU optional (CUDA 11.8+) für schnelleres Training",
    ]))

    story.append(spacer(0.3))
    story.append(Paragraph("2.2 Installationsanleitung", S["h2"]))

    steps = [
        ("Schritt 1: Repository klonen oder entpacken",
         "cd /ihr/projektordner"),
        ("Schritt 2: Virtuelle Umgebung erstellen",
         "python -m venv venv\nsource venv/bin/activate  # Linux/Mac\n# oder:\nvenv\\Scripts\\activate    # Windows"),
        ("Schritt 3: Abhängigkeiten installieren",
         "pip install -r requirements.txt"),
        ("Schritt 4: OpenAI API Key setzen",
         "# Option A: Umgebungsvariable\nexport OPENAI_API_KEY='sk-...'  # Linux/Mac\nset OPENAI_API_KEY=sk-...        # Windows\n\n# Option B: In der App-Seitenleiste eingeben"),
        ("Schritt 5: Data Extractor starten",
         "python run_extractor.py"),
        ("Schritt 6: ML Model starten",
         "python run_model.py"),
    ]

    for title, code in steps:
        story.append(Paragraph(f"<b>{title}:</b>", S["body"]))
        story.append(Paragraph(code, S["code"]))
        story.append(spacer(0.2))

    story.append(Paragraph("2.3 Konfiguration", S["h2"]))
    story.append(Paragraph(
        "Die zentrale Konfigurationsdatei <code>config.py</code> enthält alle anpassbaren Parameter:",
        S["body"],
    ))

    config_data = [
        ["Parameter", "Standard", "Beschreibung"],
        ["OPENAI_EXTRACTION_MODEL", "gpt-5-nano-2025-08-07", "OpenAI-Modell für Datenextraktion + Legal-Analyse"],
        ["epochs", str(TRAINING_CONFIG["epochs"]), "Max. Training-Epochen"],
        ["learning_rate", str(TRAINING_CONFIG["learning_rate"]), "Lernrate (AdamW)"],
        ["batch_size", str(TRAINING_CONFIG["batch_size"]), "Batch-Größe"],
        ["val_split", str(TRAINING_CONFIG["val_split"]), "Validierungsanteil"],
        ["early_stopping_patience", str(TRAINING_CONFIG["early_stopping_patience"]), "Early-Stopping-Geduld"],
    ]

    t = Table(config_data, colWidths=[5.5 * cm, 4.5 * cm, 7 * cm])
    t.setStyle(table_style(STEEL_BLUE))
    story.append(t)

    story.append(PageBreak())
    return story


def build_section_3() -> list:
    """3. Data Extractor"""
    story = []

    story.append(Paragraph("3. Datenextraktion (Programm 1)", S["h1"]))
    story.append(hr())

    story.append(Paragraph("3.1 Übersicht", S["h2"]))
    story.append(Paragraph(
        "Das erste Programm (Data Extractor) liest PDF-Urteile ein, extrahiert strukturierte "
        "Rechtsdaten via OpenAI-API und speichert alles in einem strukturierten Dataset. "
        "Es besteht aus vier Modulen: pdf_processor.py, openai_extractor.py, data_manager.py "
        "und der Streamlit-UI (app.py).",
        S["body"],
    ))

    story.append(Paragraph("3.2 PDF-Verarbeitung (pdf_processor.py)", S["h2"]))
    story.append(Paragraph(
        "PyMuPDF extrahiert den Volltext aus PDF-Urteilen. Der Extraktions-Algorithmus:",
        S["body"],
    ))
    story.extend(bullet_list([
        "Öffnet das PDF und iteriert über alle Seiten",
        "Verwendet layout-erhaltende Blockextraktion (Spalten, Absätze)",
        "Normalisiert Whitespace, entfernt PDF-Artefakte (Silbentrennung etc.)",
        "Versucht automatische Abschnittserkennung (Kläger-/Beklagten-Vorbringen, Feststellungen, etc.)",
        "Gibt Volltext + Metadaten zurück",
    ]))

    story.extend(info_box(
        "Gescannte PDFs (Bild-PDFs ohne Text-Layer) werden erkannt und mit einer "
        "Fehlermeldung übersprungen. Für solche Dokumente ist eine OCR-Vorverarbeitung notwendig."
    ))

    story.append(Paragraph("3.3 OpenAI-Extraktion (openai_extractor.py)", S["h2"]))
    story.append(Paragraph(
        "Das GPT-5-nano Modell extrahiert strukturierte Rechtsdaten aus dem Urteilstext. "
        "Das Extraktionsschema umfasst:",
        S["body"],
    ))

    schema_data = [
        ["Feld", "Typ", "Beschreibung"],
        ["datum", "String", "Urteilsdatum (YYYY-MM-DD)"],
        ["gericht", "String", "Gerichtsbezeichnung (anonymisiert)"],
        ["instanz", "Enum", "BG / LG / OLG / OGH"],
        ["streitwert_eur", "Float", "Streitwert in Euro"],
        ["anspruchsart", "Enum", f"{len(CLAIM_TYPES)} Kategorien + 'Andere'"],
        ["anspruchsgruende", "Liste", "Rechtsgrundlagen (§§ ABGB, UGB, etc.)"],
        ["einwendungen", "Objekt", f"{len(DEFENSE_TYPES)} Boolean-Felder"],
        ["klaeger_beweismittel", "Liste", "Beweismittel des Klägers"],
        ["beklagter_beweismittel", "Liste", "Beweismittel des Beklagten"],
        ["sachverstaendiger_bestellt", "Boolean", "Ob Sachverständiger bestellt"],
        ["outcome", "Int (0/1/2)", "Unterliegen / Teilweise / Obsiegen"],
        ["zugesprochener_anteil_prozent", "Float", "Zugesprochener Anteil (0-100%)"],
        ["kostenentscheidung", "Enum", "Kläger / Beklagter / Geteilt"],
    ]

    t = Table(schema_data, colWidths=[5.5 * cm, 3 * cm, 8.5 * cm])
    t.setStyle(table_style(NAVY))
    story.append(t)

    story.append(spacer(0.4))
    story.append(Paragraph("Erkannte Einwendungen (Einwendungstypen):", S["h3"]))

    defense_data = [["Typ", "Beschreibung"]] + [
        [dtype, DEFENSE_LABELS[dtype]]
        for dtype in DEFENSE_TYPES
    ]
    t2 = Table(defense_data, colWidths=[6 * cm, 11 * cm])
    t2.setStyle(table_style(STEEL_BLUE))
    story.append(t2)

    story.append(spacer(0.4))
    story.append(Paragraph("3.4 Strukturierte Legal-Analyse (GPT-5-nano)", S["h2"]))
    story.append(Paragraph(
        "Für jeden Fall wird eine detaillierte strukturierte Analyse der rechtlichen "
        "Argumente, Anspruchsgrundlagen und Einwendungen extrahiert. Diese strukturierten "
        "Features ersetzen die bisherigen Text-Embeddings und dienen als primärer Training-Input.",
        S["body"],
    ))

    la_sections = [
        ["fall_metadaten", "Rechtsgebiet, Verbrauchergeschäft, Streitwert"],
        ["klaegervorbringen_anspruchsgrundlagen", "Vertragliche, deliktische, dingliche Ansprüche"],
        ["beklagtenvorbringen_prozessual", "Unzuständigkeit, Streitanhängigkeit, Prozesshindernisse"],
        ["beklagtenvorbringen_materiell_rechtshindernd", "Geschäftsunfähigkeit, Formmangel, Irrtum"],
        ["beklagtenvorbringen_materiell_rechtsvernichtend", "Erfüllung, Aufrechnung, Rücktritt"],
        ["beklagtenvorbringen_materiell_rechtshemmend", "Verjährung, Zurückbehaltung, Fälligkeit"],
        ["beklagtenvorbringen_allgemein", "Legitimation, Mitverschulden, Bestreitung"],
    ]

    la_data = [["Schema-Sektion", "Inhalt"]] + la_sections

    t3 = Table(la_data, colWidths=[7 * cm, 10 * cm])
    t3.setStyle(table_style(STEEL_BLUE))
    story.append(t3)

    story.append(spacer(0.3))
    story.append(Paragraph("3.5 Datenspeicherung (data_manager.py)", S["h2"]))
    story.append(Paragraph(
        "Das Dataset wird in zwei Formaten gespeichert:",
        S["body"],
    ))
    story.extend(bullet_list([
        f"data/extracted/cases_dataset.json — Falldaten + Legal-Analyse (JSON, lesbar, editierbar)",
        f"data/models/ — Trainiertes Modell (PyTorch .pt) + Feature-Scaler (Pickle)",
    ]))

    story.append(PageBreak())
    return story


def build_section_4() -> list:
    """4. ML Model"""
    story = []

    story.append(Paragraph("4. Machine Learning Modell (Programm 2)", S["h1"]))
    story.append(hr())

    story.append(Paragraph("4.1 Architektur-Übersicht", S["h2"]))
    story.append(Paragraph(
        "Das LitigationClassifier-Netzwerk verarbeitet strukturierte Rechtsmerkmale "
        "(Original-Metadaten + Legal-Analyse-Features) in einem kompakten Feed-Forward-Netzwerk. "
        "Es gibt drei Klassen aus: Unterliegen (0), Teilweises Obsiegen (1), Obsiegen (2).",
        S["body"],
    ))

    arch_data = [
        ["Komponente", "Input", "Output", "Aktivierung"],
        ["Encoder Layer 1", "~77-dim\nstrukturierte Features", "128-dim", "GELU + LayerNorm\n+ Dropout"],
        ["Encoder Layer 2", "128-dim", "128-dim", "GELU + LayerNorm\n+ Dropout"],
        ["Fusion Layer", "128-dim", "64-dim", "GELU + Dropout"],
        ["Classifier", "64-dim", "3-dim", "Softmax"],
    ]

    t = Table(arch_data, colWidths=[4.5 * cm, 4 * cm, 3 * cm, 5.5 * cm])
    t.setStyle(table_style(NAVY))
    story.append(t)

    story.append(spacer(0.4))
    story.append(Paragraph("4.2 Netzwerk-Design", S["h2"]))

    design_details = [
        ("Verlustfunktion", "Focal Loss (γ=2) mit inversem Klassengewicht — robust gegen Datenungleichgewicht"),
        ("Optimizer", "AdamW (weight decay=1e-4) — verhindert Overfitting durch L2-Regularisierung"),
        ("LR-Scheduler", "ReduceLROnPlateau (Faktor 0.5, Geduld 15 Epochen)"),
        ("Early Stopping", "Geduld 30 Epochen — stoppt Training bei Stagnation"),
        ("Gradient Clipping", "Max-Norm 1.0 — verhindert explodierende Gradienten"),
        ("Gewichtsinitialisierung", "Xavier Uniform — optimale Startgewichte für tiefe Netze"),
        ("Normalisierung", "LayerNorm nach jeder linearen Schicht — stabilisiert Training"),
    ]

    dd_data = [["Aspekt", "Details"]] + [[k, v] for k, v in design_details]
    t2 = Table(dd_data, colWidths=[5 * cm, 12 * cm])
    t2.setStyle(table_style(STEEL_BLUE))
    story.append(t2)

    story.append(spacer(0.4))
    story.append(Paragraph("4.3 Feature Engineering (feature_engineer.py)", S["h2"]))
    story.append(Paragraph(
        "Strukturierte Features werden in numerische Vektoren kodiert:",
        S["body"],
    ))

    feat_data = [
        ["Feature", "Kodierung", "Dim."],
        ["Streitwert", "log(1 + streitwert) → StandardScaler", "1"],
        ["Anspruchsart", f"One-Hot ({len(CLAIM_TYPES)}+1 Klassen)", str(len(CLAIM_TYPES) + 1)],
        ["Einwendungen", f"Binary Flags ({len(DEFENSE_TYPES)} Typen)", str(len(DEFENSE_TYPES))],
        ["Kläger-Beweismittel", "Anzahl (Integer)", "1"],
        ["Beklagten-Beweismittel", "Anzahl (Integer)", "1"],
        ["Anspruchsgrundlagen", "Anzahl (Integer)", "1"],
        ["Instanz", "One-Hot (BG/LG/OLG/OGH)", "4"],
        ["Sachverständiger", "Binary Flag", "1"],
        ["Rechtsgebiet", "One-Hot (8 Kategorien)", "8"],
        ["Legal-Analyse Booleans", "42 Boolean-Felder (Ansprüche + Einwendungen)", "42"],
        ["Zitierte Normen", "Anzahl Kläger + Beklagter", "2"],
        ["GESAMT", "", "~77"],
    ]

    t3 = Table(feat_data, colWidths=[6 * cm, 8 * cm, 3 * cm])
    t3.setStyle(table_style(NAVY))
    story.append(t3)

    story.append(spacer(0.4))
    story.append(Paragraph("4.4 Training", S["h2"]))
    story.append(Paragraph(
        "Das Training erfolgt mit stratifiziertem Train/Validation-Split. "
        "Die empfohlene Mindestanzahl an beschrifteten Fällen beträgt 20 (besser: >50):",
        S["body"],
    ))
    story.extend(bullet_list([
        "< 5 Fälle: Training nicht möglich",
        "5–20 Fälle: Training möglich, aber sehr limitierte Generalisierfähigkeit",
        "20–50 Fälle: Akzeptable Ergebnisse für kanzleiinterne Nutzung",
        "> 50 Fälle: Gute Ergebnisse, robuste Vorhersagen",
        "> 100 Fälle: Beste Leistung, Vertrauen in Vorhersagen gerechtfertigt",
    ]))

    story.extend(info_box(
        "Da Kanzleidaten typischerweise unausgeglichen sind (mehr Siege als Niederlagen "
        "oder umgekehrt), verwendet das System Focal Loss und inverse Klassengewichtung, "
        "um alle Outcome-Klassen fair zu lernen."
    ))

    story.append(Paragraph("4.5 Evaluation", S["h2"]))
    story.append(Paragraph(
        "Das Modell wird mit folgenden Metriken bewertet:",
        S["body"],
    ))
    story.extend(bullet_list([
        "Accuracy: Anteil korrekt vorhergesagter Fälle (gesamt)",
        "Per-Class Precision, Recall, F1-Score",
        "Konfusionsmatrix (3×3): zeigt Verwechslungen zwischen Outcome-Klassen",
        "Probability Distribution: Histogramm der Vorhersage-Konfidenz",
    ]))

    story.append(Paragraph("4.6 Vorhersage neuer Fälle", S["h2"]))
    story.append(Paragraph(
        "Für neue Fälle werden folgende Schritte durchgeführt:",
        S["body"],
    ))
    story.extend(bullet_list([
        "1. Eingabe der Falldaten über die UI (Streitwert, Anspruchsart, Einwendungen etc.)",
        "2. Strukturierte Legal-Analyse aus GPT-5-nano oder manuelle Eingabe",
        "3. Feature-Kodierung mit dem gespeicherten Scaler (kein Neubeschriften)",
        "4. Modell-Inferenz → Wahrscheinlichkeiten für 0, 1, 2",
        "5. Anzeige der Vorhersage + Konfidenz",
    ]))

    story.append(PageBreak())
    return story


def build_section_5() -> list:
    """5. Expected Value"""
    story = []

    story.append(Paragraph("5. Erwartungswert-Kalkulation", S["h1"]))
    story.append(hr())

    story.append(Paragraph(
        "Das System berechnet einen kombinierten Erwartungswert, der das ML-Modell "
        "mit der juristischen Einschätzung des KI-Assistenten kombiniert:",
        S["body"],
    ))

    story.append(Paragraph("Formel:", S["h3"]))
    story.append(Paragraph(
        "E[outcome] = w_ML × P_ML(Obsiegen) + w_Jur × P_Juristisch(Obsiegen)",
        S["code"],
    ))

    story.append(Paragraph(
        "Monetärer Erwartungswert (netto):",
        S["h3"],
    ))
    story.append(Paragraph(
        "EV_netto = P(Voll-Obsiegen) × Streitwert + P(Teilw.) × Streitwert × 0.5 - Verfahrenskosten",
        S["code"],
    ))

    params_data = [
        ["Parameter", "Beschreibung"],
        ["P_ML(Obsiegen)", "Vom neuronalen Netz berechnete Obsiegens-Wahrscheinlichkeit"],
        ["P_Juristisch(Obsiegen)", "Einschätzung des juristischen KI-Assistenten (0.0–1.0)"],
        ["w_ML", "Gewichtung des ML-Modells (Standard: 0.5)"],
        ["w_Jur", "Gewichtung der juristischen Einschätzung (Standard: 0.5)"],
        ["Streitwert", "Eingeklagte Summe in Euro"],
        ["Verfahrenskosten", "Geschätzte Gesamtkosten bei Verlust"],
    ]

    t = Table(params_data, colWidths=[6 * cm, 11 * cm])
    t.setStyle(table_style(NAVY))
    story.append(t)

    story.append(spacer(0.4))
    story.append(Paragraph("Empfehlungsschwellen:", S["h3"]))

    rec_data = [
        ["EV-Wahrscheinlichkeit", "Empfehlung"],
        ["≥ 70%", "STARK EMPFOHLEN — Hohe Erfolgschancen"],
        ["55–69%", "EMPFOHLEN — Überwiegende Erfolgschancen"],
        ["45–54%", "NEUTRAL — Ausgeglichene Chancen, Kosten-Nutzen prüfen"],
        ["30–44%", "VORSICHT — Unterdurchschnittliche Erfolgschancen"],
        ["< 30%", "NICHT EMPFOHLEN — Geringe Erfolgschancen"],
    ]

    t2 = Table(rec_data, colWidths=[5 * cm, 12 * cm])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#e8f5e9")),
        ("BACKGROUND", (0, 2), (-1, 2), colors.HexColor("#f1f8e9")),
        ("BACKGROUND", (0, 3), (-1, 3), colors.HexColor("#fff8e1")),
        ("BACKGROUND", (0, 4), (-1, 4), colors.HexColor("#fff3e0")),
        ("BACKGROUND", (0, 5), (-1, 5), colors.HexColor("#fde8e6")),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(t2)

    story.append(PageBreak())
    return story


def build_section_6() -> list:
    """6. Privacy"""
    story = []

    story.append(Paragraph("6. Datenschutz und Sicherheit", S["h1"]))
    story.append(hr())

    story.append(Paragraph(
        "Das System wurde mit besonderer Rücksicht auf den Datenschutz im Kanzleiumfeld entwickelt.",
        S["body"],
    ))

    story.append(Paragraph("Anonymisierung:", S["h2"]))
    story.extend(bullet_list([
        "Alle extrahierten Daten werden von Namen (Richter, Parteien, Anwälte) befreit",
        "Ersetzung durch generische Platzhalter: [KLÄGER], [BEKLAGTER], [RICHTER], [ANWALT]",
        "Nur sachliche, anspruchsrelevante Daten werden gespeichert",
        "Kein Personenbezug in der Trainingsdatenbank",
    ]))

    story.append(Paragraph("Datenhaltung:", S["h2"]))
    story.extend(bullet_list([
        "Alle Daten bleiben lokal auf dem Kanzleirechner (kein Cloud-Upload der Falldaten)",
        "OpenAI-API: Nur Urteilstexte werden an OpenAI übermittelt (für Extraktion/Legal-Analyse)",
        "OpenAI's Data Usage Policy: Daten werden nicht für Training verwendet (API-Nutzung)",
        "Kein persistenter Zugriff Dritter auf das Dataset",
    ]))

    story.append(Paragraph("Empfehlung:", S["h2"]))
    story.extend(info_box(
        "Prüfen Sie vor der Nutzung die OpenAI-Datenschutzbedingungen für Enterprise/API-Kunden. "
        "Für besonders sensible Mandate empfiehlt sich die Verwendung eines lokalen LLM "
        "(z.B. Llama 3, Mistral) als Alternative zu OpenAI."
    ))

    story.append(PageBreak())
    return story


def build_section_7() -> list:
    """7. Limitations"""
    story = []

    story.append(Paragraph("7. Limitierungen und Ausblick", S["h1"]))
    story.append(hr())

    story.append(Paragraph("Bekannte Limitierungen:", S["h2"]))
    story.extend(bullet_list([
        "Kleine Datensätze (<20 Fälle): Hohe Varianz, geringe Generalisierfähigkeit",
        "Domänenverschiebung: Gericht, Richter, Region beeinflussen Ergebnisse — nicht modellierbar",
        "Extraktion gescannter PDFs: Ohne OCR kein Textinhalt verfügbar",
        "GPT-Extraktionsfehler: Manche Abschnitte werden falsch klassifiziert — manuelle Korrektur möglich",
        "Klassen-Ungleichgewicht: Bei sehr einseitigen Datasets (z.B. 90% Siege) kann das Modell überfitten",
        "Keine zeitliche Gewichtung: Ältere Urteile haben gleiches Gewicht wie neuere",
        "Keine OGH-Judikatur: Das Modell lernt aus erstinstanzlichen Urteilen, nicht aus Höchstgerichtsjudikatur",
    ]))

    story.append(Paragraph("Empfehlungen für die Praxis:", S["h2"]))
    story.extend(bullet_list([
        "Regelmäßige Nachtrainierung mit neuen Urteilen (halbjährlich empfohlen)",
        "Manuelle Überprüfung aller extrahierten Daten über die Review-UI",
        "Kombination mit der juristischen KI-Einschätzung für robustere Ergebnisse",
        "Mindestens 50 Fälle vor produktivem Einsatz der Vorhersagefunktion",
        "Ausgeglichenes Dataset anstreben (ähnliche Anzahl pro Outcome-Klasse)",
    ]))

    story.append(Paragraph("Ausblick / Erweiterungsideen:", S["h2"]))
    story.extend(bullet_list([
        "Integration von OGH-Entscheidungen als Wissensgrundlage",
        "Mehrsprachige Unterstützung (DE/AT Varianten)",
        "Zeitgewichtung: Neuere Urteile erhalten höheres Gewicht",
        "Aktive Lern-Schleife: Modell meldet unsichere Fälle für manuelle Überprüfung",
        "REST-API: Einbindung in bestehende Kanzleisoftware",
        "Erklärbarkeit: SHAP-Werte zur Interpretation der Vorhersagegründe",
    ]))

    story.append(PageBreak())
    return story


def build_section_8() -> list:
    """8. Technical Reference"""
    story = []

    story.append(Paragraph("8. Technische Referenz", S["h1"]))
    story.append(hr())

    story.append(Paragraph("Dateistruktur:", S["h2"]))
    story.append(Paragraph("""
predictive_analytics/
├── config.py                      # Zentrale Konfiguration
├── requirements.txt               # Python-Abhängigkeiten
├── run_extractor.py               # Starter: Data Extractor
├── run_model.py                   # Starter: ML Model
├── install.sh                     # Installationsskript
├── data_extractor/
│   ├── app.py                     # Streamlit UI (Extraktion)
│   ├── pdf_processor.py           # PDF-Textextraktion
│   ├── openai_extractor.py        # OpenAI API Interface
│   └── data_manager.py            # Dataset-Verwaltung
├── model/
│   ├── app.py                     # Streamlit UI (Modell)
│   ├── neural_net.py              # PyTorch Netzwerkarchitektur
│   ├── feature_engineer.py        # Feature-Kodierung
│   ├── trainer.py                 # Trainingsschleife
│   └── predictor.py               # Vorhersage & EV
├── docs/
│   └── generate_docs.py           # Dieses Dokument generieren
└── data/
    ├── extracted/
    │   └── cases_dataset.json     # Strukturiertes Dataset
    └── models/
        ├── litigation_model.pt    # Trainiertes Modell
        ├── feature_scaler.pkl     # StandardScaler
        └── training_history.json  # Trainingsverlauf
    """, S["code"]))

    story.append(spacer(0.4))
    story.append(Paragraph("API-Kosten (ca.):", S["h2"]))

    cost_data = [
        ["Operation", "Modell", "Tokens/Fall", "Kosten/Fall (ca.)"],
        ["Strukturierte Extraktion", "gpt-5-nano", "~4.000", "~$0.002"],
        ["Abschnitte extrahieren", "gpt-5-nano", "~6.000", "~$0.003"],
        ["Legal-Analyse", "gpt-5-nano", "~8.000", "~$0.004"],
        ["Gesamt pro Urteil", "—", "~18.000", "~$0.009"],
        ["1.000 Urteile", "—", "—", "~$9.00"],
        ["10.000 Urteile", "—", "—", "~$90.00"],
    ]

    t = Table(cost_data, colWidths=[5.5 * cm, 4 * cm, 3.5 * cm, 4 * cm])
    t.setStyle(table_style(STEEL_BLUE))
    story.append(t)

    story.extend(info_box(
        "OpenAI-Preise können sich ändern. Aktuelle Preise unter: platform.openai.com/pricing"
    ))

    story.append(Paragraph("Modell-Hyperparameter:", S["h2"]))
    hp_data = [
        ["Hyperparameter", "Wert", "Beschreibung"],
        ["hidden_dims", str(NN_CONFIG["hidden_dims"]), "Hidden-Layer-Dimensionen"],
        ["fusion_dims", str(NN_CONFIG["fusion_dims"]), "Schichtgrößen des Fusion-Netzwerks"],
        ["dropout", str(NN_CONFIG["dropout"]), "Dropout-Rate"],
        ["epochs", str(TRAINING_CONFIG["epochs"]), "Max. Trainingsepochen"],
        ["batch_size", str(TRAINING_CONFIG["batch_size"]), "Batch-Größe"],
        ["learning_rate", str(TRAINING_CONFIG["learning_rate"]), "Initiale Lernrate"],
        ["weight_decay", str(TRAINING_CONFIG["weight_decay"]), "L2-Regularisierungsstärke"],
        ["focal_gamma", "2.0", "Focal-Loss γ-Parameter"],
    ]
    t2 = Table(hp_data, colWidths=[5.5 * cm, 3.5 * cm, 8 * cm])
    t2.setStyle(table_style(NAVY))
    story.append(t2)

    return story


# ─── Main Document Builder ────────────────────────────────────────────────────────

def generate_documentation(output_path: str = None) -> str:
    """
    Generate the complete PDF documentation.

    Args:
        output_path: Path for the output PDF. Defaults to docs/PLA_Dokumentation.pdf

    Returns:
        Absolute path to the generated PDF.
    """
    if output_path is None:
        docs_dir = Path(__file__).parent
        docs_dir.mkdir(exist_ok=True)
        output_path = str(docs_dir / "PLA_Dokumentation.pdf")

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2.5 * cm,
        bottomMargin=2 * cm,
        title="Predictive Litigation Analytics — Dokumentation",
        author="Predictive Litigation Analytics System",
        subject="Technische Dokumentation",
    )

    story = []

    # Build all sections
    story.extend(build_title_page())
    story.extend(build_toc())
    story.extend(build_section_1())
    story.extend(build_section_2())
    story.extend(build_section_3())
    story.extend(build_section_4())
    story.extend(build_section_5())
    story.extend(build_section_6())
    story.extend(build_section_7())
    story.extend(build_section_8())

    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)

    return output_path


if __name__ == "__main__":
    import sys

    output = sys.argv[1] if len(sys.argv) > 1 else None
    print("Generiere PDF-Dokumentation...")
    path = generate_documentation(output)
    print(f"✅ Dokumentation gespeichert: {path}")
