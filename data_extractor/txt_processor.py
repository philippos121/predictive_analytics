"""
TXT Processor: Liest und vorverarbeitet österreichische OGH-Urteile als Plaintext.

Eingabe: .txt-Dateien mit Urteilen des Obersten Gerichtshofs (OGH) — Zivilsenat.
Der OGH legt den vollständigen Verfahrensgang offen, sodass aus dem OGH-Urteil
das Vorbringen der Parteien sowie die Entscheidung des ERSTGERICHTS extrahiert
werden können. Die eigentliche Extraktion erfolgt via GPT (openai_extractor.py);
dieser Prozessor übernimmt Lesen, Bereinigung und strukturelle Vorverarbeitung.

Strafurteile (OGH-Strafsenat) werden herausgefiltert — nur Zivilurteile
fließen in die Extraktionspipeline.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import EMBEDDING_SECTIONS


class TxtProcessingError(Exception):
    pass


class StrafurteilError(Exception):
    """Wird ausgelöst, wenn eine TXT-Datei als Strafurteil erkannt wird."""
    pass


# ─── Strafurteil-Erkennung ────────────────────────────────────────────────────────
# Starke Indikatoren für OGH-Strafurteile — erscheinen in Zivilurteilen nicht.
_CRIMINAL_STRONG = [
    r"\bStGB\b",                        # Strafgesetzbuch
    r"\bStPO\b",                        # Strafprozessordnung
    r"\bAngeklagte(?:n|r|m)?\b",        # Angeklagter (strafprozessuale Partei)
    r"\bFreiheitsstrafe\b",             # Freiheitsstrafe
    r"\bBewährung\b",                   # Bewährung
    r"\bTagessätze?\b",                 # Geldstrafe in Tagessätzen
    r"\bStrafsenat\b",                  # OGH-Strafsenat
    r"\bOberste[rs]?\s+Gerichtshof\b.*\bStraf", # "OGH ... Straf..."
    r"\bStaatsanwalt(?:schaft)?\b",     # Staatsanwaltschaft
    r"\bfreigesprochen\b",              # Freispruch
    r"\bverurteilt\s+(?:zu|wegen)\b",   # "verurteilt zu X Jahren"
    r"\bAnklageschrift\b",              # Anklageschrift
    r"\b(?:§|§§)\s*\d+\s+StGB\b",      # §§ StGB-Paragraphen
    r"\bSicherungsverwahrung\b",
    r"\bJugendstrafrecht\b",
]

# Österreichische Zivilindikatoren (Gegencheck)
_CIVIL_INDICATORS = [
    r"\bKläger(?:in)?\b",
    r"\bBeklagte[nrm]?\b",
    r"\bStreitwert\b",
    r"\bZPO\b",
    r"\bABGB\b",
    r"\bSchadenersatz\b",
    r"\bKaufpreis\b",
    r"\bMieter(?:in)?\b",
    r"\bRevision\b",            # OGH Zivilrevision
    r"\bRevisionswerber\b",     # typisch österreichisch
]

_RE_CRIMINAL = [re.compile(p, re.IGNORECASE) for p in _CRIMINAL_STRONG]
_RE_CIVIL = [re.compile(p, re.IGNORECASE) for p in _CIVIL_INDICATORS]


def detect_judgment_type(text: str) -> str:
    """
    Klassifiziert ein OGH-Urteil als Zivil- oder Strafurteil.

    Gibt zurück:
        "zivil"  — Zivilurteil, wird verarbeitet
        "straf"  — Strafurteil, wird herausgefiltert (StrafurteilError)
        "unklar" — nicht eindeutig klassifizierbar, wird als Zivilurteil behandelt

    Logik: Prüft die ersten 5.000 Zeichen auf starke strafrechtliche Indikatoren.
    Ab 2 Treffern oder 1 Treffer ohne Zivilindikation → Strafurteil.
    """
    sample = text[:5000]
    criminal_hits = sum(1 for pat in _RE_CRIMINAL if pat.search(sample))
    civil_hits = sum(1 for pat in _RE_CIVIL if pat.search(sample))

    if criminal_hits >= 2:
        return "straf"
    if criminal_hits == 1 and civil_hits == 0:
        return "straf"
    return "zivil"


def read_txt_file(txt_path: str | Path, encoding: str = "utf-8") -> str:
    """
    Liest eine TXT-Datei mit automatischer Encoding-Erkennung.

    Probiert: utf-8 → utf-8-sig → latin-1 → cp1252.

    Raises:
        TxtProcessingError: Datei nicht lesbar oder zu kurz.
        StrafurteilError:   Text wurde als Strafurteil erkannt.
    """
    txt_path = Path(txt_path)
    if not txt_path.exists():
        raise TxtProcessingError(f"Datei nicht gefunden: {txt_path}")
    if txt_path.suffix.lower() != ".txt":
        raise TxtProcessingError(f"Kein TXT-Format: {txt_path.name}")

    text = None
    for enc in [encoding, "utf-8-sig", "latin-1", "cp1252"]:
        try:
            text = txt_path.read_text(encoding=enc)
            break
        except (UnicodeDecodeError, LookupError):
            continue

    if text is None:
        raise TxtProcessingError(
            f"Datei konnte mit keinem unterstützten Encoding gelesen werden: {txt_path.name}"
        )

    text = _clean_text(text)

    if len(text.strip()) < 300:
        raise TxtProcessingError(
            f"Text zu kurz ({len(text)} Zeichen): {txt_path.name}"
        )

    if detect_judgment_type(text) == "straf":
        raise StrafurteilError(
            f"OGH-Strafurteil erkannt und herausgefiltert: {txt_path.name}"
        )

    return text


def _clean_text(text: str) -> str:
    """Normalisiert Zeilenenden, Whitespace und typische Plaintext-Artefakte."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Silbentrennungen am Zeilenende zusammenfügen
    text = re.sub(r"([a-zäöüßA-ZÄÖÜ])-\n([a-zäöüßA-ZÄÖÜ])", r"\1\2", text)
    # Max 2 Leerzeilen
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Mehrfache Leerzeichen
    text = re.sub(r"[ \t]+", " ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    return text.strip()


def split_into_sections(text: str) -> dict[str, str]:
    """
    Teilt einen OGH-Urteilstext in erkannte Abschnitte auf.

    OGH-Zivilurteile haben typischerweise folgende Struktur:
      1. Kopf / Rubrum (Parteien, Aktenzeichen, Revisionszulassung)
      2. Spruch (Entscheidungsformel des OGH)
      3. Entscheidungsgründe:
         a. Sachverhalt / Feststellungen (aus Erstgericht)
         b. Vorbringen der Parteien (Klage, Klagebeantwortung)
         c. Ersturteil: Entscheidung des Erstgerichts (BG/LG)
         d. Berufungsgericht: Entscheidung des OLG
         e. Revision / Revisionsbeantwortung
         f. Rechtliche Beurteilung des OGH (Senat)

    Die tatsächliche strukturierte Extraktion (GPT) nutzt den Volltext,
    diese Funktion liefert eine grobe Vorab-Segmentierung für die Darstellung
    im Review-Tab der App.
    """
    sections = {s: "" for s in EMBEDDING_SECTIONS}
    sections["full_text"] = text
    sections["erstgericht_entscheidung_text"] = ""
    sections["ogh_entscheidung"] = ""
    sections["spruch"] = ""

    patterns = {
        "klaegervorbringen": [
            r"(?i)((?:die?\s+)?kläger(?:in)?\s+(?:begehrt|bringt\s+vor|macht\s+geltend|brachte\s+vor)"
            r"|klagevorbringen\b|vorbringen\s+(?:der\s+)?kläger(?:in)?)",
        ],
        "beklagtenvorbringen": [
            r"(?i)((?:die?\s+)?beklagte(?:n|r)?\s+(?:wendet\s+ein|bringt\s+vor|bestritt|beantragt)"
            r"|vorbringen\s+(?:des\s+)?beklagten?|beklagten?-?vorbringen)",
        ],
        "feststellungen": [
            r"(?i)((?:das\s+erstgericht\s+)?(?:stellte?\s+(?:folgendes?\s+)?fest|hat\s+folgendes?\s+festgestellt)"
            r"|(?:als\s+)?feststellungen?\b|sachverhalt(?:sfeststellung)?)",
        ],
        "beweisw_rdigung": [
            r"(?i)(beweiswürdigung\b|würdigung\s+der\s+beweise"
            r"|das\s+(?:erstgericht|gericht)\s+(?:würdigte?|stütze?(?:te?)?)\s+(?:diese\s+)?feststellungen?)",
        ],
        "erstgericht_entscheidung_text": [
            r"(?i)(das\s+erstgericht\s+(?:gab|wies|erachtete|erkannte|sprach)"
            r"|ersturteil\b|das\s+(?:bezirksgericht|landesgericht)\s+(?:als\s+erstgericht\s+)?(?:gab|wies|erachtete|erkannte)"
            r"|(?:das|dem)\s+(?:BG|LG)\s+(?:als\s+erstgericht|wien|graz|linz|salzburg|innsbruck|klagenfurt))",
        ],
        "ogh_entscheidung": [
            r"(?i)(der\s+(?:erkennende\s+)?senat\s+hat\s+(?:hiezu\s+)?erwogen"
            r"|rechtliche\s+beurteilung\b|der\s+ogh\s+hat\s+erwogen"
            r"|die\s+revision\s+ist\s+(?:nicht\s+)?(?:zulässig|berechtigt))",
        ],
        "spruch": [
            r"(?i)(dem\s+(?:außerordentlichen\s+)?revisionsrekurs\s+(?:wird\s+)?(?:folge\s+gegeben|nicht\s+folge\s+gegeben)"
            r"|der\s+revision\s+(?:wird\s+)?(?:folge\s+gegeben|nicht\s+folge\s+gegeben)"
            r"|im\s+namen\s+der\s+republik\b|spruch\b)",
        ],
    }

    section_positions: list[tuple[int, str]] = []
    for section_key, pat_list in patterns.items():
        for pat in pat_list:
            for match in re.finditer(pat, text):
                section_positions.append((match.start(), section_key))
                break

    section_positions.sort(key=lambda x: x[0])

    for i, (pos, key) in enumerate(section_positions):
        end = section_positions[i + 1][0] if i + 1 < len(section_positions) else len(text)
        header_end = text.find("\n", pos)
        if header_end == -1:
            header_end = pos
        sections[key] = text[header_end + 1:end].strip()

    return sections


def get_txt_metadata(txt_path: str | Path) -> dict:
    """Gibt Datei-Metadaten zurück."""
    txt_path = Path(txt_path)
    try:
        size = txt_path.stat().st_size
        return {"filename": txt_path.name, "size_bytes": size}
    except Exception:
        return {"filename": txt_path.name, "size_bytes": 0}


def get_text_stats(text: str) -> dict:
    """Gibt einfache Textstatistiken zurück."""
    words = text.split()
    return {
        "char_count": len(text),
        "word_count": len(words),
        "line_count": text.count("\n"),
    }


def process_txt(txt_path: str | Path) -> dict:
    """
    Vollständige Pipeline: Text lesen + Abschnitte + Metadaten.

    Raises:
        TxtProcessingError: Datei nicht lesbar oder zu kurz.
        StrafurteilError:   Text wurde als Strafurteil erkannt.
    """
    txt_path = Path(txt_path)
    meta = get_txt_metadata(txt_path)
    text = read_txt_file(txt_path)
    sections = split_into_sections(text)
    stats = get_text_stats(text)
    return {
        "metadata": meta,
        "text": text,
        "sections": sections,
        "stats": stats,
    }
