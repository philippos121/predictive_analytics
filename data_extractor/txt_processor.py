"""
TXT Processor: Liest und verarbeitet deutsche Zivilurteile als Plaintext-Dateien.

Ersetzt den PDF-Prozessor. Die Eingangsurteile liegen als .txt-Dateien vor
(Amtsgerichte und Landgerichte). Strafurteile werden automatisch herausgefiltert —
nur Zivilurteile werden an die Extraktionspipeline weitergeleitet.
"""

import re
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import EMBEDDING_SECTIONS


class TxtProcessingError(Exception):
    pass


class StrafurteilError(Exception):
    """Wird ausgelöst, wenn eine Datei als Strafurteil erkannt wird."""
    pass


# ─── Strafurteil-Erkennungsmuster ────────────────────────────────────────────────
# Starke Indikatoren für Strafurteile — diese Begriffe erscheinen in
# Zivilurteilen praktisch nie.
_CRIMINAL_STRONG = [
    r"\bStGB\b",                    # Strafgesetzbuch
    r"\bStPO\b",                    # Strafprozessordnung
    r"\bAngeklagte[nrm]?\b",        # Angeklagter (strafprozessuale Partei)
    r"\bFreiheitsstrafe\b",         # Freiheitsstrafe
    r"\bBewährungsstrafe\b",        # Bewährungsstrafe
    r"\bBewährung\b",               # Bewährung
    r"\bStrafkammer\b",             # Strafkammer
    r"\bSchöffengericht\b",         # Schöffengericht
    r"\bJugendgericht\b",           # Jugendgericht
    r"\bStaatsanwalt(?:schaft)?\b", # Staatsanwaltschaft / -anwalt
    r"\bfreigesprochen\b",          # Freispruch
    r"\bverurteilt\s+(?:zu|wegen)", # "verurteilt zu X Jahren" / "verurteilt wegen"
    r"\bAnklageschrift\b",          # Anklageschrift
    r"\bHaftbefehl\b",              # Haftbefehl
    r"\bTagessätze?\b",             # Geldstrafe in Tagessätzen (nur Strafrecht)
    r"\bGeldstrafe\s+von\s+\d+\s+Tagessätzen\b",
    r"\b(?:§|§§)\s*\d+\s+StGB\b",  # §§ StGB-Paragraphen
    r"\bJugendstrafe\b",
    r"\bSicherungsverwahrung\b",
]

# Schwache Gegenindikation: Zivilrechtliche Begriffe
# (werden zur Bestätigung genutzt, aber nicht alleine)
_CIVIL_INDICATORS = [
    r"\bKläger(?:in)?\b",
    r"\bBeklagte[nrm]?\b",
    r"\bStreitwert\b",
    r"\bZPO\b",
    r"\bBGB\b",
    r"\bSchadensersatz\b",
    r"\bKaufpreis\b",
    r"\bMiete(?:r(?:in)?)?\b",
]

_RE_CRIMINAL = [re.compile(p, re.IGNORECASE) for p in _CRIMINAL_STRONG]
_RE_CIVIL = [re.compile(p, re.IGNORECASE) for p in _CIVIL_INDICATORS]


def detect_judgment_type(text: str) -> str:
    """
    Erkennt den Typ eines Urteils anhand von Schlüsselbegriffen.

    Gibt zurück:
        "zivil"    — Zivilurteil (zur weiteren Verarbeitung freigegeben)
        "straf"    — Strafurteil (soll herausgefiltert werden)
        "unklar"   — Nicht eindeutig klassifizierbar (wird als Zivilurteil behandelt)

    Logik:
    - Jeder starke Straf-Indikator zählt als Treffer
    - Ab 2 Treffern → sicher Strafurteil
    - 1 Treffer + keine Zivilindikation → wahrscheinlich Strafurteil
    - Sonst → Zivilurteil / unklar
    """
    # Nur die ersten 5000 Zeichen prüfen (Header des Urteils reicht)
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
    Liest eine TXT-Datei und gibt den bereinigten Text zurück.

    Probiert automatisch verschiedene Encodings (utf-8, latin-1, cp1252),
    falls utf-8 fehlschlägt.

    Raises:
        TxtProcessingError: Wenn die Datei nicht gelesen werden kann.
        StrafurteilError:   Wenn die Datei als Strafurteil erkannt wird.
    """
    txt_path = Path(txt_path)
    if not txt_path.exists():
        raise TxtProcessingError(f"Datei nicht gefunden: {txt_path}")
    if txt_path.suffix.lower() not in (".txt",):
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

    if len(text.strip()) < 200:
        raise TxtProcessingError(
            f"Text zu kurz ({len(text)} Zeichen) — möglicherweise leere oder beschädigte Datei: {txt_path.name}"
        )

    # Strafurteil-Filter
    judgment_type = detect_judgment_type(text)
    if judgment_type == "straf":
        raise StrafurteilError(
            f"Strafurteil erkannt und herausgefiltert: {txt_path.name}"
        )

    return text


def _clean_text(text: str) -> str:
    """Normalisiert Whitespace und bereinigt typische Textartefakte."""
    # Normalisiere Zeilenenden
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Maximal 2 aufeinanderfolgende Leerzeilen
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Trennstriche am Zeilenende (Silbentrennung) zusammenfügen
    text = re.sub(r"([a-zäöüßA-ZÄÖÜ])-\n([a-zäöüßA-ZÄÖÜ])", r"\1\2", text)
    # Mehrfache Leerzeichen normalisieren
    text = re.sub(r"[ \t]+", " ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    return text.strip()


def split_into_sections(text: str) -> dict[str, str]:
    """
    Teilt einen deutschen Zivilurteilstext in Standardabschnitte auf.

    Deutsche Zivilurteile (AG/LG) haben typischerweise:
    - Rubrum / Tatbestand (Parteien, Antrag)
    - Tatbestand / Sachverhalt
    - Entscheidungsgründe (enthält Beweiswürdigung + rechtliche Beurteilung)
    - Tenor / Urteilsformel

    Diese Funktion extrahiert die für das Modell relevanten Abschnitte.
    """
    sections = {s: "" for s in EMBEDDING_SECTIONS}
    sections["full_text"] = text
    sections["tenor"] = ""

    patterns = {
        "klaegervorbringen": [
            r"(?i)(vorbringen\s+(?:der\s+)?kläger(?:in)?|kläger(?:in)?\s+(?:trägt\s+vor|bringt\s+vor|behauptet)"
            r"|klagevorbringen|tatbestand\b)",
        ],
        "beklagtenvorbringen": [
            r"(?i)(vorbringen\s+(?:der\s+)?beklagten?|beklagte(?:r|n)?\s+(?:trägt\s+vor|bringt\s+vor|bestreitet)"
            r"|einwendungen\s+des\s+beklagten?)",
        ],
        "feststellungen": [
            r"(?i)((?:zur\s+)?sache(?:\s+selbst)?:|feststellungen\b|sachverhalt(?:sfeststellung)?"
            r"|als\s+(?:erwiesen|unstreitig)\s+(?:gilt|steht\s+fest)|unstreitiger\s+sachverhalt)",
        ],
        "beweisw_rdigung": [
            r"(?i)(beweiswürdigung\b|würdigung\s+der\s+beweise"
            r"|(?:das\s+gericht\s+)?hat\s+(?:folgende\s+)?beweis(?:e\s+erhoben|aufnahme)"
            r"|entscheidungsgründe\b)",
        ],
        "rechtliche_beurteilung": [
            r"(?i)(rechtliche?\s+(?:beurteilung|würdigung|ausführungen)"
            r"|in\s+rechtlicher\s+hinsicht|die\s+klage\s+ist\s+(?:be|un)gründet)",
        ],
        "tenor": [
            r"(?i)(im\s+namen\s+des\s+volkes|urteil\s*:|tenor\b|wird\s+(?:für\s+recht\s+erkannt|erkannt))",
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
        content_start = header_end + 1
        sections[key] = text[content_start:end].strip()

    return sections


def get_txt_metadata(txt_path: str | Path) -> dict:
    """Gibt Metadaten einer TXT-Datei zurück."""
    txt_path = Path(txt_path)
    try:
        size = txt_path.stat().st_size
        return {
            "filename": txt_path.name,
            "size_bytes": size,
        }
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
    Vollständige Pipeline: Text lesen + Abschnitte erkennen + Metadaten.

    Raises:
        TxtProcessingError: Bei Lesefehler.
        StrafurteilError:   Wenn das Urteil als Strafurteil klassifiziert wird.
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
