"""
Legal Analyzer: Juristische Fallbeurteilung via LLM mit österreichischem Rechtssuch.

Nutzt OpenAI's Web-Search-Modell, um primär auf ris.bka.gv.at und anderen
österreichischen Rechtsquellen zu suchen. Liefert eine strukturierte Einschätzung
der Erfolgsaussichten als juristisches Komplement zur ML-Vorhersage.

Quellen (bevorzugt):
  - https://ris.bka.gv.at  (Gesetze, OGH-Judikatur)
  - https://ogh.gv.at       (OGH-Entscheidungen)
"""

import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

import openai

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DEFENSE_LABELS,
    DEFENSE_TYPES,
    LEGAL_ANALYSIS_MODEL,
    OPENAI_MAX_RETRIES,
    OPENAI_RETRY_DELAY_SEC,
)

# ─── System Prompt ────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """Du bist ein erfahrener österreichischer Rechtsanwalt und Spezialist \
für österreichisches Zivilrecht (ABGB, ZPO, UGB, KSchG). Deine Aufgabe ist eine \
juristische Fallbeurteilung hinsichtlich der Erfolgsaussichten eines Zivilverfahrens \
in erster Instanz aus Sicht des Klägers.

ARBEITSWEISE:
1. Analysiere den Sachverhalt juristisch sorgfältig unter Anwendung österreichischen Rechts.
2. Suche vorrangig auf ris.bka.gv.at nach einschlägigen Gesetzestexten und OGH-Judikatur.
3. Berücksichtige aktuelle OGH-Rechtsprechung und Gesetzestexte (ABGB, ZPO, UGB etc.).
4. Gib eine fundierte Einschätzung der Erfolgsaussichten aus Kläger-Perspektive.

QUELLEN (bevorzugt, in dieser Reihenfolge):
1. ris.bka.gv.at — Rechtsinformationssystem des Bundes (Gesetze + OGH-Judikatur)
2. ogh.gv.at — OGH-Entscheidungsdatenbank
3. Sonstige österreichische Rechtsquellen

AUSGABEFORMAT:
Antworte ausschließlich mit einem validen JSON-Objekt (kein Markdown-Block):
{
  "analyse": "Detaillierte juristische Analyse (mindestens 400 Wörter, auf Deutsch)",
  "staerken_klaeger": ["Stärke 1", "Stärke 2"],
  "schwaechen_klaeger": ["Schwäche 1", "Schwäche 2"],
  "relevante_normen": ["§ 918 ABGB (Rücktritt bei Verzug)", "§ 1295 ABGB (Schadenersatz)"],
  "relevante_judikatur": ["OGH 1 Ob 123/24g (Thema)", "OGH 3 Ob 45/23f (Thema)"],
  "erfolgseinschaetzung": 0.65,
  "einschaetzung_begruendung": "Kurze Begründung der Erfolgswahrscheinlichkeit",
  "konfidenz": "hoch"
}

WICHTIG:
- erfolgseinschaetzung: float 0.0–1.0 (0 = sicher Niederlage, 1 = sicher Sieg) aus KLÄGER-Sicht
- konfidenz: "hoch" | "mittel" | "niedrig" (Qualität der juristischen Einschätzung)
- Ziel ist die Vorhersage des ERSTGERICHTLICHEN Ausgangs (nicht OGH/Berufung)
- Gib nur das JSON zurück, keine weiteren Texte."""


# ─── Prompt Builder ───────────────────────────────────────────────────────────

def _build_prompt(case_dict: dict) -> str:
    """Erstellt den User-Prompt aus den Falldaten."""
    lines = ["# Fallbeschreibung für juristische Analyse\n"]

    if case_dict.get("streitwert_eur"):
        lines.append(f"**Streitwert:** EUR {case_dict['streitwert_eur']:,.2f}")

    if case_dict.get("anspruchsart"):
        lines.append(f"**Anspruchsart:** {case_dict['anspruchsart']}")

    if case_dict.get("instanz"):
        lines.append(f"**Gericht (Erstinstanz):** {case_dict['instanz']}")

    gruende = case_dict.get("anspruchsgruende", [])
    if gruende:
        if isinstance(gruende, list):
            lines.append(f"**Rechtliche Grundlagen:** {', '.join(str(g) for g in gruende if g)}")
        else:
            lines.append(f"**Rechtliche Grundlagen:** {gruende}")

    # Einwendungen
    active_defenses = []
    for dk in DEFENSE_TYPES:
        if case_dict.get(dk) or (
            isinstance(case_dict.get("einwendungen"), dict)
            and case_dict["einwendungen"].get(dk)
        ):
            active_defenses.append(DEFENSE_LABELS.get(dk, dk))
    if active_defenses:
        lines.append(f"**Einwendungen des Beklagten:** {', '.join(active_defenses)}")

    # Beweise
    klaeger_bm = case_dict.get("klaeger_beweismittel", [])
    beklagter_bm = case_dict.get("beklagter_beweismittel", [])
    if klaeger_bm:
        n = len(klaeger_bm) if isinstance(klaeger_bm, list) else klaeger_bm
        lines.append(f"**Kläger-Beweismittel:** {n}")
    if beklagter_bm:
        n = len(beklagter_bm) if isinstance(beklagter_bm, list) else beklagter_bm
        lines.append(f"**Beklagten-Beweismittel:** {n}")
    if case_dict.get("sachverstaendiger_bestellt"):
        fachgeb = case_dict.get("sachverstaendigen_fachgebiet", "")
        lines.append(f"**Sachverständiger bestellt:** Ja{f' ({fachgeb})' if fachgeb else ''}")

    # Textvorbringen (für detaillierte Analyse)
    klaeger_text = case_dict.get("klaegervorbringen", "")
    beklagter_text = case_dict.get("beklagtenvorbringen", "")
    beweise_text = case_dict.get("aufgenommene_beweise", "")

    if klaeger_text:
        lines.append(f"\n## Kläger-Vorbringen\n{klaeger_text}")
    if beklagter_text:
        lines.append(f"\n## Beklagten-Vorbringen / Einwendungen\n{beklagter_text}")
    if beweise_text:
        lines.append(f"\n## Aufgenommene Beweise\n{beweise_text}")

    lines.append(
        "\n---\n"
        "Führe eine vollständige juristische Analyse durch. "
        "Suche nach einschlägiger österreichischer Rechtsprechung auf ris.bka.gv.at. "
        "Antworte ausschließlich mit dem JSON-Objekt."
    )
    return "\n".join(lines)


# ─── Response Parser ─────────────────────────────────────────────────────────

def _parse_response(text: str) -> dict:
    """Parst das JSON aus der LLM-Antwort; Fallback auf Rohtext."""
    # Strip markdown code blocks if present
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip()
    cleaned = re.sub(r"```\s*$", "", cleaned).strip()

    # Try full parse
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Find first { ... } block
        m = re.search(r"\{[\s\S]*\}", cleaned)
        if m:
            try:
                data = json.loads(m.group())
            except json.JSONDecodeError:
                data = {}
        else:
            data = {}

    # Normalise and validate
    result = {
        "analyse": data.get("analyse", text),
        "staerken_klaeger": data.get("staerken_klaeger", []),
        "schwaechen_klaeger": data.get("schwaechen_klaeger", []),
        "relevante_normen": data.get("relevante_normen", []),
        "relevante_judikatur": data.get("relevante_judikatur", []),
        "erfolgseinschaetzung": float(data.get("erfolgseinschaetzung", 0.5)),
        "einschaetzung_begruendung": data.get("einschaetzung_begruendung", ""),
        "konfidenz": data.get("konfidenz", "mittel"),
    }
    result["erfolgseinschaetzung"] = max(0.0, min(1.0, result["erfolgseinschaetzung"]))
    return result


# ─── LegalAnalyzer ───────────────────────────────────────────────────────────

class LegalAnalyzer:
    """
    Juristische Fallanalyse via LLM mit österreichischer Rechtsweb-Suche.

    Nutzt ein Search-fähiges GPT-Modell, sucht bevorzugt auf ris.bka.gv.at
    und gibt eine strukturierte Einschätzung der Erfolgsaussichten zurück.

    Returns (dict):
        analyse               : Juristische Analyse (Volltext)
        staerken_klaeger      : Liste juristischer Stärken
        schwaechen_klaeger    : Liste juristischer Schwächen
        relevante_normen      : Einschlägige Gesetzesnormen
        relevante_judikatur   : Einschlägige OGH-Entscheidungen
        erfolgseinschaetzung  : float 0.0–1.0 (Erfolgswahrscheinlichkeit Kläger)
        einschaetzung_begruendung : Kurze Begründung
        konfidenz             : "hoch" | "mittel" | "niedrig"
        sources               : Web-Quellen (falls verfügbar)
        error                 : Fehlermeldung oder None
    """

    def __init__(self, api_key: Optional[str] = None):
        self.client = openai.OpenAI(api_key=api_key) if api_key else openai.OpenAI()
        self.model = LEGAL_ANALYSIS_MODEL

    def analyze(
        self,
        case_dict: dict,
        progress_callback=None,
    ) -> dict:
        """
        Führt die juristische Analyse durch.

        Args:
            case_dict: Falldaten (strukturierte Felder + ggf. Textvorbringen)
            progress_callback: Optionale Funktion für Fortschritts-Updates

        Returns:
            Strukturiertes Analyseergebnis (dict)
        """
        prompt = _build_prompt(case_dict)

        if progress_callback:
            progress_callback(
                f"Juristische Analyse läuft (Modell: {self.model}, Web-Suche aktiv)…"
            )

        last_error = None
        for attempt in range(OPENAI_MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    # web_search_preview wird automatisch vom Modell genutzt
                )

                raw_text = response.choices[0].message.content or ""
                result = _parse_response(raw_text)
                result["sources"] = []
                result["error"] = None
                result["raw_response"] = raw_text
                result["model_used"] = self.model

                # Extrahiere URL-Annotations falls vorhanden (neuere SDK-Versionen)
                try:
                    annotations = response.choices[0].message.annotations or []
                    for ann in annotations:
                        url = getattr(ann, "url", None) or getattr(
                            getattr(ann, "url_citation", None), "url", None
                        )
                        title = getattr(ann, "title", None) or getattr(
                            getattr(ann, "url_citation", None), "title", url
                        )
                        if url:
                            result["sources"].append({"url": url, "title": title or url})
                except Exception:
                    pass

                return result

            except openai.RateLimitError as e:
                last_error = e
                if attempt < OPENAI_MAX_RETRIES - 1:
                    time.sleep(OPENAI_RETRY_DELAY_SEC * (2 ** attempt))
            except Exception as e:
                last_error = e
                if attempt < OPENAI_MAX_RETRIES - 1:
                    time.sleep(OPENAI_RETRY_DELAY_SEC)
                else:
                    break

        return {
            "analyse": "",
            "staerken_klaeger": [],
            "schwaechen_klaeger": [],
            "relevante_normen": [],
            "relevante_judikatur": [],
            "erfolgseinschaetzung": 0.5,
            "einschaetzung_begruendung": "Analyse fehlgeschlagen.",
            "konfidenz": "niedrig",
            "sources": [],
            "error": str(last_error),
            "raw_response": "",
            "model_used": self.model,
        }
