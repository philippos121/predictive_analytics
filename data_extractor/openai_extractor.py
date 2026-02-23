"""
OpenAI Extractor: Uses GPT-4o-mini and text-embedding-3-large to extract
structured legal data and embeddings from Austrian civil judgment text.
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CLAIM_TYPES,
    DEFENSE_TYPES,
    EMBEDDING_DIM,
    EMBEDDING_SECTIONS,
    OPENAI_EMBEDDING_MODEL,
    OPENAI_EXTRACTION_MODEL,
    OPENAI_REQUEST_DELAY_SEC,
)


# ─── Extraction Prompt ──────────────────────────────────────────────────────────

EXTRACTION_SYSTEM_PROMPT = """Du bist ein Experte für österreichisches Zivilrecht.
Deine Aufgabe ist es, aus Texten österreichischer Zivilurteile präzise strukturierte
Daten zu extrahieren. Antworte ausschließlich mit validem JSON ohne jeglichen anderen Text.

Wichtige Regeln:
- Extrahiere KEINE Namen von Richtern, Parteien oder Anwälten
- Fokussiere auf anspruchsrelevante, materiell-rechtliche Inhalte
- Bei fehlenden Informationen: null für Felder, [] für Listen, false für Boolean
- Outcome: Beziehe dich auf den Ausgang aus Sicht des KLÄGERS
"""

EXTRACTION_USER_PROMPT = """Analysiere dieses österreichische Zivilurteil und extrahiere die folgenden Informationen als JSON:

{{
  "datum": "YYYY-MM-DD oder null",
  "gericht": "z.B. BG Wien, LG Salzburg oder null (keine richternamen)",
  "instanz": "BG" oder "LG" oder "OLG" oder "OGH" oder null,
  "streitwert_eur": Zahl als float oder null,
  "streitwert_unbekannt": boolean,

  "anspruchsart": "Eine der folgenden: {claim_types_str} oder 'Andere'",
  "anspruchsgruende": ["Liste der Rechtsgrundlagen, z.B. § 1295 ABGB, § 922 ABGB"],

  "klaeger_anspruch_zusammenfassung": "Kurze sachliche Zusammenfassung was der Kläger begehrt (max 200 Wörter, KEINE Namen)",
  "beklagter_vorbringen_zusammenfassung": "Kurze sachliche Zusammenfassung der Einwendungen (max 200 Wörter, KEINE Namen)",

  "einwendungen": {{
    "mangel": boolean,
    "irrtum": boolean,
    "nichterfuellung": boolean,
    "verjaehrung": boolean,
    "aufrechnung": boolean,
    "listige_irrefuehrung": boolean,
    "unmoeglichkeit": boolean,
    "unzustaendigkeit": boolean,
    "fehlende_aktivlegitimation": boolean,
    "keine_passivlegitimation": boolean,
    "zahlung_erfolgt": boolean,
    "andere": boolean,
    "andere_beschreibung": "Beschreibung sonstiger Einwendungen oder null"
  }},

  "klaeger_beweismittel": ["Liste der Beweismittel des Klägers, z.B. Urkunden, Zeugen, Sachverständige"],
  "beklagter_beweismittel": ["Liste der Beweismittel des Beklagten"],
  "sachverstaendiger_bestellt": boolean,
  "sachverstaendigen_fachgebiet": "z.B. Bautechnik, Medizin oder null",

  "verfahrensdauer_monate": Zahl oder null,
  "anzahl_verhandlungen": Zahl oder null,

  "outcome": 0 oder 1 oder 2,
  "outcome_beschreibung": "Kurze Beschreibung des Urteilsergebnisses (KEINE Namen)",
  "zugesprochener_betrag_eur": float oder null,
  "zugesprochener_anteil_prozent": float zwischen 0 und 100 oder null,

  "kostenentscheidung": "Kläger" oder "Beklagter" oder "Geteilt" oder null,

  "besonderheiten": ["Besondere rechtliche oder sachliche Besonderheiten des Falles"]
}}

OUTCOME KODIERUNG:
- 0 = Kläger UNTERLIEGT vollständig (Klage abgewiesen)
- 1 = TEILWEISES Obsiegen/Unterliegen (Klage teilweise zugesprochen)
- 2 = Kläger OBSIEGT vollständig (Klage vollständig zugesprochen)

URTEILSTEXT:
{text}"""


# ─── Text Section Extraction Prompt ─────────────────────────────────────────────

SECTION_EXTRACTION_PROMPT = """Extrahiere aus diesem österreichischen Zivilurteil die folgenden Textabschnitte.
Gib das Ergebnis als JSON zurück. Wenn ein Abschnitt nicht vorhanden ist, gib einen leeren String zurück.
Entferne alle Namen von Personen (Richter, Parteien, Anwälte) - ersetze sie mit [KLÄGER], [BEKLAGTER], [RICHTER], [ANWALT].

{{
  "klaegervorbringen": "Vollständiger Text des Kläger-Vorbringens (anonymisiert)",
  "beklagtenvorbringen": "Vollständiger Text des Beklagten-Vorbringens (anonymisiert)",
  "feststellungen": "Vollständiger Text der Sachverhaltsfeststellungen (anonymisiert)",
  "beweisw_rdigung": "Vollständiger Text der Beweiswürdigung (anonymisiert)"
}}

URTEILSTEXT:
{text}"""


# ─── Evidence Description Generation Prompt ──────────────────────────────────────
# Generiert eine faktische Beschreibung der aufgenommenen (!) Beweise aus
# Beweiswürdigung + Feststellungen — ohne eigene Bewertung.
# Zweck: Dieses Embedding kodiert welche Beweismittel das Gericht tatsächlich
# aufgenommen hat, als neutralen Input für die Outcome-Prognose.

EVIDENCE_DESCRIPTION_PROMPT = """Du bist ein österreichischer Zivilrechtsspezialist.
Beschreibe auf Basis der folgenden Textabschnitte (Beweiswürdigung und Feststellungen)
ausschließlich faktisch und ohne eigene Bewertung, welche Beweise das Gericht
tatsächlich aufgenommen hat.

Gib ausschließlich ein JSON-Objekt zurück:
{{
  "aufgenommene_beweise": "Sachliche Beschreibung der aufgenommenen Beweise"
}}

Regeln:
- Beschreibe Art und Anzahl der Beweismittel (z.B. 'drei Zeugenvernehmungen',
  'zwei Urkunden', 'ein Sachverständigengutachten')
- Gib an, welche Partei welches Beweismittel beigebracht hat
- Gib an, ob Beweismittel das Vorbringen einer Partei stützen oder widerlegen
  (nur soweit aus Beweiswürdigung/Feststellungen eindeutig hervorgeht)
- KEINE eigene rechtliche Würdigung, KEINE Schlussfolgerungen
- KEINE Namen von Personen — ersetze sie mit [KLÄGER], [BEKLAGTER], [ZEUGE_1] usw.
- Wenn keine Beweise aufgenommen wurden, schreibe 'Keine Beweise aufgenommen.'
- Maximale Länge: 400 Wörter

BEWEISWÜRDIGUNG:
{beweisw_rdigung}

FESTSTELLUNGEN:
{feststellungen}"""


class OpenAIExtractor:
    """
    Handles all OpenAI API interactions for legal text extraction and embedding.
    """

    def __init__(self, api_key: str, progress_callback: Optional[Callable] = None):
        self.client = openai.OpenAI(api_key=api_key)
        self.progress_callback = progress_callback or (lambda msg, pct: None)

    def _log(self, msg: str, pct: float = 0.0):
        self.progress_callback(msg, pct)

    @retry(
        retry=retry_if_exception_type((openai.RateLimitError, openai.APIConnectionError)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(5),
    )
    def _chat_completion(self, messages: list[dict], temperature: float = 0.0) -> str:
        """Call GPT-4o-mini with retry logic."""
        response = self.client.chat.completions.create(
            model=OPENAI_EXTRACTION_MODEL,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content

    @retry(
        retry=retry_if_exception_type((openai.RateLimitError, openai.APIConnectionError)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(5),
    )
    def _get_embedding(self, text: str) -> list[float]:
        """Get embedding vector from text-embedding-3-large."""
        if not text or not text.strip():
            return [0.0] * EMBEDDING_DIM

        # Truncate text to avoid token limits (approx 8191 tokens max)
        text = text[:32000]  # ~8000 tokens at ~4 chars/token

        response = self.client.embeddings.create(
            model=OPENAI_EMBEDDING_MODEL,
            input=text,
            dimensions=EMBEDDING_DIM,
        )
        return response.data[0].embedding

    def extract_structured_data(self, text: str) -> dict[str, Any]:
        """
        Extract structured legal data from judgment text using GPT-4o-mini.
        Returns parsed JSON dict with case metadata and outcome.
        """
        self._log("Extrahiere strukturierte Daten via GPT-4o-mini...", 0.2)

        claim_types_str = ", ".join(f'"{c}"' for c in CLAIM_TYPES)

        # Truncate text for extraction (keep first 15000 chars = most relevant)
        truncated_text = text[:15000]
        if len(text) > 15000:
            truncated_text += f"\n\n[... Text gekürzt, Gesamtlänge: {len(text)} Zeichen]"

        prompt = EXTRACTION_USER_PROMPT.format(
            claim_types_str=claim_types_str,
            text=truncated_text,
        )

        messages = [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        raw = self._chat_completion(messages)
        time.sleep(OPENAI_REQUEST_DELAY_SEC)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"Ungültige JSON-Antwort von OpenAI: {e}\nRaw: {raw[:500]}")

        return self._validate_and_normalize(data)

    def extract_text_sections(self, text: str) -> dict[str, str]:
        """
        Extract and anonymize text sections, then generate the evidence description.

        Pipeline:
        1. GPT extrahiert klaegervorbringen, beklagtenvorbringen, feststellungen,
           beweisw_rdigung aus dem Urteilstext (anonymisiert).
        2. GPT generiert aufgenommene_beweise — eine faktische, wertungsfreie
           Beschreibung der tatsächlich aufgenommenen Beweise — aus feststellungen
           und beweisw_rdigung.

        Rückgabe enthält alle Zwischenabschnitte sowie aufgenommene_beweise.
        generate_embeddings() verwendet nur die drei Abschnitte aus EMBEDDING_SECTIONS.
        """
        self._log("Extrahiere Textabschnitte aus Urteil...", 0.4)

        truncated_text = text[:20000]
        prompt = SECTION_EXTRACTION_PROMPT.format(text=truncated_text)

        messages = [
            {
                "role": "system",
                "content": "Du extrahierst Textabschnitte aus Gerichtsurteilen und gibst JSON zurück."
                " Anonymisiere alle Personennamen.",
            },
            {"role": "user", "content": prompt},
        ]

        raw = self._chat_completion(messages)
        time.sleep(OPENAI_REQUEST_DELAY_SEC)

        try:
            raw_sections = json.loads(raw)
        except json.JSONDecodeError:
            raw_sections = {}

        klaegervorbringen = raw_sections.get("klaegervorbringen", "")
        beklagtenvorbringen = raw_sections.get("beklagtenvorbringen", "")
        feststellungen = raw_sections.get("feststellungen", "")
        beweisw_rdigung = raw_sections.get("beweisw_rdigung", "")

        # Step 2: Faktische Beweis-Beschreibung aus Beweiswürdigung + Feststellungen
        self._log("Generiere Beweis-Beschreibung (aufgenommene Beweise)...", 0.5)
        aufgenommene_beweise = self._generate_evidence_description(
            feststellungen=feststellungen,
            beweisw_rdigung=beweisw_rdigung,
        )

        return {
            "klaegervorbringen": klaegervorbringen,
            "beklagtenvorbringen": beklagtenvorbringen,
            "feststellungen": feststellungen,
            "beweisw_rdigung": beweisw_rdigung,
            "aufgenommene_beweise": aufgenommene_beweise,
        }

    def _generate_evidence_description(
        self,
        feststellungen: str,
        beweisw_rdigung: str,
    ) -> str:
        """
        Generiert eine faktische, wertungsfreie Beschreibung der aufgenommenen
        Beweise aus den Abschnitten Beweiswürdigung und Feststellungen.
        """
        if not feststellungen.strip() and not beweisw_rdigung.strip():
            return "Keine Beweise aufgenommen."

        prompt = EVIDENCE_DESCRIPTION_PROMPT.format(
            feststellungen=feststellungen[:8000],
            beweisw_rdigung=beweisw_rdigung[:8000],
        )

        messages = [
            {
                "role": "system",
                "content": "Du bist ein österreichischer Zivilrechtsspezialist. "
                "Gib ausschließlich JSON zurück.",
            },
            {"role": "user", "content": prompt},
        ]

        raw = self._chat_completion(messages)
        time.sleep(OPENAI_REQUEST_DELAY_SEC)

        try:
            result = json.loads(raw)
            return result.get("aufgenommene_beweise", "Keine Beweise aufgenommen.")
        except json.JSONDecodeError:
            return "Keine Beweise aufgenommen."

    def generate_embeddings(self, sections: dict[str, str]) -> dict[str, list[float]]:
        """
        Generate embedding vectors for each text section.
        Returns dict mapping section name to embedding vector.
        """
        embeddings = {}
        total = len(EMBEDDING_SECTIONS)

        for i, section_key in enumerate(EMBEDDING_SECTIONS):
            self._log(
                f"Generiere Embedding für: {section_key} ({i+1}/{total})...",
                0.6 + (i / total) * 0.35,
            )
            text = sections.get(section_key, "")
            embeddings[section_key] = self._get_embedding(text)
            time.sleep(OPENAI_REQUEST_DELAY_SEC)

        return embeddings

    def process_judgment(self, text: str) -> dict[str, Any]:
        """
        Full extraction pipeline for a single judgment:
        1. Extract structured data
        2. Extract text sections
        3. Generate embeddings

        Returns complete case dict ready for dataset storage.
        """
        self._log("Starte Verarbeitung...", 0.1)

        # Step 1: Structured data
        structured = self.extract_structured_data(text)
        self._log("Strukturierte Daten extrahiert.", 0.35)

        # Step 2: Text sections
        sections = self.extract_text_sections(text)
        self._log("Textabschnitte extrahiert.", 0.55)

        # Step 3: Embeddings
        embeddings = self.generate_embeddings(sections)
        self._log("Embeddings generiert.", 0.95)

        return {
            "structured": structured,
            "sections": sections,
            "embeddings": embeddings,
        }

    def embed_new_case_text(self, case_sections: dict[str, str]) -> dict[str, list[float]]:
        """
        Embed a new case's text sections for prediction.
        case_sections must contain: klaegervorbringen, beklagtenvorbringen,
        aufgenommene_beweise.
        Used when applying the model to new cases.
        """
        return self.generate_embeddings(case_sections)

    def _validate_and_normalize(self, data: dict) -> dict:
        """Validate and normalize extracted structured data."""
        # Ensure required fields
        defaults = {
            "datum": None,
            "gericht": None,
            "instanz": None,
            "streitwert_eur": None,
            "streitwert_unbekannt": False,
            "anspruchsart": "Andere",
            "anspruchsgruende": [],
            "klaeger_anspruch_zusammenfassung": "",
            "beklagter_vorbringen_zusammenfassung": "",
            "einwendungen": {d: False for d in DEFENSE_TYPES},
            "klaeger_beweismittel": [],
            "beklagter_beweismittel": [],
            "sachverstaendiger_bestellt": False,
            "sachverstaendigen_fachgebiet": None,
            "verfahrensdauer_monate": None,
            "anzahl_verhandlungen": None,
            "outcome": None,
            "outcome_beschreibung": "",
            "zugesprochener_betrag_eur": None,
            "zugesprochener_anteil_prozent": None,
            "kostenentscheidung": None,
            "besonderheiten": [],
        }

        for key, default in defaults.items():
            if key not in data:
                data[key] = default

        # Validate outcome
        if data["outcome"] not in (0, 1, 2):
            data["outcome"] = None

        # Ensure einwendungen has all defense keys
        if isinstance(data.get("einwendungen"), dict):
            for d in DEFENSE_TYPES:
                if d not in data["einwendungen"]:
                    data["einwendungen"][d] = False
        else:
            data["einwendungen"] = {d: False for d in DEFENSE_TYPES}

        # Validate streitwert
        if data["streitwert_eur"] is not None:
            try:
                data["streitwert_eur"] = float(data["streitwert_eur"])
            except (TypeError, ValueError):
                data["streitwert_eur"] = None

        return data
