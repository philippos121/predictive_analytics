"""
OpenAI Extractor: Uses GPT-4o-mini and text-embedding-3-large to extract
structured legal data and embeddings from Austrian OGH civil judgments.

Österreichisches Zivilrecht (ABGB/ZPO). Eingabe: OGH-Urteile (TXT).
Der OGH legt den vollständigen Verfahrensgang offen — damit sind aus einem
einzigen OGH-Urteil extrahierbar:
  - Vorbringen des Klägers (beim Erstgericht)
  - Vorbringen des Beklagten (beim Erstgericht)
  - Entscheidung des ERSTGERICHTS (Trainings-Label)

Entscheidungen des Berufungsgerichts (OLG) und des OGH selbst werden
als Label NICHT verwendet — Ziel ist die Vorhersage erstinstanzlicher Ergebnisse.
"""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    BEWEISMITTEL_TYPEN,
    CLAIM_TYPES,
    DEFENSE_TYPES,
    EMBEDDING_DIM,
    EMBEDDING_SECTIONS,
    OPENAI_EMBEDDING_MODEL,
    OPENAI_EXTRACTION_MODEL,
    OPENAI_REQUEST_DELAY_SEC,
    VERFAHRENSARTEN,
)


# ─── Extraction Prompt ──────────────────────────────────────────────────────────

EXTRACTION_SYSTEM_PROMPT = """Du bist ein Experte für österreichisches Zivilrecht (ABGB, ZPO).
Deine Aufgabe ist es, aus OGH-Urteilen (Oberster Gerichtshof) präzise strukturierte
Daten zu extrahieren. Antworte ausschließlich mit validem JSON ohne jeglichen anderen Text.

WICHTIG — Verfahrensgang im OGH-Urteil:
OGH-Urteile enthalten Entscheidungen MEHRERER Instanzen. Du musst klar unterscheiden:
  1. ERSTGERICHT (BG/LG): Die erste Instanz — deren Entscheidung ist das Trainings-Label.
  2. BERUFUNGSGERICHT (OLG): Zweite Instanz — für den Outcome NICHT relevant.
  3. OGH: Dritte Instanz (das vorliegende Urteil) — für den Outcome NICHT relevant.

Für das Feld "outcome" extrahiere AUSSCHLIESSLICH das Ergebnis des ERSTGERICHTS.

Weitere Regeln:
- Extrahiere KEINE Namen von Richtern, Parteien oder Anwälten
- Fokussiere auf anspruchsrelevante, materiell-rechtliche Inhalte (ABGB, ZPO)
- Bei fehlenden Informationen: null für Felder, [] für Listen, false für Boolean
- Outcome-Perspektive: aus Sicht des KLÄGERS beim ERSTGERICHT
"""

EXTRACTION_USER_PROMPT = """Analysiere dieses österreichische OGH-Zivilurteil und extrahiere
die folgenden Informationen als JSON.

ACHTUNG: "instanz", "gericht" und "outcome" beziehen sich auf das ERSTGERICHT —
nicht auf das Berufungsgericht oder den OGH.

{{
  "datum_ersturteil": "YYYY-MM-DD — Datum des Ersturteils oder null",
  "datum_ogh": "YYYY-MM-DD — Datum des OGH-Urteils oder null",
  "gericht": "Name des ERSTGERICHTS, z.B. BG Wien, LG Salzburg oder null (keine Richtername)",
  "instanz": "BG" oder "LG" (Instanz des ERSTGERICHTS) oder null,

  "streitwert_eur": Zahl als float oder null,
  "streitwert_unbekannt": boolean,

  "anspruchsart": "Eine der folgenden: {claim_types_str} oder 'Andere'",
  "anspruchsgruende": ["Rechtsgrundlagen aus dem Erstverfahren, z.B. § 1295 ABGB, § 922 ABGB, § 879 ABGB"],

  "klaeger_anspruch_zusammenfassung": "Was hat der Kläger beim Erstgericht begehrt? (max 200 Wörter, KEINE Namen)",
  "beklagter_vorbringen_zusammenfassung": "Welche Einwendungen hat der Beklagte beim Erstgericht erhoben? (max 200 Wörter, KEINE Namen)",

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

  "klaeger_beweismittel": ["Beweismittel des Klägers im Erstverfahren, z.B. Urkunden, Zeugen, Sachverständige"],
  "beklagter_beweismittel": ["Beweismittel des Beklagten im Erstverfahren"],

  "beweismitteltypen_klaeger": {{
    "urkunden": Anzahl angebotener Urkunden/Dokumente des Klägers (int, 0 wenn keine),
    "zeugen": Anzahl angebotener Zeugen des Klägers (int, 0 wenn keine),
    "sachverstaendige": Anzahl beantragter Sachverständige des Klägers (int, 0 wenn keine),
    "parteienvernehmung": boolean (hat Kläger Parteienvernehmung beantragt?)
  }},
  "beweismitteltypen_beklagter": {{
    "urkunden": Anzahl angebotener Urkunden/Dokumente des Beklagten (int, 0 wenn keine),
    "zeugen": Anzahl angebotener Zeugen des Beklagten (int, 0 wenn keine),
    "sachverstaendige": Anzahl beantragter Sachverständige des Beklagten (int, 0 wenn keine),
    "parteienvernehmung": boolean (hat Beklagter Parteienvernehmung beantragt?)
  }},

  "widerklage": boolean (hat der Beklagte beim Erstgericht Widerklage erhoben?),
  "verfahrensart": "Eine der folgenden: {verfahrensarten_str} oder null",

  "outcome": 0 oder 1 oder 2,
  "outcome_beschreibung": "Kurze Beschreibung der ERSTGERICHT-Entscheidung (KEINE Namen)",
  "zugesprochener_betrag_eur": float oder null,
  "zugesprochener_anteil_prozent": float zwischen 0 und 100 oder null,

  "kostenentscheidung_erstgericht": "Kläger" oder "Beklagter" oder "Geteilt" oder null,

  "besonderheiten": ["Besondere rechtliche oder sachliche Aspekte des Falls"]
}}

OUTCOME KODIERUNG (ERSTGERICHT):
- 0 = Kläger UNTERLIEGT vollständig beim Erstgericht (Klage abgewiesen)
- 1 = TEILWEISES Obsiegen/Unterliegen beim Erstgericht (Klage teilweise zugesprochen)
- 2 = Kläger OBSIEGT vollständig beim Erstgericht (Klage vollständig zugesprochen)

OGH-URTEILSTEXT:
{text}"""


# ─── Text Section Extraction Prompt ─────────────────────────────────────────────
# Extrahiert aus dem OGH-Urteil die für das ML-Modell relevanten Textabschnitte.
#
# WICHTIG: Das OGH-Urteil enthält Argumentation mehrerer Instanzen.
# Wir wollen NUR:
#   - Das Vorbringen der Parteien beim ERSTGERICHT (vor dem BG/LG)
#   - Die Sachverhaltsfeststellungen des ERSTGERICHTS
#   - Die Beweiswürdigung des ERSTGERICHTS
#   - Den Entscheidungstext des ERSTGERICHTS (für Label-Verifikation)
#
# Das Vorbringen vor dem Berufungsgericht, Revisionsgründe und OGH-Begründung
# werden NICHT als ML-Input verwendet (wären Data Leakage für das Label).

SECTION_EXTRACTION_PROMPT = """Du bist ein Experte für österreichisches Zivilrecht.
Dieses Urteil ist ein OGH-Urteil, das den gesamten Verfahrensgang enthält.
Extrahiere die folgenden Textabschnitte — sie beziehen sich ausschließlich auf
das ERSTGERICHT (BG oder LG), NICHT auf Berufungsgericht oder OGH.

Gib das Ergebnis als JSON zurück. Leerer String wenn Abschnitt nicht auffindbar.
Ersetze ALLE Personennamen durch [KLÄGER], [BEKLAGTER], [RICHTER], [ANWALT].

{{
  "klaegervorbringen": "Vollständiger Text des Kläger-Vorbringens WIE VOR DEM ERSTGERICHT vorgebracht (anonymisiert). Enthält Klagebegehren, Tatsachenbehauptungen und rechtliche Argumentation des Klägers.",

  "beklagtenvorbringen": "Vollständiger Text des Beklagten-Vorbringens WIE VOR DEM ERSTGERICHT vorgebracht (anonymisiert). Enthält Klagebeantwortung, Einwendungen und Gegenvorbringen des Beklagten.",

  "feststellungen": "Sachverhaltsfeststellungen des ERSTGERICHTS (anonymisiert). Was hat das Erstgericht als erwiesen angenommen? NICHT die Feststellungen späterer Instanzen.",

  "beweisw_rdigung": "Beweiswürdigung des ERSTGERICHTS (anonymisiert). Wie hat das Erstgericht die Beweise gewürdigt? NICHT die Beweiswürdigung späterer Instanzen.",

  "erstgericht_entscheidung_text": "Entscheidungstext und Begründung des ERSTGERICHTS (anonymisiert). Was hat das BG/LG entschieden und wie begründet? NICHT die Entscheidung des Berufungsgerichts oder des OGH."
}}

OGH-URTEILSTEXT:
{text}"""


# ─── Evidence Description Generation Prompt ──────────────────────────────────────
# Generiert eine faktische Beschreibung der aufgenommenen (!) Beweise aus
# der Beweiswürdigung und den Feststellungen des ERSTGERICHTS — ohne Bewertung.
# Zweck: Kodiert welche Beweismittel das Erstgericht tatsächlich aufgenommen hat.

EVIDENCE_DESCRIPTION_PROMPT = """Du bist ein österreichischer Zivilrechtsspezialist.
Beschreibe auf Basis der folgenden Textabschnitte (Beweiswürdigung und Sachverhaltsfeststellungen
des ERSTGERICHTS) ausschließlich faktisch und ohne eigene Bewertung, welche Beweise das
Erstgericht tatsächlich aufgenommen hat.

Gib ausschließlich ein JSON-Objekt zurück:
{{
  "aufgenommene_beweise": "Sachliche Beschreibung der aufgenommenen Beweise"
}}

Regeln:
- Beschreibe Art und Anzahl der Beweismittel (z.B. 'drei Zeugenvernehmungen',
  'zwei Urkunden', 'ein Sachverständigengutachten')
- Gib an, welche Partei welches Beweismittel beigebracht hat
- Gib an, ob Beweismittel das Vorbringen einer Partei stützen oder widerlegen
  (nur soweit eindeutig aus der Beweiswürdigung hervorgeht)
- KEINE eigene rechtliche Würdigung, KEINE Schlussfolgerungen
- KEINE Namen — ersetze mit [KLÄGER], [BEKLAGTER], [ZEUGE_1] usw.
- Wenn keine Beweise aufgenommen wurden: 'Keine Beweise aufgenommen.'
- Maximale Länge: 400 Wörter

BEWEISWÜRDIGUNG DES ERSTGERICHTS:
{beweisw_rdigung}

SACHVERHALTSFESTSTELLUNGEN DES ERSTGERICHTS:
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
    def _chat_completion(self, messages: list[dict]) -> str:
        """Call gpt-5-mini with retry logic (no temperature parameter)."""
        response = self.client.chat.completions.create(
            model=OPENAI_EXTRACTION_MODEL,
            messages=messages,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content

    @retry(
        retry=retry_if_exception_type((openai.RateLimitError, openai.APIConnectionError)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(5),
    )
    def _get_embedding(self, text) -> list[float]:
        """Get embedding vector from text-embedding-3-large."""
        if not isinstance(text, str):
            text = json.dumps(text, ensure_ascii=False) if isinstance(text, (dict, list)) else str(text or "")
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
        verfahrensarten_str = ", ".join(f'"{v}"' for v in VERFAHRENSARTEN)

        # Truncate text for extraction (keep first 15000 chars = most relevant)
        truncated_text = text[:15000]
        if len(text) > 15000:
            truncated_text += f"\n\n[... Text gekürzt, Gesamtlänge: {len(text)} Zeichen]"

        prompt = EXTRACTION_USER_PROMPT.format(
            claim_types_str=claim_types_str,
            verfahrensarten_str=verfahrensarten_str,
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
        Extrahiert und anonymisiert Textabschnitte aus einem OGH-Urteil.

        Pipeline:
        1. GPT extrahiert ausschließlich ERSTGERICHT-relevante Abschnitte:
           klaegervorbringen, beklagtenvorbringen, feststellungen,
           beweisw_rdigung, erstgericht_entscheidung_text (anonymisiert).
           Berufungsgericht- und OGH-Abschnitte werden NICHT extrahiert.
        2. GPT generiert aufgenommene_beweise — faktische, wertungsfreie
           Beschreibung der vom Erstgericht aufgenommenen Beweise.

        Rückgabe enthält alle Abschnitte + aufgenommene_beweise.
        generate_embeddings() verwendet nur EMBEDDING_SECTIONS:
          [klaegervorbringen, beklagtenvorbringen, aufgenommene_beweise]
        erstgericht_entscheidung_text wird nur gespeichert, nicht eingebettet
        (vermeidet Data Leakage: Label soll nicht als Feature einfließen).
        """
        self._log("Extrahiere Erstgericht-Abschnitte aus OGH-Urteil...", 0.4)

        truncated_text = text[:20000]
        prompt = SECTION_EXTRACTION_PROMPT.format(text=truncated_text)

        messages = [
            {
                "role": "system",
                "content": "Du bist ein österreichischer Zivilrechtsspezialist. "
                "Extrahiere Textabschnitte aus OGH-Urteilen — ausschließlich "
                "die Erstgericht-Ebene. Gib JSON zurück. Anonymisiere Personennamen.",
            },
            {"role": "user", "content": prompt},
        ]

        raw = self._chat_completion(messages)
        time.sleep(OPENAI_REQUEST_DELAY_SEC)

        try:
            raw_sections = json.loads(raw)
        except json.JSONDecodeError:
            raw_sections = {}

        def _to_str(val) -> str:
            """Ensure GPT response value is a plain string, not a nested dict/list."""
            if isinstance(val, str):
                return val
            if isinstance(val, dict):
                for k in ("text", "content", "value", "inhalt"):
                    if k in val and isinstance(val[k], str):
                        return val[k]
                return json.dumps(val, ensure_ascii=False)
            if val is None:
                return ""
            return str(val)

        klaegervorbringen = _to_str(raw_sections.get("klaegervorbringen", ""))
        beklagtenvorbringen = _to_str(raw_sections.get("beklagtenvorbringen", ""))
        feststellungen = _to_str(raw_sections.get("feststellungen", ""))
        beweisw_rdigung = _to_str(raw_sections.get("beweisw_rdigung", ""))
        erstgericht_entscheidung_text = _to_str(
            raw_sections.get("erstgericht_entscheidung_text", "")
        )

        # Step 2: Faktische Beweis-Beschreibung aus Beweiswürdigung + Feststellungen
        self._log("Generiere Beweis-Beschreibung (aufgenommene Beweise Erstgericht)...", 0.5)
        aufgenommene_beweise = self._generate_evidence_description(
            feststellungen=feststellungen,
            beweisw_rdigung=beweisw_rdigung,
        )

        return {
            "klaegervorbringen": klaegervorbringen,
            "beklagtenvorbringen": beklagtenvorbringen,
            "feststellungen": feststellungen,
            "beweisw_rdigung": beweisw_rdigung,
            "erstgericht_entscheidung_text": erstgericht_entscheidung_text,
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
        if not str(feststellungen).strip() and not str(beweisw_rdigung).strip():
            return "Keine Beweise aufgenommen."

        prompt = EVIDENCE_DESCRIPTION_PROMPT.format(
            feststellungen=feststellungen[:8000],
            beweisw_rdigung=beweisw_rdigung[:8000],
        )

        messages = [
            {
                "role": "system",
                "content": "Du bist ein österreichischer Zivilrechtsspezialist (ABGB/ZPO). "
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
        Generate embedding vectors for all text sections in parallel.

        Alle EMBEDDING_SECTIONS werden gleichzeitig eingebettet (ThreadPoolExecutor).
        Statt 3 sequenzieller API-Calls → 1 paralleler Batch → ~3× schneller.
        """
        self._log(
            f"Generiere {len(EMBEDDING_SECTIONS)} Embeddings parallel...", 0.65
        )

        def _embed(section_key: str) -> tuple[str, list[float]]:
            text = sections.get(section_key, "")
            return section_key, self._get_embedding(text)

        embeddings: dict[str, list[float]] = {}
        with ThreadPoolExecutor(max_workers=len(EMBEDDING_SECTIONS)) as pool:
            futures = {pool.submit(_embed, key): key for key in EMBEDDING_SECTIONS}
            for future in as_completed(futures):
                key, vec = future.result()
                embeddings[key] = vec

        self._log("Embeddings fertig.", 0.95)
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
        """Validate and normalize extracted structured data from OGH judgment."""
        # Ensure required fields — datum_ersturteil/datum_ogh statt datum
        _default_bm_typen = {t: 0 for t in BEWEISMITTEL_TYPEN}
        _default_bm_typen["parteienvernehmung"] = False

        defaults = {
            "datum_ersturteil": None,
            "datum_ogh": None,
            # Backwards-compat: accept "datum" from older prompts
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
            "beweismitteltypen_klaeger": dict(_default_bm_typen),
            "beweismitteltypen_beklagter": dict(_default_bm_typen),
            "widerklage": False,
            "verfahrensart": None,
            # legacy fields kept for backwards-compat with old extractions
            "sachverstaendiger_bestellt": False,
            "sachverstaendigen_fachgebiet": None,
            "verfahrensdauer_monate": None,
            "anzahl_verhandlungen": None,
            "outcome": None,
            "outcome_beschreibung": "",
            "zugesprochener_betrag_eur": None,
            "zugesprochener_anteil_prozent": None,
            "kostenentscheidung_erstgericht": None,
            # backwards compat
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

        # Validate verfahrensart
        if data.get("verfahrensart") not in VERFAHRENSARTEN:
            data["verfahrensart"] = None

        # Validate + normalize beweismitteltypen dicts
        for bm_key in ("beweismitteltypen_klaeger", "beweismitteltypen_beklagter"):
            bm = data.get(bm_key)
            if not isinstance(bm, dict):
                data[bm_key] = {t: 0 for t in BEWEISMITTEL_TYPEN}
                data[bm_key]["parteienvernehmung"] = False
            else:
                for t in BEWEISMITTEL_TYPEN:
                    if t == "parteienvernehmung":
                        data[bm_key][t] = bool(bm.get(t, False))
                    else:
                        try:
                            data[bm_key][t] = max(0, int(bm.get(t, 0)))
                        except (TypeError, ValueError):
                            data[bm_key][t] = 0

        return data
