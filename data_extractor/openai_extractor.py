"""
OpenAI Extractor: Uses GPT-5-nano to extract structured legal data
and a detailed legal analysis from Austrian civil judgment text.

v2.0 — No embeddings.  Instead of text-embedding-3-large vectors the extractor
now produces a fine-grained structured legal analysis (fall_metadaten,
klaegervorbringen, beklagtenvorbringen categories) that serves as the primary
training signal for the prediction model.

Includes AsyncBatchExtractor for parallel extraction (100 concurrent API calls).
"""

import asyncio
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
    OPENAI_EXTRACTION_MODEL,
    OPENAI_REQUEST_DELAY_SEC,
)


# ─── Extraction Prompt (metadata + outcome) ──────────────────────────────────────

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


# ─── Structured Legal Analysis Prompt ────────────────────────────────────────────
# Replaces embeddings: GPT extracts a detailed, schema-conformant JSON
# covering claims, defenses, and procedural aspects.

LEGAL_ANALYSIS_SYSTEM_PROMPT = """You are an expert Austrian legal AI specialized in civil law (Zivilrecht, ABGB) and civil procedure (Zivilprozessrecht, ZPO). Your task is to analyze the initial court submissions (Vorbringen) of the plaintiff (Kläger) and defendant (Beklagter) from an Austrian civil trial (Erstgericht).

Extract a highly structured JSON representation of the legal arguments, claims, and defenses. Do not hallucinate. If a specific defense, claim, or concept is not explicitly mentioned or heavily implied by the facts, default to `false` or `null`.

Return ONLY a valid JSON object matching the following exact schema:

{
  "fall_metadaten": {
    "rechtsgebiet_hauptkategorie": "Enum: [Schuldrecht_Vertrag, Schuldrecht_Gesetzlich, Sachenrecht, Familienrecht, Erbrecht, Immaterialgueterrecht, Gesellschaftsrecht, Sonstiges]",
    "verbrauchergeschaeft_kschg": "Boolean. True if the facts indicate a B2C transaction triggering KSchG or FAGG.",
    "streitwert_bekannt": "Boolean. True if a specific monetary amount is demanded."
  },

  "klaegervorbringen_anspruchsgrundlagen": {
    "vertraglich_erfuellung": "Boolean. Claiming primary performance of a contract (e.g., Kaufpreis, Werklohn, Mietzins).",
    "vertraglich_gewaehrleistung": "Boolean. Warranty claims (§§ 922 ff ABGB - Preisminderung, Wandlung, Verbesserung).",
    "vertraglich_poenale": "Boolean. Claiming a contractual penalty (Konventionalstrafe/Pönale, § 1336 ABGB).",
    "quasi_vertraglich": "Boolean. Culpa in contrahendo (c.i.c.) or Geschäftsführung ohne Auftrag (GoA, §§ 1036 ff).",
    "schadenersatz_ex_contractu": "Boolean. Contractual damages (§§ 1295 iVm 1298 ABGB).",
    "schadenersatz_ex_delicto": "Boolean. Tortious damages or strict liability (Verschuldenshaftung, EKHG, PHG).",
    "bereicherung": "Boolean. Unjust enrichment (Condictio, §§ 1431 ff, § 877 ABGB).",
    "dinglich_eigentum": "Boolean. Rei vindicatio (Eigentumsklage, § 366 ABGB) or Actio negatoria (§ 523 ABGB).",
    "dinglich_besitz": "Boolean. Possession protection (Besitzstörungsklage, § 339 ABGB).",
    "unterlassung_beseitigung": "Boolean. Injunctive relief or removal (often in IP, UWG, or property law).",
    "sonstige_anspruchsgrundlage": "Boolean. True if there is a core claim that does not fit the above categories.",
    "sonstige_anspruchsgrundlage_beschreibung": "String. If sonstige_anspruchsgrundlage is true, name the legal concept. Otherwise null.",
    "zitierte_normen_klaeger": "List of Strings. E.g., ['§ 879 ABGB', '§ 1295 ABGB']. Empty list if none."
  },

  "beklagtenvorbringen_prozessual": {
    "unzuständigkeit": "Boolean. Lack of jurisdiction (örtlich, sachlich, international).",
    "streitanhaengigkeit_rechtskraft": "Boolean. Lis pendens or res judicata (already pending or decided).",
    "mangelnde_partei_prozessfaehigkeit": "Boolean. Lack of legal capacity to be a party or stand in court.",
    "sonstiges_prozesshindernis": "Boolean. True if another formal blocker (e.g., Schiedseinrede) is raised.",
    "sonstiges_prozesshindernis_beschreibung": "String. If sonstiges_prozesshindernis is true, name it. Otherwise null."
  },

  "beklagtenvorbringen_materiell_rechtshindernd": {
    "mangelnde_geschaeftsfaehigkeit": "Boolean. Incapacity to contract (§ 865 ABGB).",
    "dissens_scherz_scheinvertrag": "Boolean. Lack of genuine agreement, joke declaration, or sham contract (§§ 869, 916 ABGB).",
    "sittenwidrigkeit_gesetzwidrigkeit": "Boolean. Immorality or illegality (§ 879 ABGB).",
    "formmangel": "Boolean. Lack of required legal form (§ 883 ABGB, e.g., Notariatsakt).",
    "irrtum_list_drohung": "Boolean. Contesting validity due to error, deceit, or duress (§§ 870, 871 ABGB).",
    "laesio_enormis_wucher": "Boolean. Verkürzung über die Hälfte (§ 934 ABGB) or unconscionability.",
    "sonstige_rechtshindernde_einwendung": "Boolean. True if another rechtshindernde Einwendung applies.",
    "sonstige_rechtshindernde_beschreibung": "String. If true, briefly describe it. Otherwise null."
  },

  "beklagtenvorbringen_materiell_rechtsvernichtend": {
    "erfuellung_zahlung": "Boolean. Claim already fulfilled or paid (§ 1412 ABGB).",
    "aufrechnung_kompensation": "Boolean. Set-off against a counterclaim (§ 1438 ABGB).",
    "ruecktritt_kuendigung": "Boolean. Valid withdrawal (e.g., KSchG, FAGG) or termination of contract (§ 918 ABGB).",
    "unmoeglichkeit": "Boolean. Subsequent impossibility of performance or force majeure (§ 1447 ABGB).",
    "verzicht_erlass": "Boolean. Waiver or release of debt (§ 1444 ABGB).",
    "sonstige_rechtsvernichtende_einwendung": "Boolean. True if another rechtsvernichtende Einwendung applies.",
    "sonstige_rechtsvernichtende_beschreibung": "String. If true, briefly describe it. Otherwise null."
  },

  "beklagtenvorbringen_materiell_rechtshemmend": {
    "verjaehrung_praeklusion": "Boolean. Statute of limitations expired (§ 1478 ABGB) or preclusion.",
    "zug_um_zug_einrede": "Boolean. Einrede des nicht erfüllten Vertrages (§ 1052 ABGB - won't perform until plaintiff performs).",
    "zurueckbehaltungsrecht": "Boolean. Right of retention (§ 471 ABGB).",
    "mangelnde_faelligkeit_stundung": "Boolean. Claim is not yet due or an extension (Stundung) was granted.",
    "sonstige_rechtshemmende_einrede": "Boolean. True if another rechtshemmende Einrede applies.",
    "sonstige_rechtshemmende_beschreibung": "String. If true, briefly describe it. Otherwise null."
  },

  "beklagtenvorbringen_allgemein": {
    "mangelnde_aktiv_passivlegitimation": "Boolean. Wrong plaintiff or wrong defendant.",
    "mitverschulden_schadensminderung": "Boolean. Plaintiff contributed to the damage (§ 1304 ABGB) or failed to mitigate.",
    "bestreitet_tatbestand_komplett": "Boolean. Complete denial of the factual events.",
    "bestreitet_nur_rechtliche_wertung": "Boolean. Admits facts, disputes legal interpretation.",
    "bestreitet_hoehe": "Boolean. Specifically disputes the amount demanded.",
    "bestreitet_verschulden": "Boolean. Specifically denies negligence or intent.",
    "sonstige_allgemeine_bestreitung": "Boolean. True if there is a major defense/denial not captured anywhere else.",
    "sonstige_allgemeine_beschreibung": "String. If true, briefly describe it. Otherwise null.",
    "zitierte_normen_beklagter": "List of Strings. Explicitly cited paragraphs. Empty list if none."
  }
}"""

LEGAL_ANALYSIS_USER_PROMPT = """Analysiere das folgende österreichische Zivilurteil und extrahiere die strukturierte rechtliche Analyse gemäß dem vorgegebenen Schema.

Wichtig:
- Antworte NUR mit validem JSON, kein anderer Text.
- Setze Boolean-Felder auf false wenn nicht explizit erwähnt oder stark impliziert.
- Setze String-Felder auf null wenn nicht zutreffend.
- Listen: leere Liste [] wenn keine Normen zitiert werden.
- rechtsgebiet_hauptkategorie: genau einer der Enum-Werte.

URTEILSTEXT:
{text}"""


# ─── Evidence Description Generation Prompt ──────────────────────────────────────

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
    Handles all OpenAI API interactions for legal text extraction.
    Uses GPT-5-nano for structured data extraction — no embeddings.
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
        """Call GPT-5-nano with retry logic."""
        response = self.client.chat.completions.create(
            model=OPENAI_EXTRACTION_MODEL,
            messages=messages,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content

    def extract_structured_data(self, text: str) -> dict[str, Any]:
        """
        Extract structured legal data from judgment text using GPT-5-nano.
        Returns parsed JSON dict with case metadata and outcome.
        """
        self._log("Extrahiere strukturierte Daten via GPT-5-nano...", 0.2)

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
        if not str(feststellungen).strip() and not str(beweisw_rdigung).strip():
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

    def extract_legal_analysis(self, text: str) -> dict[str, Any]:
        """
        Extract the detailed structured legal analysis from judgment text.

        This is the primary training signal — replaces embedding vectors.
        Uses the comprehensive Austrian civil law schema covering:
        - fall_metadaten (case metadata / legal area)
        - klaegervorbringen_anspruchsgrundlagen (plaintiff's claims)
        - beklagtenvorbringen_prozessual (procedural defenses)
        - beklagtenvorbringen_materiell_rechtshindernd (claim-blocking defenses)
        - beklagtenvorbringen_materiell_rechtsvernichtend (claim-destroying defenses)
        - beklagtenvorbringen_materiell_rechtshemmend (claim-impeding defenses)
        - beklagtenvorbringen_allgemein (general defenses)

        Returns parsed JSON dict matching the schema.
        """
        self._log("Extrahiere strukturierte Legal-Analyse via GPT-5-nano...", 0.6)

        # Use more text for the legal analysis (needs full context)
        truncated_text = text[:25000]
        if len(text) > 25000:
            truncated_text += f"\n\n[... Text gekürzt, Gesamtlänge: {len(text)} Zeichen]"

        prompt = LEGAL_ANALYSIS_USER_PROMPT.format(text=truncated_text)

        messages = [
            {"role": "system", "content": LEGAL_ANALYSIS_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        raw = self._chat_completion(messages)
        time.sleep(OPENAI_REQUEST_DELAY_SEC)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Ungültige JSON-Antwort bei Legal-Analyse: {e}\nRaw: {raw[:500]}"
            )

        return self._validate_legal_analysis(data)

    def process_judgment(self, text: str) -> dict[str, Any]:
        """
        Full extraction pipeline for a single judgment:
        1. Extract structured metadata (outcome, streitwert, etc.)
        2. Extract text sections (anonymized)
        3. Extract structured legal analysis (replaces embeddings)

        Returns complete case dict ready for dataset storage.
        """
        self._log("Starte Verarbeitung...", 0.1)

        # Step 1: Structured data
        structured = self.extract_structured_data(text)
        self._log("Strukturierte Daten extrahiert.", 0.35)

        # Step 2: Text sections
        sections = self.extract_text_sections(text)
        self._log("Textabschnitte extrahiert.", 0.55)

        # Step 3: Legal analysis (replaces embeddings)
        legal_analysis = self.extract_legal_analysis(text)
        self._log("Legal-Analyse extrahiert.", 0.95)

        return {
            "structured": structured,
            "sections": sections,
            "legal_analysis": legal_analysis,
        }

    def _validate_and_normalize(self, data: dict) -> dict:
        """Validate and normalize extracted structured data."""
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

    def _validate_legal_analysis(self, data: dict) -> dict:
        """Validate and normalize the legal analysis output."""
        from config import LEGAL_ANALYSIS_BOOL_FIELDS, RECHTSGEBIET_CATEGORIES

        # Ensure all top-level sections exist
        sections_defaults = {
            "fall_metadaten": {},
            "klaegervorbringen_anspruchsgrundlagen": {},
            "beklagtenvorbringen_prozessual": {},
            "beklagtenvorbringen_materiell_rechtshindernd": {},
            "beklagtenvorbringen_materiell_rechtsvernichtend": {},
            "beklagtenvorbringen_materiell_rechtshemmend": {},
            "beklagtenvorbringen_allgemein": {},
        }

        for section, default in sections_defaults.items():
            if section not in data or not isinstance(data[section], dict):
                data[section] = default

        # Validate rechtsgebiet_hauptkategorie
        fm = data["fall_metadaten"]
        rg = fm.get("rechtsgebiet_hauptkategorie", "Sonstiges")
        if rg not in RECHTSGEBIET_CATEGORIES:
            fm["rechtsgebiet_hauptkategorie"] = "Sonstiges"

        # Ensure all boolean fields exist and are boolean
        for section_key, fields in LEGAL_ANALYSIS_BOOL_FIELDS.items():
            section = data.get(section_key, {})
            for field in fields:
                val = section.get(field)
                section[field] = bool(val) if val is not None else False
            data[section_key] = section

        # Ensure list fields exist
        ka = data["klaegervorbringen_anspruchsgrundlagen"]
        if not isinstance(ka.get("zitierte_normen_klaeger"), list):
            ka["zitierte_normen_klaeger"] = []

        ba = data["beklagtenvorbringen_allgemein"]
        if not isinstance(ba.get("zitierte_normen_beklagter"), list):
            ba["zitierte_normen_beklagter"] = []

        # Ensure description strings default to null
        for section_key in [
            "klaegervorbringen_anspruchsgrundlagen",
            "beklagtenvorbringen_prozessual",
            "beklagtenvorbringen_materiell_rechtshindernd",
            "beklagtenvorbringen_materiell_rechtsvernichtend",
            "beklagtenvorbringen_materiell_rechtshemmend",
            "beklagtenvorbringen_allgemein",
        ]:
            section = data[section_key]
            for key in list(section.keys()):
                if key.endswith("_beschreibung") and not isinstance(section[key], str):
                    section[key] = None

        return data

    @staticmethod
    def empty_legal_analysis() -> dict:
        """Return a valid legal_analysis with all defaults (for manual entry)."""
        from config import LEGAL_ANALYSIS_BOOL_FIELDS

        analysis = {
            "fall_metadaten": {
                "rechtsgebiet_hauptkategorie": "Sonstiges",
                "verbrauchergeschaeft_kschg": False,
                "streitwert_bekannt": False,
            },
            "klaegervorbringen_anspruchsgrundlagen": {
                "vertraglich_erfuellung": False,
                "vertraglich_gewaehrleistung": False,
                "vertraglich_poenale": False,
                "quasi_vertraglich": False,
                "schadenersatz_ex_contractu": False,
                "schadenersatz_ex_delicto": False,
                "bereicherung": False,
                "dinglich_eigentum": False,
                "dinglich_besitz": False,
                "unterlassung_beseitigung": False,
                "sonstige_anspruchsgrundlage": False,
                "sonstige_anspruchsgrundlage_beschreibung": None,
                "zitierte_normen_klaeger": [],
            },
            "beklagtenvorbringen_prozessual": {
                "unzuständigkeit": False,
                "streitanhaengigkeit_rechtskraft": False,
                "mangelnde_partei_prozessfaehigkeit": False,
                "sonstiges_prozesshindernis": False,
                "sonstiges_prozesshindernis_beschreibung": None,
            },
            "beklagtenvorbringen_materiell_rechtshindernd": {
                "mangelnde_geschaeftsfaehigkeit": False,
                "dissens_scherz_scheinvertrag": False,
                "sittenwidrigkeit_gesetzwidrigkeit": False,
                "formmangel": False,
                "irrtum_list_drohung": False,
                "laesio_enormis_wucher": False,
                "sonstige_rechtshindernde_einwendung": False,
                "sonstige_rechtshindernde_beschreibung": None,
            },
            "beklagtenvorbringen_materiell_rechtsvernichtend": {
                "erfuellung_zahlung": False,
                "aufrechnung_kompensation": False,
                "ruecktritt_kuendigung": False,
                "unmoeglichkeit": False,
                "verzicht_erlass": False,
                "sonstige_rechtsvernichtende_einwendung": False,
                "sonstige_rechtsvernichtende_beschreibung": None,
            },
            "beklagtenvorbringen_materiell_rechtshemmend": {
                "verjaehrung_praeklusion": False,
                "zug_um_zug_einrede": False,
                "zurueckbehaltungsrecht": False,
                "mangelnde_faelligkeit_stundung": False,
                "sonstige_rechtshemmende_einrede": False,
                "sonstige_rechtshemmende_beschreibung": None,
            },
            "beklagtenvorbringen_allgemein": {
                "mangelnde_aktiv_passivlegitimation": False,
                "mitverschulden_schadensminderung": False,
                "bestreitet_tatbestand_komplett": False,
                "bestreitet_nur_rechtliche_wertung": False,
                "bestreitet_hoehe": False,
                "bestreitet_verschulden": False,
                "sonstige_allgemeine_bestreitung": False,
                "sonstige_allgemeine_beschreibung": None,
                "zitierte_normen_beklagter": [],
            },
        }
        return analysis


class AsyncBatchExtractor:
    """
    Parallel batch extractor using asyncio + openai.AsyncOpenAI.

    Processes cases from dataset.json with up to `max_concurrent` parallel
    API calls to GPT-5-nano.  Each case goes through the full 3-step
    pipeline (structured data, text sections, legal analysis).

    Usage:
        extractor = AsyncBatchExtractor(api_key="sk-...", max_concurrent=100)
        asyncio.run(extractor.extract_dataset(dataset_path))
    """

    def __init__(
        self,
        api_key: str,
        max_concurrent: int = 100,
        progress_callback: Optional[Callable] = None,
    ):
        self.api_key = api_key
        self.max_concurrent = max_concurrent
        self.progress_callback = progress_callback or (lambda done, total, msg: None)
        self.async_client = openai.AsyncOpenAI(api_key=api_key)
        # Synchronous extractor for validation/normalization helpers
        self._sync = OpenAIExtractor(api_key=api_key)

    async def _async_chat_completion(
        self,
        messages: list[dict],
        semaphore: asyncio.Semaphore,
        max_retries: int = 5,
    ) -> str:
        """Single async chat completion with semaphore-based concurrency limit and retry."""
        async with semaphore:
            for attempt in range(max_retries):
                try:
                    response = await self.async_client.chat.completions.create(
                        model=OPENAI_EXTRACTION_MODEL,
                        messages=messages,
                        response_format={"type": "json_object"},
                    )
                    return response.choices[0].message.content
                except (openai.RateLimitError, openai.APIConnectionError) as e:
                    if attempt < max_retries - 1:
                        wait = min(2 ** (attempt + 1), 30)
                        await asyncio.sleep(wait)
                    else:
                        raise

    async def _extract_structured_data_async(
        self, text: str, semaphore: asyncio.Semaphore,
    ) -> dict:
        """Async version of extract_structured_data."""
        claim_types_str = ", ".join(f'"{c}"' for c in CLAIM_TYPES)
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
        raw = await self._async_chat_completion(messages, semaphore)
        data = json.loads(raw)
        return self._sync._validate_and_normalize(data)

    async def _extract_text_sections_async(
        self, text: str, semaphore: asyncio.Semaphore,
    ) -> dict[str, str]:
        """Async version of extract_text_sections."""
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
        raw = await self._async_chat_completion(messages, semaphore)
        try:
            raw_sections = json.loads(raw)
        except json.JSONDecodeError:
            raw_sections = {}

        def _to_str(val) -> str:
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

        # Evidence description
        aufgenommene_beweise = await self._extract_evidence_async(
            feststellungen, beweisw_rdigung, semaphore
        )

        return {
            "klaegervorbringen": klaegervorbringen,
            "beklagtenvorbringen": beklagtenvorbringen,
            "feststellungen": feststellungen,
            "beweisw_rdigung": beweisw_rdigung,
            "aufgenommene_beweise": aufgenommene_beweise,
        }

    async def _extract_evidence_async(
        self, feststellungen: str, beweisw_rdigung: str, semaphore: asyncio.Semaphore,
    ) -> str:
        """Async version of _generate_evidence_description."""
        if not str(feststellungen).strip() and not str(beweisw_rdigung).strip():
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
        raw = await self._async_chat_completion(messages, semaphore)
        try:
            result = json.loads(raw)
            return result.get("aufgenommene_beweise", "Keine Beweise aufgenommen.")
        except json.JSONDecodeError:
            return "Keine Beweise aufgenommen."

    async def _extract_legal_analysis_async(
        self, text: str, semaphore: asyncio.Semaphore,
    ) -> dict:
        """Async version of extract_legal_analysis."""
        truncated_text = text[:25000]
        if len(text) > 25000:
            truncated_text += f"\n\n[... Text gekürzt, Gesamtlänge: {len(text)} Zeichen]"

        prompt = LEGAL_ANALYSIS_USER_PROMPT.format(text=truncated_text)
        messages = [
            {"role": "system", "content": LEGAL_ANALYSIS_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        raw = await self._async_chat_completion(messages, semaphore)
        data = json.loads(raw)
        return self._sync._validate_legal_analysis(data)

    async def _process_single_case(
        self,
        case_idx: int,
        case: dict,
        semaphore: asyncio.Semaphore,
        total: int,
    ) -> tuple[int, dict, Optional[str]]:
        """
        Process a single case through the full extraction pipeline.
        Returns (case_idx, updated_case, error_or_None).
        """
        case_id = case.get("case_id", f"idx_{case_idx}")
        try:
            # The case already has text in sections — use klaegervorbringen
            # + beklagtenvorbringen as the source text for legal analysis,
            # or re-extract from the original text if available.
            # For cases already in the dataset, we just need legal_analysis.

            # Build judgment text from existing sections for legal analysis
            sections = case.get("sections", {})
            text_parts = []
            for key in ["klaegervorbringen", "beklagtenvorbringen",
                        "feststellungen", "beweisw_rdigung"]:
                sec_text = sections.get(key, "")
                if sec_text:
                    text_parts.append(f"[{key.upper()}]\n{sec_text}")

            combined_text = "\n\n".join(text_parts)
            if not combined_text.strip():
                # Fallback: use structured summaries
                s = case.get("structured", {})
                combined_text = (
                    f"Kläger: {s.get('klaeger_anspruch_zusammenfassung', '')}\n"
                    f"Beklagter: {s.get('beklagter_vorbringen_zusammenfassung', '')}"
                )

            legal_analysis = await self._extract_legal_analysis_async(
                combined_text, semaphore
            )

            case["legal_analysis"] = legal_analysis
            done_count = self._done_counter
            self._done_counter += 1
            self.progress_callback(
                self._done_counter, total,
                f"[{self._done_counter}/{total}] {case_id} OK"
            )
            return case_idx, case, None

        except Exception as e:
            self._done_counter += 1
            error_msg = f"{case_id}: {e}"
            self.progress_callback(
                self._done_counter, total,
                f"[{self._done_counter}/{total}] {case_id} FEHLER: {e}"
            )
            return case_idx, case, error_msg

    async def extract_legal_analysis_batch(
        self,
        cases: list[dict],
    ) -> tuple[list[dict], list[str]]:
        """
        Extract legal_analysis for all cases that don't have one yet.
        Runs up to self.max_concurrent API calls in parallel.

        Args:
            cases: List of case dicts (modified in-place with legal_analysis)

        Returns:
            (updated_cases, errors) — errors is a list of error messages
        """
        # Filter to cases needing extraction
        indices_to_process = [
            i for i, c in enumerate(cases)
            if not c.get("legal_analysis")
        ]

        total = len(indices_to_process)
        if total == 0:
            self.progress_callback(0, 0, "Alle Fälle haben bereits eine Legal-Analyse.")
            return cases, []

        self.progress_callback(0, total, f"Starte Extraktion für {total} Fälle...")
        self._done_counter = 0

        semaphore = asyncio.Semaphore(self.max_concurrent)
        tasks = [
            self._process_single_case(idx, cases[idx], semaphore, total)
            for idx in indices_to_process
        ]

        results = await asyncio.gather(*tasks)

        errors = []
        for case_idx, updated_case, error in results:
            cases[case_idx] = updated_case
            if error:
                errors.append(error)

        self.progress_callback(
            total, total,
            f"Fertig: {total - len(errors)} OK, {len(errors)} Fehler"
        )
        return cases, errors

    async def extract_dataset(
        self,
        dataset_path: Path,
        save_every: int = 100,
    ) -> tuple[int, int]:
        """
        Load dataset.json, extract legal_analysis for all cases missing it,
        and save back.  Periodically saves progress every `save_every` cases.

        Returns (n_success, n_errors).
        """
        from data_extractor.data_manager import DataManager

        dm = DataManager(dataset_path=dataset_path)
        cases = dm.load_dataset()

        if not cases:
            print("Dataset ist leer.")
            return 0, 0

        # Process in chunks to enable periodic saves
        indices_to_process = [
            i for i, c in enumerate(cases)
            if not c.get("legal_analysis")
        ]
        total = len(indices_to_process)

        if total == 0:
            print("Alle Fälle haben bereits eine Legal-Analyse.")
            return len(cases), 0

        print(f"Dataset: {len(cases)} Fälle, davon {total} ohne Legal-Analyse")
        print(f"Starte parallele Extraktion (max {self.max_concurrent} gleichzeitig)...")

        all_errors = []
        self._done_counter = 0
        semaphore = asyncio.Semaphore(self.max_concurrent)

        # Process in chunks for periodic saves
        for chunk_start in range(0, len(indices_to_process), save_every):
            chunk_indices = indices_to_process[chunk_start:chunk_start + save_every]
            chunk_total = len(chunk_indices)

            tasks = [
                self._process_single_case(idx, cases[idx], semaphore, total)
                for idx in chunk_indices
            ]

            results = await asyncio.gather(*tasks)

            for case_idx, updated_case, error in results:
                cases[case_idx] = updated_case
                if error:
                    all_errors.append(error)

            # Periodic save
            dm.save_dataset(cases)
            n_done = min(chunk_start + save_every, total)
            print(f"  Gespeichert: {n_done}/{total} verarbeitet")

        n_success = total - len(all_errors)
        print(f"\nFertig: {n_success} erfolgreich, {len(all_errors)} Fehler")
        if all_errors:
            print(f"Fehler:\n" + "\n".join(f"  - {e}" for e in all_errors[:20]))
            if len(all_errors) > 20:
                print(f"  ... und {len(all_errors) - 20} weitere")

        return n_success, len(all_errors)
