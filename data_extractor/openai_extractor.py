"""
OpenAI Extractor: Extracts klaegervorbringen, beklagtenvorbringen and
erstgericht outcome from Austrian OGH civil judgments.

Pipeline per judgment (3 API calls total):
  1. One GPT call  → klaegervorbringen, beklagtenvorbringen, outcome (0/1/2)
  2. Two embedding calls in parallel → one vector per section

Österreichisches Zivilrecht (ABGB/ZPO). Eingabe: OGH-Urteile (TXT).
Extrahiert werden ausschließlich ERSTGERICHT-relevante Inhalte.
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
    EMBEDDING_DIM,
    EMBEDDING_SECTIONS,
    OPENAI_EMBEDDING_MODEL,
    OPENAI_EXTRACTION_MODEL,
    OPENAI_REQUEST_DELAY_SEC,
)


# ─── Extraction Prompt ───────────────────────────────────────────────────────
# Single call extracts all three required fields from the OGH judgment text.
# The OGH judgment spans multiple instances; the prompt enforces ERSTGERICHT focus.

EXTRACTION_PROMPT = """Du bist ein Experte für österreichisches Zivilrecht (ABGB, ZPO).
Extrahiere aus diesem OGH-Urteil ausschließlich Informationen des ERSTGERICHTS (BG/LG).

Das OGH-Urteil enthält Entscheidungen MEHRERER Instanzen. Extrahiere NUR:
  - Das Vorbringen der Parteien VOR DEM ERSTGERICHT
  - Das Ergebnis des ERSTGERICHTS (nicht OLG oder OGH)

Antworte ausschließlich mit validem JSON:
{{
  "klaegervorbringen": "Vollständiges Vorbringen des Klägers beim Erstgericht — Klagebegehren, Tatsachenbehauptungen, Rechtsgrundlagen. Anonymisiert: ersetze ALLE Personennamen durch [KLÄGER], [BEKLAGTER], [RICHTER], [ANWALT]. Leerer String wenn nicht auffindbar.",
  "beklagtenvorbringen": "Vollständiges Vorbringen des Beklagten beim Erstgericht — Einwendungen, Gegenvorbringen, Klagebeantwortung. Anonymisiert. Leerer String wenn nicht auffindbar.",
  "outcome": 0 oder 1 oder 2 (null wenn nicht bestimmbar)
}}

OUTCOME KODIERUNG — ausschließlich ERSTGERICHT, aus Sicht des Klägers:
  0 = Kläger unterliegt vollständig (Klage abgewiesen)
  1 = Teilweises Obsiegen/Unterliegen (Klage teilweise zugesprochen)
  2 = Kläger obsiegt vollständig (Klage voll zugesprochen)

OGH-URTEILSTEXT:
{text}"""


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
        """Call the extraction model with retry logic."""
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

        text = text[:32000]  # ~8 000 tokens at ~4 chars/token

        response = self.client.embeddings.create(
            model=OPENAI_EMBEDDING_MODEL,
            input=text,
            dimensions=EMBEDDING_DIM,
        )
        return response.data[0].embedding

    def extract_case(self, text: str) -> tuple[dict, dict]:
        """
        Single GPT call: extract klaegervorbringen, beklagtenvorbringen, outcome.

        Returns:
            structured: {"outcome": 0 | 1 | 2 | None}
            sections:   {"klaegervorbringen": str, "beklagtenvorbringen": str}
        """
        self._log("Extrahiere Vorbringen und Outcome (Erstgericht)...", 0.2)

        truncated = text[:20000]
        if len(text) > 20000:
            truncated += f"\n\n[... gekürzt, Gesamtlänge: {len(text)} Zeichen]"

        messages = [
            {
                "role": "system",
                "content": (
                    "Du bist ein österreichischer Zivilrechtsspezialist (ABGB/ZPO). "
                    "Antworte ausschließlich mit validem JSON ohne weiteren Text."
                ),
            },
            {"role": "user", "content": EXTRACTION_PROMPT.format(text=truncated)},
        ]

        raw = self._chat_completion(messages)
        time.sleep(OPENAI_REQUEST_DELAY_SEC)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"Ungültige JSON-Antwort von OpenAI: {e}\nRaw: {raw[:500]}")

        outcome = data.get("outcome")
        if outcome not in (0, 1, 2):
            outcome = None

        structured = {"outcome": outcome}
        sections = {
            "klaegervorbringen": str(data.get("klaegervorbringen") or ""),
            "beklagtenvorbringen": str(data.get("beklagtenvorbringen") or ""),
        }
        return structured, sections

    def generate_embeddings(self, sections: dict[str, str]) -> dict[str, list[float]]:
        """
        Generate embedding vectors for klaegervorbringen and beklagtenvorbringen
        in parallel (two concurrent API calls).
        """
        self._log("Generiere Embeddings parallel...", 0.65)

        def _embed(section_key: str) -> tuple[str, list[float]]:
            return section_key, self._get_embedding(sections.get(section_key, ""))

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
        Full extraction pipeline for a single OGH judgment (3 API calls):
          1. GPT:   klaegervorbringen + beklagtenvorbringen + outcome
          2+3. Embeddings: one per section, in parallel

        Returns complete case dict ready for DataManager.add_case().
        """
        self._log("Starte Verarbeitung...", 0.1)

        structured, sections = self.extract_case(text)
        self._log("Vorbringen und Outcome extrahiert.", 0.5)

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
        case_sections must contain: klaegervorbringen, beklagtenvorbringen.
        """
        return self.generate_embeddings(case_sections)
