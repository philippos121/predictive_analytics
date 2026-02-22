"""
PDF Processor: Extracts raw text from Austrian civil judgment PDFs.

Uses PyMuPDF (fitz) for robust PDF text extraction with layout preservation.
Handles multi-column layouts and typical court document formatting.
"""

import re
import sys
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import EMBEDDING_SECTIONS


class PDFProcessingError(Exception):
    pass


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """
    Extract full text from a PDF file using PyMuPDF.
    Returns cleaned, normalized text.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise PDFProcessingError(f"PDF not found: {pdf_path}")

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        raise PDFProcessingError(f"Cannot open PDF {pdf_path.name}: {e}")

    pages_text = []
    for page_num, page in enumerate(doc):
        try:
            # Use layout-preserving text extraction
            blocks = page.get_text("blocks", sort=True)
            page_text = "\n".join(
                b[4].strip() for b in blocks if b[4].strip()
            )
            pages_text.append(page_text)
        except Exception:
            # Fallback to simple extraction
            pages_text.append(page.get_text())

    doc.close()

    full_text = "\n\n".join(pages_text)
    return _clean_text(full_text)


def _clean_text(text: str) -> str:
    """Normalize whitespace and remove artifacts from PDF extraction."""
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Remove excessive whitespace lines (keep max 2 consecutive blank lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Fix common PDF extraction artifacts
    text = re.sub(r"([a-zäöüßA-ZÄÖÜ])-\n([a-zäöüßA-ZÄÖÜ])", r"\1\2", text)
    # Normalize spaces
    text = re.sub(r"[ \t]+", " ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    return text.strip()


def split_into_sections(text: str) -> dict[str, str]:
    """
    Attempt to split Austrian court judgment text into standard sections.

    Austrian civil judgments typically have:
    - Spruch / Urteil (verdict)
    - Sachverhalt / Vorbringen der Parteien
    - Feststellungen
    - Beweiswürdigung
    - Rechtliche Beurteilung
    """
    sections = {s: "" for s in EMBEDDING_SECTIONS}
    sections["full_text"] = text
    sections["spruch"] = ""

    # Section header patterns (case-insensitive, Austrian court style)
    patterns = {
        "klaegervorbringen": [
            r"(?i)(vorbringen\s+der\s+kläger|klägerisches\s+vorbringen"
            r"|kläger\s+bringt\s+vor|klage|vorbringen\s+des\s+klägers)",
        ],
        "beklagtenvorbringen": [
            r"(?i)(vorbringen\s+der\s+beklagten|beklagter\s+bringt\s+vor"
            r"|beklagten-?vorbringen|einwendungen\s+des\s+beklagten)",
        ],
        "feststellungen": [
            r"(?i)(feststellungen|sachverhalt|sachverhalts?feststellung"
            r"|als\s+erwiesen\s+angenommener\s+sachverhalt)",
        ],
        "beweisw_rdigung": [
            r"(?i)(beweiswürdigung|würdigung\s+der\s+beweise"
            r"|beweis(aufnahme)?würdigung)",
        ],
        "rechtliche_beurteilung": [
            r"(?i)(rechtliche?\s+beurteilung|rechtliche?\s+würdigung"
            r"|rechtliches|rechtlich)",
        ],
        "spruch": [
            r"(?i)(im\s+namen\s+der\s+republik|urteil|spruch|erkenntnis)",
        ],
    }

    # Find section positions
    section_positions: list[tuple[int, str]] = []
    for section_key, pat_list in patterns.items():
        for pat in pat_list:
            for match in re.finditer(pat, text):
                section_positions.append((match.start(), section_key))
                break  # Take first match per pattern list

    # Sort by position
    section_positions.sort(key=lambda x: x[0])

    # Extract section content
    for i, (pos, key) in enumerate(section_positions):
        end = section_positions[i + 1][0] if i + 1 < len(section_positions) else len(text)
        # Find the end of the header line
        header_end = text.find("\n", pos)
        if header_end == -1:
            header_end = pos
        content_start = header_end + 1
        sections[key] = text[content_start:end].strip()

    return sections


def get_pdf_metadata(pdf_path: str | Path) -> dict:
    """Extract metadata from PDF file."""
    pdf_path = Path(pdf_path)
    try:
        doc = fitz.open(str(pdf_path))
        meta = doc.metadata
        page_count = doc.page_count
        doc.close()
        return {
            "filename": pdf_path.name,
            "pages": page_count,
            "title": meta.get("title", ""),
            "author": meta.get("author", ""),
            "created": meta.get("creationDate", ""),
            "modified": meta.get("modDate", ""),
        }
    except Exception:
        return {"filename": pdf_path.name, "pages": 0}


def get_text_stats(text: str) -> dict:
    """Return basic text statistics."""
    words = text.split()
    return {
        "char_count": len(text),
        "word_count": len(words),
        "line_count": text.count("\n"),
    }


def process_pdf(pdf_path: str | Path) -> dict:
    """
    Full pipeline: extract text + split sections + metadata.
    Returns a dict with all extracted information.
    """
    pdf_path = Path(pdf_path)
    meta = get_pdf_metadata(pdf_path)
    text = extract_text_from_pdf(pdf_path)
    sections = split_into_sections(text)
    stats = get_text_stats(text)

    return {
        "metadata": meta,
        "text": text,
        "sections": sections,
        "stats": stats,
    }
