
from __future__ import annotations

import re
from typing import Any

from config import (
    CONVENTIONAL_KEYWORDS,
    ISLAMIC_KEYWORDS,
    KNOWN_BANKS,
    METADATA_SCHEMA,
)




def _normalise(text: str) -> str:
    """Lower-case and collapse whitespace for fuzzy matching."""
    return re.sub(r"\s+", " ", text.lower().strip())





def infer_bank(text: str, filename: str = "") -> tuple[str, str]:
    """
    Return (canonical_bank_name, bank_short_key) by scanning *text* and
    *filename* against KNOWN_BANKS.

    Strategy:
      1. Check filename first (most reliable signal for whole-doc PDFs).
      2. Fall back to scanning the text content itself.
      3. Return ("Unknown", "unknown") if nothing matches.
    """
    haystack = _normalise(f"{filename} {text}")

    for canonical, pattern in KNOWN_BANKS.items():
        if re.search(pattern, haystack):
            
            short = canonical.lower().split()[0]  
            return canonical, short

    return "Unknown", "unknown"





def infer_financing_type(text: str) -> str:
    """
    Classify *text* as 'islamic', 'conventional', 'both', or 'unknown'.

    We check for Islamic-finance terminology first (higher precision signal)
    then conventional markers.
    """
    norm = _normalise(text)
    has_islamic = any(kw in norm for kw in ISLAMIC_KEYWORDS)
    has_conv = any(kw in norm for kw in CONVENTIONAL_KEYWORDS)

    if has_islamic and has_conv:
        return "both"
    if has_islamic:
        return "islamic"
    if has_conv:
        return "conventional"
    return "unknown"




_SEGMENT_PATTERNS: dict[str, str] = {
    "salaried":      r"\bsalaried\b|\bemployee\b|\bpermanent\s+employee\b",
    "self_employed": r"\bself[\s\-]?employed\b|\bbusiness\s*(person|owner|man)\b|\bprofessional\b",
    "agriculturist": r"\bagricultur(ist|al|e)\b|\bfarmer\b|\bkisan\b",
    "nrp":           (
        r"\bnrp\b|\boverseas\s+pakistan(i)?\b|\broshan\s+digital\b"
        r"|\bnon[\s\-]?resident\b"
    ),
}


def infer_customer_segment(text: str) -> str:
    """
    Return a comma-separated string of matched customer segments, e.g.
    'salaried,self_employed'.  Returns 'general' if nothing specific detected.
    """
    norm = _normalise(text)
    matched = [
        seg
        for seg, pattern in _SEGMENT_PATTERNS.items()
        if re.search(pattern, norm)
    ]
    return ",".join(matched) if matched else "general"




_RATE_PATTERN = re.compile(
    r"\b(kibor|profit\s+rate|markup\s+rate|interest\s+rate|bps|%\s*p\.?a\.?)\b",
    re.IGNORECASE,
)
_TENURE_PATTERN = re.compile(
    r"\b(\d+\s*(year|yr|month|mnth|annum)s?\b|tenure|repayment\s+period)\b",
    re.IGNORECASE,
)
_DOWN_PAYMENT_PATTERN = re.compile(
    r"\b(down\s*pay(ment)?|equity|own\s+contribution|minimum\s+\d+%)\b",
    re.IGNORECASE,
)
_LIMIT_PATTERN = re.compile(
    r"\b(maximum\s+financ(ing|e)|loan\s+(cap|limit)|pkr\s*[\d,.]+\s*(m|million|lakh|lac)|"
    r"sbp\s+(cap|limit)|financing\s+limit)\b",
    re.IGNORECASE,
)
_VEHICLE_NEW_PATTERN = re.compile(
    r"\bnew\s+vehicle\b|\bnew\s+car\b|\bnew\s+unit\b", re.IGNORECASE
)
_VEHICLE_USED_PATTERN = re.compile(
    r"\b(used|old|reconditioned|pre[\s\-]?owned)\s+(vehicle|car|unit|cars?)\b"
    r"|\bup\s+to\s+\d+\s+years?\b",
    re.IGNORECASE,
)


def infer_financial_flags(text: str) -> dict[str, bool]:
    return {
        "has_rate_info":      bool(_RATE_PATTERN.search(text)),
        "has_tenure_info":    bool(_TENURE_PATTERN.search(text)),
        "has_down_payment":   bool(_DOWN_PAYMENT_PATTERN.search(text)),
        "has_limit_info":     bool(_LIMIT_PATTERN.search(text)),
    }


def infer_vehicle_type(text: str) -> str:
    has_new  = bool(_VEHICLE_NEW_PATTERN.search(text))
    has_used = bool(_VEHICLE_USED_PATTERN.search(text))
    if has_new and has_used:
        return "both"
    if has_new:
        return "new"
    if has_used:
        return "used"
    return "unknown"





def build_metadata(
    *,
    text: str,
    source_file: str,
    page_number: int,
    element_type: str,
    section_header: str,
    bank_name_override: str | None = None,
    chunk_index: int = 0,
    total_chunks: int = 1,
) -> dict[str, Any]:
    """
    Assemble the full metadata dict conforming to METADATA_SCHEMA.

    Parameters
    ----------
    text              : Raw text of this chunk (used for inference).
    source_file       : Filename of the originating PDF.
    page_number       : 1-indexed page number from unstructured.
    element_type      : Unstructured element type string.
    section_header    : Nearest ancestor Title/Header text.
    bank_name_override: If the bank is already known from filename parsing,
                        skip auto-detection and use this value.
    chunk_index       : 0-indexed position of this chunk within its parent element.
    total_chunks      : Total number of chunks from the same element.
    """
    combined_context = f"{source_file} {section_header} {text}"

    if bank_name_override:
        bank_name = bank_name_override
        bank_short = bank_name_override.lower().split()[0]
    else:
        bank_name, bank_short = infer_bank(combined_context, source_file)

    document_type = (
        "table"  if element_type == "Table"
        else "header" if element_type in ("Title", "Header")
        else "text"
    )

    flags = infer_financial_flags(text)

    metadata: dict[str, Any] = {
        
        "source_file":      source_file,
        "page_number":      page_number,
        
        "bank_name":        bank_name,
        "bank_short":       bank_short,
        
        "element_type":     element_type,
        "document_type":    document_type,
        "section_header":   section_header[:200],  
        
        "financing_type":   infer_financing_type(combined_context),
        "customer_segment": infer_customer_segment(text),
        
        **flags,
        
        "vehicle_type":     infer_vehicle_type(text),
        
        "chunk_index":      chunk_index,
        "total_chunks":     total_chunks,
    }

    
    missing = set(METADATA_SCHEMA.keys()) - set(metadata.keys())
    if missing:
        raise RuntimeError(f"build_metadata: missing keys {missing}")

    return metadata
