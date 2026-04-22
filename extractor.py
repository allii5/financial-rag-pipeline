
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

from config import (
    DATA_DIR,
    ELEMENT_TYPE_TABLE,
    ELEMENT_TYPES_CONTENT,
    ELEMENT_TYPES_STRUCTURAL,
    PDF_STRATEGY,
)

logger = logging.getLogger(__name__)




@dataclass
class RawElement:
    """
    Normalised representation of one unstructured element.

    Keeps only what downstream chunking needs; avoids leaking unstructured
    internal types into other modules.
    """
    element_type: str          
    text: str                 
    html: str                 
    page_number: int        
    source_file: str         
    element_id: str          


@dataclass
class ExtractedDocument:
    """All elements parsed from a single PDF file."""
    source_file: str
    file_path: Path
    elements: list[RawElement] = field(default_factory=list)

    
    titles:    list[RawElement] = field(default_factory=list)
    tables:    list[RawElement] = field(default_factory=list)
    content:   list[RawElement] = field(default_factory=list)  

    @property
    def total(self) -> int:
        return len(self.elements)





def _import_partition_pdf():
    """Lazy import so the module loads even if unstructured isn't installed."""
    try:
        from unstructured.partition.pdf import partition_pdf  
        return partition_pdf
    except ImportError as exc:
        raise ImportError(
            "The 'unstructured' package is required for PDF extraction.\n"
            "Install with: pip install 'unstructured[pdf]' "
            "or 'unstructured[all-docs]' for full support."
        ) from exc


def extract_pdf(
    pdf_path: Path,
    strategy: str = PDF_STRATEGY,
) -> ExtractedDocument:
    """
    Parse *pdf_path* with the unstructured library and return an
    ExtractedDocument containing typed RawElement objects.

    Parameters
    ----------
    pdf_path : Path to the PDF file.
    strategy : "hi_res" (default, needs detectron2) | "fast" | "auto".

    Notes
    -----
    `infer_table_structure=True`  → unstructured will attempt to convert
        detected table regions into HTML (stored in element.metadata.text_as_html).
    `include_page_breaks=False`   → we track page numbers via element.metadata.
    """
    partition_pdf = _import_partition_pdf()

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info("Extracting '%s' with strategy='%s'", pdf_path.name, strategy)

    
    raw_elements = None
    for attempt_strategy in _strategy_fallback_order(strategy):
        try:
            raw_elements = partition_pdf(
                filename=str(pdf_path),
                strategy=attempt_strategy,
                infer_table_structure=True,   
                include_page_breaks=False,
                
                ocr_languages=None,
                
                extract_images_in_pdf=False,
            )
            logger.info(
                "Extraction succeeded with strategy='%s' (%d raw elements)",
                attempt_strategy,
                len(raw_elements),
            )
            break
        except Exception as exc: 
            logger.warning(
                "strategy='%s' failed (%s). Trying fallback.", attempt_strategy, exc
            )

    if raw_elements is None:
        raise RuntimeError(f"All extraction strategies failed for '{pdf_path.name}'.")

    
    doc = ExtractedDocument(
        source_file=pdf_path.name,
        file_path=pdf_path,
    )

    for el in raw_elements:
        el_type   = type(el).__name__          
        el_text   = (el.text or "").strip()
        el_html   = getattr(el.metadata, "text_as_html", "") or ""
        el_page   = getattr(el.metadata, "page_number", 1) or 1
        el_id     = getattr(el, "id", "") or str(id(el))

        if not el_text and not el_html:
            continue  

        raw = RawElement(
            element_type=el_type,
            text=el_text,
            html=el_html,
            page_number=int(el_page),
            source_file=pdf_path.name,
            element_id=el_id,
        )

        doc.elements.append(raw)

        
        if el_type in ELEMENT_TYPES_STRUCTURAL:
            doc.titles.append(raw)
        elif el_type == ELEMENT_TYPE_TABLE:
            doc.tables.append(raw)
        elif el_type in ELEMENT_TYPES_CONTENT:
            doc.content.append(raw)
        

    logger.info(
        "'%s' → %d titles, %d tables, %d content elements",
        pdf_path.name,
        len(doc.titles),
        len(doc.tables),
        len(doc.content),
    )
    return doc


def _strategy_fallback_order(preferred: str) -> list[str]:
    """Return strategies to try in order, starting with *preferred*."""
    all_strategies = ["hi_res", "fast", "auto"]
    ordered = [preferred] + [s for s in all_strategies if s != preferred]
    return ordered





def extract_all_pdfs(
    data_dir: Path = DATA_DIR,
    strategy: str = PDF_STRATEGY,
) -> Generator[ExtractedDocument, None, None]:
    """
    Yield an ExtractedDocument for every PDF found recursively under *data_dir*.

    Skips files that fail extraction and logs the error so one bad PDF
    doesn't abort the whole pipeline.
    """
    pdf_files = sorted(data_dir.rglob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found under '%s'", data_dir)
        return

    logger.info("Found %d PDF file(s) to process.", len(pdf_files))

    for pdf_path in pdf_files:
        try:
            yield extract_pdf(pdf_path, strategy=strategy)
        except Exception as exc:  
            logger.error("Skipping '%s': %s", pdf_path.name, exc)
            continue
