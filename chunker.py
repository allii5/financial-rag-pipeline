
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Generator

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHUNK_OVERLAP_CHARS,
    ELEMENT_TYPE_TABLE,
    ELEMENT_TYPES_STRUCTURAL,
    MAX_CHUNK_CHARS,
    MIN_CHUNK_CHARS,
)
from extractor import ExtractedDocument, RawElement
from metadata_utils import build_metadata, infer_bank

logger = logging.getLogger(__name__)




def _make_splitter() -> RecursiveCharacterTextSplitter:
    """
    Text splitter tuned for dense financial prose.

    Separators prioritise paragraph breaks → sentence breaks → word breaks
    so we don't cut mid-sentence through a rate description.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHUNK_CHARS,
        chunk_overlap=CHUNK_OVERLAP_CHARS,
        separators=["\n\n", "\n", ". ", ", ", " ", ""],
        length_function=len,
    )


@dataclass
class _SectionAccumulator:
    """Holds content elements grouped under the current section header."""
    header_text: str = ""
    header_page: int = 1
    elements: list[RawElement] = field(default_factory=list)

    def flush_text(self) -> str:
        """Concatenate all accumulated element texts into one block."""
        return "\n\n".join(el.text for el in self.elements if el.text)

    def is_empty(self) -> bool:
        return not self.elements

    def reset(self, new_header: str, page: int) -> None:
        self.header_text = new_header
        self.header_page = page
        self.elements = []





class BankingDocumentChunker:
    """
    Converts an ExtractedDocument into a list of LangChain Documents,
    preserving table context and section hierarchy.

    Usage
    -----
    chunker = BankingDocumentChunker()
    docs = chunker.chunk(extracted_doc)
    """

    def __init__(self) -> None:
        self._splitter = _make_splitter()

    

    def chunk(self, doc: ExtractedDocument) -> list[Document]:
        """
        Main entry point.  Processes all elements in document order and
        returns a flat list of LangChain Documents ready for embedding.
        """
        
        bank_name, _ = infer_bank("", doc.source_file)
        bank_override = bank_name if bank_name != "Unknown" else None

        chunks: list[Document] = []
        accumulator = _SectionAccumulator()

        for el in doc.elements:

            if el.element_type in ELEMENT_TYPES_STRUCTURAL:
                
                if not accumulator.is_empty():
                    chunks.extend(
                        self._emit_text_chunks(accumulator, doc.source_file, bank_override)
                    )
                accumulator.reset(el.text, el.page_number)

            elif el.element_type == ELEMENT_TYPE_TABLE:
                
                if not accumulator.is_empty():
                    chunks.extend(
                        self._emit_text_chunks(accumulator, doc.source_file, bank_override)
                    )
                    accumulator.elements = []  # keep header, clear content

                
                table_doc = self._emit_table_chunk(
                    el, accumulator, doc.source_file, bank_override
                )
                if table_doc:
                    chunks.append(table_doc)

            else:
                
                if el.text and len(el.text) >= MIN_CHUNK_CHARS:
                    accumulator.elements.append(el)

        
        if not accumulator.is_empty():
            chunks.extend(
                self._emit_text_chunks(accumulator, doc.source_file, bank_override)
            )

        logger.info(
            "'%s' → %d final chunks (tables + text)",
            doc.source_file,
            len(chunks),
        )
        return chunks

    

    def _emit_table_chunk(
        self,
        el: RawElement,
        acc: _SectionAccumulator,
        source_file: str,
        bank_override: str | None,
    ) -> Document | None:
        """
        Create a single Document for a Table element.

        page_content contains the HTML (richest form); if HTML is absent
        (strategy=fast) we fall back to plain text.
        """
        content = el.html if el.html else el.text
        if not content or len(content) < MIN_CHUNK_CHARS:
            logger.debug("Skipping empty/tiny table on page %d", el.page_number)
            return None

        
        context_prefix = (
            f"[Section: {acc.header_text}]\n\n" if acc.header_text else ""
        )
        page_content = context_prefix + content

        metadata = build_metadata(
            text=el.text,                    
            source_file=source_file,
            page_number=el.page_number,
            element_type=el.element_type,
            section_header=acc.header_text,
            bank_name_override=bank_override,
            chunk_index=0,
            total_chunks=1,
        )

        return Document(page_content=page_content, metadata=metadata)

    def _emit_text_chunks(
        self,
        acc: _SectionAccumulator,
        source_file: str,
        bank_override: str | None,
    ) -> list[Document]:
        """
        Split accumulated text content under the current section into one or
        more Documents, each enriched with metadata.
        """
        combined_text = acc.flush_text()
        if not combined_text or len(combined_text) < MIN_CHUNK_CHARS:
            return []

        
        if len(combined_text) <= MAX_CHUNK_CHARS:
            sub_chunks = [combined_text]
        else:
            sub_chunks = self._splitter.split_text(combined_text)

        
        sub_chunks = [c for c in sub_chunks if len(c) >= MIN_CHUNK_CHARS]
        total = len(sub_chunks)

        documents: list[Document] = []
        for idx, chunk_text in enumerate(sub_chunks):
            
            if acc.header_text:
                page_content = f"[Section: {acc.header_text}]\n\n{chunk_text}"
            else:
                page_content = chunk_text

            
            page_num = acc.elements[0].page_number if acc.elements else acc.header_page

            metadata = build_metadata(
                text=chunk_text,
                source_file=source_file,
                page_number=page_num,
                element_type=acc.elements[0].element_type if acc.elements else "NarrativeText",
                section_header=acc.header_text,
                bank_name_override=bank_override,
                chunk_index=idx,
                total_chunks=total,
            )

            documents.append(Document(page_content=page_content, metadata=metadata))

        return documents





def chunk_all_documents(
    extracted_docs: list[ExtractedDocument],
) -> Generator[list[Document], None, None]:
    """
    Yield a list of Documents for each ExtractedDocument.

    Caller can flatten: `all_docs = [d for batch in chunk_all_documents(…) for d in batch]`
    """
    chunker = BankingDocumentChunker()
    for doc in extracted_docs:
        try:
            yield chunker.chunk(doc)
        except Exception as exc:  
            logger.error("Chunking failed for '%s': %s", doc.source_file, exc)
            yield []
