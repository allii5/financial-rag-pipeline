

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from config import DATA_DIR, LOGS_DIR, PDF_STRATEGY
from extractor import extract_all_pdfs
from chunker import BankingDocumentChunker
from vectorstore import get_vector_store




def _configure_logging(level: int = logging.INFO) -> None:
    log_file = LOGS_DIR / "pipeline.log"
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8"),
    ]
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)-8s] %(name)s – %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


logger = logging.getLogger(__name__)





class PipelineStats:
    def __init__(self) -> None:
        self.pdfs_attempted    = 0
        self.pdfs_succeeded    = 0
        self.raw_elements      = 0
        self.raw_tables        = 0
        self.chunks_produced   = 0
        self.chunks_upserted   = 0
        self.start_time        = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.start_time

    def report(self) -> str:
        return (
            
            
            
            f"║  PDFs attempted    : {self.pdfs_attempted:<21}║\n"
            f"║  PDFs succeeded    : {self.pdfs_succeeded:<21}║\n"
            f"║  Raw elements      : {self.raw_elements:<21}║\n"
            f"║  Tables detected   : {self.raw_tables:<21}║\n"
            f"║  Chunks produced   : {self.chunks_produced:<21}║\n"
            f"║  Chunks upserted   : {self.chunks_upserted:<21}║\n"
            f"║  Elapsed (s)       : {self.elapsed():<21.1f}║\n"
            
        )





def run_ingestion_pipeline(
    pdf_dir: Path = DATA_DIR,
    strategy: str = PDF_STRATEGY,
    reset_store: bool = False,
    batch_size: int = 100,
) -> PipelineStats:
    """
    
    ----------
    pdf_dir     : Directory containing banking PDFs.
    strategy    : Unstructured extraction strategy ("hi_res" | "fast" | "auto").
    reset_store : Wipe the ChromaDB collection before indexing.
    batch_size  : Number of chunks per embedding API call.

    Returns
    -------
    PipelineStats object with counts and elapsed time.
    """
    stats = PipelineStats()
    chunker = BankingDocumentChunker()
    store = get_vector_store(reset=reset_store)

    logger.info("═" * 60)
    logger.info("PHASE 1+2+3 — Pakistani Banking RAG Ingestion Pipeline")
    logger.info("PDF directory : %s", pdf_dir)
    logger.info("Strategy      : %s", strategy)
    logger.info("Reset store   : %s", reset_store)
    logger.info("═" * 60)

    all_chunks = []

    
    for extracted_doc in extract_all_pdfs(pdf_dir, strategy=strategy):
        stats.pdfs_attempted += 1

        try:
            stats.raw_elements += extracted_doc.total
            stats.raw_tables   += len(extracted_doc.tables)

            logger.info(
                "[%s] Extracted %d elements (%d tables)",
                extracted_doc.source_file,
                extracted_doc.total,
                len(extracted_doc.tables),
            )

            chunks = chunker.chunk(extracted_doc)
            stats.chunks_produced += len(chunks)
            all_chunks.extend(chunks)

            logger.info(
                "[%s] → %d chunks ready for embedding",
                extracted_doc.source_file,
                len(chunks),
            )
            stats.pdfs_succeeded += 1

        except Exception as exc:  
            logger.error("[%s] Failed: %s", extracted_doc.source_file, exc)

    if not all_chunks:
        logger.warning("No chunks produced.  Check PDF directory and strategy.")
        return stats

    
    logger.info("Upserting %d chunks to ChromaDB …", len(all_chunks))
    stats.chunks_upserted = store.upsert_documents(all_chunks, batch_size=batch_size)

    
    logger.info("Total vectors in store: %d", store.count())
    logger.info(stats.report())

    return stats




def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the Pakistani Banking RAG ingestion pipeline."
    )
    p.add_argument(
        "--pdf-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing banking PDF files.",
    )
    p.add_argument(
        "--strategy",
        choices=["hi_res", "fast", "auto"],
        default=PDF_STRATEGY,
        help="Unstructured extraction strategy.",
    )
    p.add_argument(
        "--reset",
        action="store_true",
        help="Wipe ChromaDB collection before indexing.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Chunks per embedding API call.",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    _configure_logging(logging.DEBUG if args.debug else logging.INFO)

    stats = run_ingestion_pipeline(
        pdf_dir=args.pdf_dir,
        strategy=args.strategy,
        reset_store=args.reset,
        batch_size=args.batch_size,
    )
    sys.exit(0 if stats.chunks_upserted > 0 else 1)
