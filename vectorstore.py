
from __future__ import annotations

import hashlib
import logging
from typing import Any

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DIR,
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
)

logger = logging.getLogger(__name__)




def get_embedding_model() -> OpenAIEmbeddings:
    """
    Return a configured OpenAIEmbeddings instance.

    Raises RuntimeError early if the API key is not set so the error is
    clear rather than a cryptic auth failure later.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is not set.\n"
            "Export it before running: export OPENAI_API_KEY='sk-...'"
        )
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY,
        
        chunk_size=500,
    )





def get_chroma_client() -> chromadb.PersistentClient:
    """Return a persistent ChromaDB client rooted at CHROMA_DIR."""
    return chromadb.PersistentClient(path=str(CHROMA_DIR))



_CHROMADB_ALLOWED = (str, int, float, bool)


def sanitise_metadata(raw: dict[str, Any]) -> dict[str, Any]:
    """
    Convert metadata dict so every value is ChromaDB-compatible.

    Rules:
      • None           → ""   (empty string)
      • list / tuple   → comma-joined string
      • dict           → JSON string
      • other          → str(value)
    """
    import json

    clean: dict[str, Any] = {}
    for k, v in raw.items():
        if v is None:
            clean[k] = ""
        elif isinstance(v, _CHROMADB_ALLOWED):
            clean[k] = v
        elif isinstance(v, (list, tuple)):
            clean[k] = ", ".join(str(i) for i in v)
        elif isinstance(v, dict):
            clean[k] = json.dumps(v)
        else:
            clean[k] = str(v)
    return clean





def doc_id(doc: Document) -> str:
    """
    SHA-256 based ID using (source_file, page_number, chunk_index, content_prefix).

    Ensures re-running the pipeline on the same PDFs doesn't create duplicates.
    """
    sig = (
        f"{doc.metadata.get('source_file', '')}::"
        f"{doc.metadata.get('page_number', 0)}::"
        f"{doc.metadata.get('chunk_index', 0)}::"
        f"{doc.page_content[:120]}"
    )
    return hashlib.sha256(sig.encode()).hexdigest()[:32]





class BankingVectorStore:
    """
    Thin wrapper around LangChain's Chroma integration that adds:
      • Batch upsert with deterministic IDs
      • Metadata sanitisation
      • Domain-specific retriever factory
      • Stats / inspection helpers
    """

    def __init__(
        self,
        embedding_model: OpenAIEmbeddings | None = None,
        collection_name: str = CHROMA_COLLECTION_NAME,
    ) -> None:
        self._embeddings = embedding_model or get_embedding_model()
        self._collection_name = collection_name
        self._client = get_chroma_client()

        
        self._store = Chroma(
            client=self._client,
            collection_name=self._collection_name,
            embedding_function=self._embeddings,
        )
        logger.info(
            "BankingVectorStore ready → collection='%s' @ '%s'",
            collection_name,
            CHROMA_DIR,
        )

    

    def upsert_documents(
        self,
        documents: list[Document],
        batch_size: int = 100,
    ) -> int:
        """
        Upsert *documents* into ChromaDB in batches.

        Uses deterministic IDs so duplicate PDFs won't create duplicate chunks.
        Returns the number of documents successfully upserted.
        """
        if not documents:
            logger.warning("upsert_documents called with empty list.")
            return 0

        
        prepped: list[Document] = []
        ids: list[str] = []

        for doc in documents:
            clean_meta = sanitise_metadata(doc.metadata)
            prepped.append(Document(page_content=doc.page_content, metadata=clean_meta))
            ids.append(doc_id(doc))

        
        total = 0
        for start in range(0, len(prepped), batch_size):
            batch_docs = prepped[start : start + batch_size]
            batch_ids  = ids[start : start + batch_size]
            try:
                self._store.add_documents(documents=batch_docs, ids=batch_ids)
                total += len(batch_docs)
                logger.info(
                    "Upserted batch %d–%d (%d docs)",
                    start,
                    start + len(batch_docs) - 1,
                    len(batch_docs),
                )
            except Exception as exc:
                logger.error("Batch %d–%d failed: %s", start, start + batch_size, exc)

        logger.info("Total upserted: %d / %d", total, len(documents))
        return total

    

    def as_retriever(
        self,
        k: int = 6,
        where: dict[str, Any] | None = None,
    ):
        """
        Return a LangChain retriever with optional ChromaDB metadata pre-filter.

        Parameters
        ----------
        k     : Number of chunks to retrieve.
        where : ChromaDB `where` filter dict, e.g.:
                  {"bank_name": "Meezan Bank"}
                  {"$and": [{"financing_type": "islamic"}, {"has_rate_info": True}]}

        Examples
        --------
        # Only Islamic banks, only table chunks:
        retriever = store.as_retriever(
            k=4,
            where={"$and": [{"financing_type": "islamic"}, {"document_type": "table"}]}
        )
        """
        search_kwargs: dict[str, Any] = {"k": k}
        if where:
            search_kwargs["filter"] = where

        return self._store.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs,
        )

    def as_mmr_retriever(
        self,
        k: int = 6,
        fetch_k: int = 20,
        lambda_mult: float = 0.7,
        where: dict[str, Any] | None = None,
    ):
        """
        Maximum Marginal Relevance retriever – reduces redundancy in results.
        Recommended for queries spanning multiple banks.

        lambda_mult: 1.0 = pure similarity, 0.0 = pure diversity.
        """
        search_kwargs: dict[str, Any] = {
            "k": k,
            "fetch_k": fetch_k,
            "lambda_mult": lambda_mult,
        }
        if where:
            search_kwargs["filter"] = where

        return self._store.as_retriever(
            search_type="mmr",
            search_kwargs=search_kwargs,
        )

    

    def count(self) -> int:
        """Return total number of indexed vectors."""
        col = self._client.get_collection(self._collection_name)
        return col.count()

    def peek(self, n: int = 5) -> list[dict]:
        """Return the first *n* raw records (for debugging)."""
        col = self._client.get_collection(self._collection_name)
        result = col.peek(limit=n)
        records = []
        for i, doc_id_ in enumerate(result.get("ids", [])):
            records.append({
                "id": doc_id_,
                "document": (result.get("documents") or [""])[i][:200],
                "metadata": (result.get("metadatas") or [{}])[i],
            })
        return records

    def get_by_bank(self, bank_name: str, limit: int = 20) -> list[Document]:
        """
        Convenience: retrieve all chunks for a specific bank without a query.
        Useful for auditing what was indexed.
        """
        col = self._client.get_collection(self._collection_name)
        result = col.get(
            where={"bank_name": bank_name},
            limit=limit,
            include=["documents", "metadatas"],
        )
        docs = []
        for text, meta in zip(
            result.get("documents", []),
            result.get("metadatas", []),
        ):
            docs.append(Document(page_content=text or "", metadata=meta or {}))
        return docs

    def delete_collection(self) -> None:
        """Wipe the collection (useful during development)."""
        self._client.delete_collection(self._collection_name)
        logger.warning("Collection '%s' deleted.", self._collection_name)
        # Re-initialise empty store
        self._store = Chroma(
            client=self._client,
            collection_name=self._collection_name,
            embedding_function=self._embeddings,
        )



_STORE_INSTANCE: BankingVectorStore | None = None


def get_vector_store(reset: bool = False) -> BankingVectorStore:
    """
    Return the singleton BankingVectorStore.

    Parameters
    ----------
    reset : If True, delete the existing collection and start fresh.
    """
    global _STORE_INSTANCE  
    if _STORE_INSTANCE is None:
        _STORE_INSTANCE = BankingVectorStore()
    if reset:
        _STORE_INSTANCE.delete_collection()
    return _STORE_INSTANCE
