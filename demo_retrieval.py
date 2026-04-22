

from __future__ import annotations

import logging
from vectorstore import get_vector_store

logging.basicConfig(level=logging.INFO, format="%(levelname)s – %(message)s")


def print_docs(title: str, docs: list) -> None:
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}")
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        print(f"\n[{i}] Bank: {meta.get('bank_name')} | "
              f"Type: {meta.get('financing_type')} | "
              f"DocType: {meta.get('document_type')} | "
              f"Page: {meta.get('page_number')}")
        print(f"    Section: {meta.get('section_header', '')[:80]}")
        print(f"    Content: {doc.page_content[:200].strip()!r}")


def main() -> None:
    store = get_vector_store()
    print(f"\nTotal indexed chunks: {store.count()}")

    
    retriever = store.as_retriever(k=3)
    docs = retriever.invoke("What is the down payment requirement for car financing?")
    print_docs("Q1: General – Down payment requirements", docs)

    
    islamic_table_retriever = store.as_retriever(
        k=4,
        where={
            "$and": [
                {"financing_type": {"$in": ["islamic", "both"]}},
                {"document_type": "table"},
            ]
        },
    )
    docs = islamic_table_retriever.invoke("profit rate charges for car ijarah")
    print_docs("Q2: Islamic banks – Table chunks – Rate/Charges", docs)

    
    meezan_retriever = store.as_retriever(
        k=4,
        where={"bank_name": "Meezan Bank"},
    )
    docs = meezan_retriever.invoke("eligibility for overseas Pakistanis NRP")
    print_docs("Q3: Meezan Bank – NRP / Overseas eligibility", docs)

    
    rate_retriever = store.as_retriever(
        k=4,
        where={
            "$and": [
                {"has_rate_info": True},
                {"financing_type": "conventional"},
            ]
        },
    )
    docs = rate_retriever.invoke("KIBOR linked markup rate for salaried")
    print_docs("Q4: Conventional – Chunks with rate info", docs)

    
    used_retriever = store.as_retriever(
        k=3,
        where={"vehicle_type": {"$in": ["used", "both"]}},
    )
    docs = used_retriever.invoke("used car age limit maximum years old")
    print_docs("Q5: Used vehicle eligibility rules", docs)

    
    mmr_retriever = store.as_mmr_retriever(
        k=5, fetch_k=20, lambda_mult=0.6,
        where={"has_down_payment": True},
    )
    docs = mmr_retriever.invoke("minimum equity contribution percentage car loan")
    print_docs("Q6: MMR – Down payment info across multiple banks", docs)

    
    hbl_chunks = store.get_by_bank("HBL", limit=5)
    print(f"\n\nAudit – HBL chunks in store: {len(hbl_chunks)}")
    for chunk in hbl_chunks[:2]:
        print(f"  section={chunk.metadata.get('section_header', '')[:60]}")


if __name__ == "__main__":
    main()
