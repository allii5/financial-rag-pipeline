# Financial RAG Pipeline - 2025 Car Financing Assistant

An AI-powered document intelligence system designed to parse, index, and query complex financial documents. This project specifically processes the **2025 Car Financing Guide** to provide accurate, context-aware answers regarding loan terms, interest rates, and eligibility.

## 🚀 Key Features
* **Automated PDF Extraction:** Sophisticated parsing of financial PDFs using `extractor.py`.
* **Vector Embeddings:** Utilizes OpenAI embeddings (or HuggingFace) for semantic search.
* **Vector Storage:** Persistent storage using **ChromaDB** for fast retrieval.
* **Contextual Retrieval:** RAG-based pipeline to minimize hallucinations in financial data.
* **Interactive UI:** (Optional: If you built a Streamlit/FastAPI frontend, mention it here).

## 🛠️ Tech Stack
* **Language:** Python 3.10+
* **Orchestration:** LangChain / LangGraph
* **Vector DB:** ChromaDB
* **LLM:** OpenAI GPT-4o (or your specific model)
* **Environment:** Docker (optional) / Virtualenv

## 📋 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/allii5/financial-rag-pipeline.git](https://github.com/allii5/financial-rag-pipeline.git)
   cd financial-rag-pipeline