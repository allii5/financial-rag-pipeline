
from __future__ import annotations

import argparse
import logging
import os
import sys
import textwrap
import time
import uuid
from typing import Any, Iterator

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI


from config import LOGS_DIR, OPENAI_API_KEY, KNOWN_BANKS
from vectorstore import BankingVectorStore, get_vector_store



logging.basicConfig(
    level=logging.WARNING,                       
    format="%(asctime)s [%(levelname)-8s] %(name)s – %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "assistant.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)



_USE_COLOR = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def clr_cyan(t: str)    -> str: return _c("96", t)
def clr_yellow(t: str)  -> str: return _c("93", t)
def clr_green(t: str)   -> str: return _c("92", t)
def clr_red(t: str)     -> str: return _c("91", t)
def clr_bold(t: str)    -> str: return _c("1",  t)
def clr_dim(t: str)     -> str: return _c("2",  t)
def clr_magenta(t: str) -> str: return _c("95", t)



CHAT_MODEL        = "gpt-4o-mini"
TEMPERATURE       = 0.1          
MAX_TOKENS        = 1_500        
MMR_K             = 6            
MMR_FETCH_K       = 20           
MMR_LAMBDA        = 0.65         



_CONTEXTUALIZE_Q_SYSTEM = """\
You are a query reformulation assistant for a Pakistani car finance knowledge base.

Your ONLY task is to rewrite the user's latest message into a fully self-contained \
search query that can be understood without seeing the chat history.

Rules:
- If the message already stands alone (e.g., "What is KIBOR?"), return it unchanged.
- If it contains pronouns or references ("their", "that bank", "the same one"), \
  resolve them using the chat history and embed the resolved entity in the query.
- Expand shorthand Pakistani banking terms: \
  "RDA" → "Roshan Digital Account", "DM" → "Diminishing Musharakah".
- Output ONLY the rewritten query. No explanation, no preamble.
"""

CONTEXTUALIZE_Q_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _CONTEXTUALIZE_Q_SYSTEM),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])



_QA_SYSTEM = """\
You are an expert Pakistani Car Finance Consultant with deep knowledge of \
Islamic and conventional vehicle financing products offered by Pakistani banks \
as of 2025. You work strictly from the retrieved document excerpts provided \
to you below.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CORE OPERATING RULES (non-negotiable)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. STRICT GROUNDING
   Answer ONLY from the <context> below. Do NOT use your general parametric \
knowledge about finance, Pakistan, or any other topic. If the answer is not \
present in the context, say exactly:
   "I don't have specific information on that in my current knowledge base. \
Please check directly with the bank or refer to their official documentation."

2. DOMAIN FIREWALL
   If the user's question is NOT related to Pakistani car financing (e.g., \
they ask about politics, sports, cooking, general knowledge, or any other \
financial product like home loans or credit cards), respond ONLY with:
   "I'm your dedicated Car Finance Consultant for Pakistani banks. I can only \
assist with vehicle financing queries — such as loan eligibility, profit rates, \
down payments, tenure, or bank comparisons. How can I help you with car finance?"

3. FINANCIAL PRECISION
   - Quote KIBOR rates, BPS spreads, SBP caps, and down payment percentages \
exactly as stated in the context. Never round or paraphrase figures.
   - Distinguish clearly between Islamic products (Ijarah, Diminishing \
Musharakah, Murabaha — using Takaful, profit rate) and conventional products \
(insurance, interest/markup rate, KIBOR-linked).
   - When the SBP financing cap (PKR 3 million) differs from a bank's own \
higher cap (e.g., Meezan's PKR 10 million), explicitly call out both.

4. COMPARISON QUERIES
   When comparing multiple banks, present information in a structured manner. \
Use a clear heading per bank. Never conflate figures from different banks.

5. BILINGUAL SUPPORT
   You understand queries written in Roman Urdu, Urdu-English mix (e.g., \
"Meezan ka down payment kitna hai?", "kya used car pe loan milta hai?"). \
Always respond in professional English UNLESS the user explicitly asks \
"reply in Urdu" or "Urdu mein batao", in which case respond in standard Urdu \
(Nastaliq script).

6. MANDATORY CITATIONS
   Every response MUST end with a "Sources:" section listing the bank name(s) \
and page number(s) you drew information from. Format:
   Sources:
   • [Bank Name] – Page [N]
   • [Bank Name] – Page [N]
   If the same bank appears on multiple pages, list each page separately.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

<context>
{context}
</context>
"""

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _QA_SYSTEM),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])



DOCUMENT_PROMPT = PromptTemplate(
    input_variables=[
        "page_content",
        "bank_name",
        "financing_type",
        "document_type",
        "page_number",
        "section_header",
        "customer_segment",
        "vehicle_type",
    ],
    template=(
        "╔══ [{bank_name}] | Financing: {financing_type} | "
        "Type: {document_type} | Page: {page_number} ══╗\n"
        "  Segment: {customer_segment} | Vehicle: {vehicle_type}\n"
        "  Section: {section_header}\n"
        "╠══════════════════════════════════════════════════════╣\n"
        "{page_content}\n"
        "╚══════════════════════════════════════════════════════╝"
    ),
)


def format_retrieved_docs(docs: list[Document]) -> str:
    """
    Render a list of retrieved Documents into the {context} string.

    Applies DOCUMENT_PROMPT to each doc so the LLM sees rich metadata
    alongside content. Falls back gracefully for any missing metadata keys.
    """
    if not docs:
        return "No relevant documents were retrieved for this query."

    rendered: list[str] = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata

        
        variables = {
            "page_content":    doc.page_content,
            "bank_name":       meta.get("bank_name", "Unknown Bank"),
            "financing_type":  meta.get("financing_type", "unknown"),
            "document_type":   meta.get("document_type", "text"),
            "page_number":     str(meta.get("page_number", "N/A")),
            "section_header":  (meta.get("section_header", "") or "")[:120],
            "customer_segment": meta.get("customer_segment", "general"),
            "vehicle_type":    meta.get("vehicle_type", "unknown"),
        }

        try:
            formatted = DOCUMENT_PROMPT.format(**variables)
        except KeyError as e:
            
            logger.warning("DOCUMENT_PROMPT missing key %s; using raw content", e)
            formatted = f"[Chunk {i} | {variables['bank_name']} p.{variables['page_number']}]\n{doc.page_content}"

        rendered.append(f"[DOCUMENT {i} of {len(docs)}]\n{formatted}")

    return "\n\n".join(rendered)



_FINANCE_SIGNALS = frozenset([
    "car", "vehicle", "auto", "loan", "finance", "financing", "bank",
    "kibor", "ijarah", "musharakah", "murabaha", "takaful", "insurance",
    "down payment", "markup", "profit rate", "tenure", "installment",
    "emi", "sbp", "nrp", "roshan", "conventional", "islamic", "shariah",
    "salaried", "self employed", "agriculturist", "gaadi", "gari",
    "qisht", "byaan", "raqam", "rate", "percent", "%", "pkr", "rs.",
    "meezan", "alfalah", "hbl", "mcb", "ubl", "allied", "faysal",
    "askari", "soneri", "habib", "standard chartered",
])

_HARD_OFF_TOPIC = frozenset([
    "cricket", "football", "hockey", "imran khan", "politics", "election",
    "recipe", "cooking", "weather", "temperature", "movie", "film", "song",
    "capital city", "president", "prime minister", "army",
])


def is_finance_query(text: str) -> bool:
    """
    Heuristic: return True if the query is plausibly car-finance related.

    Logic:
      1. If hard off-topic keywords are present → False.
      2. If any finance signal keyword is present → True.
      3. Short queries (≤ 6 words) are passed through (LLM handles them).
      4. Long unknown queries → False (forces polite decline).
    """
    lower = text.lower()

    if any(kw in lower for kw in _HARD_OFF_TOPIC):
        return False

    if any(kw in lower for kw in _FINANCE_SIGNALS):
        return True

    word_count = len(text.split())
    return word_count <= 3    


_OFF_TOPIC_RESPONSE = (
    "I'm your dedicated Car Finance Consultant for Pakistani banks. "
    "I can only assist with vehicle financing queries — such as loan eligibility, "
    "profit rates, down payments, tenure, or bank comparisons. "
    "How can I help you with car finance today?"
)




class CarFinanceAssistant:
    """
    Encapsulates the full LCEL RAG chain with per-session memory.

    Lifecycle
    ─────────
    assistant = CarFinanceAssistant.build()
    response  = assistant.ask("What is Meezan's down payment?", session_id="s1")

    The assistant keeps separate ChatMessageHistory per session_id, enabling
    concurrent users in a production deployment (e.g., FastAPI + WebSockets).
    """

    def __init__(
        self,
        chain_with_history: RunnableWithMessageHistory,
        debug: bool = False,
    ) -> None:
        self._chain    = chain_with_history
        self._debug    = debug
        self._sessions: dict[str, ChatMessageHistory] = {}

    

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """Return (or create) the ChatMessageHistory for *session_id*."""
        if session_id not in self._sessions:
            self._sessions[session_id] = ChatMessageHistory()
            logger.info("New session created: %s", session_id)
        return self._sessions[session_id]

    def clear_session(self, session_id: str) -> None:
        """Wipe the message history for *session_id*."""
        if session_id in self._sessions:
            self._sessions[session_id].clear()
            logger.info("Session cleared: %s", session_id)

    def session_turn_count(self, session_id: str) -> int:
        """Return the number of human turns in *session_id*."""
        hist = self._sessions.get(session_id)
        if hist is None:
            return 0
        return sum(1 for m in hist.messages if m.type == "human")

    

    def stream(
        self,
        user_input: str,
        session_id: str,
    ) -> Iterator[str]:
        """
        Yield answer tokens one by one.

        The caller is responsible for printing / buffering.
        Yields the empty string as a sentinel when streaming is complete.

        Off-topic queries are intercepted here and yield the canned
        _OFF_TOPIC_RESPONSE character-by-character (for consistent UX).
        """
        if not is_finance_query(user_input):
            logger.info("Off-topic query intercepted: %s", user_input[:80])
            yield from _OFF_TOPIC_RESPONSE
            return

        config = {"configurable": {"session_id": session_id}}

        try:
            for chunk in self._chain.stream({"input": user_input}, config=config):
                
                token = chunk.get("answer", "")
                if token:
                    yield token

                
                if self._debug and "context" in chunk and chunk["context"]:
                    self._print_debug_context(chunk["context"])

        except Exception as exc:
            logger.exception("Stream error for session %s", session_id)
            yield (
                f"\n\n[System Error] An error occurred while generating the response. "
                f"Please try again. (Detail: {exc})"
            )

    def _print_debug_context(self, docs: list[Document]) -> None:
        """Print retrieved chunks to stderr in debug mode."""
        print("\n" + clr_dim("─" * 60), file=sys.stderr)
        print(clr_dim(f"  DEBUG: {len(docs)} chunks retrieved"), file=sys.stderr)
        for i, doc in enumerate(docs, 1):
            m = doc.metadata
            print(
                clr_dim(
                    f"  [{i}] {m.get('bank_name','?')} "
                    f"| {m.get('financing_type','?')} "
                    f"| p.{m.get('page_number','?')} "
                    f"| {m.get('document_type','?')}"
                ),
                file=sys.stderr,
            )
            snippet = doc.page_content[:120].replace("\n", " ")
            print(clr_dim(f"      {snippet}…"), file=sys.stderr)
        print(clr_dim("─" * 60), file=sys.stderr)


    @classmethod
    def build(
        cls,
        store: BankingVectorStore | None = None,
        metadata_filter: dict[str, Any] | None = None,
        debug: bool = False,
    ) -> "CarFinanceAssistant":
        """
        Assemble the full LCEL pipeline and return a ready CarFinanceAssistant.

        Parameters
        ----------
        store           : BankingVectorStore instance (uses singleton if None).
        metadata_filter : Optional ChromaDB `where` filter applied to all
                          retrievals, e.g. {"bank_name": "Meezan Bank"} or
                          {"financing_type": "islamic"}.
        debug           : If True, print retrieved chunks to stderr.
        """
        if not OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. "
                "Run: export OPENAI_API_KEY='sk-...'"
            )

        vs = store or get_vector_store()

        
        llm = ChatOpenAI(
            model=CHAT_MODEL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            openai_api_key=OPENAI_API_KEY,
            streaming=True,     
        )

        
        base_retriever = vs.as_mmr_retriever(
            k=MMR_K,
            fetch_k=MMR_FETCH_K,
            lambda_mult=MMR_LAMBDA,
            where=metadata_filter,
        )

    
        history_aware_retriever = create_history_aware_retriever(
            llm=llm,
            retriever=base_retriever,
            prompt=CONTEXTUALIZE_Q_PROMPT,
        )

        
        question_answer_chain = create_stuff_documents_chain(
            llm=llm,
            prompt=QA_PROMPT,
            document_prompt=DOCUMENT_PROMPT,
            document_separator="\n\n" + "─" * 60 + "\n\n",
        )

        
        rag_chain = create_retrieval_chain(
            retriever=history_aware_retriever,
            combine_docs_chain=question_answer_chain,
        )

        
        instance = cls(chain_with_history=None, debug=debug)  # type: ignore[arg-type]

        chain_with_history = RunnableWithMessageHistory(
            rag_chain,
            instance.get_session_history,        
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        
        instance._chain = chain_with_history
        return instance




_BANNER = """

║          🚗  Pakistani Car Finance Consultant  🚗                    ║
║          Powered by GPT-4o-mini + ChromaDB (RAG)                    ║
║                                                                      ║
║  Commands:                                                           ║
║    /clear    → Reset conversation history                           ║
║    /debug    → Toggle retrieved-chunks display                      ║
║    /filter   → Show active metadata filter                          ║
║    /banks    → List all 22 supported banks                          ║
║    /help     → Show this menu                                       ║
║    /quit     → Exit                                                  ║

"""

_BANKS_LIST = "\n".join(
    f"  • {name}" for name in sorted(KNOWN_BANKS.keys())
)


def _parse_filter_arg(raw: str | None) -> dict[str, Any] | None:
    """
    Parse --filter CLI argument into a ChromaDB where dict.

    Supported forms:
      bank="Meezan Bank"   → {"bank_name": "Meezan Bank"}
      type=islamic          → {"financing_type": "islamic"}
      type=conventional     → {"financing_type": "conventional"}
    """
    if not raw:
        return None

    lower = raw.lower().strip()
    if lower.startswith("bank="):
        bank_val = raw.split("=", 1)[1].strip().strip('"').strip("'")
        return {"bank_name": bank_val}
    if lower.startswith("type="):
        type_val = raw.split("=", 1)[1].strip().lower()
        return {"financing_type": type_val}

    print(clr_yellow(f"  Warning: unrecognised --filter '{raw}'. Ignoring."))
    return None


def _print_streaming(token_iterator: Iterator[str]) -> str:
    """
    Print tokens to stdout as they arrive (typewriter effect).

    Returns the full assembled response string for optional logging.
    """
    full = []
    try:
        for token in token_iterator:
            print(token, end="", flush=True)
            full.append(token)
    except KeyboardInterrupt:
        print(clr_dim("\n  [Interrupted]"))
    return "".join(full)


def _handle_command(
    cmd: str,
    assistant: CarFinanceAssistant,
    session_id: str,
    active_filter: dict | None,
) -> bool:
    """
    Process a /command.  Returns True if a command was handled (skip LLM call).
    """
    cmd = cmd.strip().lower()

    if cmd in ("/quit", "/exit", "/q"):
        print(clr_cyan("\n  Goodbye! Thank you for using the Car Finance Consultant."))
        sys.exit(0)

    if cmd == "/clear":
        assistant.clear_session(session_id)
        print(clr_green("  ✓ Conversation history cleared.\n"))
        return True

    if cmd == "/debug":
        assistant._debug = not assistant._debug
        state = "ON" if assistant._debug else "OFF"
        print(clr_yellow(f"  ✓ Debug mode: {state}\n"))
        return True

    if cmd == "/banks":
        print(clr_cyan("\n  Supported Banks:\n") + _BANKS_LIST + "\n")
        return True

    if cmd == "/filter":
        if active_filter:
            print(clr_cyan(f"  Active filter: {active_filter}\n"))
        else:
            print(clr_dim("  No metadata filter active.\n"))
        return True

    if cmd == "/help":
        print(_BANNER)
        return True

    return False





def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pakistani Car Finance Consultant – RAG Chatbot"
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Print retrieved chunks to stderr after each query.",
    )
    p.add_argument(
        "--filter",
        metavar="EXPR",
        default=None,
        help=(
            'Pre-filter all retrievals. Examples: '
            '--filter \'bank="Meezan Bank"\' or --filter type=islamic'
        ),
    )
    p.add_argument(
        "--session",
        default=None,
        help="Session ID (default: auto-generated UUID). Useful for testing.",
    )
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()

    active_filter = _parse_filter_arg(args.filter)
    session_id    = args.session or str(uuid.uuid4())[:8]

    
    print(clr_bold(clr_cyan(_BANNER)))

    if active_filter:
        print(clr_yellow(f"  Active filter: {active_filter}\n"))

    print(clr_dim(f"  Session ID : {session_id}"))
    print(clr_dim(f"  Model      : {CHAT_MODEL}"))
    print(clr_dim(f"  Retrieval  : MMR k={MMR_K}, fetch_k={MMR_FETCH_K}, λ={MMR_LAMBDA}"))
    print()

    
    print(clr_dim("  Initialising vector store and building RAG chain…"), end="", flush=True)
    try:
        assistant = CarFinanceAssistant.build(
            metadata_filter=active_filter,
            debug=args.debug,
        )
    except RuntimeError as e:
        print(clr_red(f"\n  ✗ Startup failed: {e}"))
        sys.exit(1)
    print(clr_green(" ready.\n"))

    
    while True:
        
        turn = assistant.session_turn_count(session_id) + 1
        try:
            user_input = input(clr_bold(clr_green(f"You [{turn}]: "))).strip()
        except (EOFError, KeyboardInterrupt):
            print(clr_cyan("\n\n  Goodbye!"))
            sys.exit(0)

        if not user_input:
            continue

        
        if user_input.startswith("/"):
            _handle_command(user_input, assistant, session_id, active_filter)
            continue

        
        print(clr_bold(clr_cyan("\nConsultant: ")), end="", flush=True)
        t0 = time.perf_counter()

        full_response = _print_streaming(
            assistant.stream(user_input, session_id=session_id)
        )

        elapsed = time.perf_counter() - t0
        token_estimate = len(full_response.split())
        print(
            clr_dim(f"\n\n  [{elapsed:.1f}s | ~{token_estimate} tokens]\n")
        )
