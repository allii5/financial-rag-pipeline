
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))



class TestQueryIntentGuard:

    def test_finance_keyword_passes(self):
        from assistant import is_finance_query
        assert is_finance_query("What is Meezan Bank's down payment?") is True

    def test_kibor_passes(self):
        from assistant import is_finance_query
        assert is_finance_query("KIBOR linked rate for car loan") is True

    def test_roman_urdu_passes(self):
        from assistant import is_finance_query
        assert is_finance_query("meezan ka qisht kitna hoga?") is True

    def test_ijarah_passes(self):
        from assistant import is_finance_query
        assert is_finance_query("Is ijarah available for used vehicles?") is True

    def test_cricket_blocked(self):
        from assistant import is_finance_query
        assert is_finance_query("Who won the cricket world cup?") is False

    def test_cooking_blocked(self):
        from assistant import is_finance_query
        assert is_finance_query("How do I make biryani?") is False

    def test_politics_blocked(self):
        from assistant import is_finance_query
        assert is_finance_query("What did Imran Khan say about elections?") is False

    def test_short_ambiguous_passes(self):
        # Short queries (<= 6 words) are passed through to let LLM decide
        from assistant import is_finance_query
        assert is_finance_query("Hi") is True

    def test_long_unknown_blocked(self):
        from assistant import is_finance_query
        # Long text with no finance keywords should be blocked
        result = is_finance_query(
            "Tell me all about the history of the Roman Empire in great detail please"
        )
        assert result is False


# ─── Document Formatter ───────────────────────────────────────────────────────

class TestDocumentFormatter:

    def _make_doc(self, **meta_overrides):
        from langchain_core.documents import Document
        default_meta = {
            "bank_name":       "Meezan Bank",
            "financing_type":  "islamic",
            "document_type":   "table",
            "page_number":     5,
            "section_header":  "Car Ijarah Charges",
            "customer_segment": "salaried",
            "vehicle_type":    "new",
        }
        default_meta.update(meta_overrides)
        return Document(page_content="Down payment: 15%", metadata=default_meta)

    def test_bank_name_in_output(self):
        from assistant import format_retrieved_docs
        result = format_retrieved_docs([self._make_doc()])
        assert "Meezan Bank" in result

    def test_page_number_in_output(self):
        from assistant import format_retrieved_docs
        result = format_retrieved_docs([self._make_doc(page_number=12)])
        assert "12" in result

    def test_financing_type_in_output(self):
        from assistant import format_retrieved_docs
        result = format_retrieved_docs([self._make_doc(financing_type="conventional")])
        assert "conventional" in result

    def test_content_in_output(self):
        from assistant import format_retrieved_docs
        doc = self._make_doc()
        doc.page_content = "Minimum 20% equity required."
        result = format_retrieved_docs([doc])
        assert "Minimum 20% equity required." in result

    def test_empty_list_returns_message(self):
        from assistant import format_retrieved_docs
        result = format_retrieved_docs([])
        assert "No relevant documents" in result

    def test_multiple_docs_labeled(self):
        from assistant import format_retrieved_docs
        docs = [self._make_doc(), self._make_doc(bank_name="HBL")]
        result = format_retrieved_docs(docs)
        assert "DOCUMENT 1 of 2" in result
        assert "DOCUMENT 2 of 2" in result

    def test_missing_metadata_key_graceful(self):
        from assistant import format_retrieved_docs
        from langchain_core.documents import Document
        # Doc with completely empty metadata – should not raise
        doc = Document(page_content="Some content", metadata={})
        result = format_retrieved_docs([doc])
        assert "Some content" in result

    def test_section_header_truncated_to_120(self):
        from assistant import format_retrieved_docs
        long_header = "X" * 200
        result = format_retrieved_docs([self._make_doc(section_header=long_header)])
        # The truncated header (120 chars) should appear, not the full 200
        assert "X" * 121 not in result


# ─── CLI Filter Parser ─────────────────────────────────────────────────────────

class TestFilterParser:

    def test_bank_filter(self):
        from assistant import _parse_filter_arg
        result = _parse_filter_arg('bank="Meezan Bank"')
        assert result == {"bank_name": "Meezan Bank"}

    def test_type_filter_islamic(self):
        from assistant import _parse_filter_arg
        result = _parse_filter_arg("type=islamic")
        assert result == {"financing_type": "islamic"}

    def test_type_filter_conventional(self):
        from assistant import _parse_filter_arg
        result = _parse_filter_arg("type=conventional")
        assert result == {"financing_type": "conventional"}

    def test_none_returns_none(self):
        from assistant import _parse_filter_arg
        assert _parse_filter_arg(None) is None

    def test_unknown_filter_returns_none(self, capsys):
        from assistant import _parse_filter_arg
        result = _parse_filter_arg("segment=salaried")
        assert result is None


# ─── Session Management ───────────────────────────────────────────────────────

class TestSessionManagement:
    """
    Test CarFinanceAssistant session isolation without building the real chain.
    """

    def _make_assistant(self):
        """Return a CarFinanceAssistant with a mocked internal chain."""
        from assistant import CarFinanceAssistant
        mock_chain = MagicMock()
        assistant  = CarFinanceAssistant(chain_with_history=mock_chain, debug=False)
        return assistant

    def test_new_session_created(self):
        a = self._make_assistant()
        hist = a.get_session_history("session-abc")
        assert hist is not None

    def test_same_session_returned(self):
        a = self._make_assistant()
        h1 = a.get_session_history("s1")
        h2 = a.get_session_history("s1")
        assert h1 is h2

    def test_different_sessions_isolated(self):
        a = self._make_assistant()
        h1 = a.get_session_history("s1")
        h2 = a.get_session_history("s2")
        assert h1 is not h2

    def test_clear_session(self):
        from langchain_core.messages import HumanMessage
        a = self._make_assistant()
        hist = a.get_session_history("s3")
        hist.add_message(HumanMessage(content="Test"))
        assert len(hist.messages) == 1

        a.clear_session("s3")
        assert len(hist.messages) == 0

    def test_turn_count_starts_at_zero(self):
        a = self._make_assistant()
        assert a.session_turn_count("new-session") == 0

    def test_turn_count_increments(self):
        from langchain_core.messages import HumanMessage, AIMessage
        a = self._make_assistant()
        hist = a.get_session_history("s4")
        hist.add_message(HumanMessage(content="Q1"))
        hist.add_message(AIMessage(content="A1"))
        hist.add_message(HumanMessage(content="Q2"))
        assert a.session_turn_count("s4") == 2


# ─── Off-Topic Streaming Interception ────────────────────────────────────────

class TestOffTopicInterception:

    def test_off_topic_yields_canned_response(self):
        from assistant import CarFinanceAssistant, _OFF_TOPIC_RESPONSE
        mock_chain = MagicMock()
        a = CarFinanceAssistant(chain_with_history=mock_chain, debug=False)

        tokens = list(a.stream("Who won the cricket world cup?", session_id="s1"))
        full = "".join(tokens)

        # Should return canned response without ever calling the chain
        assert "Car Finance Consultant" in full
        mock_chain.stream.assert_not_called()

    def test_finance_query_calls_chain(self):
        from assistant import CarFinanceAssistant
        mock_chain = MagicMock()
        # Simulate chain yielding {"answer": "token"} chunks
        mock_chain.stream.return_value = iter([
            {"answer": "Down"},
            {"answer": " payment"},
            {"answer": " is 15%."},
        ])

        a = CarFinanceAssistant(chain_with_history=mock_chain, debug=False)
        tokens = list(a.stream("What is Meezan's down payment?", session_id="s1"))
        full = "".join(tokens)

        assert "Down payment is 15%." == full
        mock_chain.stream.assert_called_once()

    def test_empty_answer_chunks_skipped(self):
        from assistant import CarFinanceAssistant
        mock_chain = MagicMock()
        mock_chain.stream.return_value = iter([
            {"context": "some docs"},  # no "answer" key
            {"answer": ""},            # empty string
            {"answer": "Hello"},
        ])

        a = CarFinanceAssistant(chain_with_history=mock_chain, debug=False)
        tokens = list(a.stream("Meezan car loan details", session_id="s1"))
        # Only non-empty answer tokens should be yielded
        assert tokens == ["Hello"]


# ─── Prompt Validation ────────────────────────────────────────────────────────

class TestPrompts:

    def test_qa_prompt_has_context_variable(self):
        from assistant import QA_PROMPT
        assert "context" in QA_PROMPT.input_variables

    def test_qa_prompt_has_chat_history(self):
        from assistant import QA_PROMPT
        # MessagesPlaceholder injects chat_history
        var_names = [str(v) for v in QA_PROMPT.messages]
        # Check placeholder exists somewhere in the prompt
        assert any("chat_history" in str(m) for m in QA_PROMPT.messages)

    def test_contextualize_prompt_has_input(self):
        from assistant import CONTEXTUALIZE_Q_PROMPT
        assert "input" in CONTEXTUALIZE_Q_PROMPT.input_variables

    def test_document_prompt_has_required_vars(self):
        from assistant import DOCUMENT_PROMPT
        required = {"page_content", "bank_name", "financing_type",
                    "document_type", "page_number", "section_header"}
        assert required.issubset(set(DOCUMENT_PROMPT.input_variables))

    def test_system_prompt_contains_domain_rules(self):
        from assistant import _QA_SYSTEM
        # Verify key domain rules are present in the system prompt
        assert "KIBOR" in _QA_SYSTEM
        assert "Takaful" in _QA_SYSTEM
        assert "SBP" in _QA_SYSTEM
        assert "Roman Urdu" in _QA_SYSTEM
        assert "Sources:" in _QA_SYSTEM
        assert "DOMAIN FIREWALL" in _QA_SYSTEM

    def test_system_prompt_bilingual_instruction(self):
        from assistant import _QA_SYSTEM
        assert "Urdu" in _QA_SYSTEM

    def test_contextualize_prompt_system_mentions_expansion(self):
        from assistant import _CONTEXTUALIZE_Q_SYSTEM
        assert "Roshan Digital Account" in _CONTEXTUALIZE_Q_SYSTEM
