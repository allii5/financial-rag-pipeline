
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


sys.path.insert(0, str(Path(__file__).parent.parent))




class TestMetadataUtils:

    def test_infer_bank_from_filename(self):
        from metadata_utils import infer_bank
        name, short = infer_bank("", "meezan_car_financing_2025.pdf")
        assert name == "Meezan Bank"
        assert short == "meezan"

    def test_infer_bank_from_text(self):
        from metadata_utils import infer_bank
        name, _ = infer_bank("Bank Alfalah offers car ijarah financing.", "")
        assert name == "Bank Alfalah"

    def test_infer_bank_unknown(self):
        from metadata_utils import infer_bank
        name, short = infer_bank("", "random_doc.pdf")
        assert name == "Unknown"

    def test_financing_type_islamic(self):
        from metadata_utils import infer_financing_type
        assert infer_financing_type("This is an Ijarah-based car scheme") == "islamic"

    def test_financing_type_conventional(self):
        from metadata_utils import infer_financing_type
        assert infer_financing_type("Rate is KIBOR + 2.5%") == "conventional"

    def test_financing_type_both(self):
        from metadata_utils import infer_financing_type
        assert infer_financing_type("KIBOR plus Shariah-compliant Ijarah") == "both"

    def test_financing_type_unknown(self):
        from metadata_utils import infer_financing_type
        assert infer_financing_type("Customer must provide CNIC copy") == "unknown"

    def test_customer_segment_salaried(self):
        from metadata_utils import infer_customer_segment
        seg = infer_customer_segment("For salaried employees of government entities")
        assert "salaried" in seg

    def test_customer_segment_nrp(self):
        from metadata_utils import infer_customer_segment
        seg = infer_customer_segment("Overseas Pakistanis with Roshan Digital Account")
        assert "nrp" in seg

    def test_customer_segment_multiple(self):
        from metadata_utils import infer_customer_segment
        seg = infer_customer_segment("Salaried and self-employed applicants")
        assert "salaried" in seg
        assert "self_employed" in seg

    def test_financial_flags_rate(self):
        from metadata_utils import infer_financial_flags
        flags = infer_financial_flags("Rate: KIBOR + 250 bps")
        assert flags["has_rate_info"] is True
        assert flags["has_tenure_info"] is False

    def test_financial_flags_tenure(self):
        from metadata_utils import infer_financial_flags
        flags = infer_financial_flags("Loan tenure: 1 to 7 years")
        assert flags["has_tenure_info"] is True

    def test_vehicle_type_new(self):
        from metadata_utils import infer_vehicle_type
        assert infer_vehicle_type("For new vehicle purchase only") == "new"

    def test_vehicle_type_used(self):
        from metadata_utils import infer_vehicle_type
        assert infer_vehicle_type("Used cars up to 9 years old eligible") == "used"

    def test_build_metadata_schema_complete(self):
        from metadata_utils import build_metadata
        from config import METADATA_SCHEMA
        meta = build_metadata(
            text="KIBOR + 2% for salaried customers",
            source_file="hbl_car.pdf",
            page_number=3,
            element_type="NarrativeText",
            section_header="HBL Car Loan",
        )
        missing = set(METADATA_SCHEMA.keys()) - set(meta.keys())
        assert not missing, f"Missing metadata keys: {missing}"




class TestExtractor:

    def _make_mock_element(self, type_name: str, text: str, page: int = 1, html: str = ""):
        el = MagicMock()
        el.text = text
        type(el).__name__ = type_name
        el.metadata.page_number = page
        el.metadata.text_as_html = html
        el.id = f"mock_{type_name}_{page}"
        return el

    def test_extract_pdf_element_classification(self, tmp_path):
        """Verify Title / Table / NarrativeText end up in the right buckets."""
        from extractor import extract_pdf

        fake_pdf = tmp_path / "test_bank.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4 fake content")

        mock_elements = [
            self._make_mock_element("Title", "Car Financing Scheme"),
            self._make_mock_element("NarrativeText", "Minimum age 21 years."),
            self._make_mock_element("Table", "Charges table", html="<table>...</table>"),
        ]

        with patch("extractor._import_partition_pdf", return_value=lambda *a, **kw: mock_elements):
            doc = extract_pdf(fake_pdf, strategy="fast")

        assert len(doc.titles) == 1
        assert len(doc.content) == 1
        assert len(doc.tables) == 1
        assert doc.tables[0].html == "<table>...</table>"

    def test_empty_elements_skipped(self, tmp_path):
        from extractor import extract_pdf
        fake_pdf = tmp_path / "empty.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4")

        mock_elements = [
            self._make_mock_element("NarrativeText", ""), 
            self._make_mock_element("Title", "Section A"),
        ]

        with patch("extractor._import_partition_pdf", return_value=lambda *a, **kw: mock_elements):
            doc = extract_pdf(fake_pdf, strategy="fast")

        assert len(doc.elements) == 1 




class TestChunker:

    def _make_extracted_doc(self) -> "ExtractedDocument":
        from extractor import ExtractedDocument, RawElement

        doc = ExtractedDocument(source_file="meezan_car.pdf", file_path=Path("."))

        elements = [
            RawElement("Title", "Meezan Bank Car Ijarah", "", 1, "meezan_car.pdf", "id1"),
            RawElement("NarrativeText", "Shariah-compliant financing for salaried customers. Profit rate linked to KIBOR + 250bps.", "", 1, "meezan_car.pdf", "id2"),
            RawElement("Table", "Down Payment: 15%\nTenure: 1-7 years", "<table><tr><td>Down Payment</td><td>15%</td></tr></table>", 2, "meezan_car.pdf", "id3"),
            RawElement("NarrativeText", "Processing fee: 0.5% of financed amount.", "", 2, "meezan_car.pdf", "id4"),
        ]
        doc.elements = elements
        doc.titles  = [elements[0]]
        doc.tables  = [elements[2]]
        doc.content = [elements[1], elements[3]]
        return doc

    def test_chunk_produces_documents(self):
        from chunker import BankingDocumentChunker
        chunker = BankingDocumentChunker()
        docs = chunker.chunk(self._make_extracted_doc())
        assert len(docs) >= 2  # at least one text + one table chunk

    def test_table_chunk_contains_html(self):
        from chunker import BankingDocumentChunker
        chunker = BankingDocumentChunker()
        docs = chunker.chunk(self._make_extracted_doc())
        table_docs = [d for d in docs if d.metadata["element_type"] == "Table"]
        assert table_docs, "No table chunk found"
        assert "<table>" in table_docs[0].page_content

    def test_section_header_propagated(self):
        from chunker import BankingDocumentChunker
        chunker = BankingDocumentChunker()
        docs = chunker.chunk(self._make_extracted_doc())
        
        for doc in docs:
            if doc.metadata["document_type"] != "header":
                assert doc.metadata["section_header"] == "Meezan Bank Car Ijarah"

    def test_bank_name_inferred(self):
        from chunker import BankingDocumentChunker
        chunker = BankingDocumentChunker()
        docs = chunker.chunk(self._make_extracted_doc())
        assert all(d.metadata["bank_name"] == "Meezan Bank" for d in docs)

    def test_financing_type_islamic(self):
        from chunker import BankingDocumentChunker
        chunker = BankingDocumentChunker()
        docs = chunker.chunk(self._make_extracted_doc())
        types = {d.metadata["financing_type"] for d in docs}
        # At least one chunk should be tagged islamic or both
        assert types & {"islamic", "both"}

    def test_no_tiny_chunks(self):
        from chunker import BankingDocumentChunker
        from config import MIN_CHUNK_CHARS
        chunker = BankingDocumentChunker()
        docs = chunker.chunk(self._make_extracted_doc())
        for doc in docs:
            assert len(doc.page_content) >= MIN_CHUNK_CHARS




class TestVectorStore:

    def test_sanitise_metadata_none(self):
        from vectorstore import sanitise_metadata
        result = sanitise_metadata({"key": None})
        assert result["key"] == ""

    def test_sanitise_metadata_list(self):
        from vectorstore import sanitise_metadata
        result = sanitise_metadata({"segs": ["salaried", "nrp"]})
        assert result["segs"] == "salaried, nrp"

    def test_sanitise_metadata_bool_preserved(self):
        from vectorstore import sanitise_metadata
        result = sanitise_metadata({"has_rate_info": True})
        assert result["has_rate_info"] is True

    def test_doc_id_deterministic(self):
        from langchain_core.documents import Document
        from vectorstore import doc_id
        doc = Document(
            page_content="Test content",
            metadata={"source_file": "a.pdf", "page_number": 1, "chunk_index": 0},
        )
        assert doc_id(doc) == doc_id(doc)

    def test_doc_id_unique_for_different_chunks(self):
        from langchain_core.documents import Document
        from vectorstore import doc_id
        doc1 = Document(
            page_content="Chunk one",
            metadata={"source_file": "a.pdf", "page_number": 1, "chunk_index": 0},
        )
        doc2 = Document(
            page_content="Chunk two",
            metadata={"source_file": "a.pdf", "page_number": 1, "chunk_index": 1},
        )
        assert doc_id(doc1) != doc_id(doc2)
