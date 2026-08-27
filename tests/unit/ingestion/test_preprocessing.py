from pathlib import Path

import pytest

from dental_rag.domain.models import DocumentMetadata, SourceDocument
from dental_rag.ingestion.preprocessing import normalize_text, preprocess_document


def test_normalize_text_cleans_whitespace_and_preserves_paragraphs() -> None:
    raw_text = "  Clinic    services  \r\n\r\n\r\n  Dental   implants  "

    result = normalize_text(raw_text)

    assert result == "Clinic services\n\nDental implants"


def test_normalize_text_preserves_persian_half_space() -> None:
    text = "دندان\u200cپزشکی"

    result = normalize_text(text)

    assert result == "دندان\u200cپزشکی"


def test_preprocess_document_preserves_metadata() -> None:
    metadata = DocumentMetadata(source_path=Path("clinic.txt"))
    document = SourceDocument(
        content="  Clinic    hours  ",
        metadata=metadata,
    )

    result = preprocess_document(document)

    assert result.content == "Clinic hours"
    assert result.metadata == metadata
    assert result is not document


def test_preprocess_document_rejects_empty_content() -> None:
    document = SourceDocument(
        content="   \n\t\n   ",
        metadata=DocumentMetadata(source_path=Path("clinic.txt")),
    )

    with pytest.raises(ValueError, match="empty after preprocessing"):
        preprocess_document(document)