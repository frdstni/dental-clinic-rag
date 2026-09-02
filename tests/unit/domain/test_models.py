from pathlib import Path

import pytest

from dental_rag.domain.models import (
    DocumentChunk,
    DocumentMetadata,
    SourceDocument,
)


def test_document_metadata_derives_file_information_from_path() -> None:
    metadata = DocumentMetadata(source_path=Path("data/Clinic_Info.TXT"))

    assert metadata.file_name == "Clinic_Info.TXT"
    assert metadata.file_extension == ".txt"


def test_source_document_stores_content_and_metadata() -> None:
    metadata = DocumentMetadata(source_path=Path("clinic.txt"))

    document = SourceDocument(
        content="Clinic opening hours are 9 AM to 6 PM.",
        metadata=metadata,
    )

    assert document.content == "Clinic opening hours are 9 AM to 6 PM."
    assert document.metadata == metadata

def test_document_chunk_creation():
    metadata = DocumentMetadata(
        source_path=Path("clinic.txt"),
    )

    chunk = DocumentChunk(
        content="Dental implant information.",
        metadata=metadata,
        chunk_index=0,
        start_sentence_index=0,
        end_sentence_index_exclusive=2,
    )

    assert chunk.content == "Dental implant information."
    assert chunk.chunk_index == 0
    assert chunk.start_sentence_index == 0
    assert chunk.end_sentence_index_exclusive == 2

def test_document_chunk_rejects_empty_content():
    metadata = DocumentMetadata(
        source_path=Path("clinic.txt"),
    )

    with pytest.raises(ValueError):
        DocumentChunk(
            content="",
            metadata=metadata,
            chunk_index=0,
            start_sentence_index=0,
            end_sentence_index_exclusive=1,
        )
def test_document_chunk_rejects_invalid_sentence_range():
    metadata = DocumentMetadata(
        source_path=Path("clinic.txt"),
    )

    with pytest.raises(ValueError):
        DocumentChunk(
            content="Some text",
            metadata=metadata,
            chunk_index=0,
            start_sentence_index=3,
            end_sentence_index_exclusive=2,
        )