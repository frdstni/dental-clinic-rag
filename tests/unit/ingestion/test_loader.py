from pathlib import Path

import pytest

from dental_rag.ingestion.loader import load_text_document


def test_load_text_document_reads_content_and_metadata(tmp_path: Path) -> None:
    file_path = tmp_path / "clinic.txt"
    file_path.write_text("Clinic working hours are 9 AM to 6 PM.", encoding="utf-8")

    document = load_text_document(file_path)

    assert document.content == "Clinic working hours are 9 AM to 6 PM."
    assert document.metadata.source_path == file_path
    assert document.metadata.file_name == "clinic.txt"
    assert document.metadata.file_extension == ".txt"


def test_load_text_document_raises_for_missing_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError):
        load_text_document(missing_path)


def test_load_text_document_rejects_unsupported_extension(tmp_path: Path) -> None:
    file_path = tmp_path / "clinic.pdf"
    file_path.write_text("fake pdf content", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported document type"):
        load_text_document(file_path)


def test_load_text_document_rejects_directory(tmp_path: Path) -> None:
    with pytest.raises(IsADirectoryError):
        load_text_document(tmp_path)