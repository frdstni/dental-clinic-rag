from pathlib import Path

from dental_rag.ingestion.loader import load_text_document
from dental_rag.ingestion.preprocessing import preprocess_document


def test_load_and_preprocess_document(tmp_path: Path) -> None:
    file_path = tmp_path / "clinic_info.txt"
    file_path.write_text(
        "  Clinic    Services  \n\n\n  Dental   Implants  ",
        encoding="utf-8",
    )

    raw_document = load_text_document(file_path)
    processed_document = preprocess_document(raw_document)

    assert processed_document.content == "Clinic Services\n\nDental Implants"
    assert processed_document.metadata.source_path == file_path