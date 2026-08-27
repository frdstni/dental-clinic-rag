from pathlib import Path

from dental_rag.domain.models import DocumentMetadata, SourceDocument


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