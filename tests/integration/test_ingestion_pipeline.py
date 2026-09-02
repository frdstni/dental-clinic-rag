from pathlib import Path

from dental_rag.domain.models import (
    DocumentChunk,
    DocumentMetadata,
    SourceDocument,
)
from dental_rag.ingestion.loader import load_text_document
from dental_rag.ingestion.preprocessing import preprocess_document
from dental_rag.ingestion.semantic_chunker import SemanticChunker


class FakeEmbeddingModel:
    def embed(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        return [
            [float(index), 1.0]
            for index, _ in enumerate(texts)
        ]


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

def test_ingestion_pipeline_creates_semantic_chunks():
    metadata = DocumentMetadata(
        source_path=Path("clinic.txt"),
    )

    document = SourceDocument(
        content=(
            "Clinic is open. "
            "Call for appointment. "
            "Dental implants are available."
        ),
        metadata=metadata,
    )

    chunker = SemanticChunker(
        embedding_model=FakeEmbeddingModel(),
        breakpoint_percentile=50,
    )

    chunks = chunker.chunk(document)

    assert chunks

    assert all(
        isinstance(chunk, DocumentChunk)
        for chunk in chunks
    )

    assert all(
        chunk.metadata == metadata
        for chunk in chunks
    )

