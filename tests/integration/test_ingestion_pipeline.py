from pathlib import Path

from dental_rag.ingestion.loader import load_text_document
from dental_rag.ingestion.pipeline import IngestionPipeline
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


class FakeVectorStore:
    def __init__(self) -> None:
        self.vectors: list[list[float]] = []
        self.payloads: list[dict[str, object]] = []

    def add(
        self,
        vectors: list[list[float]],
        payloads: list[dict[str, object]],
    ) -> None:
        self.vectors = vectors
        self.payloads = payloads


def test_load_and_preprocess_document(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "clinic_info.txt"

    file_path.write_text(
        "  Clinic    Services  \n\n\n  Dental   Implants  ",
        encoding="utf-8",
    )

    raw_document = load_text_document(file_path)

    processed_document = preprocess_document(
        raw_document
    )

    assert processed_document.content == (
        "Clinic Services\n\nDental Implants"
    )

    assert (
        processed_document.metadata.source_path
        == file_path
    )


def test_full_ingestion_pipeline_flow(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "clinic.txt"

    file_path.write_text(
        (
            "Clinic is open. "
            "Call for appointment. "
            "Dental implants are available."
        ),
        encoding="utf-8",
    )

    raw_document = load_text_document(file_path)

    document = preprocess_document(
        raw_document
    )

    embedding_model = FakeEmbeddingModel()

    vector_store = FakeVectorStore()

    chunker = SemanticChunker(
        embedding_model=embedding_model,
        breakpoint_percentile=50,
    )

    pipeline = IngestionPipeline(
        chunker=chunker,
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    chunks = pipeline.run(document)

    assert chunks

    assert len(vector_store.vectors) == len(
        chunks
    )

    assert len(vector_store.payloads) == len(
        chunks
    )

    assert (
        vector_store.payloads[0]["source"]
        == "clinic.txt"
    )

