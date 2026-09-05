from pathlib import Path

from dental_rag.domain.models import (
    DocumentChunk,
    DocumentMetadata,
    SourceDocument,
)
from dental_rag.ingestion.pipeline import (
    IngestionPipeline,
)


class FakeChunker:
    def chunk(
        self,
        document: SourceDocument,
    ) -> list[DocumentChunk]:
        return [
            DocumentChunk(
                content="Dental implant information.",
                metadata=document.metadata,
                chunk_index=0,
                start_sentence_index=0,
                end_sentence_index_exclusive=1,
            )
        ]


class FakeEmbeddingModel:
    def embed(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        return [
            [0.1, 0.2, 0.3]
            for _ in texts
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


def test_ingestion_pipeline_stores_document_chunks() -> None:
    document = SourceDocument(
        content="Dental implant information.",
        metadata=DocumentMetadata(
            source_path=Path("clinic.txt"),
        ),
    )

    vector_store = FakeVectorStore()

    pipeline = IngestionPipeline(
        chunker=FakeChunker(),
        embedding_model=FakeEmbeddingModel(),
        vector_store=vector_store,
    )

    chunks = pipeline.run(document)

    assert len(chunks) == 1

    assert vector_store.vectors == [
        [0.1, 0.2, 0.3]
    ]

    assert vector_store.payloads == [
        {
            "content": "Dental implant information.",
            "source": "clinic.txt",
            "chunk_index": 0,
        }
    ]