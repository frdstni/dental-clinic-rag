from pathlib import Path

from qdrant_client import QdrantClient

from dental_rag.domain.models import (
    DocumentMetadata,
    SourceDocument,
)
from dental_rag.ingestion.pipeline import (
    IngestionPipeline,
)
from dental_rag.ingestion.semantic_chunker import (
    SemanticChunker,
)
from dental_rag.retrieval.retriever import (
    Retriever,
)
from dental_rag.vector_store.qdrant_store import (
    QdrantVectorStore,
)


class FakeEmbeddingModel:
    def embed(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        return [
            [1.0, 0.0]
            for _ in texts
        ]


def test_full_rag_retrieval_flow(
    tmp_path: Path,
) -> None:
    embedding_model = FakeEmbeddingModel()

    vector_store = QdrantVectorStore(
        collection_name="rag_test_collection",
    )

    vector_store.client = QdrantClient(
        path=str(tmp_path),
    )

    vector_store.create_collection(
        vector_size=2,
    )

    document = SourceDocument(
        content=(
            "Dental implants are available. "
            "The clinic provides dental services. "
            "Patients can call for appointments."
        ),
        metadata=DocumentMetadata(
            source_path=Path("clinic.txt"),
        ),
    )

    chunker = SemanticChunker(
        embedding_model=embedding_model,
        breakpoint_percentile=50,
    )

    ingestion_pipeline = IngestionPipeline(
        chunker=chunker,
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    chunks = ingestion_pipeline.run(
        document,
    )

    assert chunks

    retriever = Retriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    results = retriever.retrieve(
        query="Dental implants",
        limit=1,
    )

    assert results

    assert results[0].payload["source"] == "clinic.txt"