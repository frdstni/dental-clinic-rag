from unittest.mock import patch

from dental_rag.application.factory import (
    create_rag_service,
)
from dental_rag.application.rag_service import (
    RagService,
)
from dental_rag.query_processing.refiners.hyde import (
    HyDERefiner,
)


def test_create_rag_service_returns_rag_service() -> None:
    with (
        patch(
            "dental_rag.application.factory.OpenAILLM",
        ),
        patch(
            "dental_rag.application.factory.OpenAIEmbeddingModel",
        ),
        patch(
            "dental_rag.application.factory.QdrantVectorStore",
        ),
    ):
        service = create_rag_service()

    assert isinstance(
        service,
        RagService,
    )


def test_create_rag_service_configures_hyde_refiner() -> None:
    with (
        patch(
            "dental_rag.application.factory.OpenAILLM",
        ),
        patch(
            "dental_rag.application.factory.OpenAIEmbeddingModel",
        ),
        patch(
            "dental_rag.application.factory.QdrantVectorStore",
        ),
    ):
        service = create_rag_service()

    assert isinstance(
        service.refiner,
        HyDERefiner,
    )


def test_create_rag_service_creates_retrieval_dependencies() -> None:
    with (
        patch(
            "dental_rag.application.factory.OpenAILLM",
        ),
        patch(
            "dental_rag.application.factory.OpenAIEmbeddingModel",
        ),
        patch(
            "dental_rag.application.factory.QdrantVectorStore",
        ),
    ):
        service = create_rag_service()

    assert service.retriever is not None

    assert service.quality_checker is not None