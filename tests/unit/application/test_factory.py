from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from dental_rag.application.factory import (
    create_rag_service,
)
from dental_rag.application.rag_service import (
    RagService,
)
from dental_rag.domain.models import (
    DocumentChunk,
    DocumentMetadata,
)
from dental_rag.ingestion.sparse_index import (
    SparseIndex,
)
from dental_rag.query_processing.refiners.hyde import (
    HyDERefiner,
)
from dental_rag.retrieval.pipeline import (
    RetrievalPipeline,
)


def _factory_patches() -> ExitStack:
    stack = ExitStack()

    stack.enter_context(
        patch(
            "dental_rag.application.factory.OpenAILLM",
        )
    )

    stack.enter_context(
        patch(
            "dental_rag.application.factory.OpenAIEmbeddingModel",
        )
    )

    stack.enter_context(
        patch(
            "dental_rag.application.factory.QdrantVectorStore",
        )
    )

    stack.enter_context(
        patch(
            "dental_rag.application.factory.CrossEncoderReranker",
        )
    )

    stack.enter_context(
        patch(
            "dental_rag.application.factory.MMRSelector",
        )
    )

    return stack


def test_create_rag_service_returns_rag_service() -> None:
    with _factory_patches():
        service = create_rag_service()

    assert isinstance(
        service,
        RagService,
    )


def test_create_rag_service_configures_hyde_refiner() -> None:
    with _factory_patches():
        service = create_rag_service()

    assert isinstance(
        service.refiner,
        HyDERefiner,
    )


def test_create_rag_service_creates_retrieval_dependencies() -> None:
    with _factory_patches():
        service = create_rag_service()

    assert service.retriever is not None
    assert service.quality_checker is not None


def test_create_rag_service_uses_retrieval_pipeline_when_sparse_index_exists() -> None:
    sparse_index = SparseIndex(
        chunks=[
            DocumentChunk(
                content="dental implant information",
                metadata=DocumentMetadata(
                    source_path=Path(
                        "clinic.txt",
                    ),
                ),
                chunk_index=0,
                start_sentence_index=0,
                end_sentence_index_exclusive=1,
            )
        ],
    )

    with _factory_patches():
        service = create_rag_service(
            sparse_index=sparse_index,
        )

    assert isinstance(
        service.retriever,
        RetrievalPipeline,
    )


def test_factory_creates_mmr_selector_when_enabled() -> None:
    with ExitStack() as stack:
        mock_mmr = stack.enter_context(
            patch(
                "dental_rag.application.factory.MMRSelector",
            )
        )

        stack.enter_context(
            patch(
                "dental_rag.application.factory.OpenAILLM",
            )
        )

        stack.enter_context(
            patch(
                "dental_rag.application.factory.OpenAIEmbeddingModel",
            )
        )

        stack.enter_context(
            patch(
                "dental_rag.application.factory.QdrantVectorStore",
            )
        )

        stack.enter_context(
            patch(
                "dental_rag.application.factory.CrossEncoderReranker",
            )
        )

        create_rag_service()

    mock_mmr.assert_called_once_with(
        lambda_value=0.5,
    )