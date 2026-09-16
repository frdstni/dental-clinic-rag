from dental_rag.application.rag_service import (
    RagService,
)
from dental_rag.config.settings import settings
from dental_rag.embeddings.openai_embedding import (
    OpenAIEmbeddingModel,
)
from dental_rag.ingestion.sparse_index import (
    SparseIndex,
)
from dental_rag.llm.openai_llm import (
    OpenAILLM,
)
from dental_rag.query_processing.analyzer import (
    QueryAnalyzer,
)
from dental_rag.query_processing.decomposer import (
    QueryDecomposer,
)
from dental_rag.query_processing.expansion import (
    QueryExpander,
)
from dental_rag.query_processing.pipeline import (
    QueryPipeline,
)
from dental_rag.query_processing.refiners.hyde import (
    HyDERefiner,
)
from dental_rag.reranking.cross_encoder import (
    CrossEncoderReranker,
)
from dental_rag.retrieval.base import (
    RetrievalBackend,
)
from dental_rag.retrieval.fusion import (
    ReciprocalRankFusion,
)
from dental_rag.retrieval.hybrid_retriever import (
    HybridRetriever,
)
from dental_rag.retrieval.pipeline import (
    RetrievalPipeline,
)
from dental_rag.retrieval.quality_checker import (
    RetrievalQualityChecker,
)
from dental_rag.retrieval.retriever import (
    Retriever,
)
from dental_rag.retrieval.sparse_retriever import (
    SparseRetriever,
)
from dental_rag.vector_store.qdrant_store import (
    QdrantVectorStore,
)


def create_rag_service(
    sparse_index: SparseIndex | None = None,
) -> RagService:
    """
    Create fully configured RagService instance.
    """

    llm = OpenAILLM()

    refiner = HyDERefiner(
        llm=llm,
    )

    query_pipeline = QueryPipeline(
        analyzer=QueryAnalyzer(),
        expander=QueryExpander(),
        decomposer=QueryDecomposer(),
    )

    embedding_model = OpenAIEmbeddingModel()

    vector_store = QdrantVectorStore(
        collection_name=settings.qdrant_collection_name,
    )

    dense_retriever = Retriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    retriever: RetrievalBackend

    if sparse_index is not None:
        retriever = HybridRetriever(
            dense_retriever=dense_retriever,
            sparse_retriever=SparseRetriever(
                index=sparse_index,
            ),
            fusion=ReciprocalRankFusion(),
        )
    else:
        retriever = dense_retriever

    if settings.reranker_enabled:
        reranker = CrossEncoderReranker(
            model_name=settings.reranker_model_name,
        )

        retriever = RetrievalPipeline(
            retriever=retriever,
            reranker=reranker,
        )

    quality_checker = RetrievalQualityChecker()

    return RagService(
        retriever=retriever,
        quality_checker=quality_checker,
        query_pipeline=query_pipeline,
        refiner=refiner,
    )