from dental_rag.application.rag_service import (
    RagService,
)
from dental_rag.config.settings import settings
from dental_rag.embeddings.openai_embedding import (
    OpenAIEmbeddingModel,
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
from dental_rag.retrieval.quality_checker import (
    RetrievalQualityChecker,
)
from dental_rag.retrieval.retriever import (
    Retriever,
)
from dental_rag.vector_store.qdrant_store import (
    QdrantVectorStore,
)


def create_rag_service() -> RagService:
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

    retriever = Retriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    quality_checker = RetrievalQualityChecker()

    return RagService(
        retriever=retriever,
        quality_checker=quality_checker,
        query_pipeline=query_pipeline,
        refiner=refiner,
    )