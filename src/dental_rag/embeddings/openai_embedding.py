from collections.abc import Sequence

from openai import OpenAI

from dental_rag.config.settings import settings
from dental_rag.embeddings.base import EmbeddingModel


class OpenAIEmbeddingModel(EmbeddingModel):
    """
    OpenAI implementation of embedding generation.
    """

    def __init__(self) -> None:
        self.client = OpenAI(
            api_key=settings.openai_api_key
        )

    def embed(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
         """
         Generate embeddings for a list of texts.
         """

         inputs = list(texts)

         if not inputs:
           return []

         response = self.client.embeddings.create(
             model=settings.openai_embedding_model,
             input=inputs,
        )

         return [
        item.embedding
        for item in response.data
        ]