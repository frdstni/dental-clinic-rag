from pydantic_settings import (
    SettingsConfigDict,
)


class Settings:
    """
    Application configuration.
    """

    openai_api_key: str = ""

    openai_embedding_model: str = (
        "text-embedding-3-small"
    )

    openai_chat_model: str = (
        "gpt-4.1-mini"
    )

    qdrant_collection_name: str = (
        "dental_clinic"
    )

    reranker_enabled: bool = True

    reranker_model_name: str = (
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    mmr_enabled: bool = True

    mmr_lambda: float = 0.5

    model_config = SettingsConfigDict(
        env_file=".env"
    )


settings = Settings()