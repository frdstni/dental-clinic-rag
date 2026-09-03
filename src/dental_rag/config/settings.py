from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application configuration.
    """

    openai_api_key: str = ""
    openai_embedding_model: str = "text-embedding-3-small"

    model_config = SettingsConfigDict(
        env_file=".env"
    )


settings = Settings()