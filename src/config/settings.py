from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    # -------------------------
    # LLM
    # -------------------------

    MODEL_PROVIDER: str = "ollama"
    MODEL_NAME: str = "qwen3:8b"
    MODEL_BASE_URL: str = "http://localhost:11434/v1"
    MODEL_API_KEY: str = "ollama"

    # -------------------------
    # Embeddings
    # -------------------------

    EMBEDDING_PROVIDER: str = "ollama"
    EMBEDDING_MODEL: str = "nomic-embed-text"
    EMBEDDING_BASE_URL: str = "http://localhost:11434"
    EMBEDDING_API_KEY: str = "ollama"

    LLM_SUPPORTS_STRUCTURED_OUTPUT: bool = True
    LLM_SUPPORTS_TOOL_CALLING: bool = True

    # -------------------------
    # General
    # -------------------------

    TEMPERATURE: float = 0
    TIMEOUT: int = 60

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).parent.parent.parent / ".env"),
        extra="ignore",
    )


def get_settings() -> Settings:
    return Settings()