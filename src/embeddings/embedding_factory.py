from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import get_settings


class EmbeddingFactory:

    @staticmethod
    def create_embeddings() -> Embeddings:

        settings = get_settings()

        provider = settings.EMBEDDING_PROVIDER.lower()

        if provider == "ollama":

            return OllamaEmbeddings(
                model=settings.EMBEDDING_MODEL,
                base_url=settings.EMBEDDING_BASE_URL,
            )

        if provider == "huggingface":

            return HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL,
            )

        if provider == "openai":

            return OpenAIEmbeddings(
                model=settings.EMBEDDING_MODEL,
                api_key=settings.EMBEDDING_API_KEY,
                base_url=settings.EMBEDDING_BASE_URL,
            )

        raise ValueError(
            f"Unsupported embedding provider: {provider}"
        )