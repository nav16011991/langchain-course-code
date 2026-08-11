from langchain_openai import ChatOpenAI

from config.settings import get_settings


class ModelFactory:

    @staticmethod
    def create_chat_model():

        settings = get_settings()

        return ChatOpenAI(
            model=settings.LLM_MODEL,
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL,
            temperature=settings.TEMPERATURE,
            timeout=settings.TIMEOUT,
        )

    @staticmethod
    def create_structured_model(schema):

        settings = get_settings()

        llm = ModelFactory.create_chat_model()

        if not settings.LLM_SUPPORTS_STRUCTURED_OUTPUT:
            raise RuntimeError(
                f"Structured output is not supported "
                f"for {settings.LLM_PROVIDER}:{settings.LLM_MODEL}"
            )

        return llm.with_structured_output(schema)