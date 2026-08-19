from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from config.settings import get_settings


class ModelFactory:

    @staticmethod
    def create_chat_model():

        settings = get_settings()
        return init_chat_model(
            model_provider=settings.MODEL_PROVIDER,
            model=settings.MODEL_NAME,
            api_key=settings.MODEL_API_KEY,
            base_url=settings.MODEL_BASE_URL,
            temperature=settings.TEMPERATURE,
            max_retries=3,
            timeout=settings.TIMEOUT,
            max_tokens=2048
        )

    @staticmethod
    def create_structured_model(schema):

        settings = get_settings()
        print(f"Creating structured model with setting: {settings.MODEL_PROVIDER}:{settings.MODEL_NAME}")
        llm = ModelFactory.create_chat_model()

        if not settings.LLM_SUPPORTS_STRUCTURED_OUTPUT:
            raise RuntimeError(
                f"Structured output is not supported "
                f"for {settings.MODEL_PROVIDER}:{settings.MODEL_NAME}"
            )

        return llm.with_structured_output(schema)