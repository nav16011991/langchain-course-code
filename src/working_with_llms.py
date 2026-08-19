from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
load_dotenv()  # Load environment variables from .env file

from llm.model_factory import ModelFactory

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage


def init_model():
    """
    Initialize the model by creating a chat model using the ModelFactory.
    This function can be used to set up the model before running any demos or chains.
    """
    llm = ModelFactory.create_chat_model()
    return llm.invoke("Hello, how are you?")

def demo_message():
    model = ModelFactory.create_chat_model()

    messages = [
        SystemMessage(content="You are a famous pirate."),
        HumanMessage(content="What is the weather like today?"),
    ]

    response = model.invoke(messages)
    print(f"Response: {response}")
    messages.append(response)
    messages.append(HumanMessage(content="What about tomorrow?"))
    return response


# Exercise: Multi-model setup
def exercise_multi_model():
    """
    EXERCISE: Create a function that:
    1. Takes a question and a list of model names
    2. Gets responses from all models
    3. Returns a dict of {model_name: response}

    Test with: question="What is AI?", models=["gpt-4o-mini", "gpt-4o"]
    """

    def get_responses(question: str, model_names: list[str]) -> dict[str, str]:
        responses = {}
        for model_name in model_names:
            model = init_chat_model(
                model=model_name,
                temperature=0.7,
                streaming=False,
            )
            response = model.invoke(question)
            responses[model_name] = response.content
        return responses

    # Test the function
    results = get_responses("What is AI?", ["gpt-4o-mini", "gpt-4o"])
    for model, answer in results.items():
        print(f"Response from {model}: {answer}\n")

if __name__ == "__main__":
    # Initialize the model and print the response
    # init_response = init_model()
    response = demo_message()
    print(response)