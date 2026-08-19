### Langchain Core concept and Runnables

from llm.model_factory import ModelFactory

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def demo_basic_chain():
    ## Create a basic chain using the chat model and a prompt template

    ### Component 1: Create a prompt template using the ChatPromptTemplate class
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Answer the question as truthfully as possible."),
            ("human", "{question}"),
        ]
    )

    ## Component 2: Create a chat model using the ModelFactory
    llm = ModelFactory.create_chat_model()
    
    ### Component 3: Create a String output parser using the StrOutputParser class
    output_parser = StrOutputParser()

    #### Component 4: Create a chain by combining the chat model, prompt template, and output parser
    #### The chain is created by using the pipe operator (|) to combine the components. The order of the components matters, as the output of one component is passed as input to the next component in the chain.
    chain = prompt | llm | output_parser

    result = chain.invoke(
        {"question": "What is Retrieval Augmented Generation?"}
    )

    print(result)

def demo_batch_chain():
    ## Create a batch chain using the chat model and a prompt template

    ### Component 1: Create a prompt template using the ChatPromptTemplate class
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Translate to Hindi."),
            ("human", "{question}"),
        ]
    )

    ## Component 2: Create a chat model using the ModelFactory
    llm = ModelFactory.create_chat_model()
    
    ### Component 3: Create a String output parser using the StrOutputParser class
    output_parser = StrOutputParser()

    #### Component 4: Create a chain by combining the chat model, prompt template, and output parser
    #### The chain is created by using the pipe operator (|) to combine the components. The order of the components matters, as the output of one component is passed as input to the next component in the chain.
    chain = prompt | llm | output_parser

    inputs = [
                {"question": "Hello, how are you?"},
                {"question": "What is LangChain?"},
                {"question": "What is LangGraph?"},
            ]
    result = chain.batch(
        inputs
    )

    for i, r in enumerate(result):
        print(f"Input: {inputs[i]['question']}")
        print(f"Output: {r}")
        print("------")


def demo_streaming():
    ## Create a streaming chain using the chat model and a prompt template

    ### Component 1: Create a prompt template using the ChatPromptTemplate class
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a talented writer. Write a short story based on the topic."),
            ("human", "{topic}"),
        ]
    )

    ## Component 2: Create a chat model using the ModelFactory
    llm = ModelFactory.create_chat_model()
    
    ### Component 3: Create a String output parser using the StrOutputParser class
    output_parser = StrOutputParser()

    #### Component 4: Create a chain by combining the chat model, prompt template, and output parser
    #### The chain is created by using the pipe operator (|) to combine the components. The order of the components matters, as the output of one component is passed as input to the next component in the chain.
    chain = prompt | llm | output_parser

    # Streaming - run with streaming enabled
    print("Streaming output: ")
    for chunk in chain.stream({"topic": "nature"}):
        print(chunk, end="", flush=True)
    print()  # for newline after streaming

def demo_schema_inspection():
    ## Create a structured output chain using the chat model and a prompt template

    ### Component 1: Create a prompt template using the ChatPromptTemplate class
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a talented writer. Write a short story based on the topic."),
            ("human", "{question}"),
        ]
    )

    ## Component 2: Create a chat model using the ModelFactory
    llm = ModelFactory.create_chat_model()
    
    ### Component 3: Create a String output parser using the StrOutputParser class
    output_parser = StrOutputParser()

    #### Component 4: Create a chain by combining the chat model, prompt template, and output parser
    #### The chain is created by using the pipe operator (|) to combine the components. The order of the components matters, as the output of one component is passed as input to the next component in the chain.
    chain = prompt | llm | output_parser

    # Inspecting schema
    print("Inspecting schema: ")
    input_schema = chain.input_schema.model_json_schema()
    output_schema = chain.output_schema.model_json_schema()

    print("Input schema: ")
    print(input_schema)
    print("Output schema: ")
    print(output_schema)

# ------- Exercise the demos -------#
# Exercise: Build your first chain
def exercise_first_chain():
    """
    EXERCISE: Create a chain that:
    1. Takes a product name and target audience
    2. Generates a marketing tagline
    3. Returns just the tagline as a string

    Test with: product="AI Course", audience="developers"
    """

    ##create a prompt template using the ChatPromptTemplate class
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a marketing expert. Generate a catchy tagline for the product."),
            ("human", "Product: {product}, Audience: {audience}"),
        ]
    )

    llm = ModelFactory.create_chat_model()
    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser

    result = chain.invoke(
        {"product": "AI Course", "audience": "developers"}
    )

    print(f"Marketing Tagline: {result}")

if __name__ == "__main__":
    #demo_basic_chain()
    #demo_batch_chain()
    #demo_streaming()
    #demo_schema_inspection()
    exercise_first_chain()