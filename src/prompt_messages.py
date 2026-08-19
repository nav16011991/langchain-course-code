from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

from llm.model_factory import ModelFactory

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage

# # chatprompttemplate
# prompt = ChatPromptTemplate.from_template("Tell me a {adjective} joke about {topic}.")


# # format and inspect
# messages = prompt.format_messages(adjective="funny", topic="chickens")

# print(messages)


# multi-message templates
# prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "You are a helpful assistant that translates {input_language} to {output_language}.",
#         ),
#         ("human", "Translate the following text: {text}"),
#     ]
# )

# messages = prompt.format_messages(
#     input_language="English", output_language="French", text="I love programming."
# )

# print(messages)

# model = ModelFactory.create_chat_model()
# response = model.invoke(messages)
# print(response.content)


# Message Types:
# from langchain_core.messages import (
#     AIMessage,
#     ChatMessage,
#     HumanMessage,
#     SystemMessage,
#     ToolMessage,
# )

# messages = [
#     HumanMessage(content="Hello!"),
#     AIMessage(content="Hi there! How can I assist you today?"),
#     SystemMessage(content="This is a system message."),
#     ToolMessage(content="Tool executed successfully.", tool_call_id="call_123"),
#     ChatMessage(content="This is a general chat message."),
# ]

from langchain_core.prompts import FewShotChatMessagePromptTemplate

examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
]

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)

fewshot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Give the opposite of each word."),
        fewshot_prompt,
        ("human", "{input}"),
    ]
)


model = ModelFactory.create_chat_model()
response = model.invoke(final_prompt.format_messages(input="happy"))

print(response.content)

