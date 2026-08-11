####################################################################################################
# from llm.model_factory import ModelFactory


# llm = ModelFactory.create_chat_model()

# response = llm.invoke(
#     "Explain RAG in simple terms."
# )

# print(response.content)

####################################################################################################

# from embeddings.embedding_factory import EmbeddingFactory


# embeddings = EmbeddingFactory.create_embeddings()

# text = "What is Retrieval Augmented Generation?"

# vector = embeddings.embed_query(text)

# print("Embedding dimension:", len(vector))
# print("First 5 values:", vector[:5])

####################################################################################################

from pydantic import BaseModel, Field

from llm.model_factory import ModelFactory


class Answer(BaseModel):

    answer: str = Field(
        description="Answer to the question"
    )

    confidence: str = Field(
        description="high, medium or low"
    )


llm = ModelFactory.create_structured_model(Answer)

result = llm.invoke(
    "What is Retrieval Augmented Generation?"
)

print(type(result))
print(result)
print(result.answer)
print(result.confidence)