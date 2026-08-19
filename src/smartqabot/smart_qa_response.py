

from typing import List
from pydantic import BaseModel, Field


class SmartQAResponse(BaseModel):
    answer: str = Field(description="The answer to the user's question.")
    confidence: str = Field(description="Confidence level: high, medium, or low")
    reasoning: str = Field(description="The reasoning behind the answer provided.")
    follow_up_questions: List[str] = Field(
        description="A list of follow-up questions related to the topic.",
        default_factory=list,
    )
    sources_needed: bool = Field(
        description="Indicates whether sources are needed for the answer.",
        default=False,
    )