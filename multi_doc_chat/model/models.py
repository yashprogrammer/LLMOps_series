from pydantic import BaseModel, Field
from typing import Annotated
from enum import Enum



class ChatAnswer(BaseModel):
    """Validate chat answer type and length."""
    answer: Annotated[str, Field(min_length=1, max_length=4096)]


class PromptType(str, Enum):
    CONTEXTUALIZE_QUESTION = "contextualize_question"
    CONTEXT_QA = "context_qa"