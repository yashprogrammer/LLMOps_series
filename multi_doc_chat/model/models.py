from pydantic import BaseModel, Field, RootModel
from typing import List, Union, Annotated
from enum import Enum

class Metadata(BaseModel):
    Summary: List[str]
    Title: str
    Author: List[str]
    DateCreated: str
    LastModifiedDate: str
    Publisher: str
    Language: str
    PageCount: Union[int, str]  # Can be "Not Available"
    SentimentTone: str
class ChangeFormat(BaseModel):
    Page: str
    Changes: str

class SummaryResponse(RootModel[list[ChangeFormat]]):
    pass
class PromptType(str, Enum):
    DOCUMENT_ANALYSIS = "document_analysis"
    DOCUMENT_COMPARISON = "document_comparison"
    CONTEXTUALIZE_QUESTION = "contextualize_question"
    CONTEXT_QA = "context_qa"


# -------------------- Simple validators for chat outputs --------------------
class ChatAnswer(BaseModel):
    """Validate chat answer type and length."""
    answer: Annotated[str, Field(min_length=1, max_length=4096)]

