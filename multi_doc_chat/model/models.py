from pydantic import BaseModel, RootModel
from typing import List, Union
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