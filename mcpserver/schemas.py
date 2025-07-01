from pydantic import BaseModel
from typing import List, Dict

class ContextItem(BaseModel):
    key: str
    value: str
    metadata: Dict = {}

class StoreRequest(BaseModel):
    context_items: List[ContextItem]

class RetrieveRequest(BaseModel):
    keys: List[str]

class TailorRequest(BaseModel):
    pdf_path: str
    job_description: str
    key: str
