from pydantic import BaseModel


class IngestRequest(BaseModel):
    db_url: str


class QueryRequest(BaseModel):
    db_id: str
    db_url: str
    query: str