from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.core.config import settings
from app.services.sql_agent import ask_sql_agent, list_sql_tables


app = FastAPI(title="Standalone LangChain SQL Agent")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    db_url: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=100)


@app.get("/")
def root():
    return {
        "status": "ok",
        "model": settings.llm_model,
        "default_database_url": settings.database_url,
    }


@app.get("/tables")
def tables(db_url: str | None = None):
    return list_sql_tables(db_url)


@app.post("/ask")
def ask(request: AskRequest):
    return ask_sql_agent(
        question=request.question,
        db_url=request.db_url,
        top_k=request.top_k,
    )
