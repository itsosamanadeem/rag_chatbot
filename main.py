from fastapi import FastAPI
from sqlalchemy import text

from app.api.auth import router as auth_router
from app.api.rag import router as rag_router
from app.db.session import Base, engine


app = FastAPI(title="SQL Dump RAG with pgvector + Auth")


@app.on_event("startup")
def startup_event():
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(bind=engine)


@app.get("/")
def root():
    return {"status": "ok"}


app.include_router(auth_router)
app.include_router(rag_router)
