from fastapi import APIRouter

from app.schemas.rag import (
    IngestRequest,
    QueryRequest,
)

from app.services.ingestion_service import ingest_database
from app.services.query_service import query_database
from langchain_community.utilities import SQLDatabase
router = APIRouter(
    prefix="/rag",
    tags=["rag"]
)

@router.get("/test/connection")
def test_connection(db_url: str):
    try:
        db = SQLDatabase.from_uri(db_url)
        return {"message": "Connection successful", "tables": len(db.get_usable_table_names())} #type: ignore
    except Exception as e:
        return {"message": f"Connection failed: {str(e)}"}

@router.post("/ingest")
def ingest(req: IngestRequest):
    return ingest_database(req.db_url)


@router.post("/ask")
def ask(req: QueryRequest):
    return query_database(
        req.db_id,
        req.db_url,
        req.query
    )