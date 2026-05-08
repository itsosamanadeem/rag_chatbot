import threading
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.models import User
from app.db.session import SessionLocal, get_db
from app.schemas.rag import AskRequest
from app.services.ingest_tracker import (
    append_log,
    create_job,
    fail_job,
    finish_job,
    get_job,
    start_job,
    update_progress,
)
from app.services.rag_service import answer_with_context, ingest_sql_dump, retrieve_chunks


router = APIRouter(prefix="/rag", tags=["rag"])


def _run_ingest(job_id: str, owner_id: int, file_path: str):
    db = SessionLocal()
    try:
        start_job(job_id)
        append_log(job_id, "ingestion started")

        def on_progress(processed: int, total: int, message: str):
            update_progress(job_id, processed, total)
            append_log(job_id, message)

        ingest_sql_dump(db=db, owner_id=owner_id, file_path=file_path, progress_callback=on_progress)
        finish_job(job_id)
        append_log(job_id, "ingestion finished")
    except Exception as e:
        fail_job(job_id, str(e))
    finally:
        db.close()


@router.post("/ingest")
def ingest_sql(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    if not file.filename.lower().endswith(".sql"):
        raise HTTPException(status_code=400, detail="Only .sql files are supported")

    uploads_dir = Path("uploads")
    uploads_dir.mkdir(exist_ok=True)
    file_path = uploads_dir / f"{current_user.username}_{file.filename}"

    with file_path.open("wb") as out:
        out.write(file.file.read())

    job_id = create_job(owner_id=current_user.id, filename=file.filename)
    worker = threading.Thread(
        target=_run_ingest,
        args=(job_id, current_user.id, str(file_path)),
        daemon=True,
    )
    worker.start()
    return {"message": "Ingestion started", "job_id": job_id}


@router.get("/ingest/status/{job_id}")
def ingest_status(job_id: str, current_user: User = Depends(get_current_user)):
    job = get_job(job_id=job_id, owner_id=current_user.id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.post("/ask")
def ask_rag(
    payload: AskRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    retrieved = retrieve_chunks(db=db, owner_id=current_user.id, query=payload.query, top_k=payload.top_k)
    if not retrieved:
        return {"answer": "No indexed SQL chunks found for this user.", "retrieved": []}

    result = answer_with_context(query=payload.query, retrieved_chunks=retrieved)
    return result
