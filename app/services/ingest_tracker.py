from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone


_lock = threading.Lock()
_jobs: dict[str, dict] = {}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_job(owner_id: int, filename: str) -> str:
    job_id = str(uuid.uuid4())
    with _lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "owner_id": owner_id,
            "filename": filename,
            "status": "queued",
            "total": 0,
            "processed": 0,
            "progress": 0.0,
            "logs": [f"{_now_iso()} job queued for {filename}"],
            "error": None,
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
        }
    return job_id


def append_log(job_id: str, message: str) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            return
        job["logs"].append(f"{_now_iso()} {message}")
        job["updated_at"] = _now_iso()


def start_job(job_id: str) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            return
        job["status"] = "running"
        job["updated_at"] = _now_iso()


def update_progress(job_id: str, processed: int, total: int) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            return
        job["processed"] = processed
        job["total"] = total
        job["progress"] = (processed / total) if total else 0.0
        job["updated_at"] = _now_iso()


def finish_job(job_id: str) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            return
        job["status"] = "completed"
        job["progress"] = 1.0
        job["updated_at"] = _now_iso()


def fail_job(job_id: str, error: str) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            return
        job["status"] = "failed"
        job["error"] = error
        job["updated_at"] = _now_iso()
        job["logs"].append(f"{_now_iso()} ERROR: {error}")


def get_job(job_id: str, owner_id: int) -> dict | None:
    with _lock:
        job = _jobs.get(job_id)
        if not job or job["owner_id"] != owner_id:
            return None
        return dict(job)
