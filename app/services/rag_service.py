from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Callable

import ollama
import sqlparse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import SQLChunk


TABLE_RE = re.compile(r"(?:into|table|update|from|join)\s+`?([a-zA-Z_][\w$]*)`?", re.IGNORECASE)
INSERT_VALUES_RE = re.compile(r"^\s*INSERT\s+INTO\s+.+?\bVALUES\b\s*", re.IGNORECASE | re.DOTALL)


def _extract_table_names(sql_text: str) -> str:
    names = sorted(set(TABLE_RE.findall(sql_text)))
    return ",".join(names)


def _statement_type(sql_text: str) -> str:
    first = sql_text.strip().split(maxsplit=1)
    return first[0].upper() if first else "UNKNOWN"


def _line_bounds(source: str, start_offset: int, end_offset: int) -> tuple[int, int]:
    start_line = source.count("\n", 0, start_offset) + 1
    end_line = source.count("\n", 0, end_offset) + 1
    return start_line, end_line


def embed_text(text: str) -> list[float]:
    response = ollama.embeddings(model=settings.embedding_model, prompt=text)
    return response["embedding"]


def embed_texts(texts: list[str]) -> list[list[float]]:
    try:
        response = ollama.embed(model=settings.embedding_model, input=texts)
        vectors = response.get("embeddings", [])
        if vectors and len(vectors) == len(texts):
            return vectors
    except Exception:
        pass
    return [embed_text(t) for t in texts]


def _split_insert_values(values_part: str) -> list[str]:
    rows: list[str] = []
    in_quote = False
    quote_char = ""
    escape = False
    depth = 0
    start = -1

    for i, ch in enumerate(values_part):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_quote:
            escape = True
            continue
        if ch in ("'", '"'):
            if in_quote and ch == quote_char:
                in_quote = False
                quote_char = ""
            elif not in_quote:
                in_quote = True
                quote_char = ch
            continue
        if in_quote:
            continue
        if ch == "(":
            if depth == 0:
                start = i
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and start >= 0:
                rows.append(values_part[start : i + 1].strip())
                start = -1
    return rows


def _expand_statement(statement: str) -> list[str]:
    text = statement.strip().rstrip(";")
    match = INSERT_VALUES_RE.match(text)
    if not match:
        return [text]

    prefix = text[: match.end()].strip()
    values_part = text[match.end() :].strip()
    rows = _split_insert_values(values_part)
    if not rows:
        return [text]

    chunks: list[str] = []
    row_limit = max(10, settings.insert_rows_per_chunk)
    char_limit = max(2000, settings.insert_max_chunk_chars)
    bucket: list[str] = []
    bucket_len = len(prefix) + 16

    for row in rows:
        next_len = bucket_len + len(row) + 2
        if bucket and (len(bucket) >= row_limit or next_len > char_limit):
            chunks.append(f"{prefix} " + ",\n".join(bucket))
            bucket = []
            bucket_len = len(prefix) + 16
        bucket.append(row)
        bucket_len += len(row) + 2

    if bucket:
        chunks.append(f"{prefix} " + ",\n".join(bucket))
    return chunks if chunks else [text]


def _should_keep_statement(text: str) -> bool:
    if not settings.ingest_only_dml:
        return True
    stype = _statement_type(text)
    return stype in {"INSERT", "UPDATE", "DELETE"}


def ingest_sql_dump(
    db: Session,
    owner_id: int,
    file_path: str,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict:
    path = Path(file_path)
    raw_sql = path.read_text(encoding="utf-8", errors="ignore")
    statements = sqlparse.split(raw_sql)

    existing = db.scalars(select(SQLChunk).where(SQLChunk.owner_id == owner_id, SQLChunk.source_file == path.name)).all()
    for row in existing:
        db.delete(row)
    db.flush()

    chunks_added = 0
    cursor = 0
    non_empty = [s for s in statements if s.strip()]
    expanded: list[dict] = []
    for statement in non_empty:
        pieces = _expand_statement(statement)
        for piece in pieces:
            if _should_keep_statement(piece):
                expanded.append({"original": statement, "text": piece})

    total = len(expanded)
    if progress_callback:
        progress_callback(0, total, f"parsed {len(non_empty)} statements into {total} chunks")

    chunk_specs: list[dict] = []
    for idx, item in enumerate(expanded, start=1):
        statement = item["original"]
        text = item["text"]

        found_at = raw_sql.find(statement, cursor)
        if found_at == -1:
            found_at = cursor
        end_at = found_at + len(statement)
        cursor = end_at
        start_line, end_line = _line_bounds(raw_sql, found_at, end_at)

        stype = _statement_type(text)
        tnames = _extract_table_names(text)
        chunk_specs.append(
            {
                "idx": idx,
                "text": text,
                "stype": stype,
                "tnames": tnames,
                "start_line": start_line,
                "end_line": end_line,
            }
        )

    batch_size = max(8, settings.ingest_db_batch_size)
    started_at = time.time()
    processed = 0

    for batch_start in range(0, len(chunk_specs), batch_size):
        batch = chunk_specs[batch_start : batch_start + batch_size]
        embeddings = embed_texts([item["text"] for item in batch])

        rows: list[SQLChunk] = []
        for item, embedding in zip(batch, embeddings):
            rows.append(
                SQLChunk(
                    owner_id=owner_id,
                    chunk_id=f"{path.stem}-{item['idx']}",
                    source_file=path.name,
                    statement_type=item["stype"],
                    table_names=item["tnames"],
                    start_line=item["start_line"],
                    end_line=item["end_line"],
                    chunk_text=item["text"],
                    embedding=embedding,
                )
            )

        db.add_all(rows)
        db.commit()
        chunks_added += len(rows)
        processed += len(rows)
        elapsed = max(0.001, time.time() - started_at)
        rate = processed / elapsed
        remaining = max(0, total - processed)
        eta = int(remaining / rate) if rate > 0 else 0
        if progress_callback:
            progress_callback(
                processed,
                total,
                f"embedded {processed}/{total} chunks at {rate:.2f} chunks/s | ETA ~{eta}s",
            )

    if progress_callback:
        progress_callback(total, total, f"ingestion committed with {chunks_added} chunks")
    return {"chunks_added": chunks_added, "file": path.name}


def retrieve_chunks(db: Session, owner_id: int, query: str, top_k: int = 5):
    query_embedding = embed_text(query)

    rows = (
        db.query(
            SQLChunk,
            SQLChunk.embedding.cosine_distance(query_embedding).label("distance"),
        )
        .filter(SQLChunk.owner_id == owner_id)
        .order_by("distance")
        .limit(top_k)
        .all()
    )

    results = []
    for chunk, distance in rows:
        results.append(
            {
                "chunk_id": chunk.chunk_id,
                "source_file": chunk.source_file,
                "statement_type": chunk.statement_type,
                "table_names": chunk.table_names,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "score": float(1 - distance),
                "chunk_text": chunk.chunk_text,
            }
        )
    return results


def answer_with_context(query: str, retrieved_chunks: list[dict]) -> dict:
    context_block = "\n\n".join(
        [
            (
                f"[SOURCE: {c['chunk_id']} | {c['source_file']} | lines {c['start_line']}-{c['end_line']} | "
                f"table(s): {c['table_names']} | type: {c['statement_type']}]\n{c['chunk_text']}"
            )
            for c in retrieved_chunks
        ]
    )

    prompt = (
        "You are a SQL-dump RAG assistant. Answer ONLY from provided context. "
        "If evidence is insufficient, say so clearly. Always cite source chunk IDs used.\n\n"
        f"User Query:\n{query}\n\n"
        f"Retrieved Context:\n{context_block}\n\n"
        "Return concise answer with: 1) Answer 2) Why 3) Sources used."
    )

    response = ollama.chat(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": "You must not hallucinate."},
            {"role": "user", "content": prompt},
        ],
    )

    return {
        "answer": response["message"]["content"],
        "retrieved": retrieved_chunks,
    }
