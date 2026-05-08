from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    username: Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    chunks: Mapped[list["SQLChunk"]] = relationship(back_populates="owner", cascade="all, delete-orphan")


class SQLChunk(Base):
    __tablename__ = "sql_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    chunk_id: Mapped[str] = mapped_column(String(120), index=True)
    source_file: Mapped[str] = mapped_column(String(255), nullable=False)
    statement_type: Mapped[str] = mapped_column(String(50), index=True)
    table_names: Mapped[str] = mapped_column(String(255), default="")
    start_line: Mapped[int] = mapped_column(Integer)
    end_line: Mapped[int] = mapped_column(Integer)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(1024))

    owner: Mapped[User] = relationship(back_populates="chunks")
