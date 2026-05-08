from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg://aiuser:aiuser@localhost:5432/testdb"
    jwt_secret_key: str = "change_me_in_production"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    embedding_model: str = "mxbai-embed-large"
    llm_model: str = "qwen3:14b"
    embed_workers: int = 4
    ingest_db_batch_size: int = 64
    insert_rows_per_chunk: int = 200
    insert_max_chunk_chars: int = 12000
    ingest_only_dml: bool = True

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
