from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg://aiuser:aiuser@localhost:5432/testdb"
    llm_model: str = "qwen3.5:9b"
    response_model: str = "llama3.2:latest"
    ollama_base_url: str = "http://localhost:11434"
    llm_temperature: float = 0
    ollama_keep_alive: str = "10m"
    ollama_num_ctx: int = 4096
    ollama_num_gpu: int | None = None
    ollama_num_thread: int | None = None
    ollama_num_predict: int = 512
    ollama_request_timeout_seconds: int = 120
    sql_agent_top_k: int = 5
    sql_agent_max_iterations: int = 8

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
