"""Application configuration."""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """App settings from env."""
    binance_api_key: str = ""
    binance_api_secret: str = ""
    chroma_persist_dir: str = "./chroma_db"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # LLM API (OpenAI-compatible: OpenAI, Ollama, vLLM, Together, etc.)
    llm_api_url: str = ""
    llm_api_key: str = ""
    llm_model: str = "qwen2.5-7b-instruct"
    
    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()
