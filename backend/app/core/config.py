from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    APP_NAME: str = "paperpal"
    APP_ENV: str = "development"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    MONGODB_URI: str = "mongodb://localhost:27017"
    MONGODB_DB: str = "paperpal_db"
    MONGODB_COLLECTION_PAPERS: str = "papers"
    MONGODB_COLLECTION_RAG_CHUNKS: str = "rag_chunks"
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    CHROMA_PATH: str = ".chroma"
    CHROMA_COLLECTION: str = "paperpal_chunks_v1"
    RAG_INDEX_VERSION: str = "minilm-v2"
    RAG_CHUNK_TOKENS: int = 220
    RAG_CHUNK_OVERLAP: int = 40
    RAG_RETRIEVAL_CANDIDATES: int = 20
    RAG_CONTEXT_MAX_TOKENS: int = 1200
    RAG_CONTEXT_MAX_PASSAGES: int = 5
    FRONTEND_ORIGIN: str = "http://localhost:8501"


settings = Settings()
