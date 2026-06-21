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
    FRONTEND_ORIGIN: str = "http://localhost:8501"


settings = Settings()
