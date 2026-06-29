from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"
UPLOAD_DIR = DATA_DIR / "uploads"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    GROQ_API_KEY: str
    API_KEY: str = ""  # Set to enforce auth; leave empty to skip (dev mode)
    CHROMA_PERSIST_DIR: str = str(CHROMA_DIR)
    UPLOAD_DIR: str = str(UPLOAD_DIR)
    CHROMA_COLLECTION_NAME: str = "rag_documents"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    LLM_MODEL: str = "llama-3.3-70b-versatile"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    API_V1_PREFIX: str = "/api/v1"
    APP_NAME: str = "RAG Backend"
    DEBUG: bool = False

    def ensure_directories(self) -> None:
        for d in (self.CHROMA_PERSIST_DIR, self.UPLOAD_DIR):
            Path(d).mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
