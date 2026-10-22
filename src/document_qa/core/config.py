from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    DATABASE_PATH: str = "./data/app.db"
    UPLOAD_DIR: str = "./data/uploads"
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "document_chunks"

    EMBEDDING_MODEL_DIR: str = "./models/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q"
    EMBEDDING_MODEL_ID: str = "Qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q"
    EMBEDDING_DIM: int = 384
    EMBEDDING_MAX_TOKENS: int = 128
    EMBEDDING_BATCH_SIZE: int = 32
    EMBEDDING_THREADS: int = 2

    CHUNK_TARGET_TOKENS: int = 90
    CHUNK_MAX_TOKENS: int = 100
    CHUNK_OVERLAP_TOKENS: int = 16
    BREADCRUMB_MAX_TOKENS: int = 20
    LOW_TEXT_PAGE_CHARS: int = 40

    RETRIEVAL_DENSE_TOP_K: int = 40 # Increased from 20
    RETRIEVAL_LEXICAL_TOP_K: int = 40 # Increased from 20
    RRF_K: int = 60
    EVIDENCE_MAX: int = 15 # Increased from 8 to give model more context
    
    LEXICAL_SUPPORT_MIN_RATIO: float = 0.2 # Reduced from 0.5 to be more lenient
    CLAIM_MAX_CHARS: int = 1500
    MISSING_MAX_CHARS: int = 160
    QUOTE_MAX_CHARS: int = 600

    MAX_FILE_MB: int = 80
    MAX_PDF_PAGES: int = 2000
    MAX_DOCUMENTS_PER_CONVERSATION: int = 50
    MAX_CHUNKS_PER_CONVERSATION: int = 100000
    MAX_QUESTION_CHARS: int = 1000
    INGEST_TIMEOUT_SECONDS: int = 900
    QUESTION_DEADLINE_SECONDS: int = 150

    ALLOWED_HOSTS: str = "localhost:8000,127.0.0.1:8000"
    CORS_ORIGINS: str = "http://localhost:3000,http://127.0.0.1:3000"
    LOG_LEVEL: str = "INFO"

    LLM_BASE_URL: str | None = None
    LLM_API_KEY: str | None = None
    LLM_MODEL: str | None = None
    LLM_TIMEOUT_SECONDS: int = 60
    LLM_MAX_OUTPUT_TOKENS: int = 1024
    LLM_JSON_MODE: Literal["auto", "on", "off"] = "auto"
    LLM_EXTRA_BODY: str = "{}"
    LLM_DATA_POLICY_ACK: bool = False
    RUN_LIVE_LLM: int = 0

    @property
    def allowed_hosts_list(self) -> list[str]:
        return [h.strip() for h in self.ALLOWED_HOSTS.split(",") if h.strip()]

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

settings = Settings()