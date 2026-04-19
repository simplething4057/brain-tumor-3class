from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # DB
    database_url: str = "postgresql+asyncpg://btuser:btpass@localhost:5432/brain_tumor"

    # ML pipeline (mounted from host)
    ml_pipeline_path: str = "/ml_pipeline"

    # File storage
    upload_dir: str = "/app/uploads"
    chroma_dir: str = "/app/chroma_db"

    # LLM
    llm_backend: str = "ollama"  # "ollama" | "groq"
    ollama_base_url: str = "http://host.docker.internal:11434"
    ollama_model: str = "llama3:8b"
    groq_api_key: str = ""

    class Config:
        env_file = ".env"
        extra = "ignore"

    @property
    def ml_path(self) -> Path:
        return Path(self.ml_pipeline_path)

    @property
    def model_path(self) -> Path:
        return self.ml_path / "models" / "weights" / "meta_classifier.pkl"

    @property
    def seg_output_path(self) -> Path:
        return self.ml_path / "outputs" / "segmentation"

    @property
    def upload_path(self) -> Path:
        p = Path(self.upload_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def chroma_path(self) -> Path:
        p = Path(self.chroma_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


settings = Settings()
