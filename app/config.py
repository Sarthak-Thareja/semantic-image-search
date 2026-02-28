"""Application configuration."""

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings from env vars."""

    # API
    api_host: str = Field(default="0.0.0.0", description="API bind host")
    api_port: int = Field(default=8000, description="API port")
    api_prefix: str = Field(default="/api/v1", description="API path prefix")

    # Vector DB (ChromaDB)
    chroma_persist_dir: str = Field(
        default="./data/chroma",
        description="ChromaDB persistence directory",
    )
    chroma_collection_name: str = Field(
        default="image_embeddings",
        description="Chroma collection name",
    )

    # CLIP
    clip_model_name: str = Field(
        default="ViT-B-32",
        description="OpenCLIP model (ViT-B-32, ViT-L-14, etc.)",
    )
    clip_pretrained: str = Field(
        default="openai",
        description="OpenCLIP pretrained weights (openai, laion2b_s34b_b79k, etc.)",
    )

    # Limits
    max_image_size_mb: int = Field(default=10, description="Max image upload size (MB)")
    max_top_k: int = Field(default=100, description="Max results per query")
    default_top_k: int = Field(default=10, description="Default results per query")

    # Data
    data_dir: Path = Field(
        default=Path("./data"),
        description="Base data directory",
    )
    images_dir: Path = Field(
        default=Path("./data/images"),
        description="Directory for image files",
    )

    model_config = {"env_prefix": "RETRIEVAL_", "env_file": ".env", "extra": "ignore"}


settings = Settings()
