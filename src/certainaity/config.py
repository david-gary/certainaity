"""Runtime configuration loaded from environment variables."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CERTAINAITY_", env_file=".env")

    # File limits
    max_file_bytes: int = Field(default=50 * 1024 * 1024, description="50 MB")
    max_image_dimension: int = Field(default=20_000)
    min_image_dimension: int = Field(default=64)

    # Paths
    weights_dir: Path = Field(default=Path("weights"))
    output_dir: Path = Field(default=Path("output"))

    # Inference
    ensemble_threshold: float = Field(default=0.65)
    min_region_px: int = Field(default=64 * 64)
    use_cpu: bool = Field(default=False, description="Force CPU even if GPU is available")

    # Resilience test
    resilience_qualities: list[int] = Field(default=[70, 85, 95])
    resilience_drop_threshold: float = Field(default=0.25)

    # Feature extraction
    ela_quality: int = Field(default=75)
    noise_block_size: int = Field(default=32)
    dct_block_size: int = Field(default=8)
    feature_workers: int = Field(default=4)

    # API
    redis_url: str = Field(default="redis://localhost:6379/0")
    jwt_public_key_path: Path = Field(default=Path("secrets/jwt_public.pem"))
    rate_limit_per_minute: int = Field(default=60)


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
