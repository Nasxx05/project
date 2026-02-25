"""Application configuration using pydantic-settings."""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    app_name: str = "Hospital Forecasting System"
    app_version: str = "1.0.0"
    debug: bool = False

    # Database
    database_url: str = "postgresql://postgres:changeme@db/hospital_forecasting"

    # ML Models
    ml_models_path: str = "./ml_models"
    default_forecast_days: int = 60
    default_ensemble_size: int = 50

    # CORS
    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://localhost:80",
        "http://localhost:8000",
        "https://hospitalai-woad.vercel.app",
    ]

    # Rate limiting
    rate_limit_per_minute: int = 60

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
