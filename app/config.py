from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env")

    PROJECT_NAME: str = "Bias Mitigation Prototype"
    DEBUG: bool = True

    DATABASE_URL: str = f"sqlite:///{BASE_DIR / 'prototype.db'}"

    ARTIFACT_DIR: str = str(BASE_DIR / "app" / "artifacts")


settings = Settings()
