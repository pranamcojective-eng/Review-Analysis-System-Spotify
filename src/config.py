import logging
import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

_SECRET_ENV_KEYS = (
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "LOG_LEVEL",
    "EMBEDDING_MODEL",
    "CLUSTER_COUNT",
    "MIN_CLUSTER_SIZE",
)


def _apply_streamlit_secrets() -> None:
    """Expose Streamlit Cloud secrets as environment variables for pydantic-settings."""
    try:
        import streamlit as st
    except Exception:
        return

    try:
        secrets = st.secrets
    except Exception:
        return

    for key in _SECRET_ENV_KEYS:
        if key in secrets and not os.environ.get(key):
            os.environ[key] = str(secrets[key])


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    log_level: str = "INFO"
    embedding_model: str = "all-MiniLM-L6-v2"
    cluster_count: int = 6
    min_cluster_size: int = 2


_apply_streamlit_secrets()
settings = Settings()


def setup_logging() -> logging.Logger:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(OUTPUT_DIR / "pipeline.log", encoding="utf-8"))
    except OSError:
        pass

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger("review_discovery")
