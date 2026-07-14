"""Configuration loading: `.env` (secrets/connection) + `config.yaml` (business rules)."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Project root = the directory that contains this `etl/` package's parent.
ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "config.yaml"
ENV_PATH = ROOT / ".env"

# Load .env once at import (silent if missing — env vars may come from the shell).
load_dotenv(ENV_PATH)


@lru_cache(maxsize=1)
def settings() -> dict:
    """Business rules from config.yaml (cached)."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _env(key: str, default: str | None = None) -> str | None:
    val = os.getenv(key, default)
    return val.strip() if isinstance(val, str) else val


def source_profile(name: str | None = None) -> dict:
    """Return the MySQL connection profile for the selected source.

    `name` overrides ETL_SOURCE; one of {"staging", "local_backup"}.
    """
    name = (name or _env("ETL_SOURCE", "staging") or "staging").lower()
    if name in ("staging", "stage"):
        return {
            "profile": "staging",
            "host": _env("STAGING_HOST", "34.249.120.145"),
            "port": int(_env("STAGING_PORT", "3306")),
            "user": _env("STAGING_USER", "stageusr"),
            "password": _env("STAGING_PASSWORD", ""),
            "db": _env("STAGING_DB", "pg_1"),
        }
    if name in ("local_backup", "local", "backup"):
        return {
            "profile": "local_backup",
            "host": _env("LOCAL_HOST", "127.0.0.1"),
            "port": int(_env("LOCAL_PORT", "3306")),
            "user": _env("LOCAL_USER", "root"),
            "password": _env("LOCAL_PASSWORD", ""),
            "db": _env("LOCAL_DB", "pg_1"),
        }
    raise ValueError(f"Unknown ETL source '{name}'. Use 'staging' or 'local_backup'.")


def target_sqlite_path() -> Path:
    raw = _env("TARGET_SQLITE_PATH", "output/inventory.db")
    p = Path(raw)
    return p if p.is_absolute() else (ROOT / p)
