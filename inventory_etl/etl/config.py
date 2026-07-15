"""Configuration loading: `.env` (secrets/connection) + `config.yaml` (business rules).

Path strategy (repo layout: Inventory-Planning-Agent/inventory_etl/etl/config.py):

    ETL_ROOT    = .../Inventory-Planning-Agent/inventory_etl   (this package's home)
    REPO_ROOT   = .../Inventory-Planning-Agent                 (git repository root)
    ENV_PATH    = REPO_ROOT/.env                               (secrets; git-ignored)
    CONFIG_PATH = ETL_ROOT/config/config.yaml                  (business rules)

Relative TARGET_SQLITE_PATH values in .env resolve against REPO_ROOT, so
`inventory_etl/output/inventory.db` lands inside the ETL project. If unset,
the target defaults to ETL_ROOT/output/inventory.db.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

# --- explicit, unambiguous path anchors ---
ETL_ROOT = Path(__file__).resolve().parent.parent          # .../inventory_etl
REPO_ROOT = ETL_ROOT.parent                                # .../Inventory-Planning-Agent
ENV_PATH = REPO_ROOT / ".env"
CONFIG_PATH = ETL_ROOT / "config" / "config.yaml"
DEFAULT_SQLITE_PATH = ETL_ROOT / "output" / "inventory.db"

# Load .env from the repo root once at import (silent if missing — real
# environment variables set by the shell/CI still take effect).
load_dotenv(ENV_PATH)


@lru_cache(maxsize=1)
def settings() -> dict:
    """Business rules from config.yaml (cached). Fails loudly if missing/malformed."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"ETL configuration not found at: {CONFIG_PATH}\n"
            f"Expected 'inventory_etl/config/config.yaml' under the repo root ({REPO_ROOT}). "
            f"Run commands from the repository root and keep the config file in place."
        )
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {CONFIG_PATH} did not parse to a mapping/object.")
    return data


def _env(key: str, default: str | None = None) -> str | None:
    val = os.getenv(key, default)
    return val.strip() if isinstance(val, str) else val


def source_profile(name: str | None = None) -> dict:
    """Return the MySQL connection profile for the selected source.

    `name` overrides ETL_SOURCE; one of {"staging", "local_backup"}.
    (Passwords are read from the environment and never logged here.)
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
    """Resolve the SQLite warehouse path.

    - unset  -> ETL_ROOT/output/inventory.db (default)
    - absolute -> used as-is
    - relative -> resolved against REPO_ROOT (so 'inventory_etl/output/inventory.db' works)
    """
    raw = _env("TARGET_SQLITE_PATH")
    if not raw:
        return DEFAULT_SQLITE_PATH
    p = Path(raw)
    return p if p.is_absolute() else (REPO_ROOT / p)
