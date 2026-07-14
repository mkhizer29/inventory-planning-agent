"""Database engines and query helpers (source MySQL + target SQLite)."""
from __future__ import annotations

import urllib.parse as _url
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from . import config

SQL_DIR = Path(__file__).resolve().parent / "sql"


def source_engine(profile_name: str | None = None) -> Engine:
    """SQLAlchemy engine for the selected Magento MySQL source."""
    p = config.source_profile(profile_name)
    pw = _url.quote_plus(p["password"] or "")
    url = f"mysql+pymysql://{p['user']}:{pw}@{p['host']}:{p['port']}/{p['db']}?charset=utf8mb4"
    return create_engine(url, pool_pre_ping=True, connect_args={"connect_timeout": 20})


def target_engine() -> Engine:
    """SQLAlchemy engine for the SQLite canonical warehouse."""
    path = config.target_sqlite_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{path}")


def table_exists(engine: Engine, table: str) -> bool:
    """True if `table` exists in the connected MySQL database."""
    q = text(
        "SELECT COUNT(*) FROM information_schema.tables "
        "WHERE table_schema = DATABASE() AND table_name = :t"
    )
    with engine.connect() as c:
        return bool(c.execute(q, {"t": table}).scalar())


def load_sql(name: str, **placeholders: str) -> str:
    """Read a .sql file from etl/sql and substitute {PLACEHOLDER} tokens."""
    sql = (SQL_DIR / name).read_text(encoding="utf-8")
    for key, val in placeholders.items():
        sql = sql.replace("{" + key + "}", val)
    return sql


def read_sql(engine: Engine, sql: str) -> pd.DataFrame:
    """Run a SELECT and return a DataFrame."""
    with engine.connect() as c:
        return pd.read_sql(text(sql), c)
