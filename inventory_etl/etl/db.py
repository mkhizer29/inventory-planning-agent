"""Database engines and query helpers (source MySQL + target SQLite)."""
from __future__ import annotations

import logging
import time
import urllib.parse as _url
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError

from . import config

log = logging.getLogger("etl.db")

SQL_DIR = Path(__file__).resolve().parent / "sql"


def source_engine(profile_name: str | None = None) -> Engine:
    """SQLAlchemy engine for the selected Magento MySQL source."""
    p = config.source_profile(profile_name)
    pw = _url.quote_plus(p["password"] or "")
    url = f"mysql+pymysql://{p['user']}:{pw}@{p['host']}:{p['port']}/{p['db']}?charset=utf8mb4"
    # ssl={} forces an encrypted handshake -- some network paths silently drop the
    # plaintext MySQL auth handshake (observed: TCP connect + greeting succeed, then
    # the handshake hangs) while allowing TLS-negotiated connections through (e.g. MySQL
    # Workbench, which defaults to SSL, connects fine over the same network).
    return create_engine(
        url, pool_pre_ping=True,
        connect_args={"connect_timeout": 20, "ssl": {"ssl": {}}},
    )


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


# Waits between read_sql retries. Deliberately not tight: the observed failure is
# the whole network path going away for a while (WinError 10051 "unreachable
# network"), not a single rejected connection -- retrying instantly just burns the
# attempt budget in under a second, which is exactly what happened on 2026-08-03.
_RETRY_WAITS = (10, 20, 40, 60, 60)


def read_sql(engine: Engine, sql: str, chunk_rows: int = 50_000,
             attempts: int = len(_RETRY_WAITS) + 1) -> pd.DataFrame:
    """Run a SELECT and return a DataFrame, streaming the result server-side.

    The multi-million-row extracts (sales, shipments, delivery_geography) are
    read through a server-side cursor in `chunk_rows` batches rather than
    buffered whole in the client. A fully-buffered read of sales_shipment_item
    (2.5M rows) stalls long enough for the server's net_write_timeout (60s) to
    abort the socket mid-stream -- observed as
    `(2013, 'Lost connection to MySQL server during query [WinError 10053]')`
    after ~3.5h. Streaming keeps the client consuming steadily instead.

    Retries are from the start of the query: the extracts have no stable sort key
    to resume from, and streaming makes a restart cheap enough that this is the
    simpler correct choice. Between attempts the pool is disposed (a dropped
    socket must not be handed back out) and we back off per _RETRY_WAITS, so a
    transient link outage is ridden out rather than instantly exhausted.
    """
    last_err: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            frames: list[pd.DataFrame] = []
            opts = {"stream_results": True, "max_row_buffer": chunk_rows}
            with engine.connect().execution_options(**opts) as c:
                for chunk in pd.read_sql(text(sql), c, chunksize=chunk_rows):
                    frames.append(chunk)
            # pandas always yields >=1 frame (an empty one carries the columns).
            return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
        except OperationalError as exc:
            last_err = exc
            if attempt == attempts:
                break
            wait = _RETRY_WAITS[min(attempt - 1, len(_RETRY_WAITS) - 1)]
            log.warning("read_sql attempt %d/%d dropped (%s) -- retrying in %ds",
                        attempt, attempts,
                        type(exc.orig).__name__ if exc.orig else type(exc).__name__, wait)
            engine.dispose()  # never reuse a socket from a dropped link
            time.sleep(wait)
    raise last_err  # type: ignore[misc]


def scalar(engine: Engine, sql: str, **params) -> object:
    """Run a SELECT returning one value."""
    with engine.connect() as c:
        return c.execute(text(sql), params).scalar()


def read_sql_key_ranges(engine: Engine, make_sql, lo: int, hi: int, step: int,
                        label: str = "paged") -> pd.DataFrame:
    """Read one logical SELECT as consecutive ranges of an indexed integer key.

    `make_sql(range_lo, range_hi) -> str` must return a query restricted to
    `range_lo < key <= range_hi`. Each range is a separate short query, retried
    independently by read_sql, so a dropped link costs one range instead of the
    whole extract.

    Why this exists: the sales extract (2.3M rows over a 7-month window) failed
    three times as a single streaming read on the staging link -- at 22min, at
    54min, and once during an outage -- because one drop anywhere restarts
    everything. Ranges of the primary key make each query short and cheap to
    repeat. Keyed on the PK rather than a date because sales_order_item.created_at
    is NOT indexed (only item_id/order_id/product_id/store_id are), so date
    slicing would force a full table scan per slice.

    Rows are NOT assumed to be evenly distributed across the key space: a range
    yielding few or zero rows is normal (gaps, plus any WHERE the caller applies)
    and never treated as end-of-data. Iteration is bounded by `hi` alone.
    """
    frames: list[pd.DataFrame] = []
    total = 0
    n_ranges = max(1, -(-(hi - lo) // step))  # ceil
    for i, start in enumerate(range(lo, hi, step), start=1):
        end = min(start + step, hi)
        page = read_sql(engine, make_sql(start, end))
        frames.append(page)  # keep even empty pages: they carry the column schema
        total += len(page)
        log.info("%s: range %d/%d (key %d..%d] -> %d rows (%d total)",
                 label, i, n_ranges, start, end, len(page), total)
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def read_sql_key_ranges_cached(engine: Engine, make_sql, lo: int, hi: int, step: int,
                               parts_dir: "Path", label: str = "paged") -> pd.DataFrame:
    """read_sql_key_ranges, but each range is persisted to its own parquet part.

    A rerun skips any range whose part file already exists, so an interrupted
    extract continues instead of restarting from scratch. Parts are written
    write-then-rename so a kill mid-write can never leave a truncated part that a
    later run would silently accept as complete.

    Assembling from parts at the end also avoids holding every range in memory
    simultaneously (the concat peak in read_sql_key_ranges), which is what makes
    a multi-million-row extract viable on a small machine.
    """
    parts_dir.mkdir(parents=True, exist_ok=True)
    n_ranges = max(1, -(-(hi - lo) // step))
    fetched = reused = 0
    for i, start in enumerate(range(lo, hi, step), start=1):
        end = min(start + step, hi)
        part = parts_dir / f"part_{i:05d}.parquet"
        # An all-empty range gets a sentinel instead of a parquet: a zero-row write can
        # carry null-typed columns that refuse to unify with the real parts' types.
        done = parts_dir / f"part_{i:05d}.empty"
        if part.exists() or done.exists():
            reused += 1
            continue
        page = read_sql(engine, make_sql(start, end))
        if len(page):
            tmp = part.with_suffix(".parquet.tmp")
            page.to_parquet(tmp, index=False)
            tmp.replace(part)       # rename is atomic: no truncated part survives a kill
        else:
            done.touch()
        fetched += 1
        log.info("%s: range %d/%d (key %d..%d] -> %d rows",
                 label, i, n_ranges, start, end, len(page))
    log.info("%s: %d ranges fetched, %d already present in %s", label, fetched, reused, parts_dir)

    # Explicit file list (never the bare directory) so a stray .tmp is not read back.
    files = sorted(parts_dir.glob("part_*.parquet"))
    if not files:  # every range was empty -- still need the column schema
        return read_sql(engine, make_sql(lo, lo))

    import pyarrow as pa
    import pyarrow.dataset as pa_ds
    import pyarrow.parquet as pq

    # A column that happens to be entirely NULL within one range is written as
    # parquet type `null`, which will not unify with the `double`/`string` the same
    # column carries in other ranges (observed on base_cost). Reading the parts
    # without an explicit schema makes the first part's type win and then fails with
    # "Unsupported cast from double to null". unify_schemas(promote_options=
    # "permissive") promotes null -> the concrete type, so each part is cast in the
    # supported direction (null -> double) as it is scanned.
    unified = pa.unify_schemas([pq.read_schema(f) for f in files],
                               promote_options="permissive")
    df = pa_ds.dataset(files, schema=unified, format="parquet").to_table().to_pandas()
    log.info("%s: assembled %d rows from %d parts", label, len(df), len(files))
    return df
