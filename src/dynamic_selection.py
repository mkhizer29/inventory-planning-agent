"""Dynamic category discovery and deterministic top-N SKU selection (Phase 1).

Model-agnostic. Reads the existing SQLite warehouse
(``inventory_etl/output/inventory.db``) built by the full Magento ETL and
selects SKUs for later forecasting — it never re-runs the ETL, never writes to
the warehouse, and knows nothing about any specific forecasting model.

Public functions:
  * :func:`list_eligible_categories` — categories that contain forecast-eligible
    products, with counts and history bounds.
  * :func:`list_eligible_skus` — every eligible SKU of one exact category, unranked.
  * :func:`select_top_skus` — the deterministic top-N eligible SKUs of one exact
    category.
  * :func:`select_top_skus_detailed` — as above, plus ranking metadata.

ELIGIBILITY always uses ONLY real ecommerce ``quantity_sold`` on or before the as-of
cutoff — that rule is identical for every ranking metric. The selected SKU list later
feeds ``src/prepare_pilot_data.py`` so every experimental model is compared on the
identical SKU subset / as-of date.

RANKING depends on ``ranking_metric``:
  * ``units`` (default) — historical units sold. Uses nothing but pre-cutoff sales:
    no synthetic stock, cost, price, forecast or post-cutoff information.
  * ``stockout_risk`` — pre-forecast stockout-risk proxy from :mod:`selection_risk`.
    This metric ADDITIONALLY reads the real ``inventory_snapshot`` and per-SKU lead
    times, and under the default snapshot policy that snapshot may POSTDATE the
    cutoff (the warehouse keeps one snapshot and no stock history). Selection can
    therefore be influenced by post-cutoff inventory state — a deliberate trade-off,
    reported in the returned metadata as ``stock_snapshot_date`` /
    ``stock_is_post_cutoff`` so it is auditable rather than silent. It never reads a
    forecast, so there is no circular dependency on Phase B.

CLI::

    python src/dynamic_selection.py --list-categories --selection-cutoff 2026-06-30 --min-history-days 28
    python src/dynamic_selection.py --category "Groceries & Pets" --top-n 10 \
        --selection-cutoff 2026-06-30 --min-history-days 28 --output-file temp_selected_skus.csv
    python src/dynamic_selection.py --category "Groceries & Pets" --top-n 10 \
        --selection-cutoff 2026-07-31 --ranking-metric stockout_risk
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import sqlite3
import sys
import tempfile
from contextlib import closing
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import yaml

# ── repo-relative anchors (this module lives in src/) ───────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "inventory_etl" / "config" / "config.yaml"
DEFAULT_DB_PATH = REPO_ROOT / "inventory_etl" / "output" / "inventory.db"

# Exact output contracts (column order is part of the contract).
CATEGORY_COLUMNS = ["category", "eligible_sku_count", "historical_units",
                    "history_start", "history_end"]
SELECTION_COLUMNS = ["rank", "sku", "sku_name", "category", "sub_category", "brand",
                     "historical_units", "active_days", "history_start", "history_end"]

# Appended to SELECTION_COLUMNS only when ranking_metric == "stockout_risk", so the CSV
# carries the evidence for the ordering. prepare_pilot_data explicitly accepts and ignores
# extra selector columns (SKU is the only key it trusts), so this never affects modelling.
RISK_SELECTION_COLUMNS = [
    "stock_on_hand", "stock_snapshot_date", "lead_time_days", "lead_time_demand_mean",
    "stockout_probability", "expected_shortage_units", "proxy_days_of_cover",
    "proxy_risk_tier", "risk_assumption_flags",
]

METRIC_UNITS = "units"
METRIC_STOCKOUT_RISK = "stockout_risk"
SUPPORTED_RANKING_METRICS = (METRIC_UNITS, METRIC_STOCKOUT_RISK)
TOP_N_MIN, TOP_N_MAX = 1, 100


# ── typed errors (mapped to distinct CLI exit codes) ────────────────────────────────
class DynamicSelectionError(Exception):
    """Base error for dynamic selection."""


class MissingWarehouseError(DynamicSelectionError):
    """db_path is missing or is not a file."""


class InvalidDateError(DynamicSelectionError):
    """selection_cutoff is missing or unparseable."""


class InvalidTopNError(DynamicSelectionError):
    """top_n is not an integer in [1, 100] (booleans are rejected)."""


class UnsupportedRankingMetricError(DynamicSelectionError):
    """ranking_metric is not supported."""


class CategoryNotFoundError(DynamicSelectionError):
    """The requested category does not exist in sku_master at all."""


class CategoryEligibilityError(DynamicSelectionError):
    """The category exists but has no eligible products on/before the cutoff."""


class WarehouseSchemaError(DynamicSelectionError):
    """The warehouse could not be queried (missing tables/columns / SQL error)."""


# ── configuration (reuse project config; never duplicate channel rules) ─────────────
def _load_pilot_config() -> dict:
    """Return the ``pilot`` block from the project ``config.yaml``.

    Provides the ecommerce SOURCE channel keys and the default minimum-history
    threshold, so channel rules are not duplicated in this module.
    """
    if not CONFIG_PATH.exists():
        raise DynamicSelectionError(f"Project config not found: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    pilot = cfg.get("pilot")
    if not isinstance(pilot, dict) or "ecommerce_channel_map" not in pilot:
        raise DynamicSelectionError("config.yaml is missing a valid 'pilot.ecommerce_channel_map'")
    return pilot


def _selection_risk():
    """Import :mod:`selection_risk` lazily.

    Deferred, not top-level, for two reasons: the ``units`` path must keep working even if
    the risk module or its config block is unavailable, and ``selection_risk``'s CLI imports
    this module — a top-level import here would close that cycle.
    """
    _here = str(Path(__file__).resolve().parent)
    if _here not in sys.path:
        sys.path.insert(0, _here)
    import selection_risk                                  # noqa: PLC0415 - deliberate
    return selection_risk


def ecommerce_channels() -> list[str]:
    """Ecommerce SOURCE channel keys (before mapping), as used by prepare_pilot_data.

    These are the *keys* of ``pilot.ecommerce_channel_map`` (e.g. ``online_delivery``,
    ``naheed_web``, ``foodpanda``) — the raw ``sales_transactions.channel`` values,
    matched before any canonical-channel renaming.
    """
    return list(_load_pilot_config()["ecommerce_channel_map"].keys())


def default_min_history_days() -> int:
    """Project-default minimum distinct ecommerce transaction dates per eligible SKU."""
    return int(_load_pilot_config().get("min_history_days", 28))


# ── input validation (kept separate from DB access) ─────────────────────────────────
def _normalize_cutoff(selection_cutoff: object) -> str:
    """Normalize a date / Timestamp / 'YYYY-MM-DD' string to an inclusive 'YYYY-MM-DD'.

    Because ``transaction_date`` is stored as ISO text, the normalized string can be
    compared lexicographically in SQL. Rejects anything unparseable.
    """
    if selection_cutoff is None or (isinstance(selection_cutoff, str) and not selection_cutoff.strip()):
        raise InvalidDateError("selection_cutoff is required (YYYY-MM-DD).")
    if isinstance(selection_cutoff, bool) or not isinstance(
        selection_cutoff, (str, _dt.date, _dt.datetime, pd.Timestamp)
    ):
        raise InvalidDateError(f"Unsupported selection_cutoff type: {type(selection_cutoff).__name__}")
    try:
        ts = pd.to_datetime(selection_cutoff)
    except (ValueError, TypeError, OverflowError, pd.errors.ParserError):
        raise InvalidDateError(f"Invalid selection_cutoff: {selection_cutoff!r} (expected YYYY-MM-DD).")
    if pd.isna(ts):
        raise InvalidDateError(f"Invalid selection_cutoff: {selection_cutoff!r} (expected YYYY-MM-DD).")
    return ts.strftime("%Y-%m-%d")


def _validate_top_n(top_n: object) -> int:
    """top_n must be a real int in [1, 100]; booleans and non-ints are rejected."""
    if isinstance(top_n, bool) or not isinstance(top_n, int):
        raise InvalidTopNError(f"top_n must be an integer in [{TOP_N_MIN}, {TOP_N_MAX}], got {top_n!r}.")
    if not (TOP_N_MIN <= top_n <= TOP_N_MAX):
        raise InvalidTopNError(f"top_n must be between {TOP_N_MIN} and {TOP_N_MAX} inclusive, got {top_n}.")
    return top_n


def _validate_min_history(min_history_days: object) -> int:
    if isinstance(min_history_days, bool) or not isinstance(min_history_days, int):
        raise DynamicSelectionError(f"min_history_days must be a non-negative integer, got {min_history_days!r}.")
    if min_history_days < 0:
        raise DynamicSelectionError(f"min_history_days must be >= 0, got {min_history_days}.")
    return min_history_days


def _validate_ranking_metric(ranking_metric: str) -> str:
    if ranking_metric not in SUPPORTED_RANKING_METRICS:
        raise UnsupportedRankingMetricError(
            f"Unsupported ranking_metric {ranking_metric!r}; supported: {SUPPORTED_RANKING_METRICS}."
        )
    return ranking_metric


# ── read-only database access (isolated from ranking logic) ─────────────────────────
def _connect_readonly(db_path: str | os.PathLike) -> sqlite3.Connection:
    """Open the warehouse strictly read-only. Never creates or writes the DB."""
    p = Path(db_path)
    if not p.exists():
        raise MissingWarehouseError(f"Warehouse not found: {p}")
    if not p.is_file():
        raise MissingWarehouseError(f"Warehouse path is not a file: {p}")
    uri = f"{p.resolve().as_uri()}?mode=ro"     # e.g. file:///C:/.../inventory.db?mode=ro
    try:
        con = sqlite3.connect(uri, uri=True)
        # Keep GROUP BY / DISTINCT / ORDER BY sorters in RAM so a read-only
        # connection never needs to create an on-disk temp file (avoids
        # intermittent SQLITE_CANTOPEN), and hard-guard against any write.
        con.execute("PRAGMA query_only = TRUE")
        con.execute("PRAGMA temp_store = MEMORY")
        return con
    except sqlite3.Error as exc:                # pragma: no cover - environment specific
        raise WarehouseSchemaError(f"Could not open warehouse read-only: {exc}") from exc


def _channel_placeholders(channels: Sequence[str]) -> str:
    return ",".join("?" for _ in channels)


# Per-SKU eligibility aggregation. Aggregation happens in SQLite (no full-table
# read into memory); the cutoff and min-history filters are applied here so no
# post-cutoff transaction can influence units, active days, ranking or totals.
_PER_SKU_TEMPLATE = """
SELECT s.sku_id                              AS sku,
       s.sku_name                            AS sku_name,
       TRIM(s.category)                      AS category,
       s.sub_category                        AS sub_category,
       s.brand                               AS brand,
       SUM(st.quantity_sold)                 AS historical_units,
       COUNT(DISTINCT st.transaction_date)   AS active_days,
       MIN(st.transaction_date)              AS history_start,
       MAX(st.transaction_date)              AS history_end
FROM sales_transactions st
JOIN sku_master s ON s.sku_id = st.sku_id
WHERE st.channel IN ({channel_ph})
  AND st.transaction_date <= ?
  AND s.category IS NOT NULL AND TRIM(s.category) <> ''
  AND s.sku_id NOT LIKE 'Free%' AND s.sku_id NOT LIKE 'PACK%'
  {category_clause}
GROUP BY s.sku_id
HAVING COUNT(DISTINCT st.transaction_date) >= ?
"""


def _per_sku_query(channels: Sequence[str], cutoff: str, min_history_days: int,
                   category: str | None) -> tuple[str, list]:
    """Build the parameterized per-SKU eligibility query and its params.

    All user-controlled values (channels, cutoff, category, min-history) are bound
    parameters — nothing is concatenated into the SQL text.
    """
    category_clause = "AND TRIM(s.category) = ?" if category is not None else ""
    sql = _PER_SKU_TEMPLATE.format(
        channel_ph=_channel_placeholders(channels), category_clause=category_clause
    )
    params: list = [*channels, cutoff]
    if category is not None:
        params.append(category)
    params.append(min_history_days)
    return sql, params


def _read_sql(con: sqlite3.Connection, sql: str, params: Sequence) -> pd.DataFrame:
    try:
        return pd.read_sql_query(sql, con, params=list(params))
    except (sqlite3.Error, pd.errors.DatabaseError) as exc:
        raise WarehouseSchemaError(f"Warehouse query failed: {exc}") from exc


# ── public API ──────────────────────────────────────────────────────────────────────
def list_eligible_categories(
    db_path: str | os.PathLike,
    selection_cutoff: object,
    min_history_days: int,
) -> pd.DataFrame:
    """Return every category containing forecast-eligible SKUs.

    Columns (exact order): ``category, eligible_sku_count, historical_units,
    history_start, history_end``. A SKU is eligible when it has at least
    ``min_history_days`` distinct ecommerce transaction dates on/before
    ``selection_cutoff`` (Free*/PACK* SKUs and null/blank categories excluded).
    Sorted by historical_units desc, eligible_sku_count desc, category asc.
    """
    cutoff = _normalize_cutoff(selection_cutoff)
    min_history_days = _validate_min_history(min_history_days)
    channels = ecommerce_channels()

    per_sku_sql, per_sku_params = _per_sku_query(channels, cutoff, min_history_days, category=None)
    sql = (
        f"WITH elig AS ({per_sku_sql})\n"
        "SELECT category,\n"
        "       COUNT(*)               AS eligible_sku_count,\n"
        "       SUM(historical_units)  AS historical_units,\n"
        "       MIN(history_start)     AS history_start,\n"
        "       MAX(history_end)       AS history_end\n"
        "FROM elig\n"
        "GROUP BY category"
    )
    with closing(_connect_readonly(db_path)) as con:
        df = _read_sql(con, sql, per_sku_params)

    df = df.reindex(columns=CATEGORY_COLUMNS)
    df["category"] = df["category"].astype("string").str.strip()
    df = df.sort_values(
        ["historical_units", "eligible_sku_count", "category"],
        ascending=[False, False, True], kind="mergesort",
    ).reset_index(drop=True)
    return df[CATEGORY_COLUMNS]


def list_eligible_skus(
    db_path: str | os.PathLike,
    category: str,
    selection_cutoff: object,
    min_history_days: int,
) -> pd.DataFrame:
    """Every eligible SKU of one EXACT category, in the default units ordering.

    This is the shared candidate set: eligibility is decided here once, so a ranking
    metric never re-implements (and never drifts from) the eligibility rules. No
    ``top_n`` cap is applied — the caller ranks and truncates.

    Raises :class:`CategoryNotFoundError` if the category is absent from sku_master,
    or :class:`CategoryEligibilityError` if it exists but has no eligible SKUs.
    """
    if category is None or not str(category).strip():
        raise CategoryEligibilityError("category is required and must be non-empty.")
    category_norm = str(category).strip()
    cutoff = _normalize_cutoff(selection_cutoff)
    min_history_days = _validate_min_history(min_history_days)
    channels = ecommerce_channels()

    per_sku_sql, per_sku_params = _per_sku_query(channels, cutoff, min_history_days,
                                                 category=category_norm)
    sql = f"{per_sku_sql}\nORDER BY historical_units DESC, active_days DESC, sku ASC"
    with closing(_connect_readonly(db_path)) as con:
        eligible = _read_sql(con, sql, per_sku_params)
        if eligible.empty:
            # Distinguish "no such category" from "exists but nothing eligible".
            exists = _read_sql(
                con,
                "SELECT COUNT(*) AS n FROM sku_master "
                "WHERE category IS NOT NULL AND TRIM(category) = ?",
                [category_norm],
            )["n"].iloc[0]
            if int(exists) == 0:
                raise CategoryNotFoundError(
                    f"Category {category_norm!r} does not exist in the warehouse."
                )
            raise CategoryEligibilityError(
                f"Category {category_norm!r} exists but has no SKUs with >= {min_history_days} "
                f"distinct ecommerce transaction dates on/before {cutoff}."
            )

    # Deterministic ordering (belt-and-braces on top of SQL ORDER BY).
    return eligible.sort_values(
        ["historical_units", "active_days", "sku"],
        ascending=[False, False, True], kind="mergesort",
    ).reset_index(drop=True)


def select_top_skus_detailed(
    db_path: str | os.PathLike,
    category: str,
    top_n: int,
    selection_cutoff: object,
    min_history_days: int,
    ranking_metric: str = METRIC_UNITS,
    config_path: str | os.PathLike | None = None,
) -> tuple[pd.DataFrame, list[str], dict]:
    """:func:`select_top_skus` plus a metadata dict describing how the ranking was made.

    The metadata always carries ``ranking_metric``, ``eligible_count`` and
    ``selected_count``. For ``stockout_risk`` it also carries the risk-scan metadata
    from :mod:`selection_risk` — notably ``stock_snapshot_date`` and
    ``stock_is_post_cutoff``, which callers persist into the run record so a reviewer
    can see the SKU set was chosen with post-cutoff inventory state.
    """
    if category is None or not str(category).strip():
        raise CategoryEligibilityError("category is required and must be non-empty.")
    category_norm = str(category).strip()
    top_n = _validate_top_n(top_n)
    metric = _validate_ranking_metric(ranking_metric)
    cutoff = _normalize_cutoff(selection_cutoff)
    min_history_days = _validate_min_history(min_history_days)

    eligible = list_eligible_skus(db_path, category_norm, cutoff, min_history_days)
    eligible_count = int(len(eligible))
    warnings: list[str] = []
    meta: dict = {"ranking_metric": metric, "eligible_count": eligible_count}
    columns = list(SELECTION_COLUMNS)

    if metric == METRIC_STOCKOUT_RISK:
        # Score the FULL eligible set, then truncate — never a units-ranked shortlist,
        # which would hide any at-risk SKU that is not also a top seller.
        scored, risk_meta = _selection_risk().score_stockout_risk(
            db_path, eligible["sku"].tolist(), selection_cutoff=cutoff,
            config_path=config_path)
        ranked = _selection_risk().rank_by_stockout_risk(
            eligible.merge(scored, on="sku", how="left"))
        meta.update(risk_meta)
        columns = columns + RISK_SELECTION_COLUMNS

        scored_count = int(risk_meta.get("scored", 0))
        if scored_count == 0:
            raise CategoryEligibilityError(
                f"No SKU in category {category_norm!r} could be scored for stockout risk "
                f"(eligible={eligible_count}). "
                + " ".join(risk_meta.get("warnings", []))
            )
        if scored_count < top_n:
            warnings.append(
                f"Only {scored_count} of {eligible_count} eligible SKU(s) in "
                f"{category_norm!r} could be risk-scored; the remainder are excluded "
                f"({risk_meta.get('exclusion_reasons')}).")
        if risk_meta.get("stock_is_post_cutoff"):
            warnings.append(
                f"Stockout-risk ranking used inventory snapshot "
                f"{risk_meta.get('stock_snapshot_date')}, which postdates the selection "
                f"cutoff {cutoff}; selection was influenced by post-cutoff stock.")
        if risk_meta.get("already_out_of_stock"):
            warnings.append(
                f"{risk_meta['already_out_of_stock']} selected-pool SKU(s) are already out "
                f"of stock (P(stockout) ~ 1.0); ranked among themselves by expected "
                f"shortage units.")
        # Never let an unscored SKU occupy a Top-N slot.
        ranked = ranked[ranked["risk_scored"].astype(bool)]
        effective_count = scored_count
    else:
        ranked = eligible
        effective_count = eligible_count

    selected = ranked.head(top_n).copy()
    selected.insert(0, "rank", range(1, len(selected) + 1))
    selected = selected.reindex(columns=columns)

    if effective_count < top_n:
        warnings.append(
            f"Requested top_n={top_n} but only {effective_count} rankable SKU(s) exist in "
            f"category {category_norm!r}; selected {len(selected)}."
        )
    meta["selected_count"] = int(len(selected))
    return selected, warnings, meta


def select_top_skus(
    db_path: str | os.PathLike,
    category: str,
    top_n: int,
    selection_cutoff: object,
    min_history_days: int,
    ranking_metric: str = METRIC_UNITS,
) -> tuple[pd.DataFrame, list[str]]:
    """Select the top-N eligible SKUs of one EXACT category (trimmed match).

    Returns ``(dataframe, warnings)``. Columns are ``SELECTION_COLUMNS``: ``rank, sku,
    sku_name, category, sub_category, brand, historical_units, active_days,
    history_start, history_end`` — plus ``RISK_SELECTION_COLUMNS`` when
    ``ranking_metric='stockout_risk'``. ``rank`` starts at 1. SKU is the stable
    identifier; sku_name/sub_category/brand may be null and never break selection.

    Ordering is deterministic in both metrics:
      * ``units``         — historical_units desc, active_days desc, sku asc
      * ``stockout_risk`` — stockout_probability desc, expected_shortage_units desc, sku asc

    Use :func:`select_top_skus_detailed` when you need the ranking metadata.

    Raises :class:`CategoryNotFoundError` if the category is absent from sku_master,
    or :class:`CategoryEligibilityError` if it exists but has no eligible SKUs.
    """
    selected, warnings, _meta = select_top_skus_detailed(
        db_path, category, top_n, selection_cutoff, min_history_days, ranking_metric)
    return selected, warnings


# ── output helpers ──────────────────────────────────────────────────────────────────
def _atomic_write_csv(df: pd.DataFrame, output_file: str | os.PathLike, overwrite: bool) -> Path:
    """Write ``df`` to a UTF-8 CSV atomically (temp file + os.replace)."""
    path = Path(output_file)
    if path.exists() and not overwrite:
        raise DynamicSelectionError(
            f"Output file already exists: {path} (pass --overwrite to replace it)."
        )
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent or "."), suffix=".tmp", prefix=".dynsel_")
    os.close(fd)
    try:
        df.to_csv(tmp, index=False, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path


# ── CLI ───────────────────────────────────────────────────────────────────────────--
# Exit codes (distinct, non-zero) for scripting.
EXIT_OK = 0
EXIT_MISSING_WAREHOUSE = 2
EXIT_INVALID_DATE = 3
EXIT_MISSING_CATEGORY = 4
EXIT_MISSING_TOP_N = 5
EXIT_INVALID_TOP_N = 6
EXIT_CATEGORY_NOT_FOUND = 7
EXIT_NO_ELIGIBLE = 8
EXIT_UNSUPPORTED_METRIC = 9
EXIT_WAREHOUSE_SCHEMA = 10
EXIT_OUTPUT_EXISTS = 11
EXIT_OTHER = 1


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="dynamic_selection",
        description="Dynamic category discovery + deterministic top-N SKU selection (read-only).",
    )
    ap.add_argument("--db-path", default=str(DEFAULT_DB_PATH),
                    help=f"SQLite warehouse (default: {DEFAULT_DB_PATH}).")
    ap.add_argument("--selection-cutoff", required=True,
                    help="As-of cutoff (YYYY-MM-DD). No transaction after this influences selection.")
    ap.add_argument("--min-history-days", type=int, default=None,
                    help="Minimum distinct ecommerce transaction dates per SKU "
                         "(default: pilot.min_history_days from config.yaml).")
    ap.add_argument("--list-categories", action="store_true",
                    help="List eligible categories instead of selecting SKUs.")
    ap.add_argument("--category", default=None,
                    help="Exact category to select from (required unless --list-categories).")
    ap.add_argument("--top-n", type=int, default=None,
                    help="Number of SKUs to select (1-100; required unless --list-categories).")
    ap.add_argument("--ranking-metric", default="units",
                    help="Ranking metric (default: units).")
    ap.add_argument("--output-file", default=None,
                    help="Optional CSV output path for the selection (atomic write).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Permit replacing an existing --output-file.")
    return ap


def _print_categories(df: pd.DataFrame, cutoff: str, min_history_days: int) -> None:
    print("Category discovery")
    print(f"  selection cutoff   : {cutoff}")
    print(f"  min history days   : {min_history_days}")
    print(f"  eligible categories: {len(df)}")
    print()
    print(df.to_string(index=False) if not df.empty else "  (no eligible categories)")


def _print_selection(df: pd.DataFrame, warnings: list[str], *, category: str, top_n: int,
                     eligible_count: int, cutoff: str, min_history_days: int,
                     ranking_metric: str, output_path: Path | None) -> None:
    print("Top-N SKU selection")
    print(f"  category         : {category}")
    print(f"  requested top_n  : {top_n}")
    print(f"  eligible SKUs    : {eligible_count}")
    print(f"  selected SKUs    : {len(df)}")
    print(f"  selection cutoff : {cutoff}")
    print(f"  min history days : {min_history_days}")
    print(f"  ranking metric   : {ranking_metric}")
    for w in warnings:
        print(f"  warning          : {w}")
    if output_path is not None:
        print(f"  output written   : {output_path}")
    print()
    print(df.to_string(index=False))


def main(argv: Iterable[str] | None = None) -> int:
    """CLI entry point. Returns a distinct non-zero exit code on each failure class."""
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        min_history_days = (args.min_history_days if args.min_history_days is not None
                            else default_min_history_days())
        cutoff = _normalize_cutoff(args.selection_cutoff)  # fail fast on bad date

        if args.list_categories:
            df = list_eligible_categories(args.db_path, cutoff, min_history_days)
            _print_categories(df, cutoff, min_history_days)
            return EXIT_OK

        if args.category is None or not args.category.strip():
            print("error: --category is required when not using --list-categories", file=sys.stderr)
            return EXIT_MISSING_CATEGORY
        if args.top_n is None:
            print("error: --top-n is required when not using --list-categories", file=sys.stderr)
            return EXIT_MISSING_TOP_N

        _validate_ranking_metric(args.ranking_metric)  # fail fast with clear code
        top_n = _validate_top_n(args.top_n)

        selected, warnings = select_top_skus(
            args.db_path, args.category, top_n, cutoff, min_history_days, args.ranking_metric,
        )
        # True eligible count for the display (may exceed top_n) via the public API.
        cats = list_eligible_categories(args.db_path, cutoff, min_history_days)
        cat_norm = args.category.strip()
        match = cats.loc[cats["category"] == cat_norm, "eligible_sku_count"]
        eligible_count = int(match.iloc[0]) if not match.empty else len(selected)

        output_path: Path | None = None
        if args.output_file:
            output_path = _atomic_write_csv(selected, args.output_file, args.overwrite)

        _print_selection(
            selected, warnings, category=args.category.strip(), top_n=top_n,
            eligible_count=eligible_count, cutoff=cutoff, min_history_days=min_history_days,
            ranking_metric=args.ranking_metric, output_path=output_path,
        )
        return EXIT_OK

    except InvalidDateError as exc:
        print(f"error (invalid date): {exc}", file=sys.stderr); return EXIT_INVALID_DATE
    except InvalidTopNError as exc:
        print(f"error (invalid top_n): {exc}", file=sys.stderr); return EXIT_INVALID_TOP_N
    except UnsupportedRankingMetricError as exc:
        print(f"error (ranking metric): {exc}", file=sys.stderr); return EXIT_UNSUPPORTED_METRIC
    except MissingWarehouseError as exc:
        print(f"error (warehouse): {exc}", file=sys.stderr); return EXIT_MISSING_WAREHOUSE
    except CategoryNotFoundError as exc:
        print(f"error (category not found): {exc}", file=sys.stderr); return EXIT_CATEGORY_NOT_FOUND
    except CategoryEligibilityError as exc:
        print(f"error (no eligible products): {exc}", file=sys.stderr); return EXIT_NO_ELIGIBLE
    except WarehouseSchemaError as exc:
        print(f"error (warehouse schema/SQL): {exc}", file=sys.stderr); return EXIT_WAREHOUSE_SCHEMA
    except DynamicSelectionError as exc:
        # includes output-file-exists and config errors
        msg = str(exc)
        if "already exists" in msg:
            print(f"error (output exists): {msg}", file=sys.stderr); return EXIT_OUTPUT_EXISTS
        print(f"error: {msg}", file=sys.stderr); return EXIT_OTHER


if __name__ == "__main__":
    raise SystemExit(main())
