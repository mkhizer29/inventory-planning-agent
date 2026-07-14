"""Build the external_signals calendar (FR-A6): public holidays + payday windows.

Weather is a documented follow-up (needs an API key) and is left disabled in config.
This builder is fully offline and deterministic.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from . import config

try:
    import holidays as _holidays
except Exception:  # pragma: no cover
    _holidays = None


def build_external_signals(end_date: str) -> pd.DataFrame:
    cfg = config.settings()["external_signals"]
    start = pd.to_datetime(cfg["start_date"]).date()
    end = pd.to_datetime(end_date).date()

    dates = pd.date_range(start, end, freq="D")
    df = pd.DataFrame({"signal_date": dates})
    df["signal_date"] = df["signal_date"].dt.date

    # public holidays (Pakistan) --------------------------------------------
    hol_name = {}
    if _holidays is not None:
        pk = _holidays.country_holidays(cfg.get("country", "PK"),
                                        years=range(start.year, end.year + 1))
        hol_name = {d: n for d, n in pk.items()}
    df["is_public_holiday"] = df["signal_date"].map(lambda d: d in hol_name)
    df["holiday_name"] = df["signal_date"].map(lambda d: hol_name.get(d))

    # payday window ----------------------------------------------------------
    starts = set(cfg.get("payday_days_month_start", []))
    ends = set(cfg.get("payday_days_month_end", []))

    def _is_payday(d: dt.date) -> bool:
        last_day = (dt.date(d.year + (d.month // 12), (d.month % 12) + 1, 1)
                    - dt.timedelta(days=1)).day
        return d.day in starts or d.day in ends or d.day == last_day

    df["is_payday_window"] = df["signal_date"].map(_is_payday)

    # weather placeholder (disabled) ----------------------------------------
    df["city"] = "Karachi"
    df["weather_condition"] = None
    df["temperature_c"] = None
    return df[["signal_date", "city", "weather_condition", "temperature_c",
               "is_public_holiday", "holiday_name", "is_payday_window"]]
