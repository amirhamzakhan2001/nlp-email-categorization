import os
import csv
from datetime import datetime, timezone
from typing import Optional
from dateutil import parser as dateutil_parser

from src.config.fetch_config import DATE_OUTPUT_FORMAT, IST_ZONE
from src.fetch_gmail.fetch import normalize_slash_dates_to_hyphen


def get_latest_internal_ms_from_csv(csv_path: str) -> Optional[int]:
    """
    Reads CSV and returns latest timestamp (UTC ms) based on Date column.
    Returns None if CSV does not exist or date cannot be parsed.
    """
    if not os.path.exists(csv_path):
        return None

    latest_ms = None

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return None

        # find Date column
        date_idx = None
        for i, h in enumerate(header):
            if h.strip().lower() == "date":
                date_idx = i
                break
        if date_idx is None:
            return None

        for row in reader:
            if len(row) <= date_idx:
                continue

            raw_date = row[date_idx].strip()
            if not raw_date:
                continue

            # remove Excel apostrophe
            if raw_date.startswith("'"):
                raw_date = raw_date[1:]

            raw_date = normalize_slash_dates_to_hyphen(raw_date)

            # try strict format first
            try:
                dt = datetime.strptime(raw_date, DATE_OUTPUT_FORMAT)
                dt = dt.replace(tzinfo=IST_ZONE)
            except Exception:
                try:
                    dt = dateutil_parser.parse(raw_date, dayfirst=True)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=IST_ZONE)
                except Exception:
                    continue

            ms = int(dt.astimezone(timezone.utc).timestamp() * 1000)
            if latest_ms is None or ms > latest_ms:
                latest_ms = ms

    return latest_ms


def build_incremental_gmail_query(base_query: str, latest_internal_ms: Optional[int]) -> str:
    """
    Builds Gmail search query using 'after:' if latest date is known.
    """
    if not latest_internal_ms:
        return base_query

    dt_utc = datetime.fromtimestamp(latest_internal_ms / 1000.0, tz=timezone.utc)
    dt_ist = dt_utc.astimezone(IST_ZONE)

    # Gmail uses YYYY/MM/DD
    after_date = dt_ist.strftime("%Y/%m/%d")
    return f"{base_query} after:{after_date}"