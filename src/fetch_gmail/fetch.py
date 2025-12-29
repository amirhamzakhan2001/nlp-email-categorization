# src/gmail/fetch.py

import re
import base64
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from dateutil import parser as dateutil_parser

from src.config.fetch_config import (
    GMAIL_QUERY,
    BATCH_LISTING_SIZE,
    IST_ZONE,
    DATE_OUTPUT_FORMAT,
)


# -------------------------
# Date utilities
# -------------------------

def internal_ms_to_ist_str(internal_ms):
    if not internal_ms:
        return ""
    try:
        ts = int(internal_ms) / 1000.0
        dt_utc = datetime.fromtimestamp(ts, tz=timezone.utc)
        return dt_utc.astimezone(IST_ZONE).strftime(DATE_OUTPUT_FORMAT)
    except Exception:
        return ""


def parse_header_date_to_ist(date_header):
    if not date_header:
        return ""
    try:
        dt = parsedate_to_datetime(date_header)
    except Exception:
        try:
            dt = dateutil_parser.parse(date_header, dayfirst=True)
        except Exception:
            return ""

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(IST_ZONE).strftime(DATE_OUTPUT_FORMAT)



def normalize_slash_dates_to_hyphen(datestr):
    if not datestr:
        return ""
    s = datestr.strip()
    m = re.match(r'(\d{1,2})/(\d{1,2})/(\d{4})', s)
    if m:
        dd, mm, yy = m.groups()
        return f"{dd.zfill(2)}-{mm.zfill(2)}-{yy}"
    return s


# -------------------------
# Gmail fetch logic
# -------------------------

def fetch_all_messages(service, query: str | None = None):
    """
    Fetch all Gmail message IDs matching query.
    """
    messages = []
    page_token = None

    while True:
        resp = service.users().messages().list(
            userId="me",
            q=query or GMAIL_QUERY,
            pageToken=page_token,
            maxResults=BATCH_LISTING_SIZE,
        ).execute()

        messages.extend(resp.get("messages", []))
        page_token = resp.get("nextPageToken")

        if not page_token:
            break

    # Deduplicate by message id
    return list({m["id"]: m for m in messages}.values())


def fetch_full_message(service, msg_id: str) -> dict:
    """
    Fetch full Gmail message including raw content.
    """

    meta = service.users().messages().get(
        userId="me",
        id=msg_id,
        format="full",
    ).execute()

    headers = {
        h["name"].lower(): h["value"]
        for h in meta.get("payload", {}).get("headers", [])
    }

    date_str = (
        internal_ms_to_ist_str(meta.get("internalDate"))
        or parse_header_date_to_ist(headers.get("date", ""))
    )

    subject = headers.get("subject", "")
    from_field = headers.get("from", "")

    m = re.match(r'^(.*)<\s*([^>]+)\s*>$', from_field)
    if m:
        sender_name = m.group(1).strip().strip('"')
        sender_email = m.group(2).strip()
    else:
        sender_name = from_field
        sender_email = from_field

    raw_resp = service.users().messages().get(
        userId="me",
        id=msg_id,
        format="raw",
    ).execute()

    raw_bytes = base64.urlsafe_b64decode(
        raw_resp["raw"].encode("ascii")
    )

    return {
        "email_id": msg_id,
        "internal_ms": meta.get("internalDate"),
        "date": date_str,
        "sender_name": sender_name,
        "sender_email": sender_email,
        "subject": subject,
        "raw_bytes": raw_bytes,
    }