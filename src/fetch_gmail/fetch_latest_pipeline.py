from tqdm import tqdm

from src.config.fetch_config import (CSV_MASTER_FINAL, CSV_NEW_BUFFER, GMAIL_QUERY, CSV_MASTER )
from src.fetch_gmail.fetch import (
    fetch_all_messages,
    fetch_full_message
)
from src.fetch_gmail.incremental import (
    get_latest_internal_ms_from_csv,
    build_incremental_gmail_query
)
from src.fetch_gmail.auth import build_gmail_service
from src.fetch_gmail.email_parser import extract_clean_body_from_raw
from src.fetch_gmail.body_cleaner import preprocess_email_body, minimal_text_fix
from src.fetch_gmail.subject_cleaner import clean_subject_strip_emoji
from src.fetch_gmail.csv_writer import write_csv_header_if_missing, append_row
from src.common.logging import get_logger

logger = get_logger(__name__)


def main():
    service = build_gmail_service()
    write_csv_header_if_missing( CSV_NEW_BUFFER)

    # find last processed timestamp
    latest_ms = get_latest_internal_ms_from_csv(CSV_MASTER_FINAL)

    # build Gmail query
    query = build_incremental_gmail_query(GMAIL_QUERY, latest_ms)

    # fetch candidate messages
    messages = fetch_all_messages(service, query=query)

    saved = 0

    for msg in tqdm(messages, desc="Processing new emails"):
        meta = fetch_full_message(service, msg["id"])

        # hard guard (safety)
        if latest_ms and meta.get("internal_ms"):
            if int(meta["internal_ms"]) <= int(latest_ms):
                continue

        body = extract_clean_body_from_raw(meta["raw_bytes"], meta["email_id"])
        body = minimal_text_fix(body)

        cleaned_body = preprocess_email_body(body)
        cleaned_subject = clean_subject_strip_emoji(meta["subject"])

        append_row(
            meta["email_id"],
            meta["date"],
            meta["sender_name"],
            meta["sender_email"],
            meta["subject"],
            cleaned_subject,
            body,
            cleaned_body,
            csv_path= CSV_NEW_BUFFER
        )

        append_row(
            meta["email_id"],
            meta["date"],
            meta["sender_name"],
            meta["sender_email"],
            meta["subject"],
            cleaned_subject,
            body,
            cleaned_body,
            csv_path=CSV_MASTER
        )

        saved += 1

    logger.info(
    "Incremental fetch completed | new_emails=%d",
    saved
)


if __name__ == "__main__":
    main()