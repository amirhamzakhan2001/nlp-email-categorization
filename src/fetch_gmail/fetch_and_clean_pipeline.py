from tqdm import tqdm

from src.config.fetch_config import CSV_MASTER
from src.fetch_gmail.fetch import ( fetch_all_messages, fetch_full_message )
from src.fetch_gmail.auth import build_gmail_service
from src.fetch_gmail.email_parser import extract_clean_body_from_raw
from src.fetch_gmail.body_cleaner import preprocess_email_body, minimal_text_fix
from src.fetch_gmail.subject_cleaner import clean_subject_strip_emoji
from src.fetch_gmail.csv_writer import write_csv_header_if_missing, append_row

def main():
    service = build_gmail_service()
    write_csv_header_if_missing( CSV_MASTER)

    messages = fetch_all_messages(service)

    for msg in tqdm(messages):
        meta = fetch_full_message(service, msg["id"])

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
            csv_path= CSV_MASTER
        )

if __name__ == "__main__":
    main()