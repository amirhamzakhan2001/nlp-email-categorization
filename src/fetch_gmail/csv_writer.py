
import csv
import re
import shutil
from pathlib import Path
from bs4 import BeautifulSoup

from src.config.fetch_config import FORCE_EXCEL_TEXT_FOR_DATE
from src.fetch_gmail.fetch import normalize_slash_dates_to_hyphen
from src.fetch_gmail.body_cleaner import sanitize_field_for_csv
from src.common.logging import get_logger

logger = get_logger(__name__)

REQUIRED_HEADER = [
    "Message_ID", "Date", "Sender_Name", "Sender_Email",
    "Subject", "Cleaned_subject", "Body", "Cleaned_body"
]

def write_csv_header_if_missing( csv_path: str):
    path = Path(csv_path)
    if not path.exists():
        with path.open('w', encoding='utf-8-sig', newline='') as f:
            w = csv.writer(f, quoting=csv.QUOTE_ALL, lineterminator='\r\n')
            w.writerow(REQUIRED_HEADER)
        return

    with path.open('r', encoding='utf-8-sig', newline='') as fr:
        reader = csv.reader(fr)
        rows = list(reader)
    if not rows:
        with path.open('w', encoding='utf-8-sig', newline='') as f:
            w = csv.writer(f, quoting=csv.QUOTE_ALL, lineterminator='\r\n')
            w.writerow(REQUIRED_HEADER)
        return

    existing_header = rows[0]
    if [h.strip() for h in existing_header] == REQUIRED_HEADER:
        return

    header_map = {h.strip(): idx for idx, h in enumerate(existing_header)}
    out_rows = [REQUIRED_HEADER]
    for r in rows[1:]:
        new_r = [""] * len(REQUIRED_HEADER)
        for col in ["Date", "Sender_Name", "Sender_Email", "Subject", "Body"]:
            if col in header_map:
                idx_old = header_map[col]
                if idx_old < len(r):
                    val = r[idx_old]
                else:
                    val = ""
                new_r[REQUIRED_HEADER.index(col)] = val
        out_rows.append(new_r)

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open('w', encoding='utf-8-sig', newline='') as fw:
        writer = csv.writer(fw, quoting=csv.QUOTE_ALL, lineterminator='\r\n')
        for rr in out_rows:
            writer.writerow(rr)
    shutil.move(str(tmp_path), str(path))
    logger.info(
    "Migrated existing CSV header to new schema | path=%s",
    path
)

def append_row(msg_id, date_s, name, email_addr, subject, cleaned_subject, body, cleaned_body, csv_path: str, collapse_newlines=False ):
    date_s = normalize_slash_dates_to_hyphen(date_s)
    if FORCE_EXCEL_TEXT_FOR_DATE and date_s:
        date_s = "'" + date_s

    name_s = sanitize_field_for_csv(name, collapse_newlines=collapse_newlines)
    email_s = sanitize_field_for_csv(email_addr, collapse_newlines=collapse_newlines)
    subj_s = sanitize_field_for_csv(subject, collapse_newlines=collapse_newlines)
    csubj_s = sanitize_field_for_csv(cleaned_subject, collapse_newlines=True)
    body_s = sanitize_field_for_csv(body, collapse_newlines=collapse_newlines)
    cbody_s = sanitize_field_for_csv(cleaned_body, collapse_newlines=True)

    if isinstance(subj_s, str) and re.search(r'<[^>]+>', subj_s):
        subj_s = BeautifulSoup(subj_s, "html.parser").get_text(separator=' ', strip=True)
    if isinstance(csubj_s, str) and re.search(r'<[^>]+>', csubj_s):
        csubj_s = BeautifulSoup(csubj_s, "html.parser").get_text(separator=' ', strip=True)
    if isinstance(body_s, str) and re.search(r'<[^>]+>', body_s):
        body_s = BeautifulSoup(body_s, "html.parser").get_text(separator='\n', strip=True)
    if isinstance(cbody_s, str) and re.search(r'<[^>]+>', cbody_s):
        cbody_s = BeautifulSoup(cbody_s, "html.parser").get_text(separator='\n', strip=True)

    with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL, lineterminator='\r\n')
        w.writerow([msg_id, date_s, name_s, email_s, subj_s, csubj_s, body_s, cbody_s])


