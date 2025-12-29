from pathlib import Path
from zoneinfo import ZoneInfo


from src.common.paths import DATA_DIR, BASE_DIR

# ------------------------
# Secret directory (gitignored)
# ------------------------
SECRET_DIR = BASE_DIR / "gmail_api_secret"

CLIENT_SECRETS_FILE = SECRET_DIR / "gmail_api.json"
TOKEN_FILE = SECRET_DIR / "token.json"

# Safety checks
if not CLIENT_SECRETS_FILE.exists():
    raise FileNotFoundError(
        f"Gmail client secret not found at {CLIENT_SECRETS_FILE}"
    )

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

GMAIL_QUERY = "in:inbox"
BATCH_LISTING_SIZE = 400

IST_ZONE = ZoneInfo("Asia/Kolkata")
DATE_OUTPUT_FORMAT = "%d-%m-%Y %H:%M"
FORCE_EXCEL_TEXT_FOR_DATE = False

# CSV paths
CSV_MASTER = DATA_DIR / 'raw' / "gmail_cleaned.csv"
CSV_NEW_BUFFER = DATA_DIR / 'raw' / "gmail_new_mail_buffer.csv"
CSV_MASTER_FINAL = DATA_DIR / "processed" / "gmail_master.csv"


REPLACE_URL_WITH_TOKEN = True
URL_TOKEN = "<URL>"
REDACT_EMAIL = True
EMAIL_TOKEN = "<EMAIL>"
REDACT_PHONE = True
PHONE_TOKEN = "<PHONE>"
REMOVE_EMOJIS = True
COLLAPSE_NEWLINES = True
LOWERCASE_OUTPUT = False
TRUNCATE_CHARS = 8000
REMOVE_CURRENCY_LINES = True
REMOVE_LONG_TOKENS = True
MAX_TOKEN_LEN = 120
REMOVE_REPEATED_PUNCT = True
REPLACE_PERCENT_BLOBS = True