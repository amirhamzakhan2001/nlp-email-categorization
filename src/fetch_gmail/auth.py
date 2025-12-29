# src/gmail/auth.py

from pathlib import Path
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

from src.config.fetch_config import (
    CLIENT_SECRETS_FILE,
    TOKEN_FILE,
    SCOPES,
)


def get_credentials() -> Credentials:
    """
    Load Gmail OAuth credentials.
    Refresh if expired, otherwise trigger browser auth.
    """

    creds = None

    # Ensure token directory exists
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)

    if TOKEN_FILE.exists():
        try:
            creds = Credentials.from_authorized_user_file(
                TOKEN_FILE, SCOPES
            )
        except Exception:
            creds = None

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                CLIENT_SECRETS_FILE, SCOPES
            )
            creds = flow.run_local_server(port=0)

        # Persist refreshed / new token
        TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")

    return creds


def build_gmail_service():
    """
    Build authenticated Gmail API service.
    """
    creds = get_credentials()
    return build(
        "gmail",
        "v1",
        credentials=creds,
        cache_discovery=False
    )