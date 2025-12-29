# src/labeling/llm_client.py

import os
import time
from google import genai
from dotenv import load_dotenv

from src.config.clustering_config import (
    MAX_REQUESTS_PER_MINUTE,
    WAIT_TIME_SECONDS
)

load_dotenv()

gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

api_request_count = 0
api_request_start_time = time.time()

def _rate_limit():
    global api_request_count, api_request_start_time

    elapsed = time.time() - api_request_start_time
    if api_request_count >= MAX_REQUESTS_PER_MINUTE:
        if elapsed < WAIT_TIME_SECONDS:
            time.sleep(WAIT_TIME_SECONDS - elapsed)
        api_request_count = 0
        api_request_start_time = time.time()


def call_gemini_chat(prompt: str, max_attempts: int = 3) -> str:
    """
    Single Gemini call handler:
    - per-minute rate limiting
    - retries with exponential backoff
    """
    global api_request_count

    attempt = 0
    last_exception = None

    while attempt < max_attempts:
        attempt += 1
        try:
            _rate_limit()

            # ---- API call ----
            response = gemini_client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=prompt
            )

            api_request_count += 1
            return response.text.strip()

        except Exception as e:
            last_exception = e
            if attempt < max_attempts:
                backoff = 2 ** attempt
                time.sleep(backoff)

    raise last_exception

