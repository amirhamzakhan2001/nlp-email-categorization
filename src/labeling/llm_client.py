# src/labeling/llm_client.py

import os
import time
import requests
from dotenv import load_dotenv

from src.config.clustering_config import (
    MAX_REQUESTS_PER_MINUTE,
    WAIT_TIME_SECONDS
)

load_dotenv()

# ----------------------------
# Ollama configuration
# ----------------------------
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_GENERATE_URL = f"{OLLAMA_BASE_URL}/api/generate"
OLLAMA_TAGS_URL = f"{OLLAMA_BASE_URL}/api/tags"
OLLAMA_PULL_URL = f"{OLLAMA_BASE_URL}/api/pull"

MODEL_NAME = "qwen2.5:14b"

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


# ----------------------------
# Ensure model exists (ONE-TIME)
# ----------------------------
def ensure_model_exists():
    """
    Checks if Qwen model exists in Ollama.
    Automatically pulls it ONCE if missing.
    """
    try:
        response = requests.get(OLLAMA_TAGS_URL, timeout=10)
        response.raise_for_status()

        models = response.json().get("models", [])
        model_names = [m.get("name") for m in models]

        if MODEL_NAME not in model_names:
            print(f"[INFO] Model '{MODEL_NAME}' not found. Pulling now (one-time)...")

            pull_response = requests.post(
                OLLAMA_PULL_URL,
                json={"name": MODEL_NAME},
                timeout=3600  # allow long download
            )
            pull_response.raise_for_status()

            print(f"[INFO] Model '{MODEL_NAME}' downloaded successfully.")

    except requests.exceptions.RequestException as e:
        raise RuntimeError(
            "Ollama is not running or not reachable. "
            "Start Ollama before running the pipeline."
        ) from e


# 🔑 Ensure model is available at import time
ensure_model_exists()




def call_qwen_chat(prompt: str, max_attempts: int = 3) -> str:
    """
    Qwen 2.5 (14B) call handler:
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

            payload = {
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False
            }

            response = requests.post(
                OLLAMA_GENERATE_URL,
                json=payload,
                timeout=300
            )

            response.raise_for_status()
            data = response.json()

            api_request_count += 1
            return data["response"].strip()

        except Exception as e:
            last_exception = e
            if attempt < max_attempts:
                backoff = 2 ** attempt
                time.sleep(backoff)

    raise last_exception

