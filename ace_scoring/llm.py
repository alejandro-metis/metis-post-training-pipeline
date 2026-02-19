"""Thin wrapper around OpenAI client for LLM judge calls."""

import json
import os
import threading
import time

from openai import OpenAI

_client: OpenAI | None = None
_client_lock = threading.Lock()

MAX_RETRIES = 3
RETRY_BACKOFF = 1.0  # seconds, doubles each retry


def get_client() -> OpenAI:
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is None:
            _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        return _client


def call_judge(prompt: str, model: str | None = None, temperature: float = 0.0) -> str:
    """Call LLM judge with retry on transient errors. Returns raw text."""
    model = model or os.environ.get("ACE_JUDGE_MODEL", "gpt-4o")
    client = get_client()

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            # Retry on transient errors (rate limit, server errors)
            wait = RETRY_BACKOFF * (2 ** attempt)
            print(f"[ace_scoring] LLM call failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}. Retrying in {wait}s...")
            time.sleep(wait)


def parse_json(text: str):
    """Extract JSON from LLM response (handles ```json fences)."""
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    return json.loads(text)
