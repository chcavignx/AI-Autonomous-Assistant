#!/usr/bin/env python3
"""llm/llm_client.py."""

import logging
from typing import Any

import requests

module_name = __name__
lib_name = module_name.split(".")[1]
logger = logging.getLogger(lib_name)


def generate_llm_response(
    api_type: str,
    url: str,
    model: str,
    prompt: str,
    timeout: float = 5.0,
    api_key: str | None = None,
    payload_override: dict[str, Any] | None = None,  # pyright: ignore[reportExplicitAny]
) -> str | None:
    """Generate LLM response using the specified API type (ollama or openai)."""
    headers = {"Content-Type": "application/json"}

    if api_type == "openai":
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if payload_override:
            payload = payload_override
        else:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            }
    elif payload_override:
        payload = payload_override
    else:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }

    try:
        logger.info("Querying LLM API (%s, model: %s) at %s...", api_type, model, url)
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if response.status_code == 200:
            result_json = response.json()  # pyright: ignore[reportAny]
            if api_type == "openai":
                choices = result_json.get("choices", [])  # pyright: ignore[reportAny]
                if choices:
                    text = choices[0].get("message", {}).get("content", "").strip()  # pyright: ignore[reportAny]
                    if text:
                        return text  # pyright: ignore[reportAny]
            else:  # ollama
                text = result_json.get("response", "").strip()  # pyright: ignore[reportAny]
                if text:
                    return text  # pyright: ignore[reportAny]
        logger.warning("%s LLM API returned status %d: %s", api_type.upper(), response.status_code, response.text)
    except Exception as e:
        logger.warning("Failed to query %s LLM: %s", api_type, e)

    return None
