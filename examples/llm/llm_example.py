#!/usr/bin/env python3
"""llm_example.py
======================
End-to-End demonstration of the LLM client integration library.

Usage:
  python examples/llm/llm_example.py --prompt "What is the capital of France?"
"""

import argparse
import sys
from pathlib import Path

# Ensure src is in path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.llm import generate_llm_response
from src.utils import config as _config_module

def main() -> None:
    parser = argparse.ArgumentParser(description="E2E LLM client demonstration.")
    parser.add_argument(
        "--prompt",
        type=str,
        default="State in one short sentence what is the capital of France.",
        help="The prompt to send to the LLM.",
    )
    parser.add_argument(
        "--api-type",
        type=str,
        help="Override the LLM API provider type (e.g. ollama, openai).",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Override the model to use.",
    )
    parser.add_argument(
        "--url",
        type=str,
        help="Override the API endpoint URL.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="API key if using OpenAI API.",
    )

    args = parser.parse_args()

    # Load configuration
    try:
        config = _config_module.load_config()
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

    # Resolve variables, prioritizing CLI arguments over config file settings
    api_type = args.api_type or config.llm.api_type
    url = args.url or config.llm.url
    model = args.model or config.llm.model
    api_key = args.api_key or config.llm.api_key
    timeout = config.llm.timeout

    print("=" * 60)
    print("LLM CLIENT CONFIGURATION:")
    print(f"  API Provider Type : {api_type}")
    print(f"  Endpoint URL      : {url}")
    print(f"  Model             : {model}")
    print(f"  API Key           : {'[PROVIDED]' if api_key else '[NONE]'}")
    print(f"  Timeout           : {timeout} seconds")
    print("=" * 60)

    print(f"\nPrompt: '{args.prompt}'")
    print("Sending query to LLM...")

    response = generate_llm_response(
        api_type=api_type,
        url=url,
        model=model,
        prompt=args.prompt,
        timeout=timeout,
        api_key=api_key
    )

    print("\n" + "=" * 60)
    print("RESPONSE:")
    if response:
        print(response)
    else:
        print("Error: Could not retrieve a response from the LLM.")
        print("Note: If using Ollama, make sure the Ollama server is running locally and the model is pulled.")
        print("      If using OpenAI, ensure your API key and URL are configured correctly.")
    print("=" * 60)

if __name__ == "__main__":
    main()
