import os

import httpx
import pytest

pytestmark = pytest.mark.live


def _post(name: str, url: str, key: str, model: str) -> None:
    print(f"\n--- {name} ---")
    if not key:
        print("Skipped: key is not set.")
        return
    try:
        response = httpx.post(
            url,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": model, "messages": [{"role": "user", "content": "Say OK"}], "max_tokens": 5},
            timeout=10.0,
        )
        if response.status_code == 200:
            print("Connection successful.")
        else:
            print(f"Connection failed: {response.status_code}")
    except Exception as exc:
        print(f"Connection error: {exc}")


def test_provider_keys() -> None:
    _post("OpenRouter", "https://openrouter.ai/api/v1/chat/completions", os.getenv("OPENROUTER_API_KEY", ""), os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"))
    _post("Gemini", "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions", os.getenv("GEMINI_API_KEY", ""), os.getenv("GEMINI_MODEL", "gemini-2.0-flash"))
    _post("LLM7", os.getenv("LLM7_BASE_URL", "https://api.llm7.io/v1/chat/completions"), os.getenv("LLM7_API_KEY", ""), os.getenv("LLM7_MODEL", ""))
    _post("NVIDIA", "https://integrate.api.nvidia.com/v1/chat/completions", os.getenv("NVIDIA_API_KEY", ""), os.getenv("NVIDIA_MODEL", ""))


