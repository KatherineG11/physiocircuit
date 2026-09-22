"""Shared LLM client: reads .env from project root and provides a unified OpenAI-compatible interface."""
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # repository root
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(ENV_PATH)

DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "doubao-seed-2-0-mini-260215").strip()

# provider -> (base_url, api_key_env, model_env)
PROVIDERS = {
    "deepseek": ("https://api.deepseek.com", "OPENAI_API_KEY", "OPENAI_MODEL"),
    "zhipu": ("https://open.bigmodel.cn/api/paas/v4", "ZHIPU_API_KEY", "ZHIPU_MODEL"),
    "qwen": ("https://dashscope.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY", "QWEN_MODEL"),
    "qwen37": ("https://dashscope.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY", "QWEN37_MODEL"),
    "doubao": ("https://ark.cn-beijing.volces.com/api/v3", "DOUBAO_API_KEY", "DOUBAO_MODEL"),
    "kimi": ("https://ark.cn-beijing.volces.com/api/plan/v3", "KIMI_API_KEY", "KIMI_MODEL"),
    "minimax": ("https://ark.cn-beijing.volces.com/api/plan/v3", "MINIMAX_API_KEY", "MINIMAX_MODEL"),
    "hsdp": ("https://ark.cn-beijing.volces.com/api/plan/v3", "HSDP_API_KEY", "HSDP_MODEL"),
    "hsdb": ("https://ark.cn-beijing.volces.com/api/plan/v3", "HSDB_API_KEY", "HSDB_MODEL"),
    "doubao-seed-2-0-mini-260215": (
        "https://ark.cn-beijing.volces.com/api/plan/v3",
        "HSDB_API_KEY",
        "HSDB_MODEL",
    ),
    "deepseek-v4-flash-260801": (
        "https://ark.cn-beijing.volces.com/api/plan/v3",
        "HSDP_API_KEY",
        "HSDP_MODEL",
    ),
    "kimi-k2.7-code": (
        "https://ark.cn-beijing.volces.com/api/plan/v3",
        "KIMI_API_KEY",
        "KIMI_MODEL",
    ),
    "minimax-m3": (
        "https://ark.cn-beijing.volces.com/api/plan/v3",
        "MINIMAX_API_KEY",
        "MINIMAX_MODEL",
    ),
}


def _resolve_provider(provider):
    """Resolve a provider alias or model name; returns (base_url, api_key, model, is_plan_endpoint)."""
    provider = (provider or DEFAULT_PROVIDER).strip()
    if provider in PROVIDERS:
        base_url, key_env, model_env = PROVIDERS[provider]
        api_key = os.getenv(key_env, "")
        model = os.getenv(model_env, provider)
        is_plan = "plan/v3" in base_url
        return base_url, api_key, model, is_plan
    # Unregistered name: treat as a direct model name, use Volcengine Ark Agent Plan endpoint
    return (
        "https://ark.cn-beijing.volces.com/api/plan/v3",
        os.getenv("HSDB_API_KEY", ""),
        provider,
        True,
    )


class LLMClient:
    """OpenAI-compatible chat/completions wrapper with retries and timeout."""

    def __init__(self, provider=None, temperature=0.2, timeout=180, max_retries=3):
        base_url, api_key, model, is_plan = _resolve_provider(provider)
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        # Volcengine Ark Agent Plan and DeepSeek both use OpenAI-compatible /chat/completions
        self.endpoint = f"{self.base_url}/chat/completions"
        self.temperature = temperature

    def _headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def chat(self, messages, temperature=None, max_tokens=None, json_mode=False, **extra):
        """Send a chat request; returns (content_str, usage)."""
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "stream": False,
        }
        if max_tokens:
            body["max_tokens"] = max_tokens
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        body.update(extra)

        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(self.endpoint, headers=self._headers(), json=body, timeout=self.timeout)
                if resp.status_code != 200:
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                return content, usage
            except Exception as e:  # noqa: BLE001
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(2 * attempt)
        raise RuntimeError(f"LLM call failed after {self.max_retries} retries: {last_err}")

    def chat_json(self, messages, temperature=None, max_tokens=None, **extra):
        """Send a request with JSON mode enabled; parse and return a dict."""
        content, usage = self.chat(messages, temperature=temperature, max_tokens=max_tokens, json_mode=True, **extra)
        return parse_json_content(content), usage


def parse_json_content(content):
    """Robustly parse a JSON dict from LLM output."""
    text = content.strip()
    # Strip ```json ... ``` fences
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try extracting from the first { to the last }
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Failed to parse JSON from LLM output: {content[:400]}")


def _load_dotenv_pyproject():
    """Helper for other modules to confirm that the environment has been loaded."""
    return ENV_PATH.exists()
