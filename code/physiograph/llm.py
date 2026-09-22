"""Small, auditable LLM client abstractions used by PhysioCircuit.

The module deliberately depends only on the Python standard library.  The
``requests`` package is imported lazily by :class:`OpenAICompatibleClient`, so
the deterministic/mock test path works in a minimal Python environment.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, MutableSequence, Optional, Sequence


JsonObject = dict[str, Any]
ChatMessage = Mapping[str, Any]


@dataclass(frozen=True)
class LLMResponse:
    """One model response together with the provider audit trail.

    ``raw_response`` intentionally keeps the complete decoded provider JSON.
    This is required for reproducibility and is never interpolated into a
    later prompt by the engine.
    """

    content: str
    parsed: Optional[JsonObject] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    raw_response: Any = None


def parse_json_object(content: str) -> JsonObject:
    """Parse a JSON object, tolerating a Markdown fence and leading prose."""

    text = str(content).strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        text = text[first_newline + 1 :] if first_newline >= 0 else ""
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
        text = text.strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("model response does not contain a JSON object")
        try:
            value = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON model response: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("model response must be a JSON object")
    return value


class BaseLLMClient(ABC):
    """Interface shared by real and deterministic model clients."""

    model: str

    @abstractmethod
    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 1024,
        json_mode: bool = True,
        **request_options: Any,
    ) -> LLMResponse:
        """Return one completion and provider metadata."""

    def complete_json(
        self,
        messages: Sequence[ChatMessage],
        **request_options: Any,
    ) -> LLMResponse:
        response = self.complete(messages, json_mode=True, **request_options)
        if response.parsed is not None:
            return response
        parsed = parse_json_object(response.content)
        return LLMResponse(
            content=response.content,
            parsed=parsed,
            metadata=response.metadata,
            raw_response=response.raw_response,
        )


MockItem = Any


class MockLLMClient(BaseLLMClient):
    """Scripted LLM client for tests and offline dry runs.

    A script item may be an :class:`LLMResponse`, a JSON mapping, a string, or
    a callable receiving ``(messages, request_options, call_index)``.
    """

    def __init__(
        self,
        responses: Iterable[MockItem] | Callable[[Sequence[ChatMessage], Mapping[str, Any], int], MockItem],
        *,
        model: str = "mock-physiograph",
    ) -> None:
        self.model = model
        self._factory = responses if callable(responses) else None
        self._responses: MutableSequence[MockItem] = [] if self._factory else list(responses)  # type: ignore[arg-type]
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 1024,
        json_mode: bool = True,
        **request_options: Any,
    ) -> LLMResponse:
        index = len(self.calls)
        options = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "json_mode": json_mode,
            **request_options,
        }
        snapshot = [dict(message) for message in messages]
        self.calls.append({"messages": snapshot, "options": dict(options)})

        if self._factory is not None:
            item = self._factory(snapshot, options, index)
        else:
            if not self._responses:
                raise RuntimeError(f"mock response script exhausted at call {index}")
            item = self._responses.pop(0)

        if isinstance(item, Exception):
            raise item
        if isinstance(item, LLMResponse):
            return item
        if isinstance(item, Mapping):
            parsed = dict(item)
            content = json.dumps(parsed, ensure_ascii=False, sort_keys=True)
            raw = {"choices": [{"message": {"content": content}}], "model": self.model}
        else:
            content = str(item)
            parsed = parse_json_object(content) if json_mode else None
            raw = {"choices": [{"message": {"content": content}}], "model": self.model}
        return LLMResponse(
            content=content,
            parsed=parsed,
            metadata={
                "provider": "mock",
                "requested_model": self.model,
                "resolved_model": self.model,
                "call_index": index,
                "usage": {},
                "finish_reason": "stop",
            },
            raw_response=raw,
        )


class OpenAICompatibleClient(BaseLLMClient):
    """Synchronous client for OpenAI-compatible ``chat/completions`` APIs."""

    RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 180.0,
        max_retries: int = 3,
        provider: str = "openai-compatible",
        default_headers: Optional[Mapping[str, str]] = None,
        session: Any = None,
    ) -> None:
        if not base_url:
            raise ValueError("base_url is required")
        if not model:
            raise ValueError("model is required")
        self.base_url = base_url.rstrip("/")
        self.endpoint = (
            self.base_url
            if self.base_url.endswith("/chat/completions")
            else f"{self.base_url}/chat/completions"
        )
        self.api_key = api_key
        self.model = model
        self.timeout = float(timeout)
        self.max_retries = max(1, int(max_retries))
        self.provider = provider
        self.default_headers = dict(default_headers or {})
        self._session = session

    def _http(self) -> Any:
        if self._session is not None:
            return self._session
        try:
            import requests  # type: ignore
        except ImportError as exc:  # pragma: no cover - exercised only without requests
            raise RuntimeError(
                "OpenAICompatibleClient requires the optional 'requests' package"
            ) from exc
        self._session = requests.Session()
        return self._session

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 1024,
        json_mode: bool = True,
        **request_options: Any,
    ) -> LLMResponse:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [dict(message) for message in messages],
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_tokens),
            "stream": False,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        body.update(request_options)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.default_headers,
        }
        http = self._http()
        error: Optional[BaseException] = None
        started = time.monotonic()
        for attempt in range(1, self.max_retries + 1):
            try:
                response = http.post(
                    self.endpoint,
                    headers=headers,
                    json=body,
                    timeout=self.timeout,
                )
                status = int(response.status_code)
                if status >= 400:
                    excerpt = str(getattr(response, "text", ""))[:500]
                    exc = RuntimeError(f"LLM HTTP {status}: {excerpt}")
                    if status not in self.RETRYABLE_STATUS or attempt >= self.max_retries:
                        raise exc
                    error = exc
                    time.sleep(min(8.0, 0.5 * (2 ** (attempt - 1))))
                    continue

                raw = response.json()
                choice = raw["choices"][0]
                content = choice["message"].get("content", "")
                parsed = parse_json_object(content) if json_mode else None
                response_headers = getattr(response, "headers", {}) or {}
                request_id = (
                    response_headers.get("x-request-id")
                    or response_headers.get("request-id")
                    or raw.get("id")
                )
                metadata = {
                    "provider": self.provider,
                    "endpoint": self.endpoint,
                    "requested_model": self.model,
                    "resolved_model": raw.get("model", self.model),
                    "response_id": raw.get("id"),
                    "request_id": request_id,
                    "created": raw.get("created"),
                    "system_fingerprint": raw.get("system_fingerprint"),
                    "usage": raw.get("usage", {}),
                    "finish_reason": choice.get("finish_reason"),
                    "http_status": status,
                    "attempt": attempt,
                    "latency_ms": round((time.monotonic() - started) * 1000.0, 3),
                }
                return LLMResponse(
                    content=content,
                    parsed=parsed,
                    metadata=metadata,
                    raw_response=raw,
                )
            except Exception as exc:  # noqa: BLE001 - provider/network exceptions vary
                error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(min(8.0, 0.5 * (2 ** (attempt - 1))))
        raise RuntimeError(f"LLM request failed after {self.max_retries} attempts: {error}") from error


__all__ = [
    "BaseLLMClient",
    "LLMResponse",
    "MockLLMClient",
    "OpenAICompatibleClient",
    "parse_json_object",
]
