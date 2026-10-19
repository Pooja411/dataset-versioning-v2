from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Protocol
from urllib.parse import urlparse

from ..errors import (
    AIError,
    ExternalDisabled,
    InvalidModelOutput,
    ProviderTimeout,
    ProviderUnavailable,
    RateLimited,
    RequestTooLarge,
)

RETRYABLE = {429, 500, 502, 503, 504}
MAX_RETRY_AFTER = 30.0
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
Transport = Callable[[str, dict[str, str], bytes, float], tuple[int, dict[str, str], bytes]]


@dataclass(frozen=True)
class LLMRequest:
    system: str
    user: str
    max_output_tokens: int = 1500
    temperature: float = 0.0


@dataclass(frozen=True)
class LLMResponse:
    text: str
    provider: str
    model: str
    latency_s: float
    attempts: int
    input_tokens: int | None = None
    output_tokens: int | None = None

    def usage(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "latency_s": round(self.latency_s, 3),
            "attempts": self.attempts,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


class Provider(Protocol):
    name: str
    is_local: bool

    def complete(self, request: LLMRequest) -> LLMResponse: ...


def urllib_transport(url: str, headers: dict[str, str], body: bytes, timeout: float):
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return resp.status, dict(resp.headers), resp.read(8 * 1024 * 1024)
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers or {}), exc.read(1024 * 1024)
    except (socket.timeout, TimeoutError) as exc:
        raise ProviderTimeout("request timed out") from exc
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, (socket.timeout, TimeoutError)):
            raise ProviderTimeout("request timed out") from exc
        raise ProviderUnavailable(f"cannot reach provider: {type(exc.reason).__name__}") from exc


def _error_detail(body: bytes) -> str:
    try:
        parsed = json.loads(body)
        err = parsed.get("error", {})
        kind = err.get("type") if isinstance(err, dict) else None
        return str(kind or "unknown")[:60]
    except (ValueError, AttributeError):
        return "unparseable error body"


class _HttpProvider:
    name = "http"
    is_local = False

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str,
        timeout: float = 60.0,
        max_retries: int = 2,
        max_input_chars: int = 60_000,
        transport: Transport = urllib_transport,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if not model:
            raise AIError("a model name is required (set DATAVAULT_AI_MODEL)")
        self.model = model
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_input_chars = max_input_chars
        self._transport = transport
        self._sleep = sleep

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.model!r}, base_url={self.base_url!r})"

    def _build(self, request: LLMRequest) -> tuple[str, dict[str, str], dict[str, Any]]:
        raise NotImplementedError

    def _parse(self, payload: dict[str, Any]) -> tuple[str, int | None, int | None]:
        raise NotImplementedError

    def complete(self, request: LLMRequest) -> LLMResponse:
        if len(request.system) + len(request.user) > self.max_input_chars:
            raise RequestTooLarge(
                f"prompt is {len(request.system) + len(request.user)} characters; limit is {self.max_input_chars}"
            )
        url, headers, payload = self._build(request)
        body = json.dumps(payload).encode()
        started = time.monotonic()
        for attempt in range(1, self.max_retries + 2):
            last = attempt == self.max_retries + 1
            try:
                status, resp_headers, data = self._transport(url, headers, body, self.timeout)
            except (ProviderTimeout, ProviderUnavailable):
                if last:
                    raise
                self._sleep(min(0.5 * 2 ** (attempt - 1), 8.0))
                continue
            if status == 200:
                return self._finish(data, started, attempt)
            if status in RETRYABLE and not last:
                delay = 0.5 * 2 ** (attempt - 1)
                retry_after = {k.lower(): v for k, v in resp_headers.items()}.get("retry-after", "")
                if retry_after.replace(".", "", 1).isdigit():
                    delay = float(retry_after)
                self._sleep(min(delay, MAX_RETRY_AFTER))
                continue
            self._raise_for_status(status, data)
        raise ProviderUnavailable("retries exhausted")

    def _finish(self, data: bytes, started: float, attempt: int) -> LLMResponse:
        try:
            payload = json.loads(data)
            text, tokens_in, tokens_out = self._parse(payload)
        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
            raise InvalidModelOutput("provider returned an unexpected response shape") from exc
        if not text.strip():
            raise InvalidModelOutput("provider returned an empty response")
        return LLMResponse(
            text, self.name, self.model, time.monotonic() - started, attempt, tokens_in, tokens_out
        )

    @staticmethod
    def _raise_for_status(status: int, data: bytes) -> None:
        detail = _error_detail(data)
        if status == 429:
            raise RateLimited(f"rate limited ({detail})")
        if status in (401, 403):
            raise ProviderUnavailable(f"authentication failed (HTTP {status})")
        if status == 413:
            raise RequestTooLarge("provider rejected the request as too large")
        raise ProviderUnavailable(f"provider error (HTTP {status}, {detail})")


class AnthropicProvider(_HttpProvider):
    name = "anthropic"
    API_VERSION = "2023-06-01"

    def __init__(self, model: str, *, api_key: str | None, base_url: str = "https://api.anthropic.com", **kw: Any):
        if not api_key:
            raise AIError("an API key is required (set ANTHROPIC_API_KEY or DATAVAULT_AI_API_KEY)")
        super().__init__(model, api_key=api_key, base_url=base_url, **kw)

    def _build(self, request: LLMRequest):
        headers = {
            "content-type": "application/json",
            "x-api-key": self._api_key or "",
            "anthropic-version": self.API_VERSION,
        }
        payload = {
            "model": self.model,
            "max_tokens": request.max_output_tokens,
            "temperature": request.temperature,
            "system": request.system,
            "messages": [{"role": "user", "content": request.user}],
        }
        return f"{self.base_url}/v1/messages", headers, payload

    def _parse(self, payload: dict[str, Any]):
        text = "".join(b["text"] for b in payload["content"] if b.get("type") == "text")
        usage = payload.get("usage", {})
        return text, usage.get("input_tokens"), usage.get("output_tokens")


class OpenAICompatibleProvider(_HttpProvider):
    name = "openai-compatible"

    def __init__(self, model: str, *, base_url: str, api_key: str | None = None, **kw: Any):
        super().__init__(model, api_key=api_key, base_url=base_url, **kw)
        host = urlparse(base_url).hostname or ""
        self.is_local = host in LOCAL_HOSTS

    def _build(self, request: LLMRequest):
        headers = {"content-type": "application/json"}
        if self._api_key:
            headers["authorization"] = f"Bearer {self._api_key}"
        payload = {
            "model": self.model,
            "max_tokens": request.max_output_tokens,
            "temperature": request.temperature,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
        }
        return f"{self.base_url}/chat/completions", headers, payload

    def _parse(self, payload: dict[str, Any]):
        text = payload["choices"][0]["message"]["content"] or ""
        usage = payload.get("usage", {})
        return text, usage.get("prompt_tokens"), usage.get("completion_tokens")


class MockProvider:
    name = "mock"
    is_local = True

    def __init__(self, responder: Callable[[LLMRequest], str] | list[str]):
        self._responder = responder
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if callable(self._responder):
            text = self._responder(request)
        else:
            if not self._responder:
                raise ProviderUnavailable("mock provider has no scripted responses left")
            text = self._responder.pop(0)
        if not text.strip():
            raise InvalidModelOutput("provider returned an empty response")
        return LLMResponse(text, "mock", "mock", 0.0, 1)


def build_provider(repo_config: dict[str, Any], env: dict[str, str] | None = None) -> Provider:
    env = dict(os.environ) if env is None else env
    kind = env.get("DATAVAULT_AI_PROVIDER", "").strip().lower()
    if not kind:
        raise AIError("AI is not configured; set DATAVAULT_AI_PROVIDER (anthropic or openai-compatible)")
    model = env.get("DATAVAULT_AI_MODEL", "").strip()
    timeout = float(env.get("DATAVAULT_AI_TIMEOUT", "60"))
    common: dict[str, Any] = {"timeout": timeout}
    if kind == "anthropic":
        key = env.get("DATAVAULT_AI_API_KEY") or env.get("ANTHROPIC_API_KEY")
        provider: Provider = AnthropicProvider(
            model, api_key=key, base_url=env.get("DATAVAULT_AI_BASE_URL", "https://api.anthropic.com"), **common
        )
    elif kind in ("openai-compatible", "openai", "local"):
        base = env.get("DATAVAULT_AI_BASE_URL")
        if not base:
            raise AIError("DATAVAULT_AI_BASE_URL is required for openai-compatible providers")
        key = env.get("DATAVAULT_AI_API_KEY") or env.get("OPENAI_API_KEY")
        provider = OpenAICompatibleProvider(model, base_url=base, api_key=key, **common)
    elif kind == "mock":
        script = env.get("DATAVAULT_AI_MOCK_RESPONSE")
        if not script:
            raise AIError("DATAVAULT_AI_MOCK_RESPONSE must point to a file with a canned response (testing only)")
        canned = open(script, encoding="utf-8").read()
        return MockProvider(lambda _request: canned)
    else:
        raise AIError(f"unknown provider {kind!r}")
    opted_in = repo_config.get("ai", {}).get("allow_external") or env.get("DATAVAULT_AI_ALLOW_EXTERNAL") == "1"
    if not provider.is_local and not opted_in:
        raise ExternalDisabled(
            "this provider sends data to an external service; opt in with "
            "'datavault config set ai.allow_external true' or DATAVAULT_AI_ALLOW_EXTERNAL=1"
        )
    return provider
