"""Small OpenAI-compatible, single-turn HTTP client with no GPU dependencies.

Secrets are resolved from the environment at request time. Never persist the
return value of ``resolve_endpoint``; use ``endpoint_metadata`` in manifests.
The transport intentionally does not follow redirects with credentials.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


class APIError(RuntimeError):
    """A safe-to-log API failure; response bodies and secrets are omitted."""


def _env_name(config: dict, key: str) -> str | None:
    name = config.get(key)
    if name is None:
        return None
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(f"{key} must be an environment variable name")
    return name


def _setting(config: dict, name: str) -> str:
    variable = _env_name(config, f"{name}_env")
    value = os.environ.get(variable) if variable else None
    if not value:
        value = config.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Configure {name} or its environment variable")
    return value.strip()


def _safe_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        raise ValueError("base_url must be a valid HTTP(S) URL") from None
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment
            or any(ord(char) < 33 for char in value)):
        raise ValueError("base_url must be HTTP(S), without userinfo, query or fragment")
    if port is not None and not 0 < port <= 65535:
        raise ValueError("base_url has an invalid port")
    return urlunsplit(parsed).rstrip("/")


def endpoint_metadata(config: dict) -> dict:
    """Resolve nonsecret endpoint details for fingerprints and run manifests.

    API keys must be passed through ``api_key_env``. Literal ``api_key`` entries
    are rejected, including for local endpoints, to keep configs shareable.
    """
    if not isinstance(config, dict):
        raise ValueError("Endpoint config must be a mapping")
    allowed = {"base_url", "model", "base_url_env", "model_env", "api_key_env",
               "api_key_required", "timeout", "retries", "revision"}
    if set(config) - allowed:
        raise ValueError("Unknown endpoint field; use api_key_env for credentials")
    if "api_key" in config:
        raise ValueError("Use api_key_env; do not place API keys in configuration files")
    result = {"base_url": _safe_url(_setting(config, "base_url")),
              "model": _setting(config, "model")}
    for key in ("base_url_env", "model_env", "api_key_env"):
        variable = _env_name(config, key)
        if variable:
            result[key] = variable
    return result


def resolve_endpoint(config: dict) -> dict:
    """Resolve credentials for HTTP use only. Never serialize this return value."""
    endpoint = endpoint_metadata(config)
    variable = _env_name(config, "api_key_env")
    secret = os.environ.get(variable, "") if variable else ""
    required = config.get("api_key_required", True)
    if not isinstance(required, bool):
        raise ValueError("api_key_required must be boolean")
    if required and not secret.strip():
        raise ValueError("API key environment variable is missing or empty")
    if any(ord(char) < 32 or ord(char) == 127 for char in secret):
        raise ValueError("API key environment variable contains invalid characters")
    return {**endpoint, "api_key": secret}


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _open_request(request: Request, timeout: float):
    return build_opener(_NoRedirect()).open(request, timeout=timeout)


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces = []
        for part in content:
            if (not isinstance(part, dict) or part.get("type") != "text"
                    or not isinstance(part.get("text"), str)):
                raise APIError("API response contains unsupported content blocks")
            pieces.append(part["text"])
        return "".join(pieces)
    raise APIError("API response is missing assistant text")


def _parse_response(payload: object) -> dict:
    if not isinstance(payload, dict):
        raise APIError("API response must be an object")
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise APIError("API response must contain exactly one choice")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict) or message.get("role", "assistant") != "assistant":
        raise APIError("API response is missing an assistant message")
    if message.get("tool_calls") or message.get("function_call") or choice.get("finish_reason") in {
        "tool_calls", "function_call",
    }:
        raise APIError("Single-turn API evaluation does not support tool calls; use an official harness")
    finish_reason = choice.get("finish_reason")
    # A valid length-limited attempt can exhaust its budget in reasoning and
    # produce no final content. Preserve it as a model failure, not an API retry.
    content = message.get("content")
    text = "" if content is None and finish_reason == "length" else _content_text(content)
    reasoning = message.get("reasoning_content", message.get("reasoning", ""))
    if reasoning is None:
        reasoning = ""
    if not isinstance(reasoning, str):
        raise APIError("API response has invalid reasoning content")
    if not text.strip() and finish_reason != "length":
        raise APIError("API response has no final assistant text")
    if finish_reason not in {"stop", "length", "content_filter"}:
        raise APIError("API response has an unsupported or missing finish reason")
    usage = payload.get("usage") or {}
    if not isinstance(usage, dict):
        raise APIError("API response has invalid usage metadata")
    # Persist only numeric usage counters, not arbitrary provider-returned text.
    def counters(data):
        return {key: counters(value) if isinstance(value, dict) else value
                for key, value in data.items()
                if isinstance(key, str) and (isinstance(value, dict)
                    or isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(value) and value >= 0)}
    response_model = payload.get("model")
    if response_model is not None and not isinstance(response_model, str):
        raise APIError("API response has invalid model metadata")
    return {"text": text, "reasoning": reasoning, "finish_reason": finish_reason,
            "usage": counters(usage), "response_model": response_model}


class ChatClient:
    """Stateless, non-streaming Chat Completions client (one choice per request).

    ``parameters`` is passed through to the provider, allowing model-specific
    reasoning options and either max_tokens or max_completion_tokens. Routing,
    tools and streaming cannot be overridden through this mapping.
    """

    def __init__(self, config: dict):
        endpoint_metadata(config)
        self._config = dict(config)
        timeout = config.get("timeout", 120)
        retries = config.get("retries", 2)
        if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("timeout must be positive and finite")
        if isinstance(retries, bool) or not isinstance(retries, int) or not 0 <= retries <= 10:
            raise ValueError("retries must be an integer between 0 and 10")
        self._timeout = timeout
        self._retries = retries

    def complete(self, messages: list[dict], parameters: dict) -> dict:
        if not isinstance(messages, list) or not messages or any(not isinstance(m, dict) for m in messages):
            raise ValueError("messages must be a nonempty list of message objects")
        if any(message.get("role") in {"tool", "function"} or message.get("tool_calls")
               or message.get("function_call") for message in messages):
            raise ValueError("Tool conversation history requires an agent harness")
        if not isinstance(parameters, dict):
            raise ValueError("parameters must be a mapping")
        if any(key in parameters for key in (
            "model", "messages", "stream", "tools", "tool_choice", "functions", "function_call",
            "api_key", "base_url", "extra_headers",
        )):
            raise ValueError("parameters cannot override routing, credentials, streaming or tools")
        if parameters.get("n", 1) != 1 or isinstance(parameters.get("n", 1), bool):
            raise ValueError("Only n=1 is supported; repeat requests explicitly")
        if "max_tokens" in parameters and "max_completion_tokens" in parameters:
            raise ValueError("Set only one of max_tokens or max_completion_tokens")
        endpoint = resolve_endpoint(self._config)
        url = endpoint["base_url"]
        if not url.endswith("/chat/completions"):
            url += "/chat/completions"
        try:
            body = json.dumps({**parameters, "model": endpoint["model"], "messages": messages,
                               "n": 1, "stream": False}, ensure_ascii=False, allow_nan=False).encode("utf-8")
        except (TypeError, ValueError):
            raise ValueError("Request must contain finite JSON-serializable values") from None
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if endpoint["api_key"]:
            headers["Authorization"] = "Bearer " + endpoint["api_key"]
        for attempt in range(self._retries + 1):
            try:
                request = Request(url, data=body, headers=headers, method="POST")
                with _open_request(request, self._timeout) as response:
                    raw = response.read(16 * 1024 * 1024 + 1)
                if len(raw) > 16 * 1024 * 1024:
                    raise APIError("API response exceeded the 16 MiB limit")
                try:
                    payload = json.loads(raw)
                except (ValueError, UnicodeError):
                    raise APIError("API returned invalid JSON") from None
                return _parse_response(payload)
            except HTTPError as error:
                status = error.code
                error.close()
                retry = status in {408, 409, 429} or 500 <= status < 600
                if not retry or attempt == self._retries:
                    raise APIError(f"API request failed (HTTP {status}); response body omitted") from None
            except (URLError, TimeoutError, OSError):
                if attempt == self._retries:
                    raise APIError("API network request failed; connection details omitted") from None
            except APIError:
                raise
            except Exception:
                # HTTP implementations can include headers/URLs in exception text.
                raise APIError("API request or response failed; details omitted") from None
            time.sleep(min(2 ** attempt, 8))
        raise APIError("API request failed")  # Defensive: every loop path returns or raises.
