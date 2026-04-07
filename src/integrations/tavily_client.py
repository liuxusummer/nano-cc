from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TavilyError(RuntimeError):
    code: str
    message: str
    details: dict[str, Any]

    def __str__(self) -> str:
        return self.message


def tavily_search(
    query: str,
    *,
    max_results: int = 5,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout_seconds: float = 15.0,
) -> dict[str, Any]:
    api_key_value = (api_key or os.getenv("TAVILY_API_KEY", "")).strip()
    if not api_key_value:
        raise TavilyError("missing_api_key", "TAVILY_API_KEY is required", {})

    base = (base_url or os.getenv("TAVILY_BASE_URL", "") or "https://api.tavily.com").strip()
    base = base.rstrip("/")
    url = f"{base}/search"

    payload = {
        "api_key": api_key_value,
        "query": query,
        "max_results": int(max_results),
        "include_answer": True,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=float(timeout_seconds)) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            status = getattr(resp, "status", None)
            code = int(status) if status is not None else int(resp.getcode())
    except urllib.error.HTTPError as exc:
        raw = ""
        try:
            raw = exc.read().decode("utf-8", errors="replace")
        except Exception:
            raw = ""
        raise TavilyError(
            "upstream_http_error",
            "tavily http error",
            {"status_code": int(getattr(exc, "code", 0) or 0), "url": url, "body": raw[:800]},
        ) from exc
    except Exception as exc:
        raise TavilyError("network_error", "tavily network error", {"url": url, "error": str(exc)}) from exc

    if code < 200 or code >= 300:
        raise TavilyError(
            "upstream_http_error",
            "tavily http error",
            {"status_code": code, "url": url, "body": raw[:800]},
        )

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise TavilyError(
            "invalid_response",
            "tavily returned invalid json",
            {"url": url, "body": raw[:800]},
        ) from exc

    if not isinstance(parsed, dict):
        raise TavilyError(
            "invalid_response",
            "tavily returned unexpected response",
            {"url": url, "type": type(parsed).__name__},
        )
    return parsed

