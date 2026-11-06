"""The dashboard's only way to the system: the HTTP API (no Neo4j, no Ollama, no secrets here)."""

from __future__ import annotations

from typing import Any

import httpx


class ApiError(RuntimeError):
    """The API answered with an error, or couldn't be reached. ``str()`` is shown to the user."""


class ApiClient:
    def __init__(self, base_url: str, *, prefix: str = "/api/v1", timeout_s: float = 180.0,
                 client: httpx.Client | None = None) -> None:  # fmt: skip
        self.base_url = base_url.rstrip("/")
        self.prefix = prefix
        self._client = client or httpx.Client(base_url=self.base_url, timeout=timeout_s)

    def _call(self, method: str, path: str, **kw: Any) -> Any:
        try:
            r = self._client.request(method, f"{self.prefix}{path}", **kw)
        except httpx.TransportError as exc:
            raise ApiError(f"API not reachable at {self.base_url}: {exc}") from exc
        if r.status_code >= 400:
            try:
                body = r.json()  # {"error": {"code", "message", ...}} (core/errors.py)
                error = body.get("error") if isinstance(body.get("error"), dict) else {}
                detail = error.get("message") or body.get("detail") or body
            except ValueError:
                detail = r.text[:300]
            raise ApiError(f"HTTP {r.status_code}: {detail}")
        return r.json()

    def health(self) -> dict[str, Any]:
        # 503 means "a component is down" and still carries the report: show it.
        try:
            r = self._client.get(f"{self.prefix}/health")
        except httpx.TransportError as exc:
            raise ApiError(f"API not reachable at {self.base_url}: {exc}") from exc
        return r.json()

    def graph_stats(self) -> dict[str, Any]:
        return self._call("GET", "/graph/stats")

    def documents(self, limit: int = 100) -> dict[str, Any]:
        return self._call("GET", "/documents", params={"limit": limit})

    def search(self, query: str, mode: str, k: int, filters: dict[str, Any]) -> dict[str, Any]:
        return self._call(
            "POST",
            "/retrieval/search",
            json={"query": query, "mode": mode, "k": k, "filters": filters},
        )

    def advanced(self, query: str, k: int, filters: dict[str, Any],
                 strategy: dict[str, Any]) -> dict[str, Any]:  # fmt: skip
        return self._call("POST", "/retrieval/advanced",
                          json={"query": query, "k": k, "filters": filters,
                                "strategy": strategy})  # fmt: skip
