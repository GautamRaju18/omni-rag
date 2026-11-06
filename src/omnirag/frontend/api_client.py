"""The dashboard's only way to the system: the HTTP API (no Neo4j, no Ollama, no secrets here)."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
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

    # ---------------------------------------------------------------- chat (Phases 10-11)
    def chat(self, question: str, *, conversation_id: str | None = None,
             strategy: dict[str, Any] | None = None, k: int = 5,
             respect_check: bool | None = None) -> dict[str, Any]:  # fmt: skip
        return self._call("POST", "/chat", json={
            "question": question, "conversation_id": conversation_id, "k": k,
            "strategy": strategy, "respect_check": respect_check})  # fmt: skip

    def chat_stream(self, question: str, *, conversation_id: str | None = None,
                    strategy: dict[str, Any] | None = None, k: int = 5,
                    respect_check: bool | None = None) -> Iterator[dict[str, Any]]:  # fmt: skip
        """Server-Sent Events from ``POST /chat/stream``, parsed into dicts."""
        body = {"question": question, "conversation_id": conversation_id, "k": k,
                "strategy": strategy, "respect_check": respect_check}  # fmt: skip
        try:
            with self._client.stream("POST", f"{self.prefix}/chat/stream", json=body) as r:
                if r.status_code >= 400:
                    r.read()
                    raise ApiError(f"HTTP {r.status_code}: {r.text[:300]}")
                yield from parse_sse(r.iter_lines())
        except httpx.TransportError as exc:
            raise ApiError(f"API not reachable at {self.base_url}: {exc}") from exc

    def history(self, limit: int = 50) -> list[dict[str, Any]]:
        return self._call("GET", "/chat/history", params={"limit": limit})

    def conversation(self, conversation_id: str) -> dict[str, Any]:
        return self._call("GET", f"/chat/history/{conversation_id}")

    def delete_conversation(self, conversation_id: str) -> None:
        self._client.delete(f"{self.prefix}/chat/history/{conversation_id}")

    # ---------------------------------------------------------------- documents + ingestion
    def document(self, document_id: str) -> dict[str, Any]:
        return self._call("GET", f"/documents/{document_id}")

    def elements(self, document_id: str, limit: int = 200) -> dict[str, Any]:
        return self._call("GET", f"/documents/{document_id}/elements", params={"limit": limit})

    def original(self, document_id: str) -> bytes:
        try:
            r = self._client.get(f"{self.prefix}/documents/{document_id}/original")
        except httpx.TransportError as exc:
            raise ApiError(f"API not reachable at {self.base_url}: {exc}") from exc
        if r.status_code >= 400:
            raise ApiError(f"HTTP {r.status_code}")
        return r.content

    def upload(self, name: str, data: bytes, force: bool = False) -> dict[str, Any]:
        return self._call("POST", "/documents/upload", files={"file": (name, data)},
                          params={"force": force})  # fmt: skip

    def reingest(self, document_id: str) -> dict[str, Any]:
        return self._call("POST", "/ingestion/start", json={"document_id": document_id})

    def delete_document(self, document_id: str) -> None:
        r = self._client.delete(f"{self.prefix}/documents/{document_id}")
        if r.status_code >= 400:
            raise ApiError(f"HTTP {r.status_code}: {r.text[:300]}")

    def jobs(self, limit: int = 50) -> dict[str, Any]:
        return self._call("GET", "/ingestion/jobs", params={"limit": limit})

    def ingestion_status(self) -> dict[str, Any]:
        return self._call("GET", "/ingestion/status")

    # ---------------------------------------------------------------- graph, eval, analytics
    def graph_entities(self, limit: int = 100) -> list[dict[str, Any]]:
        return self._call("GET", "/graph/entities", params={"limit": limit})

    def graph_neighborhood(self, entity: str | None = None, document_id: str | None = None,
                           limit: int = 60) -> dict[str, Any]:  # fmt: skip
        params = {"limit": limit, "entity": entity, "document_id": document_id}
        return self._call("GET", "/graph/neighborhood",
                          params={k: v for k, v in params.items() if v})  # fmt: skip

    def graph_entity(self, name: str) -> dict[str, Any]:
        return self._call("GET", "/graph/entity", params={"name": name})

    def evaluate(self, experiments: list[str], limit: int | None, judge: bool) -> dict[str, Any]:
        return self._call(
            "POST", "/evaluate", json={"experiments": experiments, "limit": limit, "judge": judge}
        )

    def eval_runs(self) -> list[dict[str, Any]]:
        return self._call("GET", "/evaluate")

    def eval_run(self, run_id: str, full: bool = False) -> dict[str, Any]:
        return self._call("GET", f"/evaluate/{run_id}", params={"full": full})

    def analytics(self) -> dict[str, Any]:
        return self._call("GET", "/analytics")


def parse_sse(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    """``event: x`` / ``data: {...}`` blocks → the JSON data of each event."""
    data: list[str] = []
    for line in lines:
        if line.startswith("data:"):
            data.append(line[5:].strip())
        elif not line.strip() and data:
            yield json.loads("\n".join(data))
            data = []
    if data:
        yield json.loads("\n".join(data))
