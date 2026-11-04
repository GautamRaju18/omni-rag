"""Async client for the local Ollama server.

Ollama runs natively on Windows (direct GPU access, no Docker GPU passthrough). From WSL2 in
mirrored networking mode it is reachable at ``http://localhost:11434``.

Phase 1 only needs the read-only endpoints used for health checks:

* ``GET /api/version`` – server version (cheap liveness probe)
* ``GET /api/tags``    – models pulled to disk
* ``GET /api/ps``      – models currently loaded, incl. how many bytes sit in VRAM

Chat, streaming and embeddings are added in later phases on top of the same ``httpx`` client.
"""

from __future__ import annotations

import time
from typing import Any

import httpx
import structlog

from omnirag.schemas.health import ComponentHealth

log = structlog.get_logger(module=__name__)

_GB = 1024**3


def normalize_model_name(name: str) -> str:
    """Ollama stores untagged models as ``name:latest``; compare on the normalized form."""
    return name if ":" in name else f"{name}:latest"


class OllamaClient:
    """Thin, typed wrapper around the Ollama REST API."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = 120.0,
        connect_retries: int = 2,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        # One pooled client per process. `retries` only retries failed *connections*
        # (never a request the server already received), so it is safe for POSTs too.
        self._http = http_client or httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout_s, connect=5.0),
            transport=httpx.AsyncHTTPTransport(retries=connect_retries),
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _get_json(self, path: str, timeout_s: float | None = None) -> dict[str, Any]:
        timeout = httpx.Timeout(timeout_s) if timeout_s else httpx.USE_CLIENT_DEFAULT
        response = await self._http.get(path, timeout=timeout)
        response.raise_for_status()
        return response.json()

    async def version(self, timeout_s: float | None = None) -> str:
        return (await self._get_json("/api/version", timeout_s))["version"]

    async def list_models(self, timeout_s: float | None = None) -> list[str]:
        """Names of all models pulled to disk, e.g. ``["bge-m3:latest", "qwen3:4b"]``."""
        data = await self._get_json("/api/tags", timeout_s)
        return [m["name"] for m in data.get("models", [])]

    async def loaded_models(self, timeout_s: float | None = None) -> list[dict[str, Any]]:
        """Models resident in memory and how much of each is on the GPU."""
        data = await self._get_json("/api/ps", timeout_s)
        loaded = []
        for m in data.get("models", []):
            size, vram = int(m.get("size", 0)), int(m.get("size_vram", 0))
            loaded.append(
                {
                    "name": m["name"],
                    "size_gb": round(size / _GB, 2),
                    "vram_gb": round(vram / _GB, 2),
                    "gpu_percent": round(100 * vram / size, 1) if size else 0.0,
                    "context_length": m.get("context_length"),
                    "expires_at": m.get("expires_at"),
                }
            )
        return loaded

    async def health(self, required_models: list[str], timeout_s: float = 3.0) -> ComponentHealth:
        """Probe the server; ``degraded`` if reachable but required models are missing."""
        started = time.perf_counter()
        try:
            version = await self.version(timeout_s)
            available = await self.list_models(timeout_s)
            loaded = await self.loaded_models(timeout_s)
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            log.warning("ollama_unhealthy", error=repr(exc), base_url=self.base_url)
            return ComponentHealth(
                name="ollama",
                status="down",
                latency_ms=_elapsed_ms(started),
                details={"base_url": self.base_url},
                error=f"{type(exc).__name__}: {exc}",
            )

        available_set = {normalize_model_name(m) for m in available}
        missing = [m for m in required_models if normalize_model_name(m) not in available_set]
        return ComponentHealth(
            name="ollama",
            status="degraded" if missing else "ok",
            latency_ms=_elapsed_ms(started),
            details={
                "base_url": self.base_url,
                "version": version,
                "required_models": required_models,
                "missing_models": missing,
                "available_models": sorted(available),
                "loaded_models": loaded,
                "vram_in_use_gb": round(sum(m["vram_gb"] for m in loaded), 2),
            },
            error=f"Missing models: {', '.join(missing)} (run: ollama pull <name>)"
            if missing
            else None,
        )


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)
