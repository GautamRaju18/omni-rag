"""OllamaClient tests against an in-memory HTTP transport (no real server needed)."""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from omnirag.llm.ollama_client import OllamaClient, normalize_model_name

GB = 1024**3


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> OllamaClient:
    http = httpx.AsyncClient(base_url="http://ollama.test", transport=httpx.MockTransport(handler))
    return OllamaClient("http://ollama.test", http_client=http)


def _healthy_handler(request: httpx.Request) -> httpx.Response:
    routes = {
        "/api/version": {"version": "0.40.14"},
        "/api/tags": {"models": [{"name": "qwen3:4b-instruct"}, {"name": "bge-m3:latest"}]},
        "/api/ps": {
            "models": [
                {"name": "qwen3:4b-instruct", "size": 4 * GB, "size_vram": 3 * GB},
            ]
        },
    }
    return httpx.Response(200, json=routes[request.url.path])


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("bge-m3", "bge-m3:latest"), ("qwen3:4b", "qwen3:4b"), ("bge-m3:latest", "bge-m3:latest")],
)
def test_normalize_model_name(raw: str, expected: str) -> None:
    assert normalize_model_name(raw) == expected


async def test_health_ok_when_all_required_models_present() -> None:
    client = _client(_healthy_handler)
    result = await client.health(["qwen3:4b-instruct", "bge-m3"])
    assert result.status == "ok"
    assert result.error is None
    assert result.details["version"] == "0.40.14"
    assert result.details["missing_models"] == []
    loaded = result.details["loaded_models"][0]
    assert loaded["vram_gb"] == 3.0
    assert loaded["gpu_percent"] == 75.0
    assert result.details["vram_in_use_gb"] == 3.0


async def test_health_degraded_when_model_missing() -> None:
    client = _client(_healthy_handler)
    result = await client.health(["qwen3:4b-instruct", "qwen2.5vl:3b"])
    assert result.status == "degraded"
    assert result.details["missing_models"] == ["qwen2.5vl:3b"]
    assert "ollama pull" in (result.error or "")


async def test_health_down_when_unreachable() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    result = await _client(refuse).health(["bge-m3"])
    assert result.status == "down"
    assert "ConnectError" in (result.error or "")


async def test_health_down_on_server_error() -> None:
    result = await _client(lambda r: httpx.Response(500, text="boom")).health(["bge-m3"])
    assert result.status == "down"
    assert "HTTPStatusError" in (result.error or "")
