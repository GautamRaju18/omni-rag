"""Neo4jClient failure behaviour that needs no running database."""

from __future__ import annotations

import time

from omnirag.knowledge_base.neo4j_client import Neo4jClient


async def test_health_down_fast_when_nothing_listens() -> None:
    # Port 1 on localhost is never a Neo4j server: connection is refused immediately.
    client = Neo4jClient("bolt://127.0.0.1:1", "neo4j", "x", connection_timeout_s=2.0)
    started = time.perf_counter()
    try:
        result = await client.health(timeout_s=3.0)
    finally:
        await client.close()
    assert result.status == "down"
    assert result.error
    assert time.perf_counter() - started < 3.5, "health probe must fail fast, not hang"
