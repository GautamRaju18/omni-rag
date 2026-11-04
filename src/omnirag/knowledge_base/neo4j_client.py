"""Neo4j connection manager.

The official async driver keeps a connection pool; we create exactly one driver per process
(in the FastAPI lifespan) and share it. Creating the driver does not open a connection, so the
API can start even when Neo4j is down and report that through ``/health/neo4j``.

Schema, constraints, vector and full-text indexes are added in Phase 5.
"""

from __future__ import annotations

import asyncio
import time

import structlog
from neo4j import AsyncDriver, AsyncGraphDatabase
from neo4j.exceptions import DriverError, Neo4jError

from omnirag.schemas.health import ComponentHealth

log = structlog.get_logger(module=__name__)

# Neo4j 2025+ returns one row per component ("Neo4j Kernel", "Cypher"), so read all rows.
_COMPONENTS_QUERY = "CALL dbms.components() YIELD name, versions, edition RETURN *"


class Neo4jClient:
    """Owns the driver and exposes small, parameterized query helpers."""

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        *,
        database: str = "neo4j",
        connection_timeout_s: float = 5.0,
    ) -> None:
        self.uri = uri
        self.database = database
        self._driver: AsyncDriver = AsyncGraphDatabase.driver(
            uri,
            auth=(user, password),
            connection_timeout=connection_timeout_s,
            connection_acquisition_timeout=connection_timeout_s,
            # Keep retries short: a health probe must fail fast, not hang for 30 s.
            max_transaction_retry_time=connection_timeout_s,
        )

    @property
    def driver(self) -> AsyncDriver:
        return self._driver

    async def close(self) -> None:
        await self._driver.close()

    async def health(self, timeout_s: float = 5.0) -> ComponentHealth:
        """Verify connectivity + auth and read server version/edition."""
        started = time.perf_counter()
        try:
            async with asyncio.timeout(timeout_s):
                # An auto-commit `session.run` is not retried by the driver (unlike
                # `execute_query`), which is exactly what a probe wants.
                async with self._driver.session(database=self.database) as session:
                    result = await session.run(_COMPONENTS_QUERY)
                    components = {r["name"]: r async for r in result}
        # DriverError: client side (connection, result handling); Neo4jError: server side.
        except (DriverError, Neo4jError, TimeoutError, OSError) as exc:
            log.warning("neo4j_unhealthy", error=repr(exc), uri=self.uri)
            return ComponentHealth(
                name="neo4j",
                status="down",
                latency_ms=_elapsed_ms(started),
                details={"uri": self.uri, "database": self.database},
                error=f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__,
            )

        kernel = components.get("Neo4j Kernel")
        cypher = components.get("Cypher")
        return ComponentHealth(
            name="neo4j",
            status="ok",
            latency_ms=_elapsed_ms(started),
            details={
                "uri": self.uri,
                "database": self.database,
                "version": kernel["versions"][0] if kernel else None,
                "edition": kernel["edition"] if kernel else None,
                "cypher_versions": cypher["versions"] if cypher else [],
            },
        )


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)
