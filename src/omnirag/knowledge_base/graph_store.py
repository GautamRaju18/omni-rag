"""Synchronous Neo4j connection manager for graph building (runs in ingestion threads).

The API's health checks use the async ``Neo4jClient``; ingestion runs in worker threads, so the
graph writer uses the sync driver. One driver per process, with sessions per unit of work.

* ``write(fn)`` / ``read(fn)`` use *managed transactions*: the driver retries transient errors
  (leader switch, deadlock) automatically, up to ``max_transaction_retry_time``.
* ``schema(query)`` runs auto-commit statements. Neo4j doesn't allow schema changes (CREATE
  INDEX / CONSTRAINT) in the same transaction as data writes.
* All data queries are parameterised: values never go into the Cypher text.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from neo4j import Driver, GraphDatabase, ManagedTransaction, NotificationDisabledClassification
from neo4j.exceptions import DriverError, Neo4jError

T = TypeVar("T")


class GraphUnavailableError(RuntimeError):
    """Neo4j is unreachable, rejected the credentials, or failed a query."""


class GraphStore:
    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        *,
        database: str = "neo4j",
        connection_timeout_s: float = 5.0,
        max_retry_time_s: float = 15.0,
    ) -> None:
        self.uri = uri
        self.database = database
        self._driver: Driver = GraphDatabase.driver(
            uri,
            auth=(user, password),
            connection_timeout=connection_timeout_s,
            max_transaction_retry_time=max_retry_time_s,
            # "relationship type X does not exist" is normal on a fresh graph (e.g. no
            # hierarchical chunks yet, so no CHILD_OF); deprecations stay visible.
            notifications_disabled_classifications=[
                NotificationDisabledClassification.UNRECOGNIZED
            ],
        )

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> GraphStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def verify(self) -> dict[str, Any]:
        rows = self.query("CALL dbms.components() YIELD name, versions, edition "
                          "RETURN name, versions, edition")  # fmt: skip
        kernel = next(r for r in rows if r["name"] == "Neo4j Kernel")
        return {"version": kernel["versions"][0], "edition": kernel["edition"], "uri": self.uri}

    def write(self, fn: Callable[..., T], **kwargs: Any) -> T:
        return self._session_call("execute_write", fn, **kwargs)

    def read(self, fn: Callable[..., T], **kwargs: Any) -> T:
        return self._session_call("execute_read", fn, **kwargs)

    def query(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        """Read query returning plain dicts."""

        def run(tx: ManagedTransaction) -> list[dict[str, Any]]:
            return tx.run(cypher, params).data()

        return self.read(run)

    def schema(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        """Auto-commit statement (schema DDL, SHOW INDEXES)."""
        try:
            with self._driver.session(database=self.database) as session:
                return session.run(cypher, params).data()
        except (DriverError, Neo4jError) as exc:
            raise GraphUnavailableError(f"Neo4j schema statement failed: {exc}") from exc

    def _session_call(self, method: str, fn: Callable[..., T], **kwargs: Any) -> T:
        try:
            with self._driver.session(database=self.database) as session:
                return getattr(session, method)(fn, **kwargs)
        except (DriverError, Neo4jError) as exc:
            raise GraphUnavailableError(f"Neo4j {method} failed: {exc}") from exc
