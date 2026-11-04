"""Health-check response schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

HealthStatus = Literal["ok", "degraded", "down"]

_SEVERITY: dict[HealthStatus, int] = {"ok": 0, "degraded": 1, "down": 2}


def worst_status(statuses: list[HealthStatus]) -> HealthStatus:
    """Overall status is the worst component status (empty list counts as ok)."""
    return max(statuses, key=_SEVERITY.__getitem__, default="ok")


class ComponentHealth(BaseModel):
    """Health of a single dependency."""

    name: str
    status: HealthStatus
    latency_ms: float | None = Field(default=None, description="Round-trip time of the probe")
    details: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class HealthReport(BaseModel):
    """Aggregate health of the API and its dependencies."""

    status: HealthStatus
    app: str
    version: str
    environment: str
    components: list[ComponentHealth]
