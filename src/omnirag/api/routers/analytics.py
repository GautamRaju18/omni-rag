"""``GET /analytics`` (Phase 13): answer quality, performance and failures from stored traces,
plus the newest evaluation report's summary. Read-only; computed on request."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Query

from omnirag.api.dependencies import ChatDep, EvaluationsDep
from omnirag.observability.analytics import evaluation_summary, turn_analytics

router = APIRouter(tags=["analytics"])


@router.get("/analytics", summary="Answer quality, latency, failures, latest evaluation")
async def analytics(
    chat: ChatDep, runs: EvaluationsDep, limit: int = Query(default=1000, ge=1, le=10_000)
) -> dict[str, Any]:
    turns = await asyncio.to_thread(chat.store.recent_turns, limit)
    report = await asyncio.to_thread(runs.latest_report)
    return {"chat": turn_analytics(turns), "evaluation": evaluation_summary(report)}
