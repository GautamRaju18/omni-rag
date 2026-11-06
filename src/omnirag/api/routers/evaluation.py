"""Evaluation endpoints (Phase 12).

* ``POST /evaluate``           start an evaluation run in the background (one at a time: 409)
* ``GET /evaluate``            runs started here plus measured reports (``docs/benchmarks``)
* ``GET /evaluate/{run_id}``   one run: status, progress lines, and the report when done

A full run (110 questions × 7 experiments, judge on) takes about two hours on the reference
laptop, mostly CPU reranking (E5/E6) and generation; ``limit`` and fewer experiments make a
quick check.
"""

from __future__ import annotations

import asyncio
from typing import Annotated, Any

from fastapi import APIRouter, Path, Query
from pydantic import BaseModel, Field, field_validator

from omnirag.api.dependencies import EvaluationsDep
from omnirag.core.errors import ConflictError, NotFoundError
from omnirag.evaluation.rag_eval import EXPERIMENTS
from omnirag.evaluation.runs import RunConflictError, summary_of

router = APIRouter(prefix="/evaluate", tags=["evaluation"])


class EvaluateRequest(BaseModel):
    experiments: list[str] = Field(default_factory=lambda: ["E3", "E7"], min_length=1)
    limit: int | None = Field(default=None, ge=1, le=1000)  # first N questions
    judge: bool = True

    @field_validator("experiments")
    @classmethod
    def _known(cls, v: list[str]) -> list[str]:
        unknown = sorted(set(v) - set(EXPERIMENTS))
        if unknown:
            raise ValueError(f"unknown experiments {unknown}; choose from {sorted(EXPERIMENTS)}")
        return list(dict.fromkeys(v))


@router.post("", status_code=202, summary="Start an evaluation run (background)")
async def start(request: EvaluateRequest, runs: EvaluationsDep) -> dict[str, Any]:
    try:
        state = await asyncio.to_thread(runs.start, request.experiments, limit=request.limit,
                                        judge=request.judge)  # fmt: skip
    except RunConflictError as exc:
        raise ConflictError(str(exc)) from exc
    return {k: v for k, v in state.items() if k != "report"}


@router.get("", summary="Evaluation runs and measured reports")
async def list_runs(runs: EvaluationsDep) -> list[dict[str, Any]]:
    return await asyncio.to_thread(runs.list)


@router.get("/experiments", summary="The experiment definitions (E1-E7)")
async def experiments() -> dict[str, Any]:
    return EXPERIMENTS


RunId = Annotated[str, Path(pattern=r"^(eval_[0-9a-f]{12}|rag-eval-[0-9-]+)$")]


@router.get("/{run_id}", summary="One run (report summary; per-question with full=true)")
async def get_run(run_id: RunId, runs: EvaluationsDep,
                  full: bool = Query(default=False)) -> dict[str, Any]:  # fmt: skip
    state = await asyncio.to_thread(runs.get, run_id)
    if state is None:
        raise NotFoundError(f"evaluation run {run_id} not found")
    if state.get("report") and not full:
        state = {**state, "report": summary_of(state["report"])}
    return state
