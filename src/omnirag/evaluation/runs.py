"""Evaluation runs started from the API (``POST /evaluate``, Phase 12).

One run at a time, in a background thread of the API process (the same single-process design
as ingestion, ADR-0007): an evaluation loads its own retrievers and, for E5/E6, the CPU
reranker, and two runs would only slow each other down on one GPU. Status and the final report
live in ``<data_dir>/evaluations/<run_id>.json``; per-question records stream to
``<run_id>.partial.jsonl``. Reports measured from the CLI (``docs/benchmarks/rag-eval-*.json``)
are listed read-only next to them.
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from omnirag.evaluation.dataset import EvalDataset

log = structlog.get_logger(module=__name__)


class RunConflictError(RuntimeError):
    """Another evaluation is still running."""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def summary_of(report: dict[str, Any]) -> dict[str, Any]:
    """A report without per-question data (what lists and analytics show)."""
    return {
        **{k: report.get(k) for k in ("created", "dataset", "questions", "generation_model",
                                      "prompt_version", "judge_version", "chunk_config",
                                      "wall_s", "memory", "indexing")},
        "experiments": {name: {k: v for k, v in exp.items() if k != "per_question"}
                        for name, exp in report.get("experiments", {}).items()},
    }  # fmt: skip


class EvaluationRuns:
    def __init__(self, settings: Any, dataset_path: Path | None = None, runner: Any = None) -> None:
        from omnirag.evaluation import rag_eval

        self.settings = settings
        self.dir = settings.data_dir / "evaluations"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.dataset_path = dataset_path or rag_eval.DATASET
        self.runner = runner or rag_eval.run
        self._lock = threading.Lock()
        self._active: str | None = None

    # ---------------------------------------------------------------- state files
    def _path(self, run_id: str) -> Path:
        return self.dir / f"{run_id}.json"

    def _write(self, run_id: str, state: dict[str, Any]) -> None:
        tmp = self._path(run_id).with_suffix(".tmp")
        tmp.write_bytes(json.dumps(state, ensure_ascii=False).encode("utf-8"))
        tmp.replace(self._path(run_id))

    def get(self, run_id: str) -> dict[str, Any] | None:
        path = self._path(run_id)
        if path.exists() and run_id.startswith("eval_"):
            return json.loads(path.read_text(encoding="utf-8"))
        for report in self._benchmark_reports():
            if report["run_id"] == run_id:
                return report
        return None

    def _benchmark_reports(self) -> list[dict[str, Any]]:
        bdir = self.settings.benchmarks_path
        if not bdir or not Path(bdir).is_dir():
            return []
        out = []
        for path in sorted(Path(bdir).glob("rag-eval-*.json"), reverse=True):
            try:
                report = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            out.append({"run_id": path.stem, "status": "completed", "source": "benchmarks",
                        "report": report})  # fmt: skip
        return out

    def list(self) -> list[dict[str, Any]]:
        runs = []
        for path in sorted(self.dir.glob("eval_*.json"), reverse=True):
            state = json.loads(path.read_text(encoding="utf-8"))
            runs.append(
                {
                    k: state.get(k)
                    for k in (
                        "run_id",
                        "status",
                        "created",
                        "finished",
                        "experiments",
                        "limit",
                        "progress",
                        "error",
                    )
                }
            )
        runs += [{"run_id": r["run_id"], "status": "completed", "source": "benchmarks",
                  "created": r["report"].get("created"),
                  "experiments": list(r["report"].get("experiments", {}))}
                 for r in self._benchmark_reports()]  # fmt: skip
        return runs

    def latest_report(self) -> dict[str, Any] | None:
        """The newest completed report (API runs and measured benchmark reports)."""
        candidates = []
        for path in self.dir.glob("eval_*.json"):
            state = json.loads(path.read_text(encoding="utf-8"))
            if state.get("status") == "completed" and state.get("report"):
                candidates.append(state["report"])
        candidates += [r["report"] for r in self._benchmark_reports()]
        return max(candidates, key=lambda r: r.get("created", ""), default=None)

    # ---------------------------------------------------------------- run
    def start(self, experiments: list[str], *, limit: int | None = None,
              judge: bool = True) -> dict[str, Any]:  # fmt: skip
        with self._lock:
            if self._active:
                raise RunConflictError(f"evaluation {self._active} is still running")
            run_id = f"eval_{uuid.uuid4().hex[:12]}"
            self._active = run_id
        state = {"run_id": run_id, "status": "running", "created": _now(), "finished": None,
                 "experiments": experiments, "limit": limit, "judge": judge, "progress": [],
                 "error": None, "report": None}  # fmt: skip
        self._write(run_id, state)
        threading.Thread(target=self._run, args=(run_id, state), daemon=True,
                         name=f"eval-{run_id}").start()  # fmt: skip
        return state

    def _run(self, run_id: str, state: dict[str, Any]) -> None:
        def progress(line: str) -> None:
            state["progress"] = [*state["progress"][-49:], line]
            self._write(run_id, state)

        try:
            dataset = EvalDataset.load(self.dataset_path)
            report = self.runner(self.settings, dataset, state["experiments"],
                                 use_judge=state["judge"], limit=state["limit"],
                                 partial=self.dir / f"{run_id}.partial.jsonl",
                                 progress=progress)  # fmt: skip
            state.update(status="completed", report=report)
        except Exception as exc:  # noqa: BLE001 - recorded on the run, never raised
            log.exception("evaluation_failed", run_id=run_id)
            state.update(status="failed", error=f"{type(exc).__name__}: {exc}"[:500])
        finally:
            state["finished"] = _now()
            self._write(run_id, state)
            with self._lock:
                self._active = None
