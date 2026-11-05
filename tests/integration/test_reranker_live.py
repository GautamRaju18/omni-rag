"""The real BGE rerankers on this machine (weights from <data_dir>/models, never downloaded here).

uv run python -m omnirag.retrieval search "x" --mode dense+bm25+rerank   # downloads once
uv run pytest -m integration tests/integration/test_reranker_live.py
"""

from __future__ import annotations

import pytest

from omnirag.core.config import Settings
from omnirag.retrieval.reranker import MODELS, CrossEncoderReranker, sigmoid

pytestmark = pytest.mark.integration

PANDA = ("The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply "
         "panda, is a bear species endemic to China.")  # fmt: skip


def _reranker(name: str, **kw) -> CrossEncoderReranker:
    models = Settings().data_dir / "models"
    spec = MODELS[name]
    folder = models / f"{spec.repo.replace('/', '--')}-{spec.revision[:12]}"
    if not all((folder / f).exists() for f in spec.files):
        pytest.skip(f"{name} weights not downloaded yet (2.3 GB / 1.1 GB)")
    return CrossEncoderReranker(models, name, **kw)


def test_v2_m3_matches_the_model_card() -> None:
    """Model card (FlagEmbedding, fp16): -5.65234375 and [-8.1875, 5.26171875]."""
    reranker = _reranker("bge-reranker-v2-m3")
    logits = reranker.score("query", ["passage"]) + reranker.score("what is panda?", ["hi", PANDA])
    assert logits == pytest.approx([-5.65234375, -8.1875, 5.26171875], abs=0.02)  # fp32 vs fp16
    assert sigmoid(logits[0]) == pytest.approx(0.003497010252573502, abs=1e-4)


def test_base_ranks_and_falls_back_to_cpu() -> None:
    reranker = _reranker("bge-reranker-base", device="cuda")  # this torch is the CPU build
    assert reranker.device == "cpu"
    logits = reranker.score("Why did cell AUR-C103 fail?", [
        "Milestone M3: pilot line 100-cell batch, due 2026-09-30.",
        "Cell AUR-C103 failed at cycle 412 because of gas evolution at 45 °C.",
    ])  # fmt: skip
    assert logits[1] > logits[0] + 3
