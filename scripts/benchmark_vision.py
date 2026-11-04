"""Benchmark vision models on the sample images, against facts we know are in them.

For each model we measure:
* **facts read**: ground-truth strings found in the description (whitespace-insensitive);
* **invented values**: decimal numbers in the description that are not in the chart;
* cold-load time, per-image time, output tokens, and VRAM from Ollama's ``/api/ps``.

Descriptions are generated without document hints, so this measures what the model actually
sees. Results are printed as a Markdown table; record them with date and versions.

    uv run python scripts/benchmark_vision.py qwen3-vl:4b-instruct qwen3-vl:2b-instruct gemma3:4b
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import httpx
from make_office_samples import RETENTION, bar_chart_png, pouch_png

from omnirag.ingestion.vision import OllamaVision

SAMPLES = Path(__file__).resolve().parents[1] / "data" / "samples"
BASE_URL = "http://localhost:11434"

CHART_VALUES = {f"{v:.1f}" for v in RETENTION.values()}
CASES = {
    "bar_chart": (bar_chart_png(), [*sorted(CHART_VALUES), *(f"C{n}" for n in range(101, 109))]),
    "pouch_drawing": (pouch_png(), ["S1-P3"]),
    "whiteboard_photo": (
        (SAMPLES / "whiteboard_2026-03-20.jpg").read_bytes(),
        ["E-7", "QUALIFICATION PLAN", "5%", "100 cycles", "100 cells", "M3", "Marco"],
    ),
}


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def _found(fact: str, text: str) -> bool:
    if _norm(fact) in _norm(text):
        return True
    if re.fullmatch(r"\d+\.0", fact):  # "88.0" may be written "88"
        return re.search(rf"(?<![\d.]){fact[:-2]}(?![\d.])", text) is not None
    return False


def score(text: str, facts: list[str]) -> tuple[int, list[str]]:
    missing = [f for f in facts if not _found(f, text)]
    return len(facts) - len(missing), missing


def pairs(text: str) -> int:
    """Chart bars whose label and value appear together (``C105 ... 89.6`` within 20 chars).

    Retrieval needs the pairing: a list of eight values without labels can't answer
    "what is C105's retention?".
    """
    flat = re.sub(r"\s+", " ", text)
    return sum(
        re.search(rf"C{cell[1:]}\D{{0,20}}?{value:.1f}", flat) is not None
        or re.search(rf"C{cell[1:]}\D{{0,20}}?{value:g}(?![\d.])", flat) is not None
        for cell, value in RETENTION.items()
    )


def invented(text: str) -> list[str]:
    decimals = re.findall(r"\b\d{2,3}\.\d\b", text)
    return sorted({d for d in decimals if d not in CHART_VALUES})


def vram_gb(model: str) -> float | None:
    for m in httpx.get(f"{BASE_URL}/api/ps").json().get("models", []):
        if m["name"] == model:
            return round(m["size_vram"] / 1e9, 2)
    return None


def unload_all() -> None:
    """Start each model from an empty GPU so cold-load numbers are comparable."""
    for m in httpx.get(f"{BASE_URL}/api/ps").json().get("models", []):
        httpx.post(f"{BASE_URL}/api/generate", json={"model": m["name"], "keep_alive": 0})


def main(models: list[str]) -> None:
    rows = []
    for model in models:
        unload_all()
        vision = OllamaVision(BASE_URL, model, keep_alive="5m")
        first = True
        for case, (image, facts) in CASES.items():
            runs = [vision.describe(image) for _ in range(2)]  # run 2 = warm, same output
            d = runs[0]
            n, missing = score(d.text, facts)
            bad = invented(d.text) if case == "bar_chart" else []
            paired = f"{pairs(d.text)}/8" if case == "bar_chart" else ""
            cold = d.load_ms if first else None
            same = runs[0].text == runs[1].text
            rows.append((model, case, f"{n}/{len(facts)}", paired, missing, bad, cold,
                         d.duration_ms, runs[1].duration_ms, d.output_tokens, same,
                         vram_gb(model)))  # fmt: skip
            print(f"\n### {model} / {case}\n{d.text}\n", flush=True)
            first = False
        vision.unload()
    print(
        "| Model | Image | Facts | Label-value pairs | Missing | Invented values "
        "| Cold load ms | Run 1 ms | Run 2 ms | Out tokens | Same output | VRAM GB |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print("| " + " | ".join("" if v is None else str(v) for v in r) + " |")


if __name__ == "__main__":
    main(sys.argv[1:])
