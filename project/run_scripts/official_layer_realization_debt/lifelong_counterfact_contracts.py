"""Immutable bindings for the lifelong CounterFact metric correction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .lifelong_finalw_contracts import AMENDED_CHECKPOINTS


INSTRUCTION_ID = "ODEEDIT-S06-OFFICIAL-LAYER-DEBT-LIFELONG-COUNTERFACT-METRICS-V6"
SCHEMA = "odeedit.s06.layer-debt.lifelong-counterfact-metrics.v6"

FINALW_RESULT_ROOT = Path(
    "/data/janghj/ODE-edit/local/results/"
    "official-layer-realization-debt-lifelong-finalw-full10k-v1/"
    "campaign-20260903-tech-r3"
)
FINALW_INPUT_LOCK = Path(
    "/data/janghj/ODE-edit/local/state/"
    "official-layer-realization-debt-lifelong-finalw-full10k-v1/"
    "campaign-20260903-tech-r2/availability-amendment-and-input-lock.json"
)
FINALW_INPUT_LOCK_SHA256 = (
    "3684dd72e3a38e8b10544b6071e592a10eafdec881b65f0c87656eb2c662dcc7"
)

V4_RELATIVE = Path(
    "experiment-reports/servers/server4/"
    "official-layer-realization-debt-lifelong-b100x100-2026-09-03-v4-r2"
)
V5_RELATIVE = Path(
    "experiment-reports/servers/server4/"
    "official-layer-realization-debt-lifelong-b100x100-2026-09-03-v5"
)
V4_REPORT_SHA256 = "225fbe8cc7fea44a2e9e00b01a1e3648686e439cd6618a150165b3f33cc95c68"
V4_MANIFEST_SHA256 = "848dd78ea44d26105eb2f75344aa252413d249fddf2b03dd104e3a78f9cea0a9"
V4_RECEIPT_SHA256 = "bffff6260cae5f4679f0c6ac7b3b5020ac2c286955caebb7e1a4683454561b04"
V5_REPORT_SHA256 = "7efe0fb9df78cd00c9abc3abce81db411ce099a493cd1812582ff4cb89084de3"
V5_MANIFEST_SHA256 = "5599b8a351e541f214d9aae8a2b645009edb0989001dfa37b1c36e2007ac7070"
V5_RECEIPT_SHA256 = "a5b7355ccbf819095cf4aa8c4a199f11ab394b41a9bc9793a1c2d8a68b44682d"

CANONICAL_COUNTERFACT_ROOT = Path("/data/janghj/knowledge-editing-regularization")
CANONICAL_COUNTERFACT_EVALUATOR = CANONICAL_COUNTERFACT_ROOT / "experiments/py/eval_utils_counterfact.py"
CANONICAL_COUNTERFACT_SUMMARIZER = CANONICAL_COUNTERFACT_ROOT / "experiments/summarize.py"
CANONICAL_COUNTERFACT_EVALUATOR_SHA256 = (
    "f2de63e6cc68871cb042f614193294a433c4b76622dd410de515df8cb73416b8"
)
CANONICAL_COUNTERFACT_SUMMARIZER_SHA256 = (
    "9791812222b151952c16cacf28daf3e8be9645e37321e0aa84be6ab5e24a66f8"
)
PINNED_EASYEDIT_EVALUATOR_SHA256 = (
    "229875251b427c90fa4ab100da176dd46dc9fd7d5b030f4813b1831880c03713"
)
P1_EVALUATOR_SHA256 = "e22c4e62f02529162c8dcc0f65f177c1ec7330996219e07262ec93d8d5589777"
FINALW_EVALUATOR_SHA256 = "034f06f1dc08af2d2ecd17f52a9985880d2e64897cd39caa9244148bfc74a403"

ARM_ORDER = ("LM", "LA", "QM", "QA")
ARM_KEY = {
    ("llama3-8b-inst", "memit"): "LM",
    ("llama3-8b-inst", "alphaedit"): "LA",
    ("qwen2.5-7b-inst", "memit"): "QM",
    ("qwen2.5-7b-inst", "alphaedit"): "QA",
}
ARM_LABEL = {
    "LM": "Llama-3-8B-Instruct / Official MEMIT",
    "LA": "Llama-3-8B-Instruct / Official AlphaEdit",
    "QM": "Qwen2.5-7B-Instruct / Official MEMIT",
    "QA": "Qwen2.5-7B-Instruct / Official AlphaEdit",
}


class CounterFactMetricBoundary(RuntimeError):
    """Fail-close boundary for metric correction and evaluation-only backfill."""


@dataclass(frozen=True, slots=True)
class CounterFactMetricLock:
    checkpoints: tuple[int, ...] = AMENDED_CHECKPOINTS
    rewrite_prompts_per_request: int = 1
    rephrase_prompts_per_request: int = 2
    locality_prompts_per_request: int = 10
    strict_comparison: str = "LESS_THAN_TIES_FAIL"
    backfill_categories: tuple[str, ...] = ("locality_target_new",)
    evaluator_batch_size: int = 16
    edit_replay_count: int = 0
    compute_z_count: int = 0
    writer_count: int = 0
    key_count: int = 0
    solve_count: int = 0
    cache_history_mutation_count: int = 0
    backward_count: int = 0
    gradient_count: int = 0
    model_update_count: int = 0
    imputation_count: int = 0
    scientific_promotion: bool = False

    def payload(self) -> dict[str, Any]:
        return asdict(self)


__all__ = [
    "ARM_KEY",
    "ARM_LABEL",
    "ARM_ORDER",
    "CANONICAL_COUNTERFACT_EVALUATOR",
    "CANONICAL_COUNTERFACT_EVALUATOR_SHA256",
    "CANONICAL_COUNTERFACT_ROOT",
    "CANONICAL_COUNTERFACT_SUMMARIZER",
    "CANONICAL_COUNTERFACT_SUMMARIZER_SHA256",
    "CounterFactMetricBoundary",
    "CounterFactMetricLock",
    "FINALW_EVALUATOR_SHA256",
    "FINALW_INPUT_LOCK",
    "FINALW_INPUT_LOCK_SHA256",
    "FINALW_RESULT_ROOT",
    "INSTRUCTION_ID",
    "P1_EVALUATOR_SHA256",
    "PINNED_EASYEDIT_EVALUATOR_SHA256",
    "SCHEMA",
    "V4_MANIFEST_SHA256",
    "V4_RECEIPT_SHA256",
    "V4_RELATIVE",
    "V4_REPORT_SHA256",
    "V5_MANIFEST_SHA256",
    "V5_RECEIPT_SHA256",
    "V5_RELATIVE",
    "V5_REPORT_SHA256",
]
