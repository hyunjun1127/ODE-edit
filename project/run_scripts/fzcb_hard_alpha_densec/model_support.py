"""Pinned model, request, tokenizer, and Official Alpha source adapters."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from project.run_scripts.fixed_z_nonuniqueness.padding import OfficialTokenizerHook, bind_padding

from .contracts import EngineeringBoundary


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    snapshot: Path
    revision: str
    statistics_alias: str
    projector: Path
    hparams_relative: Path
    endpoint_tolerance: float


MODEL_SPECS = {
    "llama3-8b-inst": ModelSpec(
        alias="llama3-8b-inst",
        snapshot=Path("/mnt/raid5/janghj/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"),
        revision="8afb486c1db24fe5011ec46dfbe5b5dccdb575c2",
        statistics_alias="Meta-Llama-3-8B-Instruct",
        projector=Path("/mnt/raid5/janghj/EasyEdit/examples/null_space_project_Meta-Llama-3-8B-Instruct.pt"),
        hparams_relative=Path("hparams/AlphaEdit/llama3-8b.yaml"),
        endpoint_tolerance=0.013695280019564686,
    ),
    "qwen2.5-7b-inst": ModelSpec(
        alias="qwen2.5-7b-inst",
        snapshot=Path("/mnt/raid5/janghj/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"),
        revision="a09a35458c702b33eeacc393d103063234e8bc28",
        statistics_alias="Qwen2.5-7B-Instruct",
        projector=Path("/mnt/raid5/janghj/EasyEdit/examples/null_space_project_Qwen2.5-7B-Instruct.pt"),
        hparams_relative=Path("hparams/AlphaEdit/qwen2.5-7b.yaml"),
        endpoint_tolerance=0.01810729797516364,
    ),
}


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def tensor_sha(value: torch.Tensor) -> str:
    tensor = value.detach().to("cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def load_case_rows(dataset: Path, manifest: Path, count: int) -> list[dict[str, Any]]:
    rows = json.loads(dataset.read_text())
    by_id = {int(row["case_id"]): row for row in rows}
    seal = json.loads(manifest.read_text())
    selected = []
    for item in seal["cases"][:count]:
        case_id = int(item["case_id"])
        if case_id not in by_id or canonical_hash(by_id[case_id]) != item["row_sha256"]:
            raise EngineeringBoundary(f"request row identity mismatch: {case_id}")
        selected.append(by_id[case_id])
    if len(selected) != count:
        raise EngineeringBoundary("request denominator mismatch")
    return selected


def official_requests(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    requests = []
    for row in rows:
        request = row["requested_rewrite"]
        target_new = request["target_new"]["str"]
        if not target_new.startswith(" "):
            target_new = " " + target_new
        prompt = request["prompt"]
        if "{}" not in prompt:
            if request["subject"] not in prompt:
                raise EngineeringBoundary("subject absent from rewrite prompt")
            prompt = prompt.replace(request["subject"], "{}")
        requests.append(
            {
                "case_id": int(row["case_id"]),
                "prompt": prompt,
                "subject": request["subject"],
                "target_new": target_new,
                "target_true": request["target_true"]["str"],
            }
        )
    return requests


def load_model_and_tokenizer(spec: ModelSpec) -> tuple[Any, Any, dict[str, Any]]:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(
        spec.snapshot,
        local_files_only=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map={"": 0},
        trust_remote_code=False,
        attn_implementation="eager",
    )
    wrong = [(name, str(value.dtype)) for name, value in model.named_parameters() if value.dtype != torch.float32]
    if wrong or getattr(model, "is_quantized", False):
        raise EngineeringBoundary(f"FULL_FP32 model closure failed: {wrong[:4]}")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        spec.snapshot,
        local_files_only=True,
        use_fast=True,
        trust_remote_code=False,
    )
    padding = bind_padding(tokenizer, model, padding_side="right")
    return model, tokenizer, {
        "alias": spec.alias,
        "snapshot": str(spec.snapshot),
        "revision": spec.revision,
        "config_name_or_path": str(model.config._name_or_path),
        "model_dtype": "torch.float32",
        "quantized": False,
        "attention_implementation": "eager",
        "tf32": False,
        "padding": padding,
    }


def load_hparams(spec: ModelSpec, official_root: Path) -> Any:
    from easyeditor.models.alphaedit.AlphaEdit_hparams import AlphaEditHyperParams

    path = official_root / spec.hparams_relative
    hparams = AlphaEditHyperParams.from_hparams(str(path))
    # Deployment-only absolute bindings; all numerical/scientific values stay
    # byte-semantically identical to the pinned Official YAML.
    hparams.model_name = spec.statistics_alias
    hparams.stats_dir = "/mnt/raid5/janghj/EasyEdit/examples/data/stats"
    hparams.P_loc = str(spec.projector)
    hparams.device = 0
    return hparams


@contextmanager
def official_model_name(model: Any, name: str) -> Iterator[None]:
    before = model.config._name_or_path
    model.config._name_or_path = name
    try:
        yield
    finally:
        model.config._name_or_path = before


def selected_weights(model: Any, hparams: Any) -> dict[str, torch.Tensor]:
    from easyeditor.util import nethook

    return {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }


def selected_weight_identity(weights: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(weights):
        digest.update(name.encode())
        digest.update(tensor_sha(weights[name]).encode())
    return digest.hexdigest()


def restore_selected(weights: dict[str, torch.Tensor], entry: dict[str, torch.Tensor]) -> bool:
    with torch.no_grad():
        for name, value in entry.items():
            weights[name].copy_(value)
    return all(torch.equal(weights[name], value) for name, value in entry.items())


def tokenizer_hook(tokenizer: Any) -> OfficialTokenizerHook:
    return OfficialTokenizerHook(tokenizer, [])
