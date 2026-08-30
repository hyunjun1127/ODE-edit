"""Fresh Phase-B ctrl-only candidate construction, evaluation and selection."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from project.run_scripts.fixed_z_nonuniqueness.evaluation import next_token_logits, target_prefixes
from project.run_scripts.fixed_z_nonuniqueness.padding import OfficialTokenizerHook, bind_padding

from .contexts import build_constraint_bank, build_native_contexts, compare_realized_z, public_realized_z, realized_z_by_context
from .contracts import BarrierLock, MODEL_SPECS, Method, ScientificBoundary, TechnicalBoundary
from .geometry import build_axes, projector_probe_audit
from .hashing import canonical_hash, file_sha256, tensor_sha256, write_json_once
from .metrics import CandidateScore, reference_barrier, select_ctrl, teacher_kl
from .official import restore_originals, run_official_once
from .phase_a import SecondMoment, _load_hparams, _official_request, _prompt, _request_metrics
from .snapshots import EndpointSnapshot
from .transport import factor_dense


def _load_inputs(dataset: Path, source_root: Path, anchor_seal: Path) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, Any], dict[int, dict[str, Any]]]:
    rows = json.loads(dataset.read_text())
    indexed = {int(row["case_id"]): row for row in rows}
    fresh = json.loads((source_root / "project/run_scripts/barrier_usefulness/config/fresh-case-manifest-v1.json").read_text())
    prior = json.loads((source_root / "project/run_scripts/fixed_z_nonuniqueness/config/case-manifest-v1.json").read_text())
    history = [indexed[int(row["case_id"])] for row in prior["roles"]["history"]]
    cases = [indexed[int(row["case_id"])] for row in fresh["fresh_cases"]]
    seal = json.loads(anchor_seal.read_text())
    anchors = {int(row["edit_case_id"]): row for row in seal["cases"]}
    return cases, {"history": history}, seal, indexed


def _anchor_batch_with_tokenizer(tokenizer: Any, indexed: dict[int, dict[str, Any]], split: dict[str, Any]) -> tuple[list[str], torch.Tensor]:
    texts = []
    labels = []
    for selected in split["selected"]:
        row = indexed[int(selected["case_id"])]
        req = row["requested_rewrite"]
        prefixes, ids = target_prefixes(tokenizer, req["prompt"].format(req["subject"]), req["target_true"]["str"])
        observed = [int(value) for value in ids]
        if observed != selected["token_ids"]:
            raise TechnicalBoundary("sealed anchor token identity mismatch")
        texts.extend(prefixes)
        labels.extend(observed)
    return texts, torch.tensor(labels, dtype=torch.long)


def _load_teacher(path: Path, expected: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    if file_sha256(path) != expected["sha256"]:
        raise TechnicalBoundary("W0 teacher cache SHA mismatch")
    row = torch.load(path, map_location="cpu", weights_only=True)
    if tensor_sha256(row["logits"]) != expected["logits_sha256"] or tensor_sha256(row["labels"]) != expected["labels_sha256"]:
        raise TechnicalBoundary("W0 teacher tensor identity mismatch")
    return row["logits"], row["labels"]


def _history_downstream(model: Any, tokenizer: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    texts = []
    labels = []
    ranges = []
    for row in rows:
        req = row["requested_rewrite"]
        prefixes, ids = target_prefixes(tokenizer, req["prompt"].format(req["subject"]), req["target_true"]["str"])
        begin = len(texts)
        texts.extend(prefixes)
        labels.extend(int(value) for value in ids)
        ranges.append((begin, len(texts)))
    logits = next_token_logits(model, tokenizer, texts, batch_size=16).cpu()
    label_tensor = torch.tensor(labels)
    log_prob = torch.log_softmax(logits, dim=-1)
    token_nll = -log_prob[torch.arange(label_tensor.numel()), label_tensor]
    predictions = logits.argmax(dim=-1)
    return {
        "request_nll": [float(token_nll[begin:end].mean().item()) for begin, end in ranges],
        "request_strict": [bool(torch.equal(predictions[begin:end], label_tensor[begin:end])) for begin, end in ranges],
        "request_count": len(rows), "token_count": int(label_tensor.numel()),
        "logits_sha256": tensor_sha256(logits),
    }


def _observe_ctrl(model: Any, tokenizer: Any, texts: list[str], labels: torch.Tensor, teacher: torch.Tensor, floor: float, lock: BarrierLock) -> dict[str, Any]:
    candidate = next_token_logits(model, tokenizer, texts, batch_size=16).cpu()
    return {
        "barrier": reference_barrier(teacher, candidate, labels, numerical_floor=floor),
        "teacher_kl": teacher_kl(teacher, candidate, lock.cvar_alpha),
        "candidate_logits_sha256": tensor_sha256(candidate),
    }


def _save_package(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    if path.exists() or path.is_symlink():
        raise TechnicalBoundary("candidate package overwrite attempted")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    os.chmod(path, 0o600)
    return {"path": str(path), "sha256": file_sha256(path), "bytes": path.stat().st_size}


def run_case(
    *, model: Any, tokenizer: Any, method: Method, hparams: Any, row: dict[str, Any], history: list[dict[str, Any]],
    anchor: dict[str, Any], indexed: dict[int, dict[str, Any]], moment: SecondMoment, lock: BarrierLock,
    model_alias: str, case_root: Path,
) -> dict[str, Any]:
    request = _official_request(row)
    hook = OfficialTokenizerHook(tokenizer, [])
    tokenizer.padding_side = "right"
    captured = run_official_once(method=method, model=model, tokenizer=hook, request=request, hparams=hparams)
    endpoint = captured.endpoint
    if endpoint.direct_z_compute_count != 1 or endpoint.direct_z_recompute_count != 0:
        raise ScientificBoundary("fresh direct-z count failed")
    contexts = build_native_contexts(
        tokenizer=tokenizer, request=captured.normalized_request,
        context_templates=captured.compute_z_context_templates, fact_token=hparams.fact_token,
    )
    names = sorted(endpoint.originals)
    snapshot = EndpointSnapshot.capture(model, names)
    input_module = hparams.rewrite_module_tmp.format(hparams.layers[-1])
    activation_module = hparams.layer_module_tmp.format(hparams.layers[-1])
    constraints, constraint_receipt = build_constraint_bank(
        model=model, tokenizer=tokenizer, request=captured.normalized_request, contexts=contexts,
        history_templates=[item["requested_rewrite"]["prompt"] for item in history],
        history_subjects=[item["requested_rewrite"]["subject"] for item in history], input_module=input_module,
    )
    official_realized = realized_z_by_context(model, tokenizer, contexts, activation_module, endpoint.z)
    official_metrics = _request_metrics(model, tokenizer, row)
    ctrl_texts, ctrl_labels = _anchor_batch_with_tokenizer(tokenizer, indexed, anchor["ctrl"])
    teacher_path = Path(anchor["ctrl"]["teacher_cache"]["path"])
    teacher_logits, sealed_labels = _load_teacher(teacher_path, anchor["ctrl"]["teacher_cache"])
    if not torch.equal(ctrl_labels, sealed_labels):
        raise TechnicalBoundary("ctrl label binding differs")
    official_ctrl = _observe_ctrl(model, tokenizer, ctrl_texts, ctrl_labels, teacher_logits, anchor["margin_numerical_floor"], lock)
    official_history = _history_downstream(model, tokenizer, history)
    base_delta = endpoint.deltas[endpoint.last_weight_name].to(next(model.parameters()).device)
    base_action = moment.matrix_action(base_delta)
    projector = endpoint.projector.to(base_delta.device) if endpoint.projector is not None else None
    projector_audit = None if projector is None else projector_probe_audit(
        projector, constraints.to(projector.device),
        seed=int(canonical_hash(["phase-b-projector", model_alias, method.value, row["case_id"]])[:16], 16),
    )
    axes = build_axes(
        base_delta=base_delta, constraints=constraints.to(base_delta.device), matvec=moment.matvec,
        lock=lock, model=model_alias, method=method.value, case_id=str(row["case_id"]),
        count=lock.phase_b_axis_count, projector=projector, base_action=base_action,
    )
    transport = {
        name: factor_dense(delta.to(base_delta.device), tolerance=lock.fp32_relative_tolerance / 16.0)
        for name, delta in endpoint.deltas.items()
    }
    package = {
        "schema": "odeedit.s06.barrier-usefulness.fixed-candidate-bank.v1",
        "model": model_alias, "method": method.value, "case_id": int(row["case_id"]),
        "normalized_request": captured.normalized_request,
        "context_templates": captured.compute_z_context_templates,
        "z": endpoint.z.detach().cpu(), "z_sha256": tensor_sha256(endpoint.z),
        "last_weight_name": endpoint.last_weight_name, "official_delta_factors": transport,
        "axes": [{
            "axis_id": axis.axis_id, "seed": axis.seed, "output": axis.output.detach().cpu(),
            "input": axis.input.detach().cpu(), "gamma": axis.gamma, "base_action": axis.base_action,
            "unit_action": axis.unit_action, "cross": axis.cross, "total_action": axis.total_action,
            "equality": axis.equality,
        } for axis in axes],
        "A_star_sha256": constraint_receipt["A_star_sha256"],
        "official_snapshot_root": snapshot.root,
    }
    package_id = _save_package(case_root / "fixed-candidate-bank.pt", package)
    candidates = []
    strict_official = {name: value["strict"] for name, value in official_metrics.items()}
    ordered = [(axis, sign) for axis in axes for sign in (-1, 1)]
    replay = []
    for ordinal, (axis, sign) in enumerate(ordered):
        correction = axis.correction(sign)
        snapshot.apply_correction(model, endpoint.last_weight_name, correction)
        ctrl = _observe_ctrl(model, tokenizer, ctrl_texts, ctrl_labels, teacher_logits, anchor["margin_numerical_floor"], lock)
        realized = realized_z_by_context(model, tokenizer, contexts, activation_module, endpoint.z)
        realized_comparison = compare_realized_z(realized, official_realized)
        metrics = _request_metrics(model, tokenizer, row)
        history_downstream = _history_downstream(model, tokenizer, history)
        strict_equal = {name: value["strict"] for name, value in metrics.items()} == strict_official
        correction_action = axis.gamma * axis.gamma * axis.unit_action
        cross = sign * axis.gamma * axis.cross
        total = axis.base_action + 2.0 * cross + correction_action
        equality = float((correction @ constraints.to(correction.device)).norm().div(correction.norm() * constraints.to(correction.device).norm()).item())
        valid = (
            equality <= lock.fp32_relative_tolerance and abs(total / axis.base_action - 1.05) <= lock.fp32_relative_tolerance
            and abs(cross) / math.sqrt(axis.base_action * correction_action) <= lock.fp32_relative_tolerance
            and strict_equal and realized_comparison["maximum_relative"] <= lock.fp32_relative_tolerance
        )
        restore = snapshot.restore_after_candidate(model)
        candidate_id = f"{axis.axis_id}-{'plus' if sign > 0 else 'minus'}"
        candidates.append({
            "candidate_id": candidate_id, "axis_id": axis.axis_id, "sign": sign, "seed": axis.seed,
            "valid": valid, "snapshot_restore": restore, "equality_NA_star_relative": equality,
            "actual_action": {"Q_delta_off": axis.base_action, "Q_N": correction_action, "cross": cross, "Q_total": total, "ratio": total / axis.base_action},
            "realized_z": realized_comparison, "strict_equal_official": strict_equal,
            "ctrl": ctrl, "metrics": metrics, "history_downstream": history_downstream,
        })
        if ordinal in {0, 8, 16, 24, len(ordered) - 1}:
            snapshot.copy_to_model(model)
            replay.append({"after_candidate_ordinal": ordinal, "ctrl": _observe_ctrl(model, tokenizer, ctrl_texts, ctrl_labels, teacher_logits, anchor["margin_numerical_floor"], lock), "snapshot_exact": True})
    if any(not candidate["valid"] for candidate in candidates):
        raise ScientificBoundary("fresh ctrl candidate hard validity failed")
    scores = [CandidateScore(
        candidate["candidate_id"], float(candidate["ctrl"]["barrier"]["score"]),
        bool(candidate["ctrl"]["barrier"]["finite"]), float(candidate["ctrl"]["teacher_kl"]["cvar_0_875"]),
    ) for candidate in candidates]
    selectors = select_ctrl(scores, lock.action_representative_candidate_id)
    if not restore_originals(model, endpoint.originals):
        raise ScientificBoundary("fresh ctrl terminal W0 restore failed")
    return {
        "schema": "odeedit.s06.barrier-usefulness.ctrl-case.v1", "case_id": int(row["case_id"]),
        "candidate_bank": package_id, "candidate_bank_identity": canonical_hash({
            "package_sha256": package_id["sha256"], "z_sha256": package["z_sha256"],
            "candidate_ids": [candidate["candidate_id"] for candidate in candidates],
        }),
        "constraints": constraint_receipt, "native_context_count": len(contexts),
        "projector_audit": projector_audit, "official": {
            "ctrl": official_ctrl, "metrics": official_metrics,
            "realized_z": public_realized_z(official_realized), "history_downstream": official_history,
            "snapshot_root": snapshot.root,
        },
        "candidates": candidates, "selectors": selectors, "interleaved_official": replay,
        "direct_z_compute_count": 1, "direct_z_recompute_count": 0,
        "valid_candidate_count": 32, "candidate_denominator": 32,
        "w0_restore": True,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    lock = BarrierLock()
    spec = MODEL_SPECS[args.model]
    cases, roles, anchor_seal, indexed = _load_inputs(args.dataset, args.source_root, args.anchor_seal)
    if anchor_seal["model"] != args.model or anchor_seal["edited_candidate_access_count"] != 0:
        raise TechnicalBoundary("anchor seal model/leakage mismatch")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(
        spec.model_path, local_files_only=True, torch_dtype=torch.float32, low_cpu_mem_usage=True,
        device_map={"": 0}, trust_remote_code=False,
    )
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise TechnicalBoundary("Phase B ctrl FULL_FP32 failed")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(spec.model_path, local_files_only=True, use_fast=True, trust_remote_code=False)
    padding = bind_padding(tokenizer, model, padding_side="right")
    hparams_rel = spec.alpha_hparams if args.method == Method.ALPHAEDIT.value else spec.memit_hparams
    hparams = _load_hparams(Method(args.method), args.source_root / hparams_rel)
    stats_path = spec.statistics_root / spec.statistics_model_dir / "wikipedia_stats" / "model.layers.8.mlp.down_proj_float32_mom2_100000.npz"
    moment = SecondMoment(stats_path, next(model.parameters()).device, scale=float(hparams.mom2_update_weight), additive=float(getattr(hparams, "L2", 0.0)))
    anchor_by_case = {int(row["edit_case_id"]): {**row, "margin_numerical_floor": anchor_seal["margin_numerical_floor"]} for row in anchor_seal["cases"]}
    results = []
    for row in cases:
        case_root = args.raw_root / f"case-{row['case_id']}"
        if case_root.exists() or case_root.is_symlink():
            raise TechnicalBoundary("ctrl case root not create-once")
        results.append(run_case(
            model=model, tokenizer=tokenizer, method=Method(args.method), hparams=hparams,
            row=row, history=roles["history"], anchor=anchor_by_case[int(row["case_id"])], indexed=indexed,
            moment=moment, lock=lock, model_alias=args.model, case_root=case_root,
        ))
    candidate_bank_hash = canonical_hash([row["candidate_bank_identity"] for row in results])
    selected = {str(row["case_id"]): row["selectors"] for row in results}
    return {
        "schema": "odeedit.s06.barrier-usefulness.ctrl-cell.v1", "status": "CTRL_TERMINAL_VALID",
        "model": args.model, "method": args.method, "case_ids": [int(row["case_id"]) for row in cases],
        "cases": results, "candidate_bank_hash": candidate_bank_hash,
        "selected_candidate_ids": selected, "selected_candidate_ids_hash": canonical_hash(selected),
        "anchor_seal": {"path": str(args.anchor_seal), "sha256": file_sha256(args.anchor_seal), "root": anchor_seal["root_digest"]},
        "padding_binding": padding, "direct_z_compute_count": 8, "direct_z_recompute_count": 0,
        "edited_gate_access_count": 0, "replacement_count": 0, "final_audit_open_count": 0,
        "full_fp32": True, "wall_seconds": time.monotonic() - started,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        "source": {
            "head": os.popen(f"git -C {args.source_root} rev-parse HEAD").read().strip(),
            "tree": os.popen(f"git -C {args.source_root} rev-parse HEAD^{{tree}}").read().strip(),
        },
        "scientific_promotion": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--method", choices=[item.value for item in Method], required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--anchor-seal", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json_once(args.output, run(args))


if __name__ == "__main__":
    main()
