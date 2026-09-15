"""Read-only CPU reduction of a completed, exactly bound SL-ZFlow chain.

No model, evaluator, runtime or CUDA module is imported.  The optional tensor
verification is a lazy import of the CPU durable loader, never a continuation.
Public outputs contain aggregate scalars/identities only.  Prompt/item rows are
kept in memory or written to a separate explicitly requested private directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
from typing import Any


TAGS = {"RS": 1, "PS": 2, "NS": 10}
WEIGHT = "model.layers.4.mlp.down_proj.weight"
FORMAT = "SL_ZFLOW_ANALYSIS_V1"
CONTRACT_REL = "plans/global/2026-09-16-single-layer-zflow-contract-v1.json"
CONTRACT_SHA = "597d02230a0ad804387aca57948c32b9eb6b27ee35908d70afcdd1fbae6064da"


class AnalysisIntegrityError(ValueError):
    """Missing, corrupt or mismatched evidence is not an observed zero score."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisIntegrityError(message)


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def sha256(path: Path) -> str:
    before = path.stat()
    require(path.is_file() and not path.is_symlink(), f"not regular: {path}")
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    after = path.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
            f"changed while read: {path}")
    return h.hexdigest()


def member(path: Path, expected: dict | None = None) -> dict:
    result = {"path": str(path.absolute()), "bytes": path.stat().st_size,
              "sha256": sha256(path)}
    if expected is not None:
        require(result["bytes"] == expected["bytes"] and result["sha256"] == expected["sha256"],
                f"member SHA/size mismatch: {path}")
    return result


def read_json(path: Path) -> Any:
    require(path.is_file() and not path.is_symlink(), f"missing/nonregular JSON: {path}")
    def invalid(value):
        raise AnalysisIntegrityError(f"nonfinite JSON {value}: {path}")
    def unique(pairs):
        out = {}
        for key, value in pairs:
            require(key not in out, f"duplicate JSON key {key}: {path}")
            out[key] = value
        return out
    return json.loads(path.read_bytes(), parse_constant=invalid, object_pairs_hook=unique)


def child(root: Path, relative: str) -> Path:
    rel = Path(relative)
    require(not rel.is_absolute() and ".." not in rel.parts, "unsafe relative artifact path")
    out = root / rel
    require(out.resolve().is_relative_to(root.resolve()), "artifact path escape")
    for path in (out, *out.parents):
        require(not path.is_symlink(), f"artifact symlink: {path}")
        if path == root:
            break
    return out


def finite(value: Any, name: str, *, nonnegative: bool = False) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value),
            f"nonfinite/non-numeric {name}")
    require(not nonnegative or value >= 0, f"negative {name}")
    return float(value)


def integer(value: Any, name: str, *, minimum: int = 0) -> int:
    require(type(value) is int and value >= minimum, f"invalid integer {name}")
    return value


def close(a: Any, b: Any, name: str) -> None:
    require(math.isclose(finite(a, name), finite(b, name), rel_tol=1e-12, abs_tol=1e-12),
            f"arithmetic mismatch: {name}")


def quantiles(values: list[float]) -> dict:
    """Linear order-statistic interpolation, equivalent to default NumPy quantile."""
    require(bool(values), "empty metric population")
    values = sorted(finite(x, "quantile") for x in values)
    def q(p):
        position = (len(values) - 1) * p
        lo = math.floor(position); hi = math.ceil(position)
        return values[lo] + (values[hi] - values[lo]) * (position - lo)
    return {"n": len(values), "mean": statistics.fmean(values), "min": values[0],
            "p05": q(.05), "p25": q(.25), "median": q(.5), "p75": q(.75),
            "p90": q(.9), "p95": q(.95), "p99": q(.99), "max": values[-1]}


def expected_items(records: list[dict], tag: str) -> list[tuple]:
    out = []
    for record in records:
        request = record["requested_rewrite"]
        if tag == "RS":
            prompts = [request["prompt"].format(request["subject"])]
        else:
            prompts = record["paraphrase_prompts" if tag == "PS" else "neighborhood_prompts"]
        require(len(prompts) == TAGS[tag], "canonical prompt inventory mismatch")
        for index, prompt in enumerate(prompts):
            identity = digest([record["case_id"], index, prompt,
                               request["target_new"]["str"], request["target_true"]["str"]])
            out.append((record["case_id"], index, identity))
    return out


def reduce_observation(raw: dict, records: list[dict], *, binding: dict | None = None) -> tuple[list[dict], dict]:
    """Recompute native NLL-pair preference, TF strict/token and NLL tails."""
    if binding is not None:
        require(raw.get("state_binding") == binding, "observation state binding mismatch")
        require(raw.get("request_order") == binding["request_order"], "observation order root mismatch")
        require(raw.get("before_after_exact") is True and raw.get("evaluator_controller_influence") == 0,
                "observation nonmutation evidence missing")
    else:
        require(raw.get("requests") == len(records), "historical observation request count mismatch")
    require(set(raw["metrics"]) == set(TAGS), "metric inventory mismatch")
    summaries, items = [], {}
    for tag, multiplicity in TAGS.items():
        metric = raw["metrics"][tag]; rows = metric["rows"]
        expected = expected_items(records, tag)
        require(len(rows) == len(records) * multiplicity == metric["denominator"], "metric denominator mismatch")
        require([(r["case_id"], r["prompt_index"], r["identity"]) for r in rows] == expected,
                "exact case/prompt/target identity or order mismatch")
        require(len({r["identity"] for r in rows}) == len(rows), "duplicate prompt identity")
        for row in rows:
            new = finite(row["new_nll"], "new NLL", nonnegative=True)
            true = finite(row["true_nll"], "true NLL", nonnegative=True)
            success = true < new if tag == "NS" else new < true
            require(type(row["success"]) is bool and row["success"] == success, "NLL success mismatch")
            # The native reducer stores true-new even for NS; do not mistake
            # that raw field for NS's success-oriented margin.
            close(row["margin"], true - new, "native raw margin")
            for target in ("new", "true"):
                count = integer(row[target + "_token_count"], "token count", minimum=1)
                correct = integer(row[target + "_token_correct"], "correct token count")
                require(correct <= count and type(row[target + "_strict"]) is bool and
                        row[target + "_strict"] == (correct == count), "TF strict/token mismatch")
        numerator = sum(r["success"] for r in rows)
        require(metric["numerator"] == numerator, "metric numerator mismatch")
        close(metric["rate"], numerator / len(rows), "metric rate")
        if "bit_order_sha256" in metric:
            require(metric["bit_order_sha256"] == digest([(r["identity"], r["success"]) for r in rows]),
                    "success bit-order digest mismatch")
        summary = {"metric": tag, "numerator": numerator, "denominator": len(rows),
                   "percent": 100 * numerator / len(rows), "ties": sum(r["new_nll"] == r["true_nll"] for r in rows),
                   "identity_order_sha256": digest(expected), "request_count": len(records),
                   "all_prompts_preference_success_num": sum(all(r["success"] for r in rows[i:i+multiplicity])
                                                              for i in range(0, len(rows), multiplicity)),
                   "all_prompts_preference_success_den": len(records)}
        for target in ("new", "true"):
            summary.update({target + "_tf_strict_num": sum(r[target + "_strict"] for r in rows),
                            target + "_tf_strict_den": len(rows),
                            target + "_token_correct": sum(r[target + "_token_correct"] for r in rows),
                            target + "_token_den": sum(r[target + "_token_count"] for r in rows)})
            summary.update({target + "_nll_" + key: value for key, value in
                            quantiles([r[target + "_nll"] for r in rows]).items()})
        margins = [(r["new_nll"] - r["true_nll"]) if tag == "NS" else (r["true_nll"] - r["new_nll"]) for r in rows]
        summary.update({"success_margin_" + key: value for key, value in quantiles(margins).items()})
        summaries.append(summary); items[tag] = rows
    return summaries, items


def paired(left: dict, right: dict, *, comparison: str) -> list[dict]:
    result = []
    for tag in TAGS:
        a, b = left[tag], right[tag]
        require([(r["case_id"], r["prompt_index"], r["identity"]) for r in a] ==
                [(r["case_id"], r["prompt_index"], r["identity"]) for r in b], "paired identity/order mismatch")
        lost = sum(x["success"] and not y["success"] for x, y in zip(a, b))
        gained = sum(not x["success"] and y["success"] for x, y in zip(a, b))
        n0, n1 = sum(r["success"] for r in a), sum(r["success"] for r in b)
        require(n1 == n0 - lost + gained, "paired accounting mismatch")
        row = {"comparison": comparison, "metric": tag, "status": "MEASURED_EXACT_ITEM_PAIRING",
               "denominator": len(a), "left_success": n0, "right_success": n1, "lost": lost, "gained": gained,
               "retained": n0 - lost, "failed_both": len(a) - n0 - gained,
               "delta_pp": 100 * (n1 - n0) / len(a), "conditional_loss_den": n0,
               "conditional_recovery_den": len(a) - n0}
        for target in ("new", "true"):
            delta = [y[target + "_nll"] - x[target + "_nll"] for x, y in zip(a, b)]
            row.update({target + "_nll_delta_" + key: value for key, value in quantiles(delta).items()})
        result.append(row)
    return result


def request_populations(records: list[dict]) -> tuple[dict[str, set], dict]:
    """Outcome-blind historical subject/relation + target identity rule.

    A later *batch* with another target marks a supersession candidate, not
    proven causal forgetting. Conflicts within one joint batch have no ordered
    writer semantics and are reported separately. Nothing leaves the canonical
    denominator. Missing relation IDs are not silently treated as one relation.
    """
    require(len({r['case_id'] for r in records}) == len(records), 'population duplicate case')
    groups = {}
    unknown = set()
    for index, record in enumerate(records):
        request = record['requested_rewrite']; case = record['case_id']
        if not request.get('relation_id'):
            unknown.add(case)
            continue
        key = digest([request['relation_id'], request['subject']])
        groups.setdefault(key, []).append((case, index // 100, digest(request['target_new'])))
    superseded, same_batch, active = set(), set(), set()
    for members in groups.values():
        for case, batch, target in members:
            if any(b > batch and t != target for _, b, t in members):
                superseded.add(case)
            elif any(b == batch and t != target for _, b, t in members):
                same_batch.add(case)
            else:
                active.add(case)
    populations = {'ACTIVE_NO_LATER_DIFFERENT_TARGET': active,
                   'SUPERSEDED_CANDIDATE_LATER_BATCH': superseded,
                   'WITHIN_BATCH_TARGET_CONFLICT': same_batch,
                   'UNRESOLVED_MISSING_RELATION': unknown}
    require(sum(map(len, populations.values())) == len(records), 'population partition mismatch')
    return populations, dict(rule='SUBJECT_RELATION_TARGET_IDENTITY_LATER_BATCH_V1',
        outcome_filtering=False, canonical_denominator=len(records),
        counts={name: len(cases) for name, cases in populations.items()},
        interpretation='Supersession candidate is input identity, not a causal attribution or exclusion.')


def population_pairs(left: dict, right: dict, populations: dict, comparison: str) -> list[dict]:
    rows = []
    for population, cases in populations.items():
        if not cases:
            rows.extend(dict(comparison=comparison, population=population, metric=tag,
                             status='EMPTY_INPUT_POPULATION', denominator=0) for tag in TAGS)
            continue
        a = {tag: [r for r in left[tag] if r['case_id'] in cases] for tag in TAGS}
        b = {tag: [r for r in right[tag] if r['case_id'] in cases] for tag in TAGS}
        for row in paired(a, b, comparison=comparison):
            row['population'] = population
            rows.append(row)
    return rows


def source_and_sample(lock_path: Path, expected_sha: str) -> tuple[dict, list[dict], list[dict]]:
    require(sha256(lock_path) == expected_sha, "input lock SHA mismatch")
    lock = read_json(lock_path); evidence = [member(lock_path)]
    source_ref = lock["source_lock"]; source_path = Path(source_ref["path"])
    evidence.append(member(source_path, source_ref)); source = read_json(source_path)
    require(source["head"] == lock["source_head"] and source["tree"] == lock["source_tree"], "source lock identity mismatch")
    for m in source["members"]:
        evidence.append(member(Path(m["path"]), m))
    contract_path = Path(lock["source_root"]) / CONTRACT_REL
    require(sha256(contract_path) == CONTRACT_SHA, "frozen MAIN contract source mismatch")
    evidence.append(member(contract_path))
    lock = dict(lock, _analysis_expected_contract=read_json(contract_path))
    sample = lock["sample"]
    for m in sample["members"]:
        evidence.append(member(Path(m["path"]), m))
    records = read_json(Path(lock["dataset_root"]) / "counterfact.json")[:1000]
    ids = [r["case_id"] for r in records]
    require(len(ids) == len(set(ids)) == 1000, "fixed1000 unique inventory mismatch")
    require(ids == sample["case_ids"] and digest(ids) == sample["case_order_sha256"] and
            digest(records) == sample["prefix_records_sha256"], "fixed1000 prefix mismatch")
    boundaries = [{"batch_id": i + 1, "start": i * 100, "stop": (i + 1) * 100,
                   "case_order_sha256": digest(ids[i*100:(i+1)*100])} for i in range(10)]
    require(sample["batch_boundaries"] == boundaries, "B100 boundary mismatch")
    context = Path(lock["contexts_path"]); evidence.append(member(context))
    require(sha256(context) == lock["contexts"]["sha256"] and
            digest(read_json(context)) == lock["contexts"]["semantic_sha256"], "contexts mismatch")
    return lock, records, evidence


def flow_rows(flow: dict, config: dict, batch: str) -> tuple[list[dict], list[dict]]:
    accepted = integer(flow["accepted"], "accepted")
    rejected = integer(flow["rejected"], "rejected")
    calls = integer(flow["oracle_calls"], "oracle calls", minimum=1)
    require(calls == 1 + accepted + rejected <= config["integrator"]["max_oracle_calls"], "N_oracle accounting mismatch")
    require(flow["status"] in {"FIRST_ORDER_STATIONARY", "RESOURCE_STOP", "NUMERICAL_STOP"}, "unknown solver stop")
    events, trace = flow["oracle_events"], flow["trace"]
    require(len(events) == calls and len(trace) == calls - 1, "flow event/trace count mismatch")
    require(sum(t["accepted"] == 1 for t in trace) == accepted and
            all(t["accepted"] in (0, 1) for t in trace), "trace accept accounting mismatch")
    price = config["objective"]["price"]; beta = config["objective"].get("beta_essence", .0625)
    rows, groups = [], {"initial": [], "accepted": [], "rejected": []}
    last_L = None; last_C = 0.; last_F = None
    for index, event in enumerate(events):
        require(event["oracle"] == index + 1, "oracle event order mismatch")
        for key in ("edit_nll", "essence_kl", "L", "seconds", "forward_loss_seconds", "backward_seconds", "x_norm", "gradient_norm"):
            finite(event[key], key, nonnegative=key not in {"essence_kl", "L"})
        close(event["L"], event["edit_nll"] + beta * event["essence_kl"], "oracle L")
        row = {"batch": batch, "oracle": index + 1, "phase": "initial" if index == 0 else None, **event}
        if index == 0:
            row.update(C=0., F=event["L"], step=None, accepted=None)
            last_L = event["L"]; last_F = event["L"]
        else:
            item = trace[index-1]
            require(item["nfe"] == index + 1, "trace nfe order mismatch")
            for key in ("step", "F_before", "F_trial", "cost", "barrier_dual", "stationarity_before", "roundoff_limited"):
                finite(item[key], key)
            close(item["F_before"], last_F, "carried accepted objective")
            close(item["F_trial"], event["L"] + price * item["cost"], "candidate objective")
            row.update(item); row.update(C=item["cost"], F=item["F_trial"], phase="accepted" if item["accepted"] else "rejected")
            if item["accepted"]:
                last_L, last_C, last_F = event["L"], item["cost"], item["F_trial"]
        groups[row["phase"]].append(event); rows.append(row)
    close(flow["terminal"]["L"], last_L, "terminal L")
    close(flow["terminal"]["C"], last_C, "terminal C")
    close(flow["terminal"]["F"], last_F, "terminal F")
    for key, value in flow["terminal"].items():
        finite(value, "terminal " + key)
    require(flow["work"]["oracle_calls"] == calls, "work oracle count mismatch")
    compute = []
    for phase, values in groups.items():
        compute.append({"batch": batch, "component": "oracle_" + phase, "calls": len(values),
                        **{key: sum(v[key] for v in values) for key in
                           ("seconds", "forward_loss_seconds", "backward_seconds")}})
    return rows, compute


def _binding(receipt: dict, source: dict, kind: str, records: list[dict]) -> dict:
    return {"checkpoint_payload_sha256": receipt["payload_sha256"],
            "checkpoint_receipt_sha256": receipt["receipt_sha256"], "batch_id": receipt["batch_id"],
            "weight_sha256": receipt["weight_sha256"], "history_sha256": receipt["history_sha256"],
            "source": source, "kind": kind, "request_count": len(records),
            "request_order": digest([r["case_id"] for r in records]),
            "request_ids": [r["case_id"] for r in records], "evaluator_microbatch": 16}


def commit_checks(prepared: dict) -> dict:
    """Recheck recorded tests against their recorded presealed thresholds."""
    parity, cost = prepared["parity_evidence"], prepared["cost_evidence"]
    require(parity.get("passed") is True and cost.get("passed") is True,
            "commit parity/cost evidence missing")
    for key in ("max_logit_abs", "logit_relative_l2", "edit_abs_error", "kl_abs_error"):
        value = finite(parity[key], key, nonnegative=True)
        tolerance = finite(parity["tolerance"][key], "parity tolerance", nonnegative=True)
        require(value <= tolerance and parity["checks"][key] is True, "recorded parity threshold failed")
    require(parity["physical_parameter_copy"] is True and parity["entry_restored_exact"] is True and
            parity["nonselected_pointer_version_unchanged"] is True and parity["inner_history_append"] == 0,
            "actual materialization/nonmutation evidence missing")
    actual, predicted = finite(cost["actual_cost"], "actual cost", nonnegative=True), finite(cost["predicted_cost"], "predicted cost", nonnegative=True)
    relative = abs(actual-predicted) / max(actual, predicted, 1e-30)
    close(cost["relative_error"], relative, "actual cost relative error")
    require(relative <= finite(cost["tolerance"], "cost tolerance", nonnegative=True), "recorded cost threshold failed")
    close(prepared["actual_delta_cost_fp64"], actual, "durable actual cost")
    return {"actual_delta_norm": finite(cost["actual_delta_norm"], "actual delta norm", nonnegative=True),
            "cost_relative_error": relative,
            **{key: parity[key] for key in ("max_logit_abs", "logit_relative_l2", "edit_abs_error", "kl_abs_error")}}


def saved_state_arithmetic(tensors: dict, entry_weight, entry_history, prepared: dict) -> dict:
    """CPU-only independent rounded W cost and native FP32 Gram end-state.

    This verifies the saved end state, not an additional GPU replay or a claim
    that observing one endpoint alone counts every operation during a run.
    """
    import torch
    from .durable import tensor_sha256
    from .transaction import materialized_cost
    require(tensor_sha256(entry_weight) == prepared['entry_weight_sha256'] and
            tensor_sha256(entry_history) == prepared['entry_history_sha256'], 'arithmetic entry identity')
    w, m, k = tensors['W'], tensors['M'], tensors['K']
    require(all(t.device.type == 'cpu' and t.dtype == torch.float32 for t in (w,m,k,entry_weight,entry_history)),
            'saved arithmetic CPU FP32 contract')
    if prepared['accepted']:
        gram = k @ k.T
        expected = entry_history + gram
        require(torch.equal(m, expected), 'saved history differs from native CPU FP32 one-Gram append')
        del gram, expected
    else:
        require(torch.equal(w, entry_weight) and torch.equal(m, entry_history), 'saved no-update changed state')
    delta = w.double() - entry_weight.double()
    actual_cost = materialized_cost(delta, entry_history, request_count=prepared['request_count'])
    close(actual_cost, prepared['actual_delta_cost_fp64'], 'independent saved W actual cost')
    return dict(saved_history_fp32_one_gram_equal=True,
        saved_actual_cost_recomputed_fp64=actual_cost, saved_delta_norm_recomputed_fp64=float(delta.norm()),
        arithmetic_CPU_threads=torch.get_num_threads(), model_forward_calls=0)


def pinned_model_path(lock: dict, path: Path, *, full_hash: bool = False) -> Path:
    """HF snapshot symlinks are allowed only to their already pinned blob."""
    matches = [m for m in lock['model']['members'] if m['path'] == str(path.absolute())]
    require(len(matches) == 1, 'model member absent/duplicated in input lock')
    expected = matches[0]
    resolved = path.resolve(strict=True)
    require(str(resolved) == expected['realpath'] and path.is_symlink() == expected['symlink'] and
            resolved.is_file() and not resolved.is_symlink(), 'pinned model realpath changed')
    require(resolved.stat().st_size == expected['bytes'], 'pinned model size changed')
    if full_hash:
        member(resolved, expected)
    return resolved


def initial_cpu_state(lock: dict, runtime: dict):
    import torch
    from safetensors import safe_open
    from .durable import tensor_sha256
    torch.set_num_threads(8)  # Same CPU FP32 history materialization order/threads.
    model_root = Path(lock['model_path'])
    index = read_json(pinned_model_path(lock, model_root / 'model.safetensors.index.json', full_hash=True))
    shard = index['weight_map'][WEIGHT]
    require(not Path(shard).is_absolute() and '..' not in Path(shard).parts, 'model shard path escape')
    shard_path = pinned_model_path(lock, model_root / shard)
    with safe_open(str(shard_path), framework='pt', device='cpu') as stream:
        weight = stream.get_tensor(WEIGHT).to(torch.float32).clone()
    require(tensor_sha256(weight) == runtime['base_selected_weight_sha256'], 'CPU W0 selected weight tensor identity')
    return weight, torch.zeros((weight.shape[1],weight.shape[1]), dtype=torch.float32)


def analyze(root: str | Path, lock_path: str | Path, *, input_sha256: str,
            n4_path: str | Path | None = None, n4_sha256: str | None = None,
            checkpoint_loader=None) -> dict:
    """Verify and reduce a full ten-batch chain; never fill missing observations.

    checkpoint_loader is dependency injection for focused fixtures only. Normal
    execution always performs the durable loader's full file/tensor/RNG hashes.
    """
    root, lock_path = Path(root).absolute(), Path(lock_path).absolute()
    lock, records, evidence = source_and_sample(lock_path, input_sha256)
    source = {"input_lock_sha256": input_sha256, "source_head": lock["source_head"], "source_tree": lock["source_tree"]}
    runtime = read_json(root / "runtime.json"); evidence.append(member(root / "runtime.json"))
    require(runtime["source"] == source and runtime["fresh_pretrained_W0_cold_M0"] is True and
            runtime["technical_state_carried"] is False, "fresh MAIN runtime/source mismatch")
    require(runtime["model_dtype"] == "float32" and runtime["attention"] == "eager" and
            runtime["tf32_matmul"] is False and runtime["tf32_cudnn"] is False and
            runtime["writer_add_bos"] is False and runtime["writer_padding"] == "right" and
            runtime["evaluator_packing"] == "native manual left / microbatch16" and
            runtime["transformers"] == "4.44.2", "runtime contract mismatch")
    terminal = read_json(root / "TERMINAL.json"); evidence.append(member(root / "TERMINAL.json"))
    require(terminal["status"] == "SEQ1000_COMPLETED" and terminal["source"] == source and
            terminal["batches"] == 10 and terminal["unique_requests"] == terminal["attempted_requests"] == 1000,
            "not a complete canonical SEQ1000 terminal")
    require((root / "checkpoints").is_dir(), "checkpoint root absent")
    require(sorted(p.name for p in (root / "checkpoints").iterdir() if p.name.startswith("B")) ==
            [f"B{i:03d}" for i in range(1, 11)], "published checkpoint count/gap outside canonical ten")
    full_checkpoint_verification = checkpoint_loader is None
    if full_checkpoint_verification:
        from .durable import CheckpointStore
        checkpoint_loader = CheckpointStore(root / "checkpoints").load
        entry_weight, entry_history = initial_cpu_state(lock, runtime)
    nodes, batches, metrics, compute, all_items = [], [], [], [], {}
    all_current = {tag: [] for tag in TAGS}; previous = None; config = None
    parity_tolerance, cost_tolerance = None, None
    completes = []
    for index in range(1, 11):
        batch = f"B{index:03d}"; batch_root = root / batch
        complete = read_json(batch_root / "COMPLETE.json"); completes.append(complete)
        evidence.append(member(batch_root / "COMPLETE.json"))
        loaded = checkpoint_loader(batch); metadata, receipt = loaded["metadata"], loaded["receipt"]
        declared = complete["checkpoint"]
        require({k: v for k, v in receipt.items() if k != "replayed"} ==
                {k: v for k, v in declared.items() if k != "replayed"}, "COMPLETE checkpoint receipt mismatch")
        require(complete["batch"] == receipt["batch_id"] == batch and complete["batch_index"] == index and
                complete["request_count"] == 100 and complete["source"] == metadata["source"] == source and
                metadata["batch_index"] == index and metadata["next_batch_index"] == index+1,
                "checkpoint/source/batch continuity mismatch")
        config = metadata["config"] if config is None else config
        require(metadata["config"] == config == lock["_analysis_expected_contract"] and metadata["context"] == read_json(Path(lock["contexts_path"])),
                "chain config/context changed")
        prep = metadata["prepared"]
        commit_summary = commit_checks(prep)
        if full_checkpoint_verification:
            commit_summary.update(saved_state_arithmetic(loaded['tensors'], entry_weight, entry_history, prep))
            entry_weight, entry_history = loaded['tensors']['W'], loaded['tensors']['M']
        if parity_tolerance is None:
            parity_tolerance, cost_tolerance = prep["parity_evidence"]["tolerance"], prep["cost_evidence"]["tolerance"]
        require(prep["parity_evidence"]["tolerance"] == parity_tolerance and
                prep["cost_evidence"]["tolerance"] == cost_tolerance, "parity/cost tolerance changed within chain")
        expected_parent = None if previous is None else {k: previous[k] for k in ("batch_id", "payload_sha256", "receipt_sha256")}
        require(metadata["parent"] == expected_parent, "checkpoint parent receipt mismatch")
        if previous is not None:
            require(prep["entry_weight_sha256"] == previous["weight_sha256"] and
                    prep["entry_history_sha256"] == previous["history_sha256"], "interbatch W/M mismatch")
        else:
            require(prep["entry_weight_sha256"] == runtime["base_selected_weight_sha256"], "W0 selected weight mismatch")
        ledger = metadata["ledger"]; flow = ledger["flow"]
        require(prep["request_count"] == 100 and prep["accepted"] == complete["accepted"] == flow["accepted"] and
                complete["rejected"] == flow["rejected"] and complete["oracle_calls"] == flow["oracle_calls"] and
                complete["solver_status"] == flow["status"], "complete/flow/prepared count mismatch")
        require(prep["history_append"] == receipt["history_append"] == complete["history_append"] == int(flow["accepted"] > 0),
                "history exactly-once/no-update mismatch")
        if flow["accepted"] == 0:
            require(prep["status"] == "NO_UPDATE" and prep["entry_weight_sha256"] == prep["weight_sha256"] and
                    prep["entry_history_sha256"] == prep["history_sha256"], "no-update changed state")
        else:
            require(prep["status"] == "COMMITTED" and prep["parity_evidence"]["passed"] is True and
                    prep["cost_evidence"]["passed"] is True, "commit parity/cost evidence missing")
        if "edit_attempt" in ledger:
            attempt = child(root, ledger["edit_attempt"])
            for name, expected in (("flow.json", flow), ("preparation.json", ledger["preparation"])):
                require(read_json(attempt / name) == expected, "raw/checkpoint ledger mismatch")
                evidence.append(member(attempt / name))
        batch_nodes, batch_compute = flow_rows(flow, config, batch); nodes.extend(batch_nodes); compute.extend(batch_compute)
        scope_records = {"current": records[(index-1)*100:index*100]}
        if index in (5, 10): scope_records["seen-full"] = records[:index*100]
        if index == 10: scope_records["first500"] = records[:500]
        require(set(complete["artifacts"]) == set(scope_records), "COMPLETE observation inventory mismatch")
        batch_items = {}
        for kind, selected in scope_records.items():
            artifact = complete["artifacts"][kind]
            path = child(batch_root, artifact["relative_path"])
            evidence.append(member(path, artifact))
            registry_path = batch_root / "observation-registry" / kind / "000001.json"
            require(read_json(registry_path) == artifact, "COMPLETE/registry mismatch")
            evidence.append(member(registry_path))
            binding = _binding(receipt, source, kind, selected)
            require(artifact["binding"] == binding and artifact["kind"] == kind, "artifact binding mismatch")
            raw = read_json(path)
            summary, items = reduce_observation(raw, selected, binding=binding)
            for summary_row in summary:
                summary_row.update(batch=batch, batch_index=index, scope=kind,
                                   weight_sha256=receipt["weight_sha256"], source_head=source["source_head"],
                                   observation_sha256=artifact["sha256"])
                metrics.append(summary_row)
                for holder in (artifact["metrics"], complete["metrics"] if kind == "current" else artifact["metrics"]):
                    m = holder[summary_row["metric"]]
                    require(m["numerator"] == summary_row["numerator"] and m["denominator"] == summary_row["denominator"],
                            "published summary num/den mismatch")
                    close(m["rate"], summary_row["numerator"] / summary_row["denominator"], "published summary rate")
            batch_items[kind] = items; all_items[batch + "/" + kind] = items
            compute.append({"batch": batch, "component": "evaluation_" + kind,
                            "seconds": finite(artifact["evaluation_seconds"], "evaluation seconds", nonnegative=True),
                            "new_forward": artifact["new_forward"], "observed_prompt_pairs": len(selected)*13})
            if kind == "first500":
                full = read_json(child(batch_root, complete["artifacts"]["seen-full"]["relative_path"]))
                require(raw.get("new_forward") is False and raw.get("source_full_sha256") ==
                        complete["artifacts"]["seen-full"]["sha256"], "W10 first500 derivation identity mismatch")
                for tag, multiplier in TAGS.items():
                    require(items[tag] == full["metrics"][tag]["rows"][:500*multiplier], "W10 first500 rows differ")
        for tag in TAGS: all_current[tag].extend(batch_items["current"][tag])
        row = {key: complete.get(key) for key in ("batch", "batch_index", "request_count", "solver_status", "accepted", "rejected", "oracle_calls",
             "history_append", "flow_seconds", "preparation_seconds", "commit_seconds", "evaluation_seconds", "total_seconds",
             "peak_gpu_allocated", "peak_gpu_reserved", "recovery", "missing_interrupted_timing")}
        row.update(weight_sha256=receipt["weight_sha256"], history_sha256=receipt["history_sha256"],
                   checkpoint_payload_sha256=receipt["payload_sha256"], terminal_L=flow["terminal"]["L"],
                   terminal_C=flow["terminal"]["C"], terminal_F=flow["terminal"]["F"],
                   actual_delta_cost_fp64=prep["actual_delta_cost_fp64"], **commit_summary)
        checkpoint_root = root / 'checkpoints' / batch
        row['checkpoint_bundle_bytes'] = sum(p.stat().st_size for p in checkpoint_root.iterdir() if p.is_file())
        batches.append(row)
        for component, seconds in (("flow_total_including_oracle", flow["flow_seconds"]),
                                   ("preparation_total", ledger["preparation"]["total_preparation_seconds"]),
                                   ("commit_total", complete["commit_seconds"]), ("batch_total", complete["total_seconds"])):
            compute.append({"batch": batch, "component": component, "seconds": None if seconds is None else finite(seconds, component, nonnegative=True),
                            "missing_status": "NOT_RECORDED_INTERRUPTED" if seconds is None else None,
                            "additivity": "OVERLAPPING_DO_NOT_SUM_COMPONENTS"})
        for key, value in flow["work"].items():
            finite(value, "work " + key, nonnegative=True)
            compute.append({"batch": batch, "component": "work_" + key, "value": value,
                            "additivity": "AS_RECORDED_NOT_ADDITIONAL_ORACLES"})
        preparation = ledger["preparation"]
        detailed_times = {"native_keys": preparation.get("keys", {}).get("seconds"),
                          "native_geometry": preparation.get("geometry", {}).get("seconds"),
                          "prefix_teacher_combined": preparation.get("prefix_teacher_seconds"),
                          "terminal_physical_parity": prep["parity_evidence"].get("seconds"),
                          "explicit_actual_cost": ledger.get("explicit_cost_seconds"),
                          "durable_state_prepare": ledger.get("prepare_commit_seconds")}
        for component, seconds in detailed_times.items():
            compute.append({"batch": batch, "component": component,
                            "seconds": None if seconds is None else finite(seconds, component, nonnegative=True),
                            "missing_status": "NOT_RECORDED" if seconds is None else None,
                            "additivity": "SUBCOMPONENT_DO_NOT_ADD_TO_PARENT_TOTAL"})
        commit_parts = [complete['commit_seconds'], detailed_times['terminal_physical_parity'],
                        detailed_times['explicit_actual_cost'], detailed_times['durable_state_prepare']]
        if all(value is not None for value in commit_parts):
            remainder = commit_parts[0] - sum(commit_parts[1:])
            require(remainder >= -1e-6, 'commit component timing overlap/inconsistency')
            compute.append(dict(batch=batch,component='publication_hash_transfer_misc_remainder',
                seconds=remainder,additivity='DERIVED_NONEXCLUSIVE_COMMIT_REMAINDER_NOT_PURE_IO'))
        compute.append(dict(batch=batch,component='pure_filesystem_IO',seconds=None,
            missing_status='NOT_SEPARATELY_TIMED_INCLUDED_IN_COMMIT',bytes=row['checkpoint_bundle_bytes']))
        evidence.append(member(root / "checkpoints" / batch / "manifest.json"))
        previous = receipt
        # Full tensor bundles are verified one at a time, not held tenfold.
        del loaded
    require(terminal["completed"] == completes, "terminal COMPLETE inventory differs")
    require({k: v for k, v in terminal["final_checkpoint"].items() if k != "replayed"} ==
            {k: v for k, v in previous.items() if k != "replayed"}, "terminal final checkpoint mismatch")
    for key in ("accepted", "rejected", "oracle_calls"):
        require(terminal[key] == sum(r[key] for r in batches), "terminal count mismatch: " + key)
    require(terminal["history_appends"] == sum(r["history_append"] for r in batches) and
            terminal["no_update_batches"] == sum(r["accepted"] == 0 for r in batches), "terminal append/no-update mismatch")
    final = all_items["B010/seen-full"]
    populations, population_summary = request_populations(records)
    pairs = paired(all_current, final, comparison="SL_ZFLOW_ATWRITE_TO_W10")
    pairs += paired(all_items["B005/seen-full"], all_items["B010/first500"], comparison="SL_ZFLOW_W5_TO_W10_SAME_FIRST500")
    pairs += population_pairs(all_current, final, populations, "SL_ZFLOW_ATWRITE_TO_W10_BY_INPUT_POPULATION")
    if n4_path is None:
        require(n4_sha256 is None, "N4 hash without raw path")
        pairs += [{"comparison": "N4_W10_TO_SL_ZFLOW_W10", "metric": tag, "status": "NOT_MEASURED_NO_N4_RAW"} for tag in TAGS]
    else:
        require(n4_sha256 is not None and sha256(Path(n4_path)) == n4_sha256, "N4 raw SHA missing/mismatch")
        evidence.append(member(Path(n4_path)))
        n4_metrics, n4_items = reduce_observation(read_json(Path(n4_path)), records)
        pairs += paired(n4_items, final, comparison="N4_W10_TO_SL_ZFLOW_W10")
        pairs += population_pairs(n4_items, final, populations, "N4_W10_TO_SL_ZFLOW_W10_BY_INPUT_POPULATION")
        for row in n4_metrics: row.update(batch="N4_W10", batch_index=10, scope="historical-n4", observation_sha256=n4_sha256)
        metrics.extend(n4_metrics); all_items["N4_W10"] = n4_items
    return {"format": FORMAT, "status": "CPU_VERIFIED_SEQ1000", "execution_source": source,
            "runtime": runtime, "sample_unique_requests": 1000, "scientific_batches": 10,
            "attempted_requests": 1000, "node_rows": nodes, "batch_rows": batches,
            "request_population_summary": population_summary,
            "observation_inventory": {
                "new_forward_prompt_pairs": sum(r['observed_prompt_pairs'] for r in compute if r.get('new_forward') is True),
                "derived_no_forward_prompt_pairs": sum(r['observed_prompt_pairs'] for r in compute if r.get('new_forward') is False),
                "historical_n4_reused_prompt_pairs": 13000 if n4_path is not None else 0,
                "new_forward_target_sequences": 2*sum(r['observed_prompt_pairs'] for r in compute if r.get('new_forward') is True),
                "unique_scientific_requests": 1000,
                "repeated_observations_are_independent_samples": False},
            "metric_rows": metrics, "paired_rows": pairs, "compute_rows": compute,
            "input_members": evidence, "private_items": all_items,
            "checkpoint_verification": "FULL_FILE_TENSOR_RNG_SHA256" if full_checkpoint_verification else "INJECTED_FIXTURE_LOADER_NOT_A_PRODUCTION_RECEIPT",
            "saved_state_arithmetic": "CPU_FP32_ONE_GRAM_AND_FP64_STORED_DELTA_COST" if full_checkpoint_verification else "NOT_A_PRODUCTION_RECEIPT",
            "parity_tolerance": parity_tolerance, "cost_relative_tolerance": cost_tolerance,
            "limits": ["Historical N4 cudnn TF32=True versus MAIN False; cross-hardware bitwise parity NOT_CLAIMED.",
                       "Source/input verification is not a new model/evaluator correctness measurement.",
                       "Repeated observations share 1000 requests; not independent samples.",
                       "Elapsed component timers overlap; latest process time excludes earlier interrupted processes.",
                       "FIRST_ORDER_STATIONARY is not global optimality; RESOURCE_STOP is not optimality.",
                       "N4 raw SHA and exact item pairing do not independently revalidate historical model execution."],
            "terminal_counts": {k: terminal[k] for k in ("accepted", "rejected", "oracle_calls", "history_appends", "no_update_batches")},
            "latest_process_seconds": finite(terminal["latest_process_seconds"], "latest process seconds", nonnegative=True)}


def _write_json(path: Path, value: Any) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())


def _csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    require(all(not isinstance(value, (dict, list, tuple)) for row in rows for value in row.values()), "non-scalar public CSV")
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fields); writer.writeheader(); writer.writerows(rows)
        stream.flush(); os.fsync(stream.fileno())


def write_outputs(result: dict, output: str | Path, *, private_output: str | Path | None = None) -> dict:
    """Create-once public aggregates; optional case-level rows are local-only."""
    output = Path(output).absolute()
    require(not output.exists() and all(not p.is_symlink() for p in (output, *output.parents)), "output exists/symlink")
    if private_output is not None:
        private_output = Path(private_output).absolute()
        require(not private_output.exists() and all(not p.is_symlink() for p in (private_output, *private_output.parents)), "private output exists/symlink")
        require(not private_output.is_relative_to(output) and not output.is_relative_to(private_output), "public/private output overlap")
        require("local" in private_output.parts, "per-item output must be an explicit local-only path")
    output.mkdir(parents=True, mode=0o700)
    source_file = Path(__file__).resolve()
    try:
        analysis_head = subprocess.check_output(["git", "-C", str(source_file.parent), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        analysis_head = "NOT_RECORDED"
    for key, filename in (("node_rows", "node.csv"), ("batch_rows", "batch.csv"), ("metric_rows", "metrics.csv"),
                          ("paired_rows", "paired.csv"), ("compute_rows", "compute.csv")):
        _csv(output / filename, result[key])
    summary = {k: v for k, v in result.items() if k not in {"node_rows", "batch_rows", "metric_rows", "paired_rows", "compute_rows", "private_items", "runtime"}}
    summary["analysis_source"] = {"head": analysis_head, **member(source_file), "runtime_source_is_distinct": True}
    summary["runtime_contract"] = {k: result["runtime"].get(k) for k in
        ("torch", "transformers", "cuda", "device", "tf32_matmul", "tf32_cudnn", "model_dtype", "attention", "writer_add_bos", "evaluator_add_bos", "writer_padding", "evaluator_packing")}
    summary["public_outputs_raw_free"] = True
    summary["private_item_output"] = str(private_output) if private_output else "NOT_WRITTEN"
    _write_json(output / "verification.json", summary)
    if private_output is not None:
        private_output.mkdir(parents=True, mode=0o700)
        _write_json(private_output / "per-item.json", result["private_items"])
    files = [member(p) for p in sorted(output.iterdir())]
    manifest = {"format": FORMAT, "members": files, "member_root": digest(files),
                "raw_payload_broadcast": "NO_BROADCAST_NOT_REQUIRED", "model_evaluator_GPU_calls": 0}
    _write_json(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True); parser.add_argument("--input-lock", required=True)
    parser.add_argument("--input-sha256", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--private-output"); parser.add_argument("--n4-raw"); parser.add_argument("--n4-sha256")
    args = parser.parse_args()
    result = analyze(args.root, args.input_lock, input_sha256=args.input_sha256,
                     n4_path=args.n4_raw, n4_sha256=args.n4_sha256)
    manifest = write_outputs(result, args.output, private_output=args.private_output)
    print(json.dumps({"status": result["status"], "output": str(Path(args.output).absolute()),
                      "member_root": manifest["member_root"], "batches": 10, "unique_requests": 1000}))


if __name__ == "__main__":
    main()
