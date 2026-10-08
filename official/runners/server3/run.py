"""Qwen official baseline runner (native EasyEdit algorithms from ``official``).

This module never imports algorithms from the server's EasyEdit checkout.  The
checkout supplies only the pinned model, data, covariance and projector files.
The common factual evaluator is owned by server1; absence is a hard preflight
failure, not permission to substitute a second metric implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[3]
METHODS = ("FT", "MEMIT", "ALPHAEDIT", "ALPHAEDIT_BLUE", "MEMIT_FE", "SPHERE")
HISTORY = ("ALPHAEDIT", "ALPHAEDIT_BLUE", "SPHERE")
PHASES = (0, 5, 10, 15, 20)
L2_GRID = (1, 10, 95)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_once(path, value):
    from official.experiments.prepare import write_new
    write_new(path, value)


def _sha(path):
    from official.experiments.prepare import file_sha
    return file_sha(path)


def _git(*args):
    result = subprocess.run(["git", *args], cwd=ROOT, check=True, text=True,
                            capture_output=True)
    return result.stdout.strip()


def frozen_source_identity():
    """Bind the published source, including the server-specific runner bytes."""
    from official.runners.server3.submit import official_tree_sha256
    actual_tree = official_tree_sha256(ROOT)
    if (ROOT / ".git").exists():
        commit = _git("rev-parse", "HEAD")
        if _git("status", "--porcelain", "--", "official"):
            raise ValueError("OFFICIAL_SOURCE_DIRTY")
        if subprocess.run(["git", "merge-base", "--is-ancestor", commit, "origin/main"],
                          cwd=ROOT).returncode != 0:
            raise ValueError("OFFICIAL_SOURCE_NOT_ON_MAIN")
    else:
        lock_path = os.environ.get("ODEEDIT_SOURCE_LOCK")
        if not lock_path:
            raise ValueError("SOURCE_ARCHIVE_LOCK_REQUIRED")
        lock = _read(lock_path)
        commit = lock.get("code_commit")
        if not commit or lock.get("official_tree_sha256") != actual_tree:
            raise ValueError("SOURCE_ARCHIVE_TREE_MISMATCH")
        if os.environ.get("ODEEDIT_CODE_COMMIT") != commit or \
                os.environ.get("ODEEDIT_OFFICIAL_TREE_SHA256") != actual_tree:
            raise ValueError("SOURCE_ARCHIVE_ENV_LOCK_MISMATCH")
    return dict(code_commit=commit, official_tree_sha256=actual_tree)


def validate_config(config):
    """Verify a sealed physical Qwen row or a derived selected zsRE BLUE row."""
    from official.experiments.prepare import digest, load_plan, build_matrix
    if config.get("model") != "qwen25" or config.get("method") not in METHODS:
        raise ValueError("SERVER3_SCOPE")
    if config.get("dataset") not in ("cf", "zsre"):
        raise ValueError("DATASET_SCOPE")
    if config.get("execution_kind") == "selected_grid_alias":
        raise ValueError("CF_BLUE_ALIAS_NOT_PHYSICAL_CHAIN")
    unsigned = {k: v for k, v in config.items() if k != "config_sha256"}
    if config.get("config_sha256") != digest(unsigned):
        raise ValueError("CONFIG_DIGEST")
    contract, profiles = load_plan()
    allowed = {row["run_id"]: row for row in build_matrix(contract, profiles)
               if row["model"] == "qwen25"}
    original = allowed.get(config.get("run_id"))
    if original is None:
        raise ValueError("UNREGISTERED_QWEN_ROW")
    if config == original:
        return config
    # The sole permitted derived scientific config is zsRE BLUE with the CF
    # winner's L2.  Its source row and selection proof are checked separately.
    if not (config["run_id"] == "qwen25-zsre-alphaedit_blue"
            and original["hparams"]["L2"] is None
            and config["hparams"].get("L2") in L2_GRID):
        raise ValueError("CONFIG_NOT_CANONICAL")
    changed = []
    for key in original:
        if original[key] != config.get(key):
            changed.append(key)
    if set(changed) != {"hparams", "config_sha256"}:
        raise ValueError("DERIVED_BLUE_CONFIG_CHANGED_OTHER_FIELDS")
    if {k: v for k, v in config["hparams"].items() if k != "L2"} != \
            {k: v for k, v in original["hparams"].items() if k != "L2"}:
        raise ValueError("DERIVED_BLUE_HPARAMS_CHANGED")
    return config


def validate_stream(path, dataset):
    from official.experiments.prepare import digest
    path = Path(path)
    lock = _read(path.with_name(f"{dataset}-stream.lock.json"))
    if lock["dataset"] != dataset or lock["requests"] != 2000 or lock["batch_size"] != 100:
        raise ValueError("STREAM_SCOPE")
    if lock["stream_sha256"] != _sha(path):
        raise ValueError("STREAM_SHA")
    rows = _read(path)
    if len(rows) != 2000 or [r["occurrence_index"] for r in rows] != list(range(1, 2001)):
        raise ValueError("STREAM_OCCURRENCE_ORDER")
    if digest([r["case_id"] for r in rows]) != lock["ordered_case_ids_sha256"]:
        raise ValueError("STREAM_CASE_ORDER")
    if len(lock["batches"]) != 20:
        raise ValueError("STREAM_BATCH_COUNT")
    for index, batch in enumerate(lock["batches"]):
        start = index * 100
        if batch != dict(batch=index + 1, start_inclusive=start, end_exclusive=start + 100,
                         ordered_case_ids_sha256=digest([r["case_id"] for r in rows[start:start + 100]])):
            raise ValueError("STREAM_BATCH_LOCK")
    return lock, rows


def checkpoint_identity(config, stream_lock, source, asset_receipt, tokenizer_receipt):
    return dict(config_sha256=config["config_sha256"],
                stream_sha256=stream_lock["stream_sha256"],
                code_commit=source["code_commit"],
                official_tree_sha256=source["official_tree_sha256"],
                model_revision=config["model_identity"]["revision"],
                tokenizer_sha256=tokenizer_receipt["tokenizer_sha256"],
                assets_sha256=asset_receipt["assets_sha256"])


def _require_evaluator():
    from official.runners.server3.factual import require_shared_factual
    module = require_shared_factual()
    # A shared CPU/GPU forward adapter must expose the same raw case schema as
    # official.evaluation.reduce.  Refuse a partial package before loading W0.
    if not callable(getattr(module, "evaluate", None)) or not callable(
            getattr(module, "build_zsre_w0_reference", None)):
        raise RuntimeError("SHARED_FACTUAL_EVALUATE_AND_ZSRE_W0_API_REQUIRED")
    return module


def _factual_identity(source, assets, stream_lock, tokenizer_receipt):
    """Stable cold-W0/edited identity, independent of the physical method."""
    return dict(model="Qwen/Qwen2.5-7B-Instruct",
                revision="a09a35458c702b33eeacc393d103063234e8bc28",
                tokenizer_sha256=tokenizer_receipt["tokenizer_sha256"],
                stream_sha256=stream_lock["stream_sha256"],
                assets_sha256=assets["assets_sha256"],
                runtime_packages=assets["runtime"]["packages"],
                code_commit=source["code_commit"],
                official_tree_sha256=source["official_tree_sha256"])


def _evaluate_factual(module, model, tokenizer, records, dataset, *,
                      w0_reference=None, identity):
    from official.evaluation.reduce import counterfact, zsre
    observed = module.evaluate(model, tokenizer, records, dataset,
                               w0_reference=w0_reference, batch_size=16,
                               identity=identity)
    cases, summary = observed["cases"], observed["summary"]
    if len(cases) != len(records):
        raise ValueError("FACTUAL_CASE_CARDINALITY")
    expected = counterfact(cases) if dataset == "cf" else zsre(cases)
    if summary != expected:
        raise ValueError("FACTUAL_SHARED_REDUCER_MISMATCH")
    return observed


def _load_model(snapshot, revision):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if torch.cuda.device_count() != 1:
        raise RuntimeError("EXACTLY_ONE_VISIBLE_GPU_REQUIRED")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True,
                                               revision=revision, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True,
                                                 revision=revision, torch_dtype=torch.float32,
                                                 attn_implementation="eager").to("cuda:0")
    model.eval()
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("MODEL_NOT_FP32")
    return model, tokenizer


def _editable_weights(model, names):
    import torch
    lookup = dict(model.named_parameters())
    missing = set(names) - set(lookup)
    if missing:
        raise ValueError("EDITABLE_WEIGHT_MISSING:" + ",".join(sorted(missing)))
    result = {name: lookup[name].detach().to("cpu", dtype=torch.float32).clone()
              for name in names}
    if not result or any(not torch.isfinite(value).all() for value in result.values()):
        raise ValueError("NONFINITE_EDITABLE_WEIGHT")
    return result


def _weight_hashes(weights):
    return {name: hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()
            for name, tensor in weights.items()}


def _state_hashes(weights, native, cursor):
    """Exact technical B3 parity evidence; hashes only, never uploads tensors."""
    from official.experiments.checkpoint import rng_snapshot
    from official.experiments.prepare import digest
    cache = native.cache_for_checkpoint()
    return dict(weights=_weight_hashes(weights), history=_weight_hashes(cache),
                context=hashlib.sha256(pickle.dumps(native.context_snapshot(), protocol=5)).hexdigest(),
                rng=hashlib.sha256(pickle.dumps(rng_snapshot(), protocol=5)).hexdigest(),
                evaluation_cursor=digest(cursor))


def _run_receipt(config, lock, source, identity, *, batch, factual, generation=None,
                 checkpoint=None, placement=None, seconds=None):
    return dict(run_id=config["run_id"], config_sha256=config["config_sha256"],
                stream_sha256=lock["stream_sha256"], code_commit=source["code_commit"],
                official_tree_sha256=source["official_tree_sha256"],
                checkpoint_identity=identity, completed_batch=batch,
                checkpoint_sha256=checkpoint["sha256"] if checkpoint else None,
                factual_sha256=factual["cases_sha256"] if factual else None,
                factual=factual, generation=generation, checkpoint=checkpoint,
                placement=placement, seconds=seconds)


def _verified_w0_receipt(folder, dataset, lock, source, asset_receipt):
    from official.experiments.prepare import digest
    folder = Path(folder)
    receipt = _read(folder / "w0-receipt.json")
    expected = dict(model="qwen25", dataset=dataset, endpoint="W0",
                    stream_sha256=lock["stream_sha256"],
                    code_commit=source["code_commit"],
                    official_tree_sha256=source["official_tree_sha256"],
                    assets_sha256=asset_receipt["assets_sha256"],
                    observed_requests=2000)
    if any(receipt.get(key) != value for key, value in expected.items()) or \
            receipt.get("receipt_sha256") != digest({
                key: value for key, value in receipt.items() if key != "receipt_sha256"}):
        raise ValueError("SHARED_W0_RECEIPT_IDENTITY_OR_DIGEST")
    factual = receipt.get("factual") or {}
    if _sha(folder / "w0-cases.json") != factual.get("cases_sha256"):
        raise ValueError("SHARED_W0_CASE_HASH")
    if dataset == "zsre" and _sha(folder / "w0-zsre-reference.json") != \
            factual.get("w0_reference_sha256"):
        raise ValueError("SHARED_ZSRE_W0_REFERENCE_HASH")
    return receipt


def _model_seed(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _tracker(output, *, arm, writer, dataset, source_sha, config_sha, assets):
    """Start the shared scalar-only online logger before a model is loaded."""
    from official.tracking import init
    from official.tracking.method import OFFICIAL_SCHEMA
    from official.tracking.schema import OFFICIAL_INSTRUCTION, NATIVE_GENERATION_PROFILE
    env_file = os.environ.get("ODEEDIT_WANDB_ENV_FILE")
    attempt = os.environ.get("ODEEDIT_ATTEMPT_ID")
    if not env_file or not attempt:
        raise RuntimeError("ONLINE_LOGGER_ENV_OR_ATTEMPT_MISSING")
    values = dict(server="server3", task_id="official-baselines-20261008",
                  arm=arm, attempt=attempt, source_sha=source_sha,
                  config_sha=config_sha, model="qwen25", model_family="qwen2",
                  writer=writer, baseline=writer, role="scientific",
                  metric_schema=OFFICIAL_SCHEMA, instruction_id=OFFICIAL_INSTRUCTION,
                  dataset=dataset)
    if dataset == "cf":
        values.update(generation_metric_schema="counterfact-cake-generation-metrics-v1",
                      generation_profile=NATIVE_GENERATION_PROFILE,
                      generation_eval_seed=20261007,
                      reference_assets_sha256=assets["generation_reference"]["identity_sha256"],
                      generation_source_sha=assets["generation_reference"]["native_generator_sha256"],
                      generation_repair_instruction=OFFICIAL_INSTRUCTION,
                      generation_schedule="W0_AND_W20_FIRST2000")
    return init(env_file=env_file, spool=Path(output) / "wandb" / attempt,
                config=values)


def _factual_scalars(dataset, cases, summary, prefix, edits):
    """Report official request-macro scores; never relabel them as prompt success."""
    values = dict(edits=edits, post_state_edits=edits)
    if prefix == "current/post":
        values["pre_state_edits"] = edits - 100
    if len(cases) != summary["requests"]:
        raise ValueError("FACTUAL_SUMMARY_REQUEST_DENOMINATOR")
    group = f"official/{prefix}"
    for field in ("Efficacy", "Generalization", "Specificity", "Score",
                  "Score_AlphaEdit_display", "Specificity_loc_ans", "requests"):
        if field in summary:
            values[f"{group}/{field}"] = summary[field]
    return values


def _milestone_scalars(dataset, cases, summary, edits):
    """Split current 100 from a measured all-seen endpoint without a forward."""
    from official.evaluation.reduce import counterfact, zsre
    if edits not in (500, 1000, 1500, 2000) or len(cases) != edits:
        raise ValueError("MILESTONE_RAW_CARDINALITY")
    reducer = counterfact if dataset == "cf" else zsre
    current = cases[-100:]
    values = _factual_scalars(dataset, cases, summary, "all_seen/post", edits)
    values.update(_factual_scalars(dataset, current, reducer(current),
                                   "current/post", edits))
    return values


def _accuracy_scalars(dataset, observed, prefix):
    """Map measured TF counts; zsRE loc_ans is distinct from W0 agreement."""
    values = {}
    groups = [("rewrite", "R"), ("paraphrase", "P")]
    if dataset == "cf":
        groups.append(("neighborhood", "N"))
    for kind, label in groups:
        accuracy = observed["accuracy"].get(kind)
        if accuracy is None:
            continue
        stem = f"{prefix}/{label}"
        values[stem + "/count"] = accuracy["prompt_count"]
        for key in ("token_acc_pct", "prompt_acc_pct", "strict_acc_pct"):
            values[stem + "/" + key] = accuracy[key]
    return values


def _current_accuracy_scalars(dataset, cases):
    """Use the shared evaluator's own TF reducer on measured last-100 raw."""
    from official.evaluation.factual import _accuracy
    if len(cases) != 100:
        raise ValueError("CURRENT_TF_RAW_CARDINALITY")
    accuracy = {}
    for kind in ("rewrite", "paraphrase", "neighborhood"):
        if dataset == "cf":
            rows = [observation["target_" + observation["desired_target"]]
                    for case in cases for observation in case[kind + "_observations"]]
        else:
            rows = [observation for case in cases
                    for observation in case[kind + "_observations"]]
        accuracy[kind] = _accuracy(rows)
    return _accuracy_scalars(dataset, {"accuracy": accuracy}, "current/post")


def _generation_summary_for_replay(output, path, identity_sha256):
    """Read the sealed native endpoint and its raw rows without model work."""
    from official.evaluation.generation.native_observer import read_observed
    output = Path(output).resolve()
    path = Path(path)
    if not path.is_file() or path.is_symlink() or \
            path.resolve() != output / "generation" / "endpoints" / f"{identity_sha256}.json":
        raise ValueError("SCALAR_REPLAY_GENERATION_PATH_OR_IDENTITY")
    observed = read_observed(path)
    if observed["identity_sha256"] != identity_sha256:
        raise ValueError("SCALAR_REPLAY_GENERATION_IDENTITY")
    return observed["summary"]


def _w0_scalar_values(dataset, cases, factual, generation_summary=None):
    values = _factual_scalars(dataset, cases, factual["summary"], "W0_first2000", 0)
    values.update(_accuracy_scalars(dataset, factual, "W0_first2000"))
    if generation_summary is not None:
        from official.evaluation.generation.metrics import generation_payload
        values.update(generation_payload("W0_first2000", generation_summary))
    return values


def _batch_scalar_values(dataset, batch, cases, factual, generation_summary=None):
    if batch == 3:
        values = _factual_scalars(dataset, cases, factual["summary"],
                                  "current/post", batch * 100)
    else:
        values = _milestone_scalars(dataset, cases, factual["summary"], batch * 100)
        values.update(_current_accuracy_scalars(dataset, cases[-100:]))
    values.update(_accuracy_scalars(dataset, factual,
                  "current/post" if batch == 3 else "all_seen/post"))
    if generation_summary is not None:
        from official.evaluation.generation.metrics import generation_payload
        values.update(generation_payload("all_seen/post", generation_summary))
    return values


def _logging_accepted(output, label, source_receipt_sha256):
    """A local SDK queue acknowledgment is not proof of remote readback."""
    folder = Path(output) / "logging"
    paths = [folder / f"{label}.json", *sorted(folder.glob(f"{label}-replay-*.json"))]
    for path in paths:
        if not path.is_file():
            continue
        receipt = _read(path)
        if receipt.get("label") != label or \
                receipt.get("source_receipt_sha256") != source_receipt_sha256 or \
                type(receipt.get("sdk_queue_accepted")) is not bool or \
                receipt.get("remote_readback") != "NOT_ESTABLISHED_BY_SDK_QUEUE":
            raise ValueError("SCALAR_LOGGING_RECEIPT_IDENTITY")
        if receipt["sdk_queue_accepted"]:
            return True
    return False


def _log_scalar_receipt(tracker, output, label, values, source_receipt_sha256):
    """Keep local transport evidence separate from scientific completion."""
    if _logging_accepted(output, label, source_receipt_sha256):
        return True
    config_values = getattr(tracker, "config_values", None)
    if isinstance(config_values, dict) and config_values.get("dataset") == "zsre":
        from official.tracking import official_zsre_metrics
        values = dict(values)
        for endpoint in ("current/pre", "current/post", "all_seen/post", "W0_first2000"):
            prefix = f"official/{endpoint}/"
            # These are the measured reducer-summary scalars already selected
            # above. Raw cases/tokens never cross the transport boundary.
            summary = {field: values[prefix + field] for field in
                       ("Efficacy", "Generalization", "Specificity",
                        "Specificity_loc_ans", "Score", "requests")
                       if prefix + field in values}
            if summary:
                values.update(official_zsre_metrics(
                    summary, config_values=config_values, endpoint=endpoint,
                    edits=values["edits"],
                    pre_state_edits=values.get("pre_state_edits"),
                    post_state_edits=values.get("post_state_edits")))
    accepted = tracker.log(values)
    try:
        folder = Path(output) / "logging"
        path = folder / f"{label}.json"
        if path.exists():
            path = folder / f"{label}-replay-{tracker.run_id}.json"
        _write_once(path,
                    dict(label=label, sdk_queue_accepted=accepted,
                         source_receipt_sha256=source_receipt_sha256,
                         remote_readback="NOT_ESTABLISHED_BY_SDK_QUEUE",
                         run_id=tracker.run_id, spool=str(tracker.spool),
                         dropped_points_at_call=tracker.dropped))
    except Exception:
        # The tracker has its own local spool; receipt I/O cannot roll back a
        # committed native edit or replace its scientific error.
        pass
    return accepted


def _preflight(args, *, require_frozen=False):
    from official.runners.server3.assets import preflight as asset_preflight
    config = validate_config(_read(args.config)) if args.command != "w0" else None
    dataset = config["dataset"] if config else args.dataset
    stream_lock, records = validate_stream(args.stream, dataset)
    asset_receipt = asset_preflight(args.assets, hash_large=False,
                                    require_generation=dataset == "cf")
    absence = {"MODEL_SNAPSHOT_MISSING", "GENERATION_MANIFEST_MISSING", "FILE_MISSING",
               "PATH_NOT_SET", "RUNTIME_PYTHON_MISSING", "WANDB_CREDENTIAL_MISSING"}
    if require_frozen and not any(item["code"] in absence for item in asset_receipt["blockers"]):
        asset_receipt = asset_preflight(args.assets, hash_large=True,
                                        require_generation=dataset == "cf")
    source = frozen_source_identity() if require_frozen else None
    blockers = [item["code"] + (":" + str(item["path"]) if item.get("path") else "")
                for item in asset_receipt["blockers"]]
    try:
        _require_evaluator()
    except (ModuleNotFoundError, RuntimeError) as error:
        blockers.append(str(error))
    if config is not None and config["method"] == "ALPHAEDIT_BLUE" and \
            config["hparams"]["L2"] is None:
        blockers.append("QWEN_BLUE_L2_SELECTION_REQUIRED")
    result = dict(ready=not blockers, blockers=blockers,
                  config_sha256=config["config_sha256"] if config else None,
                  stream_sha256=stream_lock["stream_sha256"],
                  source=source, assets=asset_receipt, requests=len(records),
                  dataset=dataset, checkpoint="LATEST_ONE_PLUS_W20")
    return result, config, stream_lock, records, asset_receipt, source


def preflight(args):
    from official.experiments.prepare import digest
    result, config, lock, _, asset_receipt, _ = _preflight(
        args, require_frozen=bool(args.resume))
    if args.resume and result["ready"]:
        source = result["source"]
        snapshot = asset_receipt["assets"]["model_snapshot"]["path"]
        tok_receipt = _tokenizer_receipt(snapshot, None, lock)
        identity = checkpoint_identity(config, lock, source, asset_receipt, tok_receipt)
        folder = Path(args.output) / "checkpoint"
        pointer = folder / "latest.json"
        if not pointer.is_file():
            result["blockers"].append("RESUME_CHECKPOINT_MISSING")
        else:
            ref = _read(pointer)
            member = folder / ref.get("file", "")
            if Path(ref.get("file", "")).name != ref.get("file") or \
                    ref.get("identity_sha256") != digest(identity) or \
                    type(ref.get("batch")) is not int or not 0 <= ref["batch"] <= 20 or \
                    not member.is_file() or _sha(member) != ref.get("sha256"):
                result["blockers"].append("RESUME_CHECKPOINT_IDENTITY_OR_HASH")
            else:
                result["resume_from_batch"] = ref["batch"]
                if ref["batch"] > 0 and not (Path(args.output) / "commits" /
                        f"b{ref['batch']:02d}.json").is_file() and not list(
                        (Path(args.output) / "commits").glob(
                            f"pending-b{ref['batch']:02d}-*.json")):
                    result["blockers"].append("RESUME_RECEIPT_AND_PENDING_MISSING")
    result["ready"] = not result["blockers"]
    print(json.dumps(result, sort_keys=True, default=str))
    return 0 if result["ready"] else 2


def _tokenizer_receipt(snapshot, tokenizer, stream_lock):
    from official.experiments.prepare import digest
    files = {}
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                 "vocab.json", "merges.txt", "added_tokens.json"):
        path = Path(snapshot) / name
        if path.is_file():
            files[name] = _sha(path)
    if not files:
        raise ValueError("TOKENIZER_FILES_MISSING")
    audited = _read(ROOT / "official/hparams/tokenizers.lock.json")["audits"][
        "qwen25-" + stream_lock["dataset"]]
    if files != audited["tokenizer_files_sha256"] or \
            digest(files) != audited["tokenizer_sha256"]:
        raise ValueError("TOKENIZER_FILE_SHA_MISMATCH")
    if tokenizer is not None and (type(tokenizer).__name__ != audited["tokenizer_class"] or
            getattr(tokenizer, "add_bos_token", None) != audited["add_bos_token"] or
            tokenizer.bos_token_id != audited["bos_token_id"] or
            tokenizer.padding_side != audited["padding_side"]):
        raise ValueError("TOKENIZER_RUNTIME_CONTRACT_MISMATCH")
    return dict(tokenizer_sha256=digest(files), files=files,
                tokenizer_class=type(tokenizer).__name__ if tokenizer is not None else None,
                add_bos_token=getattr(tokenizer, "add_bos_token", None),
                bos_token_id=tokenizer.bos_token_id if tokenizer is not None else None,
                padding_side=tokenizer.padding_side if tokenizer is not None else None,
                stream_sha256=stream_lock["stream_sha256"])


def _generation(model, tokenizer, records, assets, output, endpoint, state_identity,
                tracker, native=None):
    from official.evaluation.generation.native_observer import NativeGenerationObserver
    from official.evaluation.generation.assets import load_assets
    from official.evaluation.generation.native_profile import PROFILE
    from official.tracking import official_generation_progress
    reference = load_assets(assets["generation_reference"]["manifest_path"])
    config = dict(model_identity="qwen25@a09a35458c702b33eeacc393d103063234e8bc28",
                  generation_source_sha=assets["generation_reference"]["native_generator_sha256"],
                  profile=PROFILE, eval_seed=20261007)
    def native_state():
        if native is None:
            return {"history": "W0_ZERO", "context": "W0_UNEDITED"}
        return {"history": _weight_hashes(native.cache_for_checkpoint()),
                "context": hashlib.sha256(pickle.dumps(
                    native.context_snapshot(), protocol=5)).hexdigest()}
    progress_counts = {"sdk_queue_accepted": 0, "sdk_queue_rejected": 0}
    def progress(row):
        accepted = tracker.log(official_generation_progress(row, endpoint=endpoint))
        progress_counts["sdk_queue_accepted" if accepted else "sdk_queue_rejected"] += 1
    observer = NativeGenerationObserver(model, tokenizer, reference, config,
                                        Path(output) / "generation",
                                        state_callback=native_state,
                                        progress_callback=progress)
    completed = False
    try:
        result = observer.observe(records, endpoint=endpoint, cohort="first2000",
                                  state_identity=state_identity)
        completed = True
        return result
    finally:
        try:
            _write_once(Path(output) / "logging" /
                        f"generation-progress-{endpoint}-{tracker.run_id}.json",
                        dict(endpoint=endpoint, generation_complete=completed,
                             remote_readback="NOT_ESTABLISHED_BY_SDK_QUEUE",
                             logger_status=tracker.status, **progress_counts))
        except Exception:
            # Transport receipt failure must not replace the science exception.
            pass


def _recover_batch_receipts(out, payload, checkpoint_ref, config, lock, source, identity):
    """Finish a committed checkpoint's small receipts after a process crash.

    The pending receipt is bound into the checkpoint evaluation cursor before
    the large checkpoint is saved.  A crash before the pointer advances leaves
    the old batch current; a crash after it advances can therefore be repaired
    without re-running the model or inventing an evaluation.
    """
    from official.experiments.prepare import digest
    batch = payload["batch"]
    if batch == 0:
        return
    if checkpoint_ref["batch"] != batch:
        raise ValueError("RECEIPT_CHECKPOINT_BATCH_MISMATCH")
    pending_sha = payload["evaluation_cursor"].get("pending_receipt_sha256")
    if not pending_sha or not isinstance(pending_sha, str) or len(pending_sha) != 64:
        raise ValueError("RECEIPT_PENDING_BINDING_MISSING")
    pending_path = Path(out) / "commits" / f"pending-b{batch:02d}-{pending_sha}.json"
    if not pending_path.is_file():
        raise ValueError("RECEIPT_PENDING_MISSING")
    pending = _read(pending_path)
    if digest(pending) != pending_sha or pending["batch"] != batch or \
            pending["config_sha256"] != config["config_sha256"] or \
            pending["stream_sha256"] != lock["stream_sha256"] or \
            pending["code_commit"] != source["code_commit"] or \
            pending["official_tree_sha256"] != source["official_tree_sha256"] or \
            pending["checkpoint_identity"] != identity:
        raise ValueError("RECEIPT_PENDING_IDENTITY_MISMATCH")
    factual = pending["factual"]
    if factual is not None and _sha(factual["cases_path"]) != factual["cases_sha256"]:
        raise ValueError("RECEIPT_FACTUAL_HASH_MISMATCH")
    generation = pending["generation"]
    if generation is not None and not Path(generation["rows_path"]).is_file():
        raise ValueError("RECEIPT_GENERATION_ROWS_MISSING")
    receipt = _run_receipt(config, lock, source, identity, batch=batch,
                           factual=factual, generation=generation,
                           checkpoint=checkpoint_ref, placement=pending["placement"],
                           seconds=pending["seconds"])
    if pending["state_hashes"] is not None:
        receipt["state_hashes"] = pending["state_hashes"]
    _write_once(Path(out) / "commits" / f"b{batch:02d}.json", receipt)
    if factual is not None:
        _write_once(Path(out) / "evaluations" / f"w{batch:02d}.json", receipt)


def _replay_w0_logging(out, dataset, receipt, tracker):
    out = Path(out)
    source_sha = receipt["receipt_sha256"]
    if _logging_accepted(out, "w0", source_sha):
        return False
    cases_path = out / "w0-cases.json"
    if _sha(cases_path) != receipt["factual"]["cases_sha256"]:
        raise ValueError("SCALAR_REPLAY_W0_CASE_HASH")
    cases = _read(cases_path)
    generation_summary = None
    if dataset == "cf":
        generation_summary = _generation_summary_for_replay(
            out, receipt["generation_rows_path"], receipt["generation_identity_sha256"])
    values = _w0_scalar_values(dataset, cases, receipt["factual"], generation_summary)
    _log_scalar_receipt(tracker, out, "w0", values, source_sha)
    return True


def _replay_committed_batch_logging(out, payload, config, lock, source, identity, tracker):
    """Retry each measured endpoint bound to the restored chain in order."""
    out = Path(out)
    start = payload["batch"]
    endpoints = payload["evaluation_cursor"]["evaluated_endpoints"]
    eligible = {batch for batch in endpoints if batch in PHASES and 0 < batch <= start}
    if start >= 3 and (out / "commits" / "b03.json").is_file():
        eligible.add(3)  # The bounded continuity test measures B3 only.
    replayed = False
    for batch in sorted(eligible):
        commit_path = out / "commits" / f"b{batch:02d}.json"
        if not commit_path.is_file():
            raise ValueError("SCALAR_REPLAY_COMMIT_MISSING")
        receipt = _read(commit_path)
        expected = dict(run_id=config["run_id"], config_sha256=config["config_sha256"],
                        stream_sha256=lock["stream_sha256"],
                        code_commit=source["code_commit"],
                        official_tree_sha256=source["official_tree_sha256"],
                        checkpoint_identity=identity, completed_batch=batch)
        if any(receipt.get(key) != value for key, value in expected.items()):
            raise ValueError("SCALAR_REPLAY_COMMIT_IDENTITY")
        factual = receipt.get("factual")
        if factual is None or receipt.get("factual_sha256") != factual.get("cases_sha256"):
            raise ValueError("SCALAR_REPLAY_FACTUAL_BINDING")
        source_sha = _sha(commit_path)
        label = f"w{batch:02d}"
        if _logging_accepted(out, label, source_sha):
            continue
        cases_path = out / "evaluations" / f"w{batch:02d}-cases.json"
        if factual.get("cases_path") != str(cases_path.resolve()) or \
                _sha(cases_path) != factual["cases_sha256"]:
            raise ValueError("SCALAR_REPLAY_FACTUAL_HASH")
        generation_summary = None
        if batch == 20 and config["dataset"] == "cf":
            generation = receipt.get("generation") or {}
            generation_summary = _generation_summary_for_replay(
                out, generation["rows_path"], generation["identity_sha256"])
        values = _batch_scalar_values(config["dataset"], batch, _read(cases_path),
                                      factual, generation_summary)
        _log_scalar_receipt(tracker, out, label, values, source_sha)
        replayed = True
    return replayed


def w0(args):
    import torch
    from official.experiments.prepare import digest
    result, _, lock, records, asset_receipt, source = _preflight(args, require_frozen=True)
    if not result["ready"]:
        raise RuntimeError("W0_PREFLIGHT_BLOCKED:" + ";".join(result["blockers"]))
    module = _require_evaluator()
    assets = asset_receipt["assets"]
    out = Path(args.output)
    if (out / "w0-receipt.json").exists():
        receipt = _verified_w0_receipt(out, args.dataset, lock, source, asset_receipt)
        if not _logging_accepted(out, "w0", receipt["receipt_sha256"]):
            with _tracker(out, arm=f"qwen25-{args.dataset}-W0", writer="W0",
                          dataset=args.dataset, assets=assets,
                          source_sha=source["code_commit"],
                          config_sha=lock["stream_sha256"]) as tracker:
                _replay_w0_logging(out, args.dataset, receipt, tracker)
        return 0
    with _tracker(out, arm=f"qwen25-{args.dataset}-W0", writer="W0",
                  dataset=args.dataset, assets=assets,
                  source_sha=source["code_commit"],
                  config_sha=lock["stream_sha256"]) as tracker:
        _model_seed(0)
        model, tok = _load_model(assets["model_snapshot"]["path"],
                                 "a09a35458c702b33eeacc393d103063234e8bc28")
        tok_receipt = _tokenizer_receipt(assets["model_snapshot"]["path"], tok, lock)
        factual_identity = _factual_identity(source, asset_receipt, lock, tok_receipt)
        with torch.inference_mode():
            if args.dataset == "zsre":
                zsre_reference = module.build_zsre_w0_reference(
                    model, tok, records, identity=factual_identity, batch_size=16)
                observed = zsre_reference["evaluation"]
            else:
                zsre_reference = None
                observed = _evaluate_factual(module, model, tok, records, args.dataset,
                                             identity=factual_identity)
            generation = _generation(model, tok, records, assets, out, "W0",
                                     dict(source=source, stream_sha256=lock["stream_sha256"]),
                                     tracker) \
                if args.dataset == "cf" else None
        cases, summary = observed["cases"], observed["summary"]
        out.mkdir(parents=True, exist_ok=True)
        _write_once(out / "w0-cases.json", cases)
        if zsre_reference is not None:
            _write_once(out / "w0-zsre-reference.json", zsre_reference)
        reference_sha = _sha(out / "w0-zsre-reference.json") if zsre_reference else None
        receipt = dict(model="qwen25", dataset=args.dataset, endpoint="W0",
                       run_id=f"qwen25-{args.dataset}-W0", stream_sha256=lock["stream_sha256"],
                       code_commit=source["code_commit"],
                       official_tree_sha256=source["official_tree_sha256"],
                       tokenizer_sha256=tok_receipt["tokenizer_sha256"],
                       assets_sha256=asset_receipt["assets_sha256"],
                       factual=dict(summary=summary, accuracy=observed["accuracy"],
                                    work=observed["work"],
                                    observation_identity_sha256=observed["identity_sha256"],
                                    cases_sha256=_sha(out / "w0-cases.json"),
                                    w0_reference_sha256=reference_sha),
                       generation_rows_path=generation["rows_path"] if generation else None,
                       generation_identity_sha256=generation["identity_sha256"] if generation else None,
                       observed_requests=len(records), receipt_sha256=None)
        receipt["receipt_sha256"] = digest({k: v for k, v in receipt.items() if k != "receipt_sha256"})
        _write_once(out / "w0-receipt.json", receipt)
        scalars = _w0_scalar_values(args.dataset, cases, receipt["factual"],
                                    generation["summary"] if generation else None)
        _log_scalar_receipt(tracker, out, "w0", scalars, receipt["receipt_sha256"])
        return 0


def execute(args):
    import torch
    from official.baselines import registry
    from official.experiments import checkpoint
    from official.runners.server3.native_state import NativeState
    result, config, lock, records, asset_receipt, source = _preflight(args, require_frozen=True)
    if not result["ready"]:
        raise RuntimeError("RUN_PREFLIGHT_BLOCKED:" + ";".join(result["blockers"]))
    shared = Path(asset_receipt["storage"]["output_root"]) / "shared-w0" / \
             f"qwen25-{config['dataset']}"
    w0_ref = _verified_w0_receipt(shared, config["dataset"], lock, source, asset_receipt)
    if config["dataset"] == "zsre":
        reference_path = shared / "w0-zsre-reference.json"
        zsre_reference = _read(reference_path)
    else:
        zsre_reference = None
    module = _require_evaluator()
    out = Path(args.output)
    checkpoint_dir = out / "checkpoint"
    if not args.resume and checkpoint_dir.exists():
        raise ValueError("EXISTING_RUN_REQUIRES_RESUME_OR_NEW_OUTPUT")
    if args.resume and not (checkpoint_dir / "latest.json").exists():
        raise ValueError("RESUME_CHECKPOINT_MISSING")
    with _tracker(out, arm=config["run_id"], writer=config["method"],
                  dataset=config["dataset"], assets=asset_receipt["assets"],
                  source_sha=source["code_commit"],
                  config_sha=config["config_sha256"]) as tracker:
        _model_seed(config["edit_seed"])
        model, tok = _load_model(asset_receipt["assets"]["model_snapshot"]["path"],
                                 config["model_identity"]["revision"])
        tok_receipt = _tokenizer_receipt(asset_receipt["assets"]["model_snapshot"]["path"], tok, lock)
        if w0_ref.get("tokenizer_sha256") != tok_receipt["tokenizer_sha256"]:
            raise ValueError("SHARED_W0_TOKENIZER_IDENTITY")
        factual_identity = _factual_identity(source, asset_receipt, lock, tok_receipt)
        identity = checkpoint_identity(config, lock, source, asset_receipt, tok_receipt)
        hparams = registry.hparams(config["method"], "qwen25", overrides=config["hparams"])
        native = NativeState(config["method"], model, hparams, asset_receipt["assets"])
        names = native.editable_parameter_names()
        start = 0
        if args.resume:
            payload = checkpoint.load(checkpoint_dir, identity)
            if payload["method"] != config["method"]:
                raise ValueError("RESUME_METHOD_MISMATCH")
            native.restore_from_checkpoint(payload)
            start = payload["batch"]
            checkpoint_ref = _read(checkpoint_dir / "latest.json")
            _recover_batch_receipts(out, payload, checkpoint_ref, config, lock,
                                    source, identity)
            _replay_committed_batch_logging(out, payload, config, lock, source, identity,
                                            tracker)
            _write_once(out / f"resume-from-b{start:02d}.json",
                        dict(start_batch=start, checkpoint_sha256=_sha(
                             checkpoint_dir / _read(checkpoint_dir / "latest.json")["file"]),
                             config_sha256=config["config_sha256"],
                             stream_sha256=lock["stream_sha256"],
                             code_commit=source["code_commit"],
                             official_tree_sha256=source["official_tree_sha256"]))
        else:
            weights = _editable_weights(model, names)
            checkpoint.save(checkpoint_dir, batch=0, weights=weights,
                            cache_c=native.cache_for_checkpoint(),
                            contexts=native.context_snapshot(),
                            evaluation_cursor=dict(completed_batch=0, w0_receipt=str(shared / "w0-receipt.json"),
                                                   evaluated_endpoints=[0], per_request_records=[]),
                            identity=identity, method=config["method"], evaluation_complete=True)
        max_batch = args.stop_after_batch if args.stop_after_batch is not None else 20
        if not start <= max_batch <= 20:
            raise ValueError("BATCH_LIMIT_NOT_FORWARD")
        for batch in range(start + 1, max_batch + 1):
                before = time.monotonic()
                current = records[(batch - 1) * 100:batch * 100]
                with torch.enable_grad():
                    model = native.apply(tok, registry.requests(current, config["method"], "qwen25"))
                weights = _editable_weights(model, names)
                evaluated = batch in PHASES or (args.qualify_b3_metrics and batch == 3)
                factual = None
                generation = None
                if evaluated:
                    eval_records = current if batch == 3 else records[:batch * 100]
                    with torch.inference_mode():
                        observed = _evaluate_factual(module, model, tok, eval_records,
                                                     config["dataset"],
                                                     w0_reference=zsre_reference,
                                                     identity=factual_identity)
                        cases, summary = observed["cases"], observed["summary"]
                    evaldir = out / "evaluations"
                    evaldir.mkdir(parents=True, exist_ok=True)
                    case_path = evaldir / f"w{batch:02d}-cases.json"
                    _write_once(case_path, cases)
                    factual = dict(summary=summary, accuracy=observed["accuracy"],
                                   work=observed["work"],
                                   observation_identity_sha256=observed["identity_sha256"],
                                   cases_sha256=_sha(case_path),
                                   cases_path=str(case_path.resolve()))
                    if batch == 20 and config["dataset"] == "cf":
                        with torch.inference_mode():
                            generation = _generation(model, tok, records, asset_receipt["assets"],
                                                     out, "W20", dict(checkpoint_identity=identity,
                                                                       batch=batch,
                                                                       weight_hashes=_weight_hashes(weights)),
                                                     tracker, native)
                cursor = dict(completed_batch=batch, w0_receipt=str(shared / "w0-receipt.json"),
                              evaluated_endpoints=[n for n in PHASES if n <= batch],
                              per_request_records=[dict(occurrence_index=r["occurrence_index"],
                                                        case_id=r["case_id"]) for r in records[:batch * 100]])
                state_hashes = _state_hashes(weights, native, cursor) \
                    if args.state_hash_at_batch3 and batch == 3 else None
                pending = dict(batch=batch, config_sha256=config["config_sha256"],
                               stream_sha256=lock["stream_sha256"],
                               code_commit=source["code_commit"],
                               official_tree_sha256=source["official_tree_sha256"],
                               checkpoint_identity=identity, factual=factual,
                               generation=(dict(rows_path=generation["rows_path"],
                                                identity_sha256=generation["identity_sha256"])
                                           if generation else None),
                               placement=native.placement,
                               seconds=time.monotonic() - before,
                               state_hashes=state_hashes)
                from official.experiments.prepare import digest
                pending_sha = digest(pending)
                _write_once(out / "commits" /
                            f"pending-b{batch:02d}-{pending_sha}.json", pending)
                cursor["pending_receipt_sha256"] = pending_sha
                ck = checkpoint.save(checkpoint_dir, batch=batch, weights=weights,
                                     cache_c=native.cache_for_checkpoint(),
                                     contexts=native.context_snapshot(), evaluation_cursor=cursor,
                                     identity=identity, method=config["method"],
                                     evaluation_complete=True)
                _recover_batch_receipts(out,
                    dict(batch=batch, evaluation_cursor=cursor), ck,
                    config, lock, source, identity)
                if factual is not None:
                    scalars = _batch_scalar_values(config["dataset"], batch, cases,
                        factual, generation["summary"] if generation else None)
                    _log_scalar_receipt(tracker, out, f"w{batch:02d}", scalars,
                                        _sha(out / "commits" / f"b{batch:02d}.json"))
        if args.stop_after_batch == 1:
            _write_once(out / "smoke-receipt.json", dict(completed_batch=1,
                        run_id=config["run_id"], config_sha256=config["config_sha256"],
                        stream_sha256=lock["stream_sha256"], code_commit=source["code_commit"],
                        official_tree_sha256=source["official_tree_sha256"],
                        checkpoint_identity=identity))
        return 0


def qualify(args):
    """Actual target-model B3 continuity versus B2 checkpoint reload test.

    Each child starts a new Python/model process.  The stopped B2 path is
    deserialized in a third process for B3.  No qualification state is passed
    into a production chain; those start again from fresh W0.
    """
    config = validate_config(_read(args.config))
    if config["dataset"] != "cf" or config["method"] not in METHODS:
        raise ValueError("CF_NATIVE_QUALIFICATION_SCOPE")
    _, lock, _ = validate_stream(args.stream, "cf")
    out = Path(args.output)
    receipt_path = out / "resume-parity.json"
    if receipt_path.exists():
        receipt = _read(receipt_path)
        if receipt.get("status") != "PASS" or receipt.get("config_sha256") != config["config_sha256"] \
                or receipt.get("stream_sha256") != lock["stream_sha256"]:
            raise ValueError("QUALIFICATION_EXISTING_RECEIPT_MISMATCH")
        return 0
    base = [sys.executable, "-B", "-m", "official.runners.server3.run", "execute",
            "--config", str(args.config), "--assets", str(args.assets),
            "--stream", str(args.stream)]
    runs = (("continuous", ["--stop-after-batch", "3", "--qualify-b3-metrics",
                            "--state-hash-at-batch3"]),
            ("stopped", ["--stop-after-batch", "2"]),
            ("resumed", ["--resume", "--stop-after-batch", "3",
                          "--qualify-b3-metrics", "--state-hash-at-batch3"]))
    for label, extra in runs:
        path = out / ("continuous" if label == "continuous" else "resumed")
        env = dict(os.environ, ODEEDIT_ATTEMPT_ID="qual-" + config["method"].lower() + "-" + label)
        command = [*base, "--output", str(path), *extra]
        result = subprocess.run(command, cwd=ROOT, env=env, check=False)
        if result.returncode:
            raise RuntimeError(f"QUALIFICATION_CHILD_FAILED:{label}:{result.returncode}")
    continuous = _read(out / "continuous" / "commits" / "b03.json")
    resumed = _read(out / "resumed" / "commits" / "b03.json")
    reload_receipt = _read(out / "resumed" / "resume-from-b02.json")
    checks = dict(weight_history_context_rng_cursor_hashes_equal=(
                      continuous["state_hashes"] == resumed["state_hashes"]),
                  b3_raw_factual_equal=(continuous["factual_sha256"] == resumed["factual_sha256"]),
                  b3_summary_equal=(continuous["factual"]["summary"] == resumed["factual"]["summary"]),
                  identical_checkpoint_identity=(
                      continuous["checkpoint_identity"] == resumed["checkpoint_identity"]),
                  reloaded_exact_b2_checkpoint=(
                      reload_receipt["start_batch"] == 2 and
                      reload_receipt["checkpoint_sha256"] ==
                      _read(out / "resumed" / "commits" / "b02.json")["checkpoint_sha256"]))
    if not all(checks.values()):
        raise RuntimeError("B3_RESUME_PARITY_FAILED:" + json.dumps(checks, sort_keys=True))
    receipt = dict(status="PASS", method=config["method"], run_id=config["run_id"],
                   config_sha256=config["config_sha256"], stream_sha256=lock["stream_sha256"],
                   code_commit=continuous["code_commit"],
                   official_tree_sha256=continuous["official_tree_sha256"],
                   completed_batch=3, checks=checks,
                   state_hashes=continuous["state_hashes"],
                   factual_sha256=continuous["factual_sha256"],
                   qualification_cost="THREE_COLD_PROCESS_STARTS_SIX_EDIT_BATCH_COMPUTATIONS",
                   W20_science_completion=False)
    _write_once(receipt_path, receipt)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "execute", "w0", "qualify"):
        sub = commands.add_parser(command)
        if command != "w0":
            sub.add_argument("--config", type=Path, required=True)
        else:
            sub.add_argument("--dataset", choices=("cf", "zsre"), required=True)
        sub.add_argument("--assets", type=Path, required=True)
        sub.add_argument("--stream", type=Path, required=True)
        sub.add_argument("--output", type=Path, required=True)
        if command == "execute":
            sub.add_argument("--resume", action="store_true")
            sub.add_argument("--stop-after-batch", type=int)
            sub.add_argument("--qualify-b3-metrics", action="store_true")
            sub.add_argument("--state-hash-at-batch3", action="store_true")
        if command == "preflight":
            sub.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "preflight":
        return preflight(args)
    if args.command == "w0":
        return w0(args)
    if args.command == "qualify":
        return qualify(args)
    return execute(args)


if __name__ == "__main__":
    sys.exit(main())
