"""rent official baseline runner: W0, actual GPU qualification, 2K chains, zsRE smoke.

Adapted from server2's GPT-J runner for all three models. Algorithms, hparams,
factual evaluation, checkpointing and W&B transport are the shared ``official``
modules. FLU/CON is deferred (``DEFERRED_CHECKPOINT_EVALUATION``): CF chains
keep the final W20 checkpoint for a later evaluation run and log no generation.
"""
import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from official.baselines import registry
from official.evaluation import reduce as factual_reduce
from official.experiments import checkpoint
from official.experiments.prepare import ROOT, build_matrix, digest, file_sha, load_plan, read, write_new
from official.runners.rent.blue_selection import selected_configuration
from official.runners.rent.freeze import tree
from official.runners.rent.native import METHODS, NativeEngine, model_spec

INSTRUCTION = "USER-OFFICIAL-BASELINES-20261008-R1"
TASK = "official-baselines-20261008"
MODES = ("w0", "qualification", "chain", "smoke")


def require(value, code):
    if not value:
        raise ValueError(code)


def member(path):
    path = Path(path).resolve()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=file_sha(path))


def configuration(run_id):
    contract, profiles = load_plan()
    rows = [row for row in build_matrix(contract, profiles) if row["run_id"] == run_id]
    require(len(rows) == 1 and rows[0]["method"] in METHODS, "OFFICIAL_RUN_ID")
    return rows[0]


def native_hparams(config):
    return registry.hparams(config["method"], config["model"], overrides=config["hparams"])


def checkpoint_identity(manifest, config):
    return dict(config_sha256=config["config_sha256"],
                stream_sha256=manifest["streams"][config["dataset"]]["lock"]["stream_sha256"],
                code_commit=manifest["code_commit"], official_tree_sha256=manifest["official_tree_sha256"],
                model_revision=manifest["model_revision"], tokenizer_sha256=manifest["tokenizer_sha256"],
                assets_sha256=manifest["assets_identity_sha256"])


def equal(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(left, torch.Tensor):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return left.dtype == right.dtype and np.array_equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
    return left == right


def tensor_hash(value):
    array = value.detach().to("cpu").contiguous().numpy()
    sha = hashlib.sha256()
    sha.update(str(array.dtype).encode())
    sha.update(json.dumps(list(array.shape)).encode())
    sha.update(memoryview(array).cast("B"))
    return sha.hexdigest()


def signature(model, engine=None):
    values = {}
    for kind, iterator in (("parameter", model.named_parameters()), ("buffer", model.named_buffers())):
        for name, value in iterator:
            values[kind + ":" + name] = (value.data_ptr(), value._version,
                list(value.shape), str(value.dtype), str(value.device), value.requires_grad)
    modules = {name: dict(training=module.training,
        hooks={key: list(getattr(module, key).keys()) for key in
               ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks")})
        for name, module in model.named_modules()}
    return dict(tensors=values, modules=modules, config=model.config.to_dict(),
                native=None if engine is None else engine.state_identity())


def verify_manifest(manifest, path):
    """Frozen source, manifest location and scientific runtime must all match."""
    run_root = Path(manifest["run_root"])
    require(manifest["schema"] == "official-rent-execution-v1" and manifest["server"] == "rent"
            and manifest["instruction_id"] == INSTRUCTION
            and Path(path).resolve() == (run_root / f"manifest-{manifest['model']}.json").resolve(),
            "OFFICIAL_RENT_EXECUTION_IDENTITY")
    require(ROOT.resolve() == (run_root / "source/official").resolve(), "RUN_FROM_FROZEN_SOURCE_ONLY")
    frozen = read(run_root / "source.json")
    require(frozen["code_commit"] == manifest["code_commit"]
            and tree(run_root / "source") == (manifest["official_tree_sha256"], frozen["members"]),
            "OFFICIAL_IMPORTED_SOURCE_BYTES_CHANGED")
    import transformers
    runtime = manifest["runtime"]
    require(torch.__version__ == runtime["torch"] and transformers.__version__ == runtime["transformers"],
            "OFFICIAL_SCIENTIFIC_RUNTIME_CHANGED")
    for name, row in manifest["model_files"].items():
        target = Path(manifest["model_snapshot"]) / name
        require(target.resolve().stat().st_size == row["bytes"], "MODEL_FILE_CHANGED: " + name)
    return True


def tracking_config(manifest, config, mode, out):
    """Shared official schema; CF generation is explicitly deferred."""
    spec = model_spec(config["model"])
    prefix = dict(w0="", qualification="QUALIFICATION_", smoke="ZSRE_SMOKE_", chain="")[mode]
    values = dict(server="rent", task_id=TASK,
                  arm="W0_BASE_MODEL" if mode == "w0" else prefix + config["run_id"],
                  attempt=out.name, source_sha=manifest["code_commit"], config_sha=config["config_sha256"],
                  model=config["model"], model_family=spec["family"],
                  writer="none" if mode == "w0" else config["method"], role="scientific",
                  metric_schema="official-baselines-scalar-v1",
                  baseline="none" if mode == "w0" else config["method"],
                  instruction_id=INSTRUCTION, dataset=config["dataset"])
    if config["dataset"] == "cf":
        values["generation_schedule"] = "DEFERRED_CHECKPOINT_EVALUATION"
    return values


class TransportAudit:
    """Receipt-only adapter around the common transport, not another logger."""
    def __init__(self, common, out, config_values=None):
        self.common, self.out, self.config_values = common, out, config_values
        self.accepted = self.rejected = self.errors = 0
        self.finished, self.finish_status = False, "NOT_FINISHED"

    def log(self, values):
        try:
            result = self.common.log(values)
        except Exception:
            self.errors += 1
            return False
        if result is False:
            self.rejected += 1
        else:
            self.accepted += 1
        return result

    def finish(self, exit_code=0):
        try:
            result = self.common.finish(exit_code=exit_code)
            self.finish_status = "BOUNDED_COMMON_FINISH_RETURNED_NOT_SCIENTIFIC_COMPLETION"
            return result
        except Exception:
            self.errors += 1
            self.finish_status = "LOGGING_FINISH_FAILED_SANITIZED"
        finally:
            self.finished = True
            write_new(self.out / "logging-transport.json", self.receipt())

    def receipt(self):
        return dict(accepted_calls=self.accepted, rejected_calls=self.rejected,
                    sanitized_errors=self.errors, bounded_finish=self.finished,
                    finish_status=self.finish_status,
                    status="LOGGING_DEGRADED" if self.rejected or self.errors else "SDK_ACCEPTED_NOT_REMOTE_ACK",
                    common_spool=str(self.out / "tracking"), no_scientific_retry=True)


def tracker(manifest, config, mode, out):
    binding = manifest["tracking"]
    require(binding.get("namespace") == "official.tracking", "OFFICIAL_TRACKING_SHARED_API_NOT_BOUND")
    module = importlib.import_module(binding["namespace"])
    require(file_sha(module.__file__) == binding["source_sha256"], "OFFICIAL_TRACKING_SOURCE_CHANGED")
    values = tracking_config(manifest, config, mode, out)
    # The common helper binds the rent DDHHMM job number from the Job env and
    # reads the existing credential; no key or full environment is copied.
    common = module.init(env_file=binding["env_file"], spool=str(out / "tracking"), config=values)
    return TransportAudit(common, out, values)


def log(tracking, values):
    accepted = tracking.log(values)
    # Raw results stay authoritative; degraded transport never triggers a retry.
    return "LOGGING_DEGRADED_NOT_REMOTE_ACK" if accepted is False else "SDK_ASYNC_NOT_REMOTE_ACK"


def load_model(manifest):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "ONE_ALLOCATED_GPU_REQUIRED")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    tok = AutoTokenizer.from_pretrained(manifest["model_snapshot"], local_files_only=True, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(manifest["model_snapshot"], local_files_only=True,
        torch_dtype=torch.float32, attn_implementation="eager").to("cuda:0").eval()
    require(all(value.dtype == torch.float32 for value in model.parameters()), "OFFICIAL_FP32_MODEL")
    return model, tok


def factual_identity(manifest, dataset):
    return dict(model_revision=manifest["model_revision"], tokenizer_sha256=manifest["tokenizer_sha256"],
                stream_sha256=manifest["streams"][dataset]["lock"]["stream_sha256"],
                runtime=manifest["runtime"], source=manifest["code_commit"])


def factual(model, tok, manifest, records, dataset, out, endpoint, engine=None, w0_reference=None):
    api = importlib.import_module("official.evaluation.factual")
    before, rng = signature(model, engine), checkpoint.rng_snapshot()
    observed = api.evaluate(model, tok, records, dataset, w0_reference=w0_reference, batch_size=16,
                            device="cuda:0", identity=factual_identity(manifest, dataset))
    require(signature(model, engine) == before and equal(checkpoint.rng_snapshot(), rng),
            "OFFICIAL_FACTUAL_NATIVE_STATE_RNG_MUTATED")
    require(len(observed["cases"]) == len(records)
            and [case["occurrence_index"] for case in observed["cases"]]
            == [record["occurrence_index"] for record in records], "OFFICIAL_FACTUAL_COHORT_IDENTITY")
    value = dict(observed, endpoint=endpoint, dataset=dataset, requests=len(records),
                 observer_no_mutation=True, RNG_restored=True)
    path = out / "factual" / (endpoint + ".json")
    write_new(path, value)
    return value, member(path)


def w0_reference(model, tok, records, manifest, out):
    api = importlib.import_module("official.evaluation.factual")
    before, rng = signature(model), checkpoint.rng_snapshot()
    value = api.build_zsre_w0_reference(model, tok, records, identity=factual_identity(manifest, "zsre"),
                                        device="cuda:0", batch_size=16)
    require(signature(model) == before and equal(checkpoint.rng_snapshot(), rng),
            "OFFICIAL_ZSRE_W0_STATE_RNG_MUTATED")
    write_new(out / "W0-reference.json", value)
    return value


def save(engine, out, identity, evaluations, checkpoint_folder=None):
    return checkpoint.save(checkpoint_folder or out / "checkpoints", batch=engine.batch,
        weights=engine.weights(), cache_c=engine.history(), contexts=engine.contexts(False),
        evaluation_cursor=evaluations, identity=identity, method=engine.method, evaluation_complete=True)


def evaluate_payload(observed, endpoint, edits, *, current=False, config_values=None):
    # Request-macro E/G/S under the shared official whitelist; legacy prompt-pair
    # diagnostics keep their own namespace.
    prefix = "W0_first2000" if endpoint == "W0" else "current/post" if current else "all_seen/post"
    dataset = observed.get("dataset", observed.get("identity", {}).get("dataset"))
    if dataset == "zsre":
        from official.tracking import official_zsre_metrics
        return official_zsre_metrics(observed["summary"], config_values=config_values, endpoint=prefix,
                                     edits=edits, pre_state_edits=max(0, edits - 100), post_state_edits=edits)
    fields = ("Efficacy", "Generalization", "Specificity", "Score", "Score_AlphaEdit_display", "requests")
    value = {"official/" + prefix + "/" + key: score for key, score in observed["summary"].items()
             if key in fields and type(score) in (int, float)}
    if "Score_AlphaEdit_display" in observed["summary"]:
        for kind, label in (("rewrite", "Efficacy"), ("paraphrase", "Generalization"),
                            ("neighborhood", "Specificity")):
            rates = []
            for case in observed["cases"]:
                bits = [float(row["target_true"]["mean_nll"] < row["target_new"]["mean_nll"]
                        if kind == "neighborhood" else
                        row["target_new"]["mean_nll"] < row["target_true"]["mean_nll"])
                        for row in case[kind + "_observations"]]
                require(bool(bits), "DISPLAY_EMPTY_REQUEST")
                rates.append(np.mean(bits))
            value["official/" + prefix + "/" + label + "_AlphaEdit_display"] = float(np.around(np.mean(rates) * 100, 2))
    value.update(cf_diagnostics(observed["cases"], prefix))
    return dict(value, edits=edits, pre_state_edits=max(0, edits - 100), post_state_edits=edits)


def cf_diagnostics(cases, prefix):
    """Already measured native prompt-pair/token rows, not request-macro aliases."""
    value, successes = {}, []
    for kind, label in (("rewrite", "R"), ("paraphrase", "P"), ("neighborhood", "N")):
        rows = [row for case in cases for row in case[kind + "_observations"]]
        require(bool(rows), "EMPTY_CF_DIAGNOSTICS")
        desired = [row["target_true" if kind == "neighborhood" else "target_new"] for row in rows]
        require(all(row["desired_target"] == ("true" if kind == "neighborhood" else "new") for row in rows),
                "CF_DIAGNOSTICS_DESIRED_TARGET")
        count = len(rows)
        success = sum((row["target_true"]["mean_nll"] < row["target_new"]["mean_nll"])
                      if kind == "neighborhood" else
                      (row["target_new"]["mean_nll"] < row["target_true"]["mean_nll"]) for row in rows)
        token_count = sum(row["token_count"] for row in desired)
        true_nll = math.fsum(row["target_true"]["mean_nll"] for row in rows) / count
        new_nll = math.fsum(row["target_new"]["mean_nll"] for row in rows) / count
        item = dict(count=count, success_count=success, success_pct=100 * success / count,
                    token_acc_pct=100 * sum(row["token_correct_count"] for row in desired) / token_count,
                    prompt_acc_pct=100 * math.fsum(row["token_correct_count"] / row["token_count"]
                                                   for row in desired) / count,
                    strict_acc_pct=100 * sum(row["strict_correct"] for row in desired) / count,
                    true_nll=true_nll, new_nll=new_nll, margin_true_minus_new=true_nll - new_nll)
        value.update({prefix + "/" + label + "/" + key: scalar for key, scalar in item.items()})
        successes.append(item["success_pct"])
    value[prefix + "/success_harmonic_pct"] = factual_reduce.harmonic(successes)
    return value


def current_subset(observed, records, dataset):
    """Last incoming 100 rows of the same actual milestone, no extra forward."""
    require(len(records) == 100, "CURRENT_COHORT_BATCH100")
    cases = observed["cases"][-100:]
    require([(row["case_id"], row["occurrence_index"]) for row in cases]
            == [(row["case_id"], row["occurrence_index"]) for row in records], "CURRENT_SUBSET_ORDERED_OCCURRENCE")
    summary = factual_reduce.counterfact(cases) if dataset == "cf" else factual_reduce.zsre(cases)
    return dict(cases=cases, summary=summary, dataset=dataset,
                parent_identity_sha256=observed["identity_sha256"], work=dict(new_model_forward_calls=0))


def qualification(model, tok, engine, manifest, config, records, out, tracking):
    """Same native continuous B3 versus actual durable B2 -> new restore -> B3."""
    require(engine.batch == 0 and config["dataset"] == "cf", "QUALIFICATION_COLD_CF_REQUIRED")
    identity = checkpoint_identity(manifest, config)
    engine.contexts()
    cursor, commits = {}, []
    save(engine, out, identity, cursor)
    for batch in (1, 2):
        receipt = engine.apply(records[(batch - 1) * 100:batch * 100], batch)
        log(tracking, dict(batch=batch, edits=batch * 100))
        _, observed_member = factual(model, tok, manifest, records[:batch * 100], "cf", out,
                                     "qualification-W" + str(batch), engine)
        cursor[str(batch)] = observed_member
        commits.append(dict(receipt, checkpoint=save(engine, out, identity, cursor)))
    durable_B2 = read(out / "checkpoints/latest.json")
    receipt3 = engine.apply(records[200:300], 3)
    continuous, continuous_member = factual(model, tok, manifest, records[200:300], "cf", out,
                                            "continuous-B3", engine)
    continuous_weights = {key: value.detach().cpu().clone() for key, value in engine.weights().items()}
    continuous_history = {key: value.clone() for key, value in engine.history().items()}
    continuous_contexts, continuous_rng = engine.contexts(False), checkpoint.rng_snapshot()
    # Load the real fsync+SHA-bound B2 payload; nothing is inferred from raw
    # observations, and native target fits/FT optimizer restart naturally at B3.
    saved = checkpoint.load(out / "checkpoints", identity)
    require(saved["batch"] == 2, "QUALIFICATION_DURABLE_B2_REQUIRED")
    engine.restore(saved)
    checkpoint.rng_restore(saved["rng"])
    del saved
    resumed_receipt = engine.apply(records[200:300], 3)
    resumed, resumed_member = factual(model, tok, manifest, records[200:300], "cf", out, "resumed-B3", engine)
    flags = dict(weights_equal=equal(continuous_weights,
                                     {key: value.detach().cpu() for key, value in engine.weights().items()}),
                 history_equal=equal(continuous_history, engine.history()),
                 contexts_equal=equal(continuous_contexts, engine.contexts(False)),
                 rng_equal=equal(continuous_rng, checkpoint.rng_snapshot()),
                 metrics_equal=equal(continuous["cases"], resumed["cases"])
                 and equal(continuous["summary"], resumed["summary"]))
    value = dict(status="PASS_ACTUAL_QUALIFICATION" if all(flags.values()) else "FAIL_ACTUAL_QUALIFICATION",
                 actual_GPU=True, server="rent", model=config["model"], dataset="cf", run_id=config["run_id"],
                 method=engine.method, code_commit=manifest["code_commit"],
                 official_tree_sha256=manifest["official_tree_sha256"], checkpoint_identity=identity,
                 continuous_batches=3, resume_after_batch=2, resumed_batches=[3], **flags,
                 weight_sha256=dict(continuous={k: tensor_hash(v) for k, v in continuous_weights.items()},
                                    resumed={k: tensor_hash(v) for k, v in engine.weights().items()}),
                 continuous_summary=continuous["summary"], resumed_summary=resumed["summary"],
                 durable_B2=durable_B2, continuous_metric=continuous_member, resumed_metric=resumed_member,
                 native_commits=commits + [receipt3, resumed_receipt], actual_native_batch_calls=4,
                 hparams_deployment_overrides=engine.hparams_overrides,
                 CPU_fixture_is_not_actual_proof=True)
    write_new(out / "qualification.json", value)
    require(all(flags.values()), "ACTUAL_B2_B3_NATIVE_RESUME_PARITY_FAILED")
    return value


def w0_folder(manifest, dataset):
    return Path(manifest["run_root"]) / "w0" / f"{manifest['model']}-{dataset}"


def cold_w0(model, tok, manifest, records, dataset, out, tracking):
    if dataset == "zsre":
        observed = w0_reference(model, tok, records, manifest, out)["evaluation"]
        write_new(out / "factual/W0.json", dict(observed, endpoint="W0", dataset=dataset, requests=len(records)))
        factual_member = member(out / "factual/W0.json")
    else:
        observed, factual_member = factual(model, tok, manifest, records, dataset, out, "W0")
    log(tracking, evaluate_payload(observed, "W0", 0, config_values=tracking.config_values))
    ready = dict(status="READY_COLD_W0_COMPLETE", actual_GPU=True, server="rent", model=manifest["model"],
                 dataset=dataset, code_commit=manifest["code_commit"],
                 official_tree_sha256=manifest["official_tree_sha256"], model_revision=manifest["model_revision"],
                 tokenizer_sha256=manifest["tokenizer_sha256"],
                 stream_sha256=manifest["streams"][dataset]["lock"]["stream_sha256"], factual=factual_member,
                 generation_status="DEFERRED_NOT_MEASURED" if dataset == "cf" else None,
                 w0_reference=member(out / "W0-reference.json") if dataset == "zsre" else None)
    write_new(out / "READY.json", ready)
    write_new(w0_folder(manifest, dataset) / "READY.json", ready)
    return ready


def read_w0(manifest, dataset):
    ready = read(w0_folder(manifest, dataset) / "READY.json")
    require(ready["status"] == "READY_COLD_W0_COMPLETE" and ready["actual_GPU"] is True
            and all(ready[key] == manifest[key] for key in
                    ("model_revision", "tokenizer_sha256", "code_commit", "official_tree_sha256"))
            and ready["stream_sha256"] == manifest["streams"][dataset]["lock"]["stream_sha256"],
            "OFFICIAL_W0_READY_IDENTITY")
    require(member(ready["factual"]["path"]) == ready["factual"], "OFFICIAL_W0_FACTUAL_RAW_CHANGED")
    reference = None
    if dataset == "zsre":
        require(member(ready["w0_reference"]["path"]) == ready["w0_reference"], "OFFICIAL_ZSRE_W0_RAW_CHANGED")
        reference = read(ready["w0_reference"]["path"])
    return ready, reference


def chain(model, tok, engine, manifest, config, records, out, tracking, resume=None, smoke=False):
    dataset = config["dataset"]
    identity = checkpoint_identity(manifest, config)
    endpoints, cursor, commits = {}, {}, []
    ready, reference = read_w0(manifest, dataset)
    endpoints["W0"] = ready["factual"]
    checkpoint_folder = Path(resume).resolve() if resume else out / "checkpoints"
    require(checkpoint_folder.name == "checkpoints" and Path(manifest["run_root"]).resolve()
            in checkpoint_folder.parents and not checkpoint_folder.is_symlink(), "OWN_TASK_CHECKPOINT_FOLDER_SCOPE")
    if resume:
        saved = checkpoint.load(checkpoint_folder, identity)
        engine.restore(saved)
        checkpoint.rng_restore(saved["rng"])
        cursor = saved["evaluation_cursor"]
        endpoints.update({key: value for key, value in cursor.items() if key in ("W5", "W10", "W15", "W20")})
        commits = list(cursor.get("commit_members", []))
        current = Path(cursor["current_commit_path"]) if cursor.get("current_commit_path") else None
        if current is not None:
            if not current.exists():
                # The durable checkpoint carries the completed native receipt;
                # recover only this metadata if the process died before writing it.
                receipt = cursor["current_commit_receipt"]
                require(receipt["batch"] == saved["batch"], "CHECKPOINT_COMMIT_CURSOR_BATCH")
                write_new(current, dict(receipt, checkpoint=read(checkpoint_folder / "latest.json"),
                                        metadata_recovered_from_durable_checkpoint=True))
            require(read(current)["batch"] == saved["batch"], "RESUME_LAST_COMMIT_BATCH")
            commits.append(member(current))
        require(len(commits) == saved["batch"], "RESUME_COMPLETED_COMMIT_PROVENANCE_REQUIRED")
        del saved
    else:
        engine.contexts()
        cursor["W0"] = ready["factual"]
        save(engine, out, identity, cursor)
    commit_root = checkpoint_folder.parent / "commits"
    end = 1 if smoke else 20
    for batch in range(engine.batch + 1, end + 1):
        previous, started = engine.batch, time.monotonic()
        receipt = engine.apply(records[(batch - 1) * 100:batch * 100], batch)
        log(tracking, {"batch": batch, "edits": batch * 100, "time/phase_seconds": time.monotonic() - started})
        if batch in (5, 10, 15, 20) or smoke:
            observed, observed_member = factual(model, tok, manifest, records[:batch * 100], dataset, out,
                                                "W" + str(batch), engine, reference)
            endpoints["W" + str(batch)] = cursor["W" + str(batch)] = observed_member
            if not smoke:
                log(tracking, evaluate_payload(observed, "W" + str(batch), batch * 100,
                                               config_values=tracking.config_values))
            incoming = current_subset(observed, records[(batch - 1) * 100:batch * 100], dataset)
            log(tracking, evaluate_payload(incoming, "W" + str(batch), batch * 100, current=True,
                                           config_values=tracking.config_values))
        path = commit_root / ("batch-" + str(batch).zfill(2) + ".json")
        cursor["commit_members"] = list(commits)
        cursor["current_commit_path"] = str(path)
        current_receipt = dict(receipt, previous_batch=previous, checkpoint_identity=identity,
                               native_state_identity=engine.state_identity(),
                               factual_endpoint=endpoints.get("W" + str(batch)))
        cursor["current_commit_receipt"] = current_receipt
        ref = save(engine, out, identity, cursor, checkpoint_folder)
        write_new(path, dict(current_receipt, checkpoint=ref))
        commits.append(member(path))
    common = dict(actual_GPU=True, server="rent", model=config["model"], dataset=dataset, run_id=config["run_id"],
                  method=engine.method, code_commit=manifest["code_commit"],
                  official_tree_sha256=manifest["official_tree_sha256"], checkpoint_identity=identity,
                  factual_endpoints=endpoints, commits=commits)
    if smoke:
        value = dict(common, status="PASS_ACTUAL_SMOKE", batches=1)
        write_new(out / "smoke.json", value)
        return value
    value = dict(common, status="SCIENTIFIC_COMPLETE", batches=20, checkpoint_folder=str(checkpoint_folder),
                 W0_READY=member(w0_folder(manifest, dataset) / "READY.json"),
                 generation_status="DEFERRED_NOT_MEASURED" if dataset == "cf" else None,
                 completion_scope="EDIT_FACTUAL_CHECKPOINT_ONLY" if dataset == "cf" else "FULL_SCHEDULE",
                 checkpoint_evaluation_consumer_pending=dataset == "cf", native_only=True)
    write_new(out / "result.json", value)
    return value


def output_folder(manifest, config, mode, job):
    name = "W0" if mode == "w0" else config["run_id"]
    return Path(manifest["run_root"]) / "runs" / config["model"] / name / f"{mode}-job{job}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-id", required=True, help="official matrix run_id, e.g. llama3-cf-memit")
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--resume", help="checkpoints folder of an earlier attempt of the same run")
    parser.add_argument("--selected-config", type=Path,
                        help="derived zsRE BLUE config written by blue_selection (CF grid winner L2) with its proof")
    args = parser.parse_args()
    import os
    job = os.environ.get("ODEEDIT_RENT_JOB_ID", "local")
    manifest, config = read(args.manifest), configuration(args.run_id)
    if args.selected_config is not None:
        # Only the zsRE BLUE row may be replaced, and only by the proven CF grid winner (blue_selection).
        config = selected_configuration(args.selected_config)
        require(config["run_id"] == args.run_id, "SELECTED_CONFIG_RUN_ID")
    require(config["model"] == manifest["model"], "RUN_ID_MODEL_MANIFEST_MISMATCH")
    out = output_folder(manifest, config, args.mode, job)
    require(not out.exists(), "ATTEMPT_ALREADY_EXECUTED_USE_NEW_JOB")
    out.mkdir(parents=True)
    started, tracking, result, error, engine = time.monotonic(), None, None, None, None
    try:
        verify_manifest(manifest, args.manifest)
        require(args.mode in ("w0", "qualification") or manifest["on_main"], "SCIENTIFIC_RUN_REQUIRES_MAIN_COMMIT")
        require(args.mode != "qualification" or (config["dataset"] == "cf" and not args.resume),
                "ACTUAL_QUALIFICATION_COLD_CF_ONLY")
        require(args.mode != "smoke" or config["dataset"] == "zsre", "ZSRE_SMOKE_ONLY")
        require(args.mode == "chain" or not args.resume, "RESUME_CHAIN_ONLY")
        records = read(manifest["streams"][config["dataset"]]["path"])
        require(len(records) == 2000 and [r["occurrence_index"] for r in records] == list(range(1, 2001)),
                "OFFICIAL_ORDERED_STREAM_2000")
        importlib.import_module("official.evaluation.factual")
        hparams = None if args.mode == "w0" else native_hparams(config)
        tracking = tracker(manifest, config, args.mode, out)
        model, tok = load_model(manifest)
        if args.mode == "w0":
            result = cold_w0(model, tok, manifest, records, config["dataset"], out, tracking)
        else:
            engine = NativeEngine(model, tok, config["method"], manifest, hparams)
            if args.mode == "qualification":
                result = qualification(model, tok, engine, manifest, config, records, out, tracking)
            else:
                result = chain(model, tok, engine, manifest, config, records, out, tracking,
                               resume=args.resume, smoke=args.mode == "smoke")
    except BaseException as exc:
        error = exc
        write_new(out / "failure.json", dict(status="TECHNICAL_FAILURE", exception_type=type(exc).__name__,
            error_code=str(exc)[:160] if isinstance(exc, ValueError) else "EXCEPTION_DETAILS_LOCAL_STDERR",
            elapsed_seconds=time.monotonic() - started,
            completed_native_batches=None if engine is None else engine.batch, automatic_retry=False))
        raise
    finally:
        if tracking is not None:
            try:
                tracking.finish(exit_code=1 if error else 0)
            except Exception:
                pass  # Never hide the scientific exception or block result persistence.
        write_new(out / "terminal.json", dict(status="FAILED" if error else "COMPLETED",
            scientific_status=None if result is None else result["status"],
            elapsed_seconds=time.monotonic() - started, actual_GPU=torch.cuda.is_initialized(),
            rent_job_id=job, tracking=None if tracking is None else tracking.receipt(),
            checkpoint_policy="LATEST1_FINAL_W20", automatic_retry=False))


if __name__ == "__main__":
    main()
