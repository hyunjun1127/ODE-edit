"""Server1 official-only bindings; no downloads or silently substituted inputs."""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import time
import uuid

from official.experiments.prepare import digest, file_sha, load_plan, write_new

METHODS = ("FT", "MEMIT", "MEMIT_FE")
LOCAL_ROOT = Path("/mnt/raid5/janghj/ODE-edit/local/official-baselines/server1")
MILESTONES = (5, 10, 15, 20)
DEFERRED_W20 = "DEFERRED_TO_SAVED_W20_CHECKPOINT"
CF_CHECKPOINT_AUTHORITY = "USER-DIRECT-SERVER1-CF-CHECKPOINT-20261009"


def generation_at_W20(config):
    policy = config.get("cf_W20_generation", "INLINE")
    require(policy in ("INLINE", DEFERRED_W20), "UNKNOWN_W20_GENERATION_POLICY")
    if policy == DEFERRED_W20:
        require(config.get("dataset") == "cf" and
                config.get("scope_override") == CF_CHECKPOINT_AUTHORITY,
                "DEFERRED_W20_EXPLICIT_CF_AUTHORITY_REQUIRED")
    return policy == "INLINE"


def require(value, code):
    if not value:
        raise ValueError(code)


def read(path):
    return json.loads(Path(path).read_text())


def member(path):
    path = Path(path).absolute()
    require(path.is_file() and not path.is_symlink(), "REGULAR_LOCAL_MEMBER_REQUIRED")
    before = path.stat()
    checksum = file_sha(path)
    after = path.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), "MEMBER_CHANGED_DURING_HASH")
    return dict(path=str(path), bytes=after.st_size, sha256=checksum)


def verify(row):
    actual = member(row["path"])
    require(all(actual[key] == row[key] for key in actual), "IMMUTABLE_INPUT_MEMBER_CHANGED")
    return Path(row["path"])


def local_output(path):
    path = Path(path).absolute()
    require(path.is_relative_to(LOCAL_ROOT) and not path.is_symlink(), "OWN_IGNORED_OUTPUT_REQUIRED")
    # A symlink ancestor must not redirect checkpoints or private raw elsewhere.
    for parent in (path, *path.parents):
        if parent == LOCAL_ROOT.parent:
            break
        require(not parent.is_symlink(), "LOCAL_OUTPUT_SYMLINK_ANCESTOR")
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_config(value):
    require(value.get("schema") == "official-server1-runtime-v1", "RUNTIME_CONFIG_SCHEMA")
    require(value.get("model") == "llama3" and value.get("method") in METHODS
            and value.get("dataset") in ("cf", "zsre"), "SERVER1_OFFICIAL_SCOPE")
    require(value.get("stream") == dict(requests=2000, batch_size=100, batches=20,
            order="existing_file_first2000", shuffle=False), "FIXED_NATIVE_STREAM")
    contract, profiles = load_plan()
    expected = profiles["llama3/" + value["method"]]["hparams"]
    require(value.get("hparams") == expected and value.get("edit_seed") == contract["edit_seed"] == 0,
            "NATIVE_HPARAMS_OR_SEED_CHANGED")
    import copy
    evaluation = copy.deepcopy(contract["evaluation"])
    if not generation_at_W20(value):
        evaluation["generation"]["edited_endpoints"] = []
        evaluation["generation"]["deferred_to_checkpoint"] = 20
    require(value.get("precision") == contract["precision"] and
            value.get("evaluation") == evaluation, "OFFICIAL_PRECISION_OR_EVALUATION_CHANGED")
    unsigned = {key: item for key, item in value.items() if key != "config_sha256"}
    require(value.get("config_sha256") == digest(unsigned), "CONFIG_DIGEST")
    from .native_parity import PLAN as native_reference_plan
    require(value.get("qualification_plan") == dict(schema="official-native-resume-plan-v1",
            batches=[1, 2, 3], external_batch_size=100, split_after_batch=2,
            selected_weight_comparison="EXACT_SHA256", RNG_comparison="EXACT",
            context_comparison="EXACT", factual_comparison="EXACT_VALUES_WITHOUT_ELAPSED_WORK",
            extra_generation=False, extra_science_chain=False,
            native_reference_matched_B3=native_reference_plan), "QUALIFICATION_PLAN_CHANGED")
    return value


def source_binding(path, config_path):
    lock = read(path)
    require(lock.get("schema") == "official-server1-execution-lock-v1"
            and lock.get("server") == "server1", "EXECUTION_LOCK_SCHEMA")
    require(lock["frozen_source"].get("main_membership_verified") is True
            and lock["frozen_source"].get("official_tree_verified") is True, "REVIEWED_MAIN_SOURCE_REQUIRED")
    root = Path(lock["frozen_source"]["directory"]).resolve()
    require(Path(__file__).resolve().is_relative_to(root / "official"), "NOT_FROZEN_OFFICIAL_RUNTIME")
    for row in lock["frozen_source"]["members"]:
        require(Path(row["path"]).is_relative_to(root / "official"), "EXTERNAL_RUNTIME_MEMBER")
        verify(row)
    expected = member(config_path)
    require(expected in lock["job_configs"], "CONFIG_NOT_IN_EXECUTION_LOCK")
    for row in lock["inputs"]:
        verify(row)
    require(os.environ.get("OFFICIAL_CODE_COMMIT") == lock["source"]["main_commit"]
            and os.environ.get("OFFICIAL_TREE_SHA256") == lock["source"]["official_tree"],
            "SOURCE_ENV_LOCK_CONFLICT")
    return lock


def bindings(config, lock):
    from .assets import verify_manifest
    assets = read(verify(config["assets_member"]))
    # The execution archive has already been checked member-by-member by
    # source_binding(). Preparation-source paths are historical provenance,
    # not a request to execute a future mutable worktree's bytes.
    verify_manifest(assets, verify_preparation_source=False)
    bundle = read(verify(config["stream_bundle_member"]))
    stream_member = bundle["datasets"][config["dataset"]]["stream"]
    stream_path = verify(stream_member)
    value = read(stream_path)
    # prepare_stream writes records separately from the compact stream lock.
    records = value
    require(isinstance(records, list) and len(records) == 2000 and
            [r["occurrence_index"] for r in records] == list(range(1, 2001)), "EXACT_ORDERED_2000_STREAM")
    require(file_sha(stream_path) == assets["streams"][config["dataset"]]["stream_sha256"],
            "DERIVED_STREAM_ASSET_IDENTITY")
    runtime = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy")}
    require(runtime == assets["runtime"]["dependency_versions"], "SCIENTIFIC_RUNTIME_CHANGED")
    identity = dict(config_sha256=config["config_sha256"],
        stream_sha256=assets["streams"][config["dataset"]]["stream_sha256"],
        code_commit=lock["source"]["main_commit"], official_tree_sha256=lock["source"]["official_tree"],
        model_revision=assets["model"]["identity"]["revision"], tokenizer_sha256=assets["model"]["tokenizer_sha256"],
        assets_sha256=assets["assets_sha256"])
    external = dict(model="llama3", model_revision=identity["model_revision"],
        tokenizer_sha256=identity["tokenizer_sha256"], assets_sha256=identity["assets_sha256"],
        stream_sha256=identity["stream_sha256"], code_commit=identity["code_commit"],
        official_tree=identity["official_tree_sha256"], runtime=runtime,
        precision="FP32_EAGER_TF32_OFF_NO_AUTOCAST", raw_local_only=True)
    return assets, records, identity, external


def seed_edit():
    import numpy as np
    import torch
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)


def load_model(assets):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
            "ONE_SLURM_VISIBLE_GPU_REQUIRED")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(assets["model"]["tokenizer_path"], local_files_only=True)
    require(tokenizer.eos_token_id is not None, "PINNED_TOKENIZER_EOS_REQUIRED")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(assets["model"]["snapshot"],
        local_files_only=True, use_safetensors=True, dtype=torch.float32,
        low_cpu_mem_usage=True, attn_implementation="eager")
    require(model.config.model_type == "llama" and model.config.hidden_size == 4096
            and model.config.intermediate_size == 14336 and model.config.num_hidden_layers == 32,
            "ACTUAL_NATIVE_LLAMA3_MODEL_CONFIG")
    model = model.to("cuda:0").eval()
    model.config.use_cache = False
    for parameter in model.parameters():
        require(not parameter.is_floating_point() or parameter.dtype == torch.float32,
                "MODEL_NOT_FP32")
        parameter.requires_grad_(False)
    return model, tokenizer


def rng_content(value):
    """Small encoded state-content proof; tensors remain RAM/checkpoint local."""
    import numpy as np
    import torch
    def encode(item):
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            return dict(dtype=str(tensor.dtype), shape=list(tensor.shape),
                        sha256=hashlib.sha256(tensor.numpy().tobytes()).hexdigest())
        if isinstance(item, np.ndarray):
            return dict(dtype=str(item.dtype), shape=list(item.shape),
                        sha256=hashlib.sha256(item.tobytes()).hexdigest())
        if isinstance(item, dict):
            return {key: encode(child) for key, child in item.items()}
        if isinstance(item, (tuple, list)):
            return [encode(child) for child in item]
        if isinstance(item, np.generic):
            return item.item()
        return item
    return encode(value)


def rng_digest(value):
    """Content identity, not nondeterministic torch.save archive timestamps."""
    return digest(rng_content(value))


def restore_checkpoint(model, engine, payload, identity):
    import torch
    from official.experiments.checkpoint import rng_restore
    require(payload["identity"] == identity and payload["method"] == engine.method,
            "RESUME_NATIVE_IDENTITY")
    require(set(payload["weights"]) == set(engine.selected_weights) and not payload["cache_c"],
            "RESUME_SELECTED_WEIGHT_OR_NON_HISTORY_SCHEMA")
    with torch.no_grad():
        for name, tensor in payload["weights"].items():
            target = engine.selected_weights[name]
            require(target.shape == tensor.shape and tensor.dtype == target.dtype == torch.float32
                    and torch.isfinite(tensor).all().item(), "RESUME_TENSOR_SHAPE_DTYPE_FINITE")
            target.copy_(tensor.to(target.device))
    engine.restore_contexts(payload["contexts"])
    require(engine.successful_calls == payload["batch"], "RESUME_NATIVE_CONTEXT_BATCH_CURSOR")
    # Loading a fresh model/C0 must not change the resumed native edit RNG.
    rng_restore(payload["rng"])
    return payload["batch"]


class Tracking:
    """Single reviewed official transport. Absence is an explicit startup block."""
    def __init__(self, config, output, identity, *, mode, method, dataset):
        contract = config["tracking"]
        require(contract.get("module") == "official.tracking.client"
                and contract.get("entity") == "wkdguswns2256"
                and contract.get("project") == "layer allocation", "OFFICIAL_TRACKING_ROUTE_NOT_SEALED")
        try:
            transport = importlib.import_module(contract["module"])
        except ModuleNotFoundError as error:
            raise ValueError("OFFICIAL_WANDB_TRANSPORT_NOT_PUBLISHED") from error
        invocation = uuid.uuid4().hex
        values = dict(server="server1", task_id="official-baselines-20261008",
            model="llama3", model_family="llama", writer=method.lower() if method else "none",
            role="scientific", arm=f"{dataset}-{method or mode}", attempt=contract["attempt"] + "-" + invocation[:12],
            baseline=method or "W0", metric_schema="official-baselines-scalar-v1",
            instruction_id="USER-OFFICIAL-BASELINES-20261008-R1", dataset=dataset,
            source_sha=identity["code_commit"], config_sha=identity["config_sha256"])
        if dataset == "cf" and not generation_at_W20(config) and mode != "base_w0":
            values["generation_schedule"] = "DEFERRED_CHECKPOINT_EVALUATION"
        elif dataset == "cf":
            assets = read(verify(config["assets_member"]))
            reference = read(assets["generation_reference"]["manifest"]["path"])
            values.update(generation_schedule="W0_AND_W20_FIRST2000",
                generation_metric_schema="counterfact-cake-generation-metrics-v1",
                generation_profile="cf-cake-native-casebatch-kv-total100-globalrng-v1",
                generation_eval_seed=20261007, reference_assets_sha256=reference["identity_sha256"],
                generation_source_sha=identity["code_commit"],
                generation_repair_instruction="USER-OFFICIAL-BASELINES-20261008-R1")
        # Parent process passes only whitelisted identity metadata, never full env.
        self.tracker = transport.init(env_file=contract["env_file"],
            spool=Path(output) / "tracking" / invocation, config=values)
        require(bool(self.tracker), "OFFICIAL_TRACKING_INIT_FAILED")

    def log(self, payload):
        require(all(type(value) in (int, float, str, bool) for value in payload.values()),
                "SCALAR_ONLY_TRACKING")
        require(self.tracker.log(payload) is not False, "TRACKING_ACCEPTANCE_FAILURE_NOT_SILENT_DROP")

    def finish(self, exit_code=0):
        return self.tracker.finish(exit_code=exit_code, timeout=45)


def factual_payload(endpoint, prefix, edits):
    require(prefix in ("W0_first2000", "all_seen/post"), "OFFICIAL_ENDPOINT_PREFIX")
    value = {"edits": edits, "pre_state_edits": edits, "post_state_edits": edits}
    for key, item in endpoint["summary"].items():
        if type(item) in (int, float):
            value[f"official/{prefix}/{key}"] = item
    # CF prompt-pair diagnostics are measured separately from request macro.
    # zsRE official token/request metrics must not masquerade as NLL preference.
    if endpoint["identity"]["dataset"] == "cf":
        successes = []
        for kind, letter in (("rewrite", "R"), ("paraphrase", "P"), ("neighborhood", "N")):
            rows = [observation for case in endpoint["cases"] for observation in case[kind + "_observations"]]
            true = [row["target_true"]["mean_nll"] for row in rows]
            new = [row["target_new"]["mean_nll"] for row in rows]
            success = sum(t < n if letter == "N" else n < t for t, n in zip(true, new))
            stats = endpoint["accuracy"][kind]
            fields = dict(count=len(rows), success_count=success, success_pct=100 * success / len(rows),
                true_nll=math.fsum(true) / len(rows), new_nll=math.fsum(new) / len(rows),
                token_acc_pct=stats["token_acc_pct"], prompt_acc_pct=stats["prompt_acc_pct"],
                strict_acc_pct=stats["strict_acc_pct"])
            fields["margin_true_minus_new"] = fields["true_nll"] - fields["new_nll"]
            value.update({f"{prefix}/{letter}/{key}": item for key, item in fields.items()})
            successes.append(fields["success_pct"])
        from official.tracking.method import harmonic
        value[prefix + "/success_harmonic_pct"] = harmonic(successes)
    return value


def generation_payload(observed, edits, *, prefix="all_seen/post"):
    from official.evaluation.generation.metrics import generation_payload as scalar_payload
    value = dict(edits=edits, pre_state_edits=edits, post_state_edits=edits)
    value.update(scalar_payload(prefix, observed["summary"]))
    return value


def immutable_observation(path, value):
    """Resume compares exact scientific bytes; elapsed/work is a separate receipt.

    Removing work is not a hash bypass: tokens/raw/summary/state/cohort/source and
    endpoint links remain immutable. A conflicting scientific row still fails.
    """
    scientific = {key: item for key, item in value.items() if key != "work"}
    write_new(path, scientific)
    work_path = Path(path).parent / "observation-cost" / (Path(path).stem + "-" + uuid.uuid4().hex + ".json")
    write_new(work_path, dict(observation_member=member(path), work=value.get("work", {})))
    return member(path)
