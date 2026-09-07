"""Read-only, CPU-only terminal/member audit for the sealed SH4 L8 takeover.

This module never imports a model runtime, calls a scheduler, or changes inputs.
An inventory SHA is computed once per raw member. Checkpoint tensor identities
are independently reconstructed on CPU using the existing tensor hash helper.
The older Server2 reference is explicitly a publication-subset verification,
not a claim to have accessed Server2 raw files.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
from typing import Any

SAMPLE_ROOT = "40fe1afb318a4a5da9630425213644a729a1e85fb4e78c18e9a4dcd2870eefdd"
SOURCE_HEAD = "44602a1a80554c67da0ef9646b43d843104785f2"
SOURCE_TREE = "cdb089a139f180c822d1e6bf44fe3d27b858c1ce"
OFFICIAL_HEAD = "14cea8245f06715684592ab55184939b99d70784"
OFFICIAL_TREE = "9c52aadbc0883da422badf0a730fff21aaa3a8a7"
CHAINS = ((4, "llama3-8b-inst"), (5, "qwen2.5-7b-inst"))
KINDS = {"rewrite_target_new": 1, "rewrite_target_true": 1,
         "rephrase_target_new": 2, "rephrase_target_true": 2,
         "locality_target_new": 10, "locality_target_true": 10}


class IntegrityError(RuntimeError):
    """A concrete evidence mismatch; never converted into partial success."""


def require(value: bool, label: str) -> None:
    if not value:
        raise IntegrityError(label)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_count(value: Any, label: str = "root") -> int:
    if isinstance(value, float):
        require(math.isfinite(value), f"NONFINITE:{label}")
        return 1
    if isinstance(value, dict):
        return sum(finite_count(v, f"{label}.{k}") for k, v in value.items())
    if isinstance(value, list):
        return sum(finite_count(v, f"{label}[{i}]") for i, v in enumerate(value))
    return 0


def snapshot_members(root: Path) -> list[dict]:
    require(root.is_dir() and not root.is_symlink(), "ROOT_DIRECTORY_TYPE")
    result = []
    for path in sorted(root.rglob("*")):
        info = path.lstat()
        require(not stat.S_ISLNK(info.st_mode), f"SYMLINK:{path.relative_to(root)}")
        if stat.S_ISDIR(info.st_mode):
            continue
        require(stat.S_ISREG(info.st_mode), f"NONREGULAR:{path.relative_to(root)}")
        before = (info.st_size, info.st_mtime_ns, info.st_ino)
        sha = file_hash(path)
        after = path.stat()
        require(before == (after.st_size, after.st_mtime_ns, after.st_ino),
                f"MEMBER_CHANGED_DURING_HASH:{path.relative_to(root)}")
        result.append(dict(path=str(path.relative_to(root)), bytes=info.st_size,
                           mode=f"{stat.S_IMODE(info.st_mode):04o}", uid=info.st_uid,
                           sha256=sha, mtime_ns=info.st_mtime_ns, inode=info.st_ino))
    return result


def confirm_unchanged(root: Path, inventory: list[dict]) -> None:
    actual = sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())
    require(actual == [r["path"] for r in inventory], "RAW_MEMBER_SET_CHANGED")
    for row in inventory:
        info = (root / row["path"]).lstat()
        require(stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode), "RAW_TYPE_CHANGED")
        require((info.st_size, info.st_mtime_ns, info.st_ino) ==
                (row["bytes"], row["mtime_ns"], row["inode"]), f"RAW_CHANGED:{row['path']}")


def audit_publication(root: Path, by_path: dict[str, dict], load) -> dict:
    prefix = "external-main/"
    manifest = load(prefix + "package-manifest.json")
    receipt = load(prefix + "rooted-package-receipt.json")
    require(by_path[prefix + "package-manifest.json"]["sha256"] == receipt["manifest_sha256"],
            "PUBLICATION_MANIFEST_SHA")
    require(canonical_hash(manifest) == receipt["root_sha256"], "PUBLICATION_ROOT")
    require(manifest["completed_chains"] == [0, 1, 2, 3], "PUBLICATION_CHAINS")
    require(len(manifest["members"]) == 48, "PUBLICATION_MEMBER_COUNT")
    for member in manifest["members"]:
        name = member["path"]
        require(Path(name).name == name, "PUBLICATION_PATH_SAFETY")
        actual = by_path[prefix + name]
        require((actual["sha256"], actual["bytes"]) == (member["sha256"], member["bytes"]),
                f"PUBLICATION_MEMBER:{name}")
    require(by_path[prefix + "factual-report-ko.md"]["sha256"] == receipt["report_sha256"],
            "PUBLICATION_REPORT_SHA")
    return dict(status="RAW_FREE_PUBLICATION_SUBSET_REHASH_PASS", members=48,
                manifest_sha256=receipt["manifest_sha256"], root_sha256=receipt["root_sha256"],
                source_head=manifest["source_head"],
                external_raw_free_tables=manifest.get("external_raw_free_tables"),
                unavailable_server2_raw_rehashed=False,
                guarantee="48 locally published members and rooted package; Server2 raw/checkpoint bytes not accessed")


def audit_metrics(panel: dict, records: list[dict], label: str) -> None:
    """Validate category cardinality and exact request/prompt order, not scores."""
    for kind, count in KINDS.items():
        values = panel[kind]
        require(len(values) == len(records) * count, f"METRIC_CARDINALITY:{label}:{kind}")
        identity_key = "request_sha256" if all("request_sha256" in r for r in values) else "case_id"
        expected = [(r[identity_key], i) for r in records for i in range(count)]
        actual = [(r[identity_key], r["prompt_index"]) for r in values]
        require(actual == expected, f"METRIC_ORDER:{label}:{kind}")
        require(len(set(actual)) == len(actual), f"METRIC_DUPLICATE:{label}:{kind}")


def audit_checkpoint(path: Path, receipt: dict, commit: dict, metadata: dict) -> dict:
    import torch
    from project.run_scripts.ordered_response_barrier_ode.fp32_overlay import tensor_sha256, tensor_set_sha256
    # map_location prevents device restoration; mmap avoids a duplicate full-state buffer.
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    weights = payload["weights"]
    require(set(weights) == {f"model.layers.{i}.mlp.down_proj.weight" for i in range(4, 9)},
            "CHECKPOINT_LAYER_INVENTORY")
    identities = []
    for name, tensor in sorted(weights.items()):
        require(tensor.device.type == "cpu" and tensor.dtype == torch.float32,
                f"CHECKPOINT_WEIGHT_DTYPE:{name}")
        require(bool(torch.isfinite(tensor).all()), f"CHECKPOINT_NONFINITE:{name}")
        identities.append(dict(name=name, shape=list(tensor.shape), dtype=str(tensor.dtype),
                               sha256=tensor_sha256(tensor)))
    cache = payload["alpha_cache"]
    require(cache.device.type == "cpu" and cache.dtype == torch.float32, "CHECKPOINT_CACHE_DTYPE")
    require(bool(torch.isfinite(cache).all()), "CHECKPOINT_CACHE_NONFINITE")
    wsha, msha = tensor_set_sha256(weights), tensor_sha256(cache)
    require(wsha == receipt["selected_weight_sha256"] == commit["committed_weight_sha256"],
            "CHECKPOINT_WEIGHT_HASH")
    require(msha == receipt["M_sha256"] == commit["committed_M_content_sha256"],
            "CHECKPOINT_CACHE_HASH")
    require(payload["commit"] == commit, "CHECKPOINT_COMMIT")
    require(payload["metadata"] == metadata, "CHECKPOINT_METADATA")
    result = dict(path=str(path), status="CPU_TENSOR_FULL_REHASH_PASS", weights=identities,
                  weight_set_sha256=wsha, cache_sha256=msha, cache_shape=list(cache.shape),
                  cache_dtype=str(cache.dtype), cache_c_new=payload["cache_c_new"],
                  model_loaded=False, gpu_actions=0)
    del payload, weights, cache
    gc.collect()
    return result


def git_blob(repo: Path, head: str, path: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repo), "show", f"{head}:{path}"])


def audit_source(repo: Path, source: dict) -> dict:
    require((source["head"], source["tree"]) == (SOURCE_HEAD, SOURCE_TREE), "SOURCE_IDENTITY")
    tree = subprocess.check_output(["git", "-C", str(repo), "rev-parse", f"{SOURCE_HEAD}^{{tree}}"], text=True).strip()
    require(tree == SOURCE_TREE, "GIT_SOURCE_TREE")
    require((source["official"]["head"], source["official"]["tree"]) ==
            (OFFICIAL_HEAD, OFFICIAL_TREE), "OFFICIAL_SEAL")
    for member in source["kernel_members"]:
        blob = git_blob(repo, SOURCE_HEAD, member["path"])
        require((len(blob), hashlib.sha256(blob).hexdigest()) == (member["bytes"], member["sha256"]),
                f"SOURCE_KERNEL:{member['path']}")
    relative = "project/run_scripts/alpha_native_response_ode_v31_sequential/runtime.py"
    runtime = git_blob(repo, SOURCE_HEAD, relative)
    require(b"tok.padding_side='right'" in runtime, "PINNED_RIGHT_PADDING_SOURCE")
    require(b"tok.pad_token_id=tok.eos_token_id" in runtime and
            b"model.config.pad_token_id=tok.pad_token_id" in runtime, "PAD_EOS_BINDING_SOURCE")
    require(runtime.count(b"f.compute_fixed_z()") == 1, "ONCE_PER_BATCH_TARGET_SOURCE")
    return dict(head=SOURCE_HEAD, tree=SOURCE_TREE, kernel_member_count=len(source["kernel_members"]),
                kernel_status="GIT_OBJECT_FULL_REHASH_PASS", official_head=OFFICIAL_HEAD,
                official_tree=OFFICIAL_TREE, official_raw_source_audit="SEALED_RUNTIME_AUTHORITY_REUSED",
                padding_side="right", padding_source_path=relative,
                padding_source_sha256=hashlib.sha256(runtime).hexdigest(),
                runtime_padding_token_id_telemetry="NOT_RECORDED_RUNTIME_SCHEMA",
                padding_gate="PINNED_SOURCE_BINDING_REUSED_NO_TOKENIZER_OR_MODEL_LOAD")


def runtime_comparison(raw_root: Path, inventory_path: Path, output: Path) -> dict:
    """Compare sealed cross-host identities, without asserting host bit parity."""
    inventory = json.loads(inventory_path.read_text())
    by_path = {r["path"]: r for r in inventory["members"]}
    inputs = {}
    def read(name):
        path = raw_root / name
        require(file_hash(path) == by_path[name]["sha256"], f"RUNTIME_INPUT_SHA:{name}")
        inputs[name] = by_path[name]["sha256"]
        return json.loads(path.read_text())
    local_assets, old_assets = read("assets.lock.json")["models"], read("external-main/assets.lock.json")["models"]
    require(read("science.lock.json") == read("external-main/science.lock.json"), "CROSS_HOST_SCIENCE_LOCK")
    rows = []
    for index, alias in CHAINS:
        local = read(f"chain-{index}-{alias}-L8_ONLY_NATIVE/runtime.lock.json")
        a, b = local_assets[alias], old_assets[alias]
        fields = ("revision", "snapshot_config_sha256", "snapshot_tokenizer_sha256", "native_name")
        require(all(a[k] == b[k] for k in fields), "CROSS_HOST_MODEL_TOKENIZER_IDENTITY")
        require(a["hparams"]["AlphaEdit"]["sha256"] == b["hparams"]["AlphaEdit"]["sha256"], "CROSS_HOST_HPARAM_IDENTITY")
        require(a["projector"]["observed_sha256"] == b["projector"]["observed_sha256"], "CROSS_HOST_PROJECTOR_IDENTITY")
        require([(x["layer"], x["observed_sha256"]) for x in a["statistics"]] ==
                [(x["layer"], x["observed_sha256"]) for x in b["statistics"]], "CROSS_HOST_STATS_IDENTITY")
        for control_index, control_arm in ((index - 4, "O_NATIVE"), (index - 2, "JV_NATIVE")):
            control = read(f"external-main/chain-{control_index}-{alias}-{control_arm}.runtime.lock.json")
            require(local["contexts_sha256"] == control["contexts_sha256"], "CROSS_HOST_CONTEXT_IDENTITY")
            require(local["dtype"] == control["dtype"] and local["parameter_inventory"] == control["parameter_inventory"], "CROSS_HOST_FP32_INVENTORY")
            require((local["official"]["head"], local["official"]["tree"]) ==
                    (control["official"]["head"], control["official"]["tree"]), "CROSS_HOST_OFFICIAL_SOURCE")
            require(Path(local["snapshot"]).name == Path(control["snapshot"]).name == a["revision"], "CROSS_HOST_SNAPSHOT_REVISION")
            row = dict(alias=alias, local_arm="L8_ONLY_NATIVE", control_arm=control_arm,
                       local_cuda_device=local["cuda_device"], control_cuda_device=control["cuda_device"],
                       local_snapshot=local["snapshot"], control_snapshot=control["snapshot"],
                       model_revision=a["revision"], model_config_sha256=a["snapshot_config_sha256"],
                       tokenizer_sha256=a["snapshot_tokenizer_sha256"],
                       AlphaEdit_hparams_sha256=a["hparams"]["AlphaEdit"]["sha256"],
                       projector_sha256=a["projector"]["observed_sha256"],
                       statistics=[dict(layer=x["layer"], sha256=x["observed_sha256"]) for x in a["statistics"]],
                       contexts_sha256=local["contexts_sha256"], dtype_inventory=local["parameter_inventory"],
                       official_head=local["official"]["head"], official_tree=local["official"]["tree"],
                       identities_status="SEALED_ASSET_IDENTITY_MATCH", host_bitwise_equivalence="NOT_ESTABLISHED",
                       performance_interpretation="Matched science/assets/W0; GPU architecture differs. Wall-time ratio is cross-host descriptive, not pure algorithm overhead.")
            for version in ("torch_version", "cuda_version", "transformers_version", "python_version", "driver_version"):
                row[f"local_{version}"] = local.get(version, "NOT_RECORDED_RUNTIME_SCHEMA")
                row[f"control_{version}"] = control.get(version, "NOT_RECORDED_RUNTIME_SCHEMA")
            rows.append(row)
    result = dict(status="CROSS_HOST_SEALED_IDENTITY_AUDIT_PASS", rows=rows, input_sha256=inputs,
                  raw_inventory_sha256=file_hash(inventory_path), scheduler_calls=0, model_load=0,
                  cross_host_numerical_bit_parity="NOT_ESTABLISHED_NOT_REQUIRED_FOR_DESCRIPTIVE_COMPARISON")
    result["root_sha256"] = canonical_hash(result)
    save(output, result)
    return result


def run(raw_root: Path, output: Path, source_repo: Path) -> dict:
    raw_root, output = raw_root.absolute(), output.absolute()
    require(not output.exists(), "CREATE_ONCE_OUTPUT")
    require(not output.is_relative_to(raw_root), "OUTPUT_MUST_NOT_BE_RAW_CHILD")
    output.mkdir(parents=True, mode=0o700)
    inventory = snapshot_members(raw_root)
    by_path = {r["path"]: r for r in inventory}
    scanned_floats = 0
    cache = {}
    def load(name):
        nonlocal scanned_floats
        if name not in cache:
            with (raw_root / name).open() as stream:
                cache[name] = json.load(stream)
            scanned_floats += finite_count(cache[name], name)
        return cache[name]
    for member in inventory:
        if member["path"].endswith(".json"):
            load(member["path"])
    require(not any(Path(r["path"]).name == "failure.json" for r in inventory), "FAILURE_MEMBER_PRESENT")
    source = load("source.lock.json")
    source_audit = audit_source(source_repo, source)
    sample = load("sample.lock.json")
    records = sample["records"]
    require(sample["ordered_root"] == SAMPLE_ROOT == canonical_hash(records), "SAMPLE_ROOT")
    require(len(records) == len({r["case_id"] for r in records}) == 1000, "SAMPLE_UNIQUE_COUNT")
    require([r["ordinal"] for r in records] == list(range(1000)), "SAMPLE_ORDINALS")
    require(all((r["batch_index"], r["batch_ordinal"]) == (i // 100 + 1, i % 100)
                for i, r in enumerate(records)), "SAMPLE_PARTITION")
    publication = audit_publication(raw_root, by_path, load)
    require(load("external-main/sample.lock.json") == sample, "SHARED_PUBLICATION_SAMPLE")
    arms, batches, checkpoints = [], [], []
    for index, alias in CHAINS:
        prefix = f"chain-{index}-{alias}-L8_ONLY_NATIVE"
        terminal = load(prefix + "/terminal-receipt.json")
        runtime = load(prefix + "/runtime.lock.json")
        require(terminal["status"] == "TERMINAL_VALID" and terminal["completed_batches"] == 10 and
                terminal["requested"] == 1000 and terminal["W0_restored"] is True, "TERMINAL_VALIDITY")
        require(terminal["source"] == runtime["source"] == source, "RUNTIME_TERMINAL_SOURCE")
        require(terminal["sample_root"] == runtime["sample_root"] == SAMPLE_ROOT, "TERMINAL_SAMPLE")
        require(terminal["alias"] == runtime["alias"] == alias and
                terminal["arm"] == runtime["arm"] == "L8_ONLY_NATIVE", "TERMINAL_MAPPING")
        require(terminal["history_appends"] == 10 and terminal["cold_init_count"] == 1 and
                terminal["technical_failure_count"] == terminal["imputation_count"] == 0, "TERMINAL_COUNTERS")
        dtype = runtime["dtype"]
        require(dtype["loaded_model_dtype"] == "torch.float32" and dtype["bf16_fp16_cast_count"] == 0
                and not dtype["autocast_enabled"] and not dtype["quantized"] and not runtime["tf32"], "FULL_FP32")
        require(set(runtime["parameter_inventory"]["parameter_tensors_by_dtype"]) == {"torch.float32"}, "DTYPE_INVENTORY")
        reference_index = 0 if index == 4 else 1
        reference = load(f"external-main/chain-{reference_index}-{alias}-O_NATIVE.terminal-receipt.json")
        require(terminal["W0_sha256"] == reference["W0_sha256"], "MATCHED_W0")
        previous, z_count, node_count = None, 0, 0
        for batch in range(1, 11):
            b = f"{prefix}/batch-{batch:02}"
            entry, target, writer, commit, complete = [load(f"{b}/{name}.json") for name in
                                                       ("entry", "target-reference", "writer", "commit", "complete")]
            selected = records[(batch - 1) * 100:batch * 100]
            require(entry["case_ids"] == commit["case_ids"] == [r["case_id"] for r in selected], "BATCH_CASE_ORDER")
            require(entry["request_order_sha256"] == canonical_hash([r["request_sha256"] for r in selected]), "BATCH_ORDER_ROOT")
            require(entry["source_head"] == SOURCE_HEAD and entry["sample_root"] == SAMPLE_ROOT, "BATCH_SOURCE_SAMPLE")
            require(entry["previous_commit"] == previous, "PREVIOUS_COMMIT_FULL_IDENTITY")
            if previous is None:
                require(entry["W_entry"] == terminal["W0_sha256"] and entry["cold_initialization_count"] == 1, "B1_W0")
            else:
                require(entry["W_entry"] == previous["committed_weight_sha256"] and
                        entry["M_entry"] == previous["committed_M_content_sha256"] and
                        entry["cold_initialization_count"] == 0, "W_M_CONTINUITY")
            require(commit["W_entry"] == commit["entry_weight_sha256"] == target["W_entry"] == entry["W_entry"], "ENTRY_W_BINDING")
            require(commit["M_entry"] == target["M_entry"] == entry["M_entry"], "ENTRY_M_BINDING")
            require(target["fixed_z_request_count"] == 100 and target["inner_z_reoptimization_count"] == 0, "TARGET_COUNTS")
            z_count += target["fixed_z_request_count"]
            zsha = target["fixed_z_sha256"]
            require(zsha == writer["fixed_z_sha256"] == commit["fixed_z_sha256"], "FIXED_Z_BINDING")
            require(target["metric"]["capture_count"] == 1 and len(target["metric"]["raw_layer_actions"]) == 5,
                    "ALL_FIVE_ENTRY_DIRECTIONS_REFERENCE")
            require(commit["history_append_count"] == writer["endpoint"]["history_append_count"] == 1, "HISTORY_APPEND_ONCE")
            require(writer["endpoint"]["terminal_history_key_capture_count"] ==
                    writer["write_including_endpoint"]["history_key_captures"] == 5, "HISTORY_FULL_INVENTORY")
            require(all(commit[k] == 0 for k in ("writer_recompute_count", "fixed_z_recompute_count", "model_forward_count", "evaluator_count")),
                    "COMMIT_RECOMPUTATION")
            require(writer["endpoint"]["selected_weight_endpoint_sha256"] == commit["committed_weight_sha256"], "WRITER_COMMIT_W")
            require(complete["commit"] == commit and complete["status"] == "BATCH_COMMITTED" and
                    complete["requested"] == 100 and complete["seen_requested"] == 100 * batch, "BATCH_TERMINAL")
            require(complete["technical_failure_count"] == complete["semantic_success_filtering_count"] == 0, "BATCH_FAILURE_FILTERING")
            require(len(writer["nodes"]) == writer["main_jvp_count"] == 4, "NODE_JVP_COUNT")
            for ordinal, node in enumerate(writer["nodes"]):
                require(node == load(f"{b}/nodes/node-{ordinal:02}.json"), "NODE_MEMBER_IDENTITY")
                require(node["node"] == ordinal and node["active_layers"] == [8] and
                        node["h"] == .5 and node["lambda_value"] == .1, "NODE_SCIENCE_MAPPING")
                require(node["fixed_z_sha256"] == zsha and node["source_entry_sha256"] == entry["W_entry"], "NODE_Z_W")
                require(all(node[k] == 0 for k in ("controller_heldout_access_count", "inner_history_append_count", "inner_weight_mutation_count")), "NODE_MUTATION")
                require(all(r["actual_step_DeltaW_squared"] == 0 and r["actual_nonzero"] == 0
                            for r in node["actual_physical"] if r["layer"] != 8), "L8_SUPPORT_ONLY")
                node_count += 1
            audit_metrics(load(f"{b}/current-full-raw.json"), selected, f"{alias}:B{batch}:current")
            if batch in (1, 5, 10):
                seen = load(f"{b}/seen-full.json")
                require(seen["request_count"] == 100 * batch and seen["controller_or_writer_influence_count"] == 0, "SEEN_COUNT_LEAKAGE")
                grouped = {kind: [r for r in seen["rows"] if r["kind"] == kind] for kind in KINDS}
                audit_metrics(grouped, records[:100 * batch], f"{alias}:B{batch}:seen")
                cpname = f"{prefix}/checkpoints/W{batch}-M{batch}.pt"
                cpreceipt = load(cpname.replace(".pt", ".receipt.json"))
                require(complete["checkpoint"] == cpreceipt, "CHECKPOINT_SUMMARY_IDENTITY")
                require((by_path[cpname]["sha256"], by_path[cpname]["bytes"]) ==
                        (cpreceipt["sha256"], cpreceipt["bytes"]), "CHECKPOINT_FILE_SHA_BYTES")
                metadata = dict(source=source, sample_root=SAMPLE_ROOT, arm="L8_ONLY_NATIVE", alias=alias, batch=batch)
                checkpoints.append(audit_checkpoint(raw_root / cpname, cpreceipt, commit, metadata))
            batches.append(dict(alias=alias, batch=batch, requests=100, nodes=4,
                                W_entry=entry["W_entry"], W_exit=commit["committed_weight_sha256"],
                                M_entry=entry["M_entry"], M_exit=commit["committed_M_content_sha256"],
                                fixed_z_sha256=zsha, target_requests=100, target_recompute=0,
                                history_append=1, status="PASS"))
            previous = commit
        require(terminal["final_commit"] == previous, "TERMINAL_FINAL_COMMIT")
        arms.append(dict(alias=alias, arm="L8_ONLY_NATIVE", slurm_child=runtime["slurm_job"],
                         status="TERMINAL_VALID_REHASH_PASS", batches=10, requests=1000,
                         W_chain_links=9, M_chain_links=9, target_requests=z_count, target_recompute=0,
                         target_count_authority="100 request targets/batch plus once-per-batch source; standalone invocation counter not recorded",
                         nodes=node_count, history_appends=10, history_key_captures=50,
                         W0_restored=True, dtype=dtype, technical_failure_count=0, imputation_count=0,
                         terminal_receipt_sha256=by_path[prefix + "/terminal-receipt.json"]["sha256"]))
    confirm_unchanged(raw_root, inventory)
    identity = [{k: r[k] for k in ("path", "bytes", "mode", "sha256")} for r in inventory]
    manifest = dict(schema="sh4.l8-takeover.raw-inventory.v1", raw_root=str(raw_root),
                    members=inventory, member_root=canonical_hash(identity), member_count=len(inventory),
                    total_bytes=sum(r["bytes"] for r in inventory), sha_passes_per_raw_member=1)
    save(output / "raw-member-inventory.json", manifest)
    save(output / "batch-integrity.json", batches)
    save(output / "checkpoint-integrity.json", checkpoints)
    with (output / "raw-member-inventory.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(inventory[0])); writer.writeheader(); writer.writerows(inventory)
    result = dict(status="TERMINAL_FULL_REHASH_PASS", arms=arms, source=source_audit,
                  sample_root=SAMPLE_ROOT, publication=publication, batches=20, requests=2000,
                  checkpoints=len(checkpoints), nodes=80, finite_float_values_checked=scanned_floats,
                  nonfinite_count=0, failure_member_count=0, duplicate_metric_rows=0,
                  raw_member_count=len(inventory), raw_member_root=manifest["member_root"],
                  raw_inventory_sha256=file_hash(output / "raw-member-inventory.json"),
                  checkpoint_integrity_sha256=file_hash(output / "checkpoint-integrity.json"),
                  batch_integrity_sha256=file_hash(output / "batch-integrity.json"),
                  read_only_raw=True, scheduler_calls=0, model_load=0, gpu_actions=0, evaluator_runs=0,
                  caveats=["Original Server2 raw/checkpoints unavailable and not rehashed; its exact 48-member publication is reused.",
                           "Pointer/version restoration and native target once semantics are sealed runtime/source assertions; offline bytes validate checkpoints and chain links.",
                           "Runtime token IDs are NOT_RECORDED; exact pinned right-padding source binding is retained."])
    result["root_sha256"] = canonical_hash(result)
    save(output / "integrity-receipt.json", result)
    return result


def save(path: Path, value: Any) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-repo", type=Path, default=Path.cwd())
    parser.add_argument("--runtime-comparison-inventory", type=Path,
                        help="Only compare cross-host runtime seals against an already rehashed inventory; output is a new JSON file")
    args = parser.parse_args()
    if args.runtime_comparison_inventory is not None:
        result = runtime_comparison(args.raw_root, args.runtime_comparison_inventory, args.output)
        print(json.dumps({"status": result["status"], "rows": len(result["rows"]), "root_sha256": result["root_sha256"]}))
        return
    result = run(args.raw_root, args.output, args.source_repo)
    print(json.dumps({k: result[k] for k in ("status", "batches", "requests", "checkpoints", "raw_member_root", "root_sha256")}))


if __name__ == "__main__":
    main()
