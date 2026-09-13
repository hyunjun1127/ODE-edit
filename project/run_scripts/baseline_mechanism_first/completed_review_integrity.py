"""Read-only E01 observation receipt/source audit; no torch/model imports.

This is not the metric reducer. It verifies publication hashes, ordered shard
coverage, exact merge reuse, source diff and the precise recorded guard scope.
Large checkpoint tensor validation belongs to the separate CPU tensor audit.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess


OLD = "b51dcf5ab825608bee81dd13549318d8d267e835"
EXEC = "58f50a25809779918b22ad0aceded732c097eab4"
REL = "project/run_scripts/baseline_mechanism_first/"
ADDED = {REL + s for s in (
    "endpoint_observation_resume.py", "launch_endpoint_observation.sh",
    "performance_schema.py", "prepare_fullseen_resume.py",
    "tests/test_performance_schema.py")}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def audit(root, execution_worktree, repo):
    checked = {}
    failures = []

    def require(condition, code):
        if not condition:
            failures.append(code)

    def file(path, expected=None, large=False):
        p = Path(path)
        st = p.lstat()
        require(stat.S_ISREG(st.st_mode) and not p.is_symlink(), "NONREGULAR:" + str(p))
        value = dict(path=str(p), bytes=st.st_size, mode=oct(stat.S_IMODE(st.st_mode)))
        if large:
            value.update(sha256=None, check="SIZE_ACCESS_ONLY_SEALED_SHA_REUSED",
                         sealed_sha256=None if expected is None else expected["sha256"])
        else:
            value.update(sha256=sha(p.read_bytes()), check="FULL_SHA256_REHASH")
        if expected:
            if "bytes" in expected:
                require(st.st_size == expected["bytes"], "SIZE:" + str(p))
            if not large:
                require(value["sha256"] == expected["sha256"], "SHA:" + str(p))
        checked[str(p)] = value
        return value

    def ref(ref):
        return file(ref["path"], ref, large=str(ref["path"]).endswith(".pt") or ref.get("bytes", 0) > 100_000_000)

    def load(path, expected=None):
        file(path, expected)
        return json.loads(Path(path).read_text())

    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args])

    changes = [line.split("\t") for line in git("diff", "--name-status", OLD, EXEC).decode().splitlines()]
    require({x[1] for x in changes} == ADDED and all(x[0] == "A" for x in changes), "UNEXPECTED_SOURCE_DIFF")
    source_files = sorted(ADDED | {REL + s for s in (
        "terminal_performance.py", "performance_compare.py", "evaluation.py",
        "fixtures.py", "contracts.py", "cold_analysis.py", "case_population.py")})
    for name in source_files:
        blob = git("show", EXEC + ":" + name)
        local = file(Path(execution_worktree) / name)
        require(local["sha256"] == sha(blob), "SOURCE_BLOB:" + name)
    runner = git("show", EXEC + ":" + REL + "endpoint_observation_resume.py").decode()
    require("native_runner" not in runner and "apply_AlphaEdit" not in runner and "compute_z(" not in runner,
            "NATIVE_ENTRYPOINT_IN_OBSERVER")
    cells = []
    for entry, expected_shards in ((5000, 47), (9000, 78)):
        cell = Path(root) / f"n{entry}"
        lock = load(cell / "input.lock.json")
        terminal = load(cell / "output/terminal.json")
        restored = load(cell / "output/restore.json")
        repair = load(cell / "repair-receipt.json")
        parent = load(lock["parent_input"]["path"], lock["parent_input"])
        for m in lock["members"]:
            ref(m)
        for m in parent["source_members"]:
            ref(m)
        for p, m in terminal["evaluator_binding"]["sources"].items():
            file(p, m)
        preserve = load(lock["preserved"]["path"], lock["preserved"])
        # Reuse the old native-attempt preservation index; do not re-audit all
        # old outputs merely because the final observation has now completed.
        require(preserve["unmodified"] is True, "PRESERVATION_RECEIPT")
        require(lock["source_head"] == terminal["source_head"] == EXEC, "SOURCE_IDENTITY")
        require(terminal["status"] == "ENDPOINT_CURRENT_FULLSEEN_OBSERVATION_COMPLETE", "TERMINAL_STATUS")
        require(not (cell / "output/failure.json").exists(), "FAILURE_PRESENT")
        require(terminal["unique_seen_requests"] == entry + 1000, "FULL_COUNT")
        require(terminal["evaluated_past_requests"] == entry + 900 and terminal["reused_current_requests"] == 100, "SPLIT_COUNT")
        require(terminal["endpoint"] == lock["endpoint"] == restored["source_endpoint"], "ENDPOINT_REFERENCE")
        r = restored["receipt"]
        require(r["weight_sha256"] == lock["expected_W"] and r["history_sha256"] == lock["expected_M"], "RESTORED_W_M")
        require(r["next_batch_index"] == lock["terminal_batch"] + 1 and r["physical_layer"] == 4, "RESTORED_MAPPING")
        require(r["rng_restored_exact"] is True, "RESTORED_RNG")
        final_guard = terminal["restore"]
        require(final_guard["status"] == "RESTORED" and final_guard["pointer_bytes_exact"] and final_guard["rng_restored_exact"], "FINAL_RESTORE")
        require(final_guard["original_exception"] is None, "FINAL_EXCEPTION")
        for name in ("new_native_batches", "new_z", "new_history_append"):
            require(lock[name] == terminal[name] == restored[name] == 0, "NEW_ACTION:" + name)
        require(not terminal["full_E01_complete"] and not terminal["trajectory_equivalence"] and not terminal["scientific_promotion"], "CLAIM_BOUNDARY")
        refs = terminal["shards"]
        require(len(refs) == expected_shards and len({m["path"] for m in refs}) == expected_shards, "SHARD_COUNT")
        listed = set()
        offset = 0
        all_rows = {tag: [] for tag in ("RS", "PS", "NS")}
        eval_seconds = 0.0
        for m in refs:
            receipt = load(m["path"], m)
            listed.add(str(Path(m["path"])))
            end = min(offset + 128, entry + 900)
            require((receipt["begin"], receipt["end"]) == (offset, end), "SHARD_ORDER")
            require(receipt["requests"] == end - offset and receipt["prompt_pairs"] == 13 * (end - offset), "SHARD_SIZE")
            require(receipt["endpoint"] == lock["endpoint"], "SHARD_ENDPOINT")
            guard = receipt["guard"]
            require(guard["weight_sha256"] == lock["expected_W"] and guard["history_sha256"] == lock["expected_M"], "SHARD_W_M")
            require(all(guard[x] is True for x in ("weight_history_bytes_exact", "all_parameter_pointer_versions_exact", "RNG_exact")), "SHARD_GUARD")
            require(all(guard[x] == 0 for x in ("writer_calls", "z_calls", "history_append")), "SHARD_NEW_ACTION")
            eval_seconds += guard["seconds"]
            raw = load(receipt["raw"]["path"], receipt["raw"])
            listed.add(str(Path(receipt["raw"]["path"])))
            require(raw["requests"] == end - offset, "RAW_COUNT")
            for tag in all_rows:
                all_rows[tag].extend(raw["metrics"][tag]["rows"])
            offset = end
        require(offset == entry + 900, "SHARD_FINAL_BOUNDARY")
        require(listed == {str(p) for p in (cell / "output/shards").iterdir()}, "UNINDEXED_SHARD_OR_PARTIAL")
        current = load(lock["reuse_current"]["path"], lock["reuse_current"])
        full = load(terminal["fullseen"]["path"], terminal["fullseen"])
        for tag in all_rows:
            # Exact JSON row equality verifies merge nonmutation, not a second
            # metric reducer nor a full original-vs-replay identity comparison.
            require(all_rows[tag] + current["metrics"][tag]["rows"] == full["metrics"][tag]["rows"], "MERGE_ROWS:" + tag)
        ref(terminal["paired_summary"])
        initial = load(cell / "output/INITIAL_VALID.json")
        require(initial["fullseen_complete"] is False, "INITIAL_NOT_TERMINAL")
        cells.append(dict(entry_n=entry, terminal_batch=lock["terminal_batch"],
            status=terminal["status"], expected_W=lock["expected_W"], expected_M=lock["expected_M"],
            shard_count=len(refs), shard_requests=offset, reused_current_requests=100,
            fullseen_requests=terminal["unique_seen_requests"], exact_merge_reuse=True,
            endpoint_reference=lock["endpoint"], restore=final_guard,
            observation_guard_seconds_sum=eval_seconds, terminal_elapsed_seconds=terminal["elapsed_seconds"],
            model_load_seconds=terminal["model_load_seconds"], new_native_batches=0, new_z=0,
            new_history_append=0, native_batches_reused=repair["reused_native_batches"],
            old_failure=repair["original_error"]["error"], old_allocated_gpu_seconds=repair["old_allocation_GPU_seconds"],
            model_revision=parent["model_revision"], torch=parent["torch"], transformers=parent["transformers"],
            tf32_matmul=parent["tf32_matmul"], tf32_cudnn=parent["tf32_cudnn"],
            complete_E01=False, trajectory_equivalence=False))
    members = sorted(checked.values(), key=lambda x: x["path"])
    return dict(schema="E01_COMPLETED_SOURCE_INTEGRITY_REVIEW_V1", status="PASS_WITH_EXPLICIT_LIMITATIONS" if not failures else "HOLD",
        execution_source=EXEC, execution_tree=git("rev-parse", EXEC + "^{tree}").decode().strip(),
        predecessor_source=OLD, source_diff=changes, native_math_existing_file_changes=0,
        science_equation_target_history_precision_tolerance_changes=0,
        observed_action_counters="Declared zero counters cross-checked against read-only observer call graph; not independent native event instrumentation",
        pair_semantics="RS/PS new_nll<true_nll; NS true_nll<new_nll; ties fail; case_id/prompt_index/prompt+targets hash keyed",
        guard_scope=dict(per_shard="Selected L4 W and history full SHA; all parameter pointers/versions; Python/NumPy/Torch/CUDA RNG",
            final="Full captured model parameters and buffers + history compared with pre-restore CPU backup by pointer and byte hash",
            version="One selected parameter version changed through restore; original version not restored or claimed",
            hooks="Fixture source reinstalls saved registries; no separate recorded per-hook equality statistic",
            buffers="Per-shard buffer-byte guard not separately recorded; final transaction covers captured buffers",
            base_authority="Restoration relative to freshly loaded pinned model; no independent whole model snapshot file hash supplied by this review"),
        limitations=["No independent full-parameter byte rehash per shard; do not upgrade pointer/version guard to that claim.",
            "No native/compute-z/history append occurred in observation code; loading/copying saved endpoint is fixture restoration, not a new edit.",
            "No trajectory parity claim; old W mismatch cause remains unresolved.",
            "Large endpoint checkpoint file SHA/tensor comparisons delegated to parent CPU tensor audit; this audit reuses sealed hashes and checks size/access.",
            "Numerator/NLL/distribution recomputation is owned by separate independent reducer; this audit checks exact publication+merge identities.",
            "No model forward, GPU, Slurm or remote action performed in review."],
        cells=cells, failures=failures, checked_members=members,
        checked_member_count=len(members), full_sha_count=sum(m["check"] == "FULL_SHA256_REHASH" for m in members),
        access_only_count=sum(m["check"] != "FULL_SHA256_REHASH" for m in members),
        checked_member_root=sha(canonical(members)), scientific_promotion=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--execution-worktree", required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    result = audit(a.root, a.execution_worktree, a.repo)
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as f:
        json.dump(result, f, ensure_ascii=False, sort_keys=True, indent=2)
        f.write("\n")
    print(json.dumps({k: result[k] for k in ("status", "failures", "checked_member_count", "full_sha_count", "access_only_count", "checked_member_root")}))


if __name__ == "__main__":
    main()
