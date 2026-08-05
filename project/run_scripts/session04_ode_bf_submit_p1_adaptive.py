#!/usr/bin/env python3
"""One-shot submitter for the adaptive-tau P1R4 history-view R1 pair."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts import session04_ode_bf_submit_p0 as preserved_p0
from project.run_scripts import session04_ode_bf_submit_p1 as preserved_p1
from project.run_scripts import session04_ode_bf_submit_p1_diag as preserved_diag
from project.run_scripts.ode_bf.artifacts import (
    ODEBFArtifactGuard,
    load_rooted_json,
    sha256_file,
    sha256_regular_tree,
)
from project.run_scripts.ode_bf.contracts import MODEL_ALIASES, ODEBFContractError
from project.run_scripts.ode_bf.firewall import (
    FORBIDDEN_IMPORT_FRAGMENTS,
    assert_ast_firewall,
    assert_no_alias_specific_controller_branch,
)
from project.run_scripts.ode_bf.p1_adaptive import (
    ADAPTIVE_INSTRUCTION_ID,
    ADAPTIVE_VARIANTS,
    adaptive_lock,
    refinement_ratio,
)
from project.run_scripts.ode_bf.p1_controller import P1ControllerLock
from project.run_scripts.ode_bf.p1_runtime import (
    expected_p1r4_adaptive_result_name,
)
from project.run_scripts.ode_bf.p1_selection import (
    verify_p1_population_seal,
    verify_p1_stream_seal,
)
from project.run_scripts.ode_bf.resource import (
    MEMORY_CAP_MIB_PER_GPU,
    assert_node_local_capacity,
    forecast_p1_adaptive_b10_memory,
    forecast_p1_adaptive_b10_time,
)
from project.run_scripts.session04_ode_bf_p1_adaptive import (
    ADAPTIVE_RESULT_TOKEN,
)
from project.run_scripts.session04_ode_bf_p1_adaptive_dry_plan import JOB_NAMES


SESSION_ID = "019fc5ec-f85b-7770-a73a-1d19be1cd491"
BRANCH = "codex/odeeditsh2-ode-bf-v1"
BASE_HEAD = "6bdf456f168ababb35b7b69eaf1ddc2ea4a229c0"
SCIENTIFIC_LINEAGE_PARENT = "40d7811313ff61077e28ef571af3d9286de2db2e"
REPAIR_INSTRUCTION_ID = "ODEEDIT-S04-ODE-BF-P1R4-HISTORY-VIEW-R1-V1"
PACKAGE_ROOT = REPO_ROOT / "project/run_scripts/ode_bf"
LOCK_ROOT = PACKAGE_ROOT / "locks"
SBATCH = REPO_ROOT / "project/run_scripts/session04_ode_bf_p1_adaptive.sbatch"
DRY_PLAN = REPO_ROOT / "project/run_scripts/session04_ode_bf_p1_adaptive_dry_plan.py"
SOURCE_MANIFEST = LOCK_ROOT / "source_manifest_p1r4_adaptive_r1.json"
ARTIFACT_LOCK = LOCK_ROOT / "p0_artifact_lock.json"
BASE_ARTIFACT_LOCK = REPO_ROOT / "project/run_scripts/ode_alloc/p0_artifact_lock_r1.json"

ALLOWED_CHANGED_PATHS = {
    "project/run_scripts/ode_bf/locks/source_manifest_p1r4_adaptive_r1.json",
    "project/run_scripts/ode_bf/p1_adaptive_runtime.py",
    "project/run_scripts/ode_bf/p1_runtime.py",
    "project/run_scripts/ode_bf/tests/test_p1_adaptive_submission.py",
    "project/run_scripts/ode_bf/tests/test_p1_state.py",
    "project/run_scripts/session04_ode_bf_p1_adaptive.py",
    "project/run_scripts/session04_ode_bf_p1_adaptive.sbatch",
    "project/run_scripts/session04_ode_bf_p1_adaptive_dry_plan.py",
    "project/run_scripts/session04_ode_bf_submit_p1_adaptive.py",
}

FROZEN_PATHS = (
    "project/run_scripts/ode_bf/alpha_backend.py",
    "project/run_scripts/ode_bf/barriers.py",
    "project/run_scripts/ode_bf/evaluator.py",
    "project/run_scripts/ode_bf/functional.py",
    "project/run_scripts/ode_bf/history.py",
    "project/run_scripts/ode_bf/p1_controller.py",
    "project/run_scripts/ode_bf/p1_evaluator.py",
    "project/run_scripts/ode_bf/p1_replay.py",
    "project/run_scripts/ode_bf/p1_state.py",
    "project/run_scripts/ode_bf/routing.py",
    "project/run_scripts/ode_bf/sampling.py",
    "project/run_scripts/ode_bf/transaction.py",
    "project/run_scripts/ode_bf/woodbury.py",
    "project/run_scripts/ode_bf/locks/numerical_lock_p1r2.json",
    "project/run_scripts/ode_bf/locks/p1r2_seqb10_stream_seal.json",
    "project/run_scripts/ode_bf/locks/p1r2_p_population_seal.json",
)

P1R4_RESULT_SHA256 = {
    "s04-p1r4-full-residual-arms-llama3-8b-inst-v1": (
        "866a4bc9adef1eec7a030c58d06dd0eac1b90bb4c462ac9fbde7705ab4e1cc4d"
    ),
    "s04-p1r4-full-residual-arms-qwen2.5-7b-inst-v1": (
        "d0d6fb7581dfbf56fc83ca51718f7a2d4b303ef8d3b8f41fe13c43a59ab6818d"
    ),
}
P1R4_FILE_SHA256 = {
    "local/odebf/state/s04-p1r4-full-residual-arms-diag-v1.submission-intent.json": (
        "76ce912e796e51c724c8a9c8acd3df416dc5691a672048d331e776abcde4b17a"
    ),
    "local/odebf/state/s04-p1r4-full-residual-arms-diag-v1.submission-receipt.json": (
        "01ed795ba631d01fa2670e5ef7d715c2b29190021931459f41d92740122d432c"
    ),
    "local/odebf/logs/odebf_s04_p1r4full_llama-16681.err": (
        "b527c775c9645b59bad1bd8e633a298797f782279296ff60890860d17c20d071"
    ),
    "local/odebf/logs/odebf_s04_p1r4full_llama-16681.out": (
        "8b52d00e1851bd8b1d3d9d55f8d871a02d6c76b973e43b36dd7ef3415f35bd0a"
    ),
    "local/odebf/logs/odebf_s04_p1r4full_qwen-16682.err": (
        "7e277b8de5787a43d9aee96048aa8d54fb41218458a923229d6075953f474ffa"
    ),
    "local/odebf/logs/odebf_s04_p1r4full_qwen-16682.out": (
        "ca10cf8e3075793076a11c4ef559ac2e49b7c489d869c46337ab91ce61c478ce"
    ),
}

FAILED_ADAPTIVE_RESULT_SHA256 = {
    "s04-p1r4-adaptive-tau-llama3-8b-inst-v1": (
        "11bec73eefd013eba74133f0be341a42b05da8f2028521973755969a95da97c7"
    ),
    "s04-p1r4-adaptive-tau-qwen2.5-7b-inst-v1": (
        "ca06c2551118cd31b0a30832545d1ebc7da75fac172dd1523efd12b30f416ab5"
    ),
}
FAILED_ADAPTIVE_FILE_SHA256 = {
    "local/odebf/state/s04-p1r4-adaptive-tau-causal-v1.submission-intent.json": (
        "4a7d8ce4e7699fa33aa825a24696de9b94734884c689365e298402bbbd0dedc1"
    ),
    "local/odebf/state/s04-p1r4-adaptive-tau-causal-v1.submission-receipt.json": (
        "4fbe54bfa58d9b20a8d8216a179cf902a1936b29a2bb0737eeb28768a72b8b39"
    ),
    "local/odebf/logs/odebf_s04_p1r4adaptive_llama-16766.out": (
        "2e5292000e3663a5260a3b7f57557ca1da7780de833f6acdc26930241ce3cf9c"
    ),
    "local/odebf/logs/odebf_s04_p1r4adaptive_llama-16766.err": (
        "625949a9ef03ffb976a5e8fb038fff4eed2ef3483e2f4803a74d9c4226f5cfa1"
    ),
    "local/odebf/logs/odebf_s04_p1r4adaptive_qwen-16767.out": (
        "2e5292000e3663a5260a3b7f57557ca1da7780de833f6acdc26930241ce3cf9c"
    ),
    "local/odebf/logs/odebf_s04_p1r4adaptive_qwen-16767.err": (
        "730a603cea81e49a90ee9cd6652c9211cc2c186802be8fcfca977e400cb14302"
    ),
}


def _run(
    args: Sequence[str],
    *,
    check: bool = True,
    cwd: Path = REPO_ROOT,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        cwd=cwd,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )


def _write_once(path: Path, value: dict[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(payload).hexdigest()


def _p1r4_immutability_gate() -> dict[str, str]:
    observed: dict[str, str] = {}
    parent = REPO_ROOT / "local/odebf/results"
    for name, expected in P1R4_RESULT_SHA256.items():
        root = parent / name
        if root.is_symlink() or not root.is_dir():
            raise ODEBFContractError("P1R4 result root identity differs")
        digest, count = sha256_regular_tree(root)
        if digest != expected or count != 113:
            raise ODEBFContractError("P1R4 result root immutability differs")
        observed[name] = digest
    for relative, expected in P1R4_FILE_SHA256.items():
        path = REPO_ROOT / relative
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
            raise ODEBFContractError("P1R4 log/state immutability differs")
        observed[relative] = expected
    return observed


def _failed_adaptive_immutability_gate() -> dict[str, str]:
    observed: dict[str, str] = {}
    parent = REPO_ROOT / "local/odebf/results"
    for name, expected in FAILED_ADAPTIVE_RESULT_SHA256.items():
        root = parent / name
        if root.is_symlink() or not root.is_dir():
            raise ODEBFContractError("failed adaptive result root identity differs")
        digest, count = sha256_regular_tree(root)
        if digest != expected or count != 5:
            raise ODEBFContractError("failed adaptive result immutability differs")
        observed[name] = digest
    for relative, expected in FAILED_ADAPTIVE_FILE_SHA256.items():
        path = REPO_ROOT / relative
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
            raise ODEBFContractError("failed adaptive log/state immutability differs")
        observed[relative] = expected
    return observed


def _base_frozen_gate() -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative in FROZEN_PATHS:
        baseline = _run(["git", "show", f"{BASE_HEAD}:{relative}"]).stdout.encode(
            "utf-8"
        )
        path = REPO_ROOT / relative
        if path.is_symlink() or not path.is_file() or path.read_bytes() != baseline:
            raise ODEBFContractError("adaptive frozen scientific bytes changed")
        observed[relative] = sha256_file(path)
    return observed


def _source_manifest_gate() -> str:
    value, raw_sha256 = load_rooted_json(
        SOURCE_MANIFEST,
        expected_schema="ode-edit-s04-ode-bf-p1r4-adaptive-r1-source-manifest/v1",
    )
    entries = value.get("entries")
    if (
        value.get("instruction_id") != REPAIR_INSTRUCTION_ID
        or value.get("expected_parent") != BASE_HEAD
        or not isinstance(entries, list)
        or not entries
    ):
        raise ODEBFContractError("adaptive source manifest provenance differs")
    locked = {
        str(entry["path"]): (str(entry["sha256"]), int(entry["size"]))
        for entry in entries
    }
    if len(locked) != len(entries):
        raise ODEBFContractError("adaptive source manifest repeats a path")
    tracked = _run(["git", "ls-tree", "-r", "--name-only", "HEAD"]).stdout.splitlines()
    expected_paths = {
        path
        for path in tracked
        if (
            path.startswith("project/run_scripts/ode_bf/")
            or path.startswith("project/run_scripts/session04_ode_bf_")
        )
        and path
        != "project/run_scripts/ode_bf/locks/source_manifest_p1r4_adaptive_r1.json"
    }
    if set(locked) != expected_paths:
        raise ODEBFContractError("adaptive source manifest path set differs")
    for relative, (digest, size) in sorted(locked.items()):
        path = REPO_ROOT / relative
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != size
            or sha256_file(path) != digest
        ):
            raise ODEBFContractError("adaptive source manifest content differs")
    return raw_sha256


def _source_gate() -> str:
    head = _run(["git", "rev-parse", "HEAD"]).stdout.strip()
    parent = _run(["git", "rev-parse", "HEAD^"]).stdout.strip()
    branch = _run(["git", "branch", "--show-current"]).stdout.strip()
    if parent != BASE_HEAD or head == BASE_HEAD or branch != BRANCH:
        raise ODEBFContractError("adaptive checkpoint ancestry/branch differs")
    if _run(["git", "status", "--porcelain", "--untracked-files=no"]).stdout:
        raise ODEBFContractError("adaptive tracked source is not clean")
    untracked = _run(
        ["git", "ls-files", "--others", "--exclude-standard"]
    ).stdout.splitlines()
    if any(
        not (
            path.startswith("agents/server2/")
            or path.startswith("audits/servers/server2/")
        )
        for path in untracked
    ):
        raise ODEBFContractError("foreign untracked path overlaps adaptive gate")
    changed = set(
        _run(["git", "diff", "--name-only", f"{BASE_HEAD}..{head}"]).stdout.splitlines()
    )
    if changed != ALLOWED_CHANGED_PATHS:
        raise ODEBFContractError("adaptive checkpoint path inventory differs")
    _run(["git", "diff", "--check", f"{BASE_HEAD}..{head}"])
    _source_manifest_gate()
    return head


def _adaptive_ast_firewall() -> str:
    paths = (
        PACKAGE_ROOT / "p1_adaptive.py",
        PACKAGE_ROOT / "p1_adaptive_runtime.py",
        PACKAGE_ROOT / "p1_stepwise.py",
        REPO_ROOT / "project/run_scripts/session04_ode_bf_p1_adaptive.py",
        DRY_PLAN,
    )
    assert_ast_firewall(paths)
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "generate"
            ):
                raise ODEBFContractError("adaptive path called model.generate")
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                lowered = node.value.lower()
                if "session03" in lowered or "knowledge-revision" in lowered:
                    raise ODEBFContractError("adaptive path references foreign namespace")
    runtime = (PACKAGE_ROOT / "p1_adaptive_runtime.py").read_text(encoding="utf-8")
    required = (
        "trust_radius_fixed_on_reject",
        "target_weight_shared_delta_tau",
        "action_frozen_before_open",
        "load_counterfact_cases_after_freeze",
        "FULL_CURRENT_RESIDUAL_DEFINITION",
        "LEGACY_PRE_SHARED_RESIDUAL_DEFINITION",
        "counter_delta",
        "coefficient_l2_norm",
        "entry_current_z_by_layer",
    )
    if any(fragment not in runtime for fragment in required):
        raise ODEBFContractError("adaptive runtime source guard differs")
    if (
        runtime.index("action-freeze.json")
        > runtime.index("cases = load_counterfact_cases_after_freeze")
    ):
        raise ODEBFContractError("adaptive held-out loader preceded action freeze")
    if "K_total=20" in runtime or "eta=1/20" in runtime:
        raise ODEBFContractError("superseded simple-K20 semantics reappeared")
    if (
        ".solve_key_view" in runtime
        or ".risk_key_view" in runtime
        or "history.solve_keys" not in runtime
        or "history.risk_keys" not in runtime
    ):
        raise ODEBFContractError("adaptive history view interface differs")
    assert_no_alias_specific_controller_branch(
        [
            PACKAGE_ROOT / "p1_adaptive.py",
            PACKAGE_ROOT / "p1_adaptive_runtime.py",
            PACKAGE_ROOT / "p1_backend.py",
        ]
    )
    return canonical_source_digest(paths)


def canonical_source_digest(paths: Sequence[Path]) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                str(path.relative_to(REPO_ROOT)): sha256_file(path)
                for path in sorted(paths)
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _lock_and_seal_gate() -> dict[str, Any]:
    stream = verify_p1_stream_seal(
        json.loads((LOCK_ROOT / "p1r2_seqb10_stream_seal.json").read_text())
    )
    population = verify_p1_population_seal(
        json.loads((LOCK_ROOT / "p1r2_p_population_seal.json").read_text()),
        stream=stream,
    )
    numerical, numerical_sha256 = load_rooted_json(
        LOCK_ROOT / "numerical_lock_p1r4_adaptive.json",
        expected_schema="ode-edit-s04-ode-bf-p1r4-adaptive-numerical-lock/v1",
    )
    expected_locks = {
        item.value: adaptive_lock(item).identity() for item in ADAPTIVE_VARIANTS
    }
    time_forecast = forecast_p1_adaptive_b10_time()
    memory_identities = {
        alias: forecast_p1_adaptive_b10_memory(
            ARTIFACT_LOCK, BASE_ARTIFACT_LOCK, alias
        ).identity()
        for alias in MODEL_ALIASES
    }
    resource_lock = numerical.get("resource", {})
    if (
        numerical.get("instruction_id") != ADAPTIVE_INSTRUCTION_ID
        or numerical.get("accepted_lineage_parent") != SCIENTIFIC_LINEAGE_PARENT
        or numerical.get("controller_identity_sha256") != P1ControllerLock().identity()
        or numerical.get("variant_lock_sha256") != expected_locks
        or numerical.get("stream_root_digest") != stream["root_digest"]
        or numerical.get("p_population_root_digest") != population["root_digest"]
        or numerical.get("variants") != [item.value for item in ADAPTIVE_VARIANTS]
        or numerical.get("scientific_promotion_authorized") is not False
        or resource_lock.get("maximum_trials")
        != time_forecast.maximum_trial_count
        or resource_lock.get("maximum_postfreeze_states")
        != time_forecast.maximum_postfreeze_state_count
        or resource_lock.get("time_forecast_identity_sha256")
        != time_forecast.identity()
        or resource_lock.get("llama_memory_forecast_identity_sha256")
        != memory_identities["llama3-8b-inst"]
        or resource_lock.get("qwen_memory_forecast_identity_sha256")
        != memory_identities["qwen2.5-7b-inst"]
    ):
        raise ODEBFContractError("adaptive numerical/seal lock differs")
    toy = refinement_ratio()
    if toy.get("passed") is not True:
        raise ODEBFContractError("adaptive float64 refinement gate failed")
    return {
        "numerical_lock_sha256": numerical_sha256,
        "stream_root_digest": stream["root_digest"],
        "population_root_digest": population["root_digest"],
        "variant_lock_sha256": expected_locks,
        "refinement_toy": toy,
    }


def _cpu_static_gate(source_head: str) -> dict[str, Any]:
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get(
        "TRANSFORMERS_OFFLINE"
    ) != "1":
        raise ODEBFContractError("adaptive canonical offline environment differs")
    _run(["scripts/check-session-boundary.sh", SESSION_ID])
    _run(["scripts/check-agent-access.sh", "--all-changed"])
    frozen = _base_frozen_gate()
    lock_gate = _lock_and_seal_gate()
    adaptive_firewall_sha256 = _adaptive_ast_firewall()
    runtime_firewall_sha256 = preserved_diag._assert_p1_runtime_ast_firewall()
    cuda_source_sha256 = preserved_p1._cuda_preflight_source_gate()
    immutable = {
        "p1r2": preserved_diag._p1r2_immutability_gate(),
        "p1r3": preserved_diag._p1r3_immutability_gate(),
        "p1r4": _p1r4_immutability_gate(),
        "failed_adaptive": _failed_adaptive_immutability_gate(),
    }
    preserved_p0._r0_immutability_gate()
    preserved_p0._r1_immutability_gate()
    preserved_p0._r2_immutability_gate()
    preserved_p0._r3_immutability_gate()
    preserved_p1._r4_immutability_gate()
    preserved_p1._p1_r0_immutability_gate()
    preserved_p1._p1_r1_immutability_gate()

    compile_paths = sorted(PACKAGE_ROOT.rglob("*.py")) + [
        REPO_ROOT / "project/run_scripts/session04_ode_bf_p1_adaptive.py",
        DRY_PLAN,
        Path(__file__),
    ]
    for path in compile_paths:
        compile(path.read_text(encoding="utf-8"), str(path), "exec", dont_inherit=True)
    _run(["bash", "-n", str(SBATCH)])

    runtime = "/mnt/raid5/janghj/EasyEdit/.venv/bin/python"
    test_env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": f"{REPO_ROOT}:/mnt/raid5/janghj/EasyEdit",
        "MPLCONFIGDIR": str(REPO_ROOT / "local/odebf/matplotlib-p1r4-adaptive-r1-gate"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    tests = _run(
        [
            runtime,
            "-W",
            "error",
            "-m",
            "unittest",
            "discover",
            "-s",
            "project/run_scripts/ode_bf/tests",
            "-p",
            "test_*.py",
        ],
        env=test_env,
    )
    test_count = re.search(r"Ran ([0-9]+) tests?", tests.stderr)
    if test_count is None or int(test_count.group(1)) < 181 or "\nOK\n" not in tests.stderr:
        raise ODEBFContractError("adaptive full warnings-as-errors suite failed")
    repair = _run(
        [
            runtime,
            "-W",
            "error",
            "-m",
            "unittest",
            "project.run_scripts.ode_alloc.tests.test_p1_runtime_contracts",
            "project.run_scripts.ode_alloc.tests.test_p1_submission_contracts",
        ],
        env=test_env,
    )
    repair_count = re.search(r"Ran ([0-9]+) tests?", repair.stderr)
    if repair_count is None or int(repair_count.group(1)) != 17 or "\nOK\n" not in repair.stderr:
        raise ODEBFContractError("preserved CUDA 17-test gate failed")

    dry_args = [runtime, str(DRY_PLAN), "--source-head", source_head]
    dry_first = _run(dry_args, env=test_env).stdout
    dry_second = _run(dry_args, env=test_env).stdout
    if dry_first != dry_second:
        raise ODEBFContractError("adaptive dry plan is not byte-repeatable")
    dry = json.loads(dry_first)
    if (
        dry.get("variants") != [item.value for item in ADAPTIVE_VARIANTS]
        or dry.get("edit_batch_size") != 10
        or dry.get("sequential_batch_count") != 1
        or dry.get("accepted_state_field_refresh_only") is not True
        or dry.get("reject_advances_tau") is not False
        or dry.get("scientific_promotion_authorized") is not False
        or dry.get("postfreeze_states")
        != "W0-N32-and-every-unique-accepted-snapshot"
    ):
        raise ODEBFContractError("adaptive dry scientific scope differs")

    artifacts: dict[str, Any] = {}
    forecasts: dict[str, Any] = {}
    time_forecast = forecast_p1_adaptive_b10_time()
    for alias in MODEL_ALIASES:
        guard = ODEBFArtifactGuard(REPO_ROOT, ARTIFACT_LOCK, alias)
        receipt = guard.preflight()
        guard.assert_unchanged()
        memory = forecast_p1_adaptive_b10_memory(
            ARTIFACT_LOCK, BASE_ARTIFACT_LOCK, alias
        )
        if (
            memory.forecast_gpu_peak_mib > 65_000
            or memory.forecast_host_peak_mib > 65_000
            or memory.dense_fp64_full_delta
            or time_forecast.forecast_seconds > 24 * 60 * 60
        ):
            raise ODEBFContractError("adaptive memory/time forecast gate failed")
        artifacts[alias] = {
            "lock_sha256": receipt.lock_sha256,
            "revision": receipt.base_model_revision,
            "projector_sha256": receipt.projector_sha256,
        }
        forecasts[alias] = memory.raw_free_payload()
    return {
        "source_head": source_head,
        "test_count": int(test_count.group(1)),
        "preserved_cuda_test_count": int(repair_count.group(1)),
        "source_manifest_sha256": _source_manifest_gate(),
        "lock_gate": lock_gate,
        "frozen_source_sha256": canonical_source_digest(
            [REPO_ROOT / value for value in FROZEN_PATHS]
        ),
        "adaptive_firewall_sha256": adaptive_firewall_sha256,
        "runtime_firewall_sha256": runtime_firewall_sha256,
        "cuda_preflight_source_sha256": cuda_source_sha256,
        "immutability": immutable,
        "dry_plan_sha256": hashlib.sha256(dry_first.encode("utf-8")).hexdigest(),
        "artifacts": artifacts,
        "memory_forecasts": forecasts,
        "time_forecast": time_forecast.raw_free_payload(),
    }


def _output_gate() -> dict[str, Path]:
    result_parent = REPO_ROOT / "local/odebf/results"
    log_parent = REPO_ROOT / "local/odebf/logs"
    state_parent = REPO_ROOT / "local/odebf/state"
    for parent in (result_parent, log_parent, state_parent):
        if parent.exists() and (parent.is_symlink() or not parent.is_dir()):
            raise ODEBFContractError("adaptive output parent differs")
        parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    roots = {
        alias: result_parent / expected_p1r4_adaptive_result_name(alias)
        for alias in MODEL_ALIASES
    }
    if any(path.exists() or path.is_symlink() for path in roots.values()):
        raise ODEBFContractError("adaptive result root exists")
    for job_name in JOB_NAMES.values():
        if list(log_parent.glob(f"{job_name}-*")):
            raise ODEBFContractError("adaptive log namespace exists")
    intent = state_parent / "s04-p1r4-adaptive-tau-causal-r1-v1.submission-intent.json"
    receipt = state_parent / "s04-p1r4-adaptive-tau-causal-r1-v1.submission-receipt.json"
    if intent.exists() or intent.is_symlink() or receipt.exists() or receipt.is_symlink():
        raise ODEBFContractError("adaptive pair was already attempted")
    roots["__intent__"] = intent
    roots["__receipt__"] = receipt
    roots["__logs__"] = log_parent
    return roots


def _submit_one(
    *, alias: str, output_root: Path, source_head: str, log_parent: Path
) -> str:
    job_name = JOB_NAMES[alias]
    result = _run(
        [
            "sbatch",
            "--parsable",
            "--job-name",
            job_name,
            "--output",
            str(log_parent / f"{job_name}-%j.out"),
            "--error",
            str(log_parent / f"{job_name}-%j.err"),
            str(SBATCH),
            alias,
            str(output_root),
            source_head,
            ADAPTIVE_RESULT_TOKEN,
        ]
    )
    job_id = result.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise ODEBFContractError("sbatch did not return an adaptive job ID")
    return job_id


def main() -> int:
    source_head = _source_gate()
    gate = _cpu_static_gate(source_head)
    paths = _output_gate()
    _p1r4_immutability_gate()
    before = preserved_p0._scheduler_jobs()
    if any(record.job_name in JOB_NAMES.values() for record in before):
        raise ODEBFContractError("adaptive scheduler name exists")
    local_before, cluster_before = assert_node_local_capacity(
        before, new_gpu_count=2
    )
    if 2 * 65_000 > 2 * MEMORY_CAP_MIB_PER_GPU:
        raise ODEBFContractError("adaptive pair host memory exceeds server2 cap")
    intent = {
        "schema": "ode-edit-s04-ode-bf-p1r4-adaptive-r1-submission-intent/v1",
        "instruction_id": REPAIR_INSTRUCTION_ID,
        "scientific_instruction_id": ADAPTIVE_INSTRUCTION_ID,
        "authorized_attempt": "HISTORY_VIEW_REPAIR_R1_CAUSAL_PAIR",
        "source_head": source_head,
        "jobs": JOB_NAMES,
        "results": {
            alias: str(paths[alias].relative_to(REPO_ROOT)) for alias in MODEL_ALIASES
        },
        "node_local_gpu_before": local_before,
        "cluster_project_gpu_diagnostic": cluster_before,
        "new_gpu_count": 2,
        "causal_diagnostic_only": True,
        "scientific_promotion_authorized": False,
        "cpu_static_gate": gate,
        "retry_or_resubmit": False,
    }
    intent_sha256 = _write_once(paths["__intent__"], intent)
    submitted: dict[str, str] = {}
    failure: dict[str, str] | None = None
    try:
        first = MODEL_ALIASES[0]
        submitted[first] = _submit_one(
            alias=first,
            output_root=paths[first],
            source_head=source_head,
            log_parent=paths["__logs__"],
        )
        assert_node_local_capacity(preserved_p0._scheduler_jobs(), new_gpu_count=1)
        second = MODEL_ALIASES[1]
        submitted[second] = _submit_one(
            alias=second,
            output_root=paths[second],
            source_head=source_head,
            log_parent=paths["__logs__"],
        )
    except Exception as exc:
        failure = {
            "exception_class": type(exc).__name__,
            "exception_message_sha256": hashlib.sha256(
                str(exc).encode("utf-8")
            ).hexdigest(),
        }
    receipt = {
        "schema": "ode-edit-s04-ode-bf-p1r4-adaptive-r1-submission-receipt/v1",
        "instruction_id": REPAIR_INSTRUCTION_ID,
        "scientific_instruction_id": ADAPTIVE_INSTRUCTION_ID,
        "authorized_attempt": "HISTORY_VIEW_REPAIR_R1_CAUSAL_PAIR",
        "status": "SUBMITTED_PAIR" if failure is None else "PARTIAL_OR_FAILED_NO_RETRY",
        "source_head": source_head,
        "intent_sha256": intent_sha256,
        "submitted": submitted,
        "failure": failure,
        "retry_or_resubmit": False,
    }
    receipt_sha256 = _write_once(paths["__receipt__"], receipt)
    if failure is not None:
        raise ODEBFContractError("adaptive one-shot submission did not complete")
    print(
        json.dumps(
            {
                "status": "SUBMITTED_PAIR",
                "source_head": source_head,
                "jobs": submitted,
                "intent_sha256": intent_sha256,
                "receipt_sha256": receipt_sha256,
                "node": "server2",
                "retry_or_resubmit": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
