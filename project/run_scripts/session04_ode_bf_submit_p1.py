#!/usr/bin/env python3
"""One-shot fail-closed submission of the sealed ODE-BF P1 model pair."""

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
from project.run_scripts.ode_bf.artifacts import (
    ODEBFArtifactGuard,
    load_rooted_json,
    sha256_file,
    sha256_regular_tree,
)
from project.run_scripts.ode_bf.contracts import MODEL_ALIASES, ODEBFContractError
from project.run_scripts.ode_bf.firewall import (
    assert_ast_firewall,
    assert_no_alias_specific_controller_branch,
)
from project.run_scripts.ode_bf.p1_controller import P1ControllerLock
from project.run_scripts.ode_bf.p1_runtime import expected_p1_result_name
from project.run_scripts.ode_bf.p1_selection import (
    verify_p1_population_seal,
    verify_p1_stream_seal,
)
from project.run_scripts.ode_bf.resource import (
    MEMORY_CAP_MIB_PER_GPU,
    assert_node_local_capacity,
    forecast_p1_b10_memory,
)
from project.run_scripts.session04_ode_bf_p1 import RUN_TOKEN
from project.run_scripts.session04_ode_bf_p1_dry_plan import JOB_NAMES


SESSION_ID = "019fc5ec-f85b-7770-a73a-1d19be1cd491"
BRANCH = "codex/odeeditsh2-ode-bf-v1"
BASE_HEAD = "e753972da50a5d6fa9789ef2e9083c9b0549c3d0"
INSTRUCTION_ID = "ODEEDIT-S04-ODE-BF-TRUST-RATIO-MEAN-P-P1R2-V1"
SCIENTIFIC_INSTRUCTION_ID = INSTRUCTION_ID
SCIENTIFIC_BASE_HEAD = BASE_HEAD
PACKAGE_ROOT = REPO_ROOT / "project/run_scripts/ode_bf"
LOCK_ROOT = PACKAGE_ROOT / "locks"
SBATCH = REPO_ROOT / "project/run_scripts/session04_ode_bf_p1.sbatch"
DRY_PLAN = REPO_ROOT / "project/run_scripts/session04_ode_bf_p1_dry_plan.py"
SOURCE_MANIFEST = LOCK_ROOT / "source_manifest_p1r2.json"
ARTIFACT_LOCK = LOCK_ROOT / "p0_artifact_lock.json"
BASE_ARTIFACT_LOCK = REPO_ROOT / "project/run_scripts/ode_alloc/p0_artifact_lock_r1.json"
P1_LOCK_SHA256 = {
    "numerical_lock_p1r2.json": "0cdb4ff528f0eddea7b40b9d36a103fa372dca8433da8a0a2cf36f77ad2fa903",
    "p1r2_seqb10_stream_seal.json": "689dcfbf95b074a946088a31b626f2b64082401a1b525e326ab1e8c46b8a488f",
    "p1r2_p_population_seal.json": "6b36d234d676afb7cb22c70b042f0319bb8687cb12ed4e2f45ce98002979a96f",
}
PRESERVED_LOCK_SHA256 = {
    "p0_artifact_lock.json": "623482b05c670b703eebc65493779e863021509f0c0855d5fafb75c0b7b24910",
    "numerical_lock_p0_r3.json": "40421f3f8ef0e47df842268bb68b9c9548398e27e0a9afecb125ab1b6f853fa1",
}
R4_RESULT_SHA256 = {
    "s04-p0-w64-receipt-field-r4-llama3-8b-inst-40421f3f": (
        "d2b9cd78a7d1120b582eef7358609ef078b965cea439d12c48601a998a2eb416"
    ),
    "s04-p0-w64-receipt-field-r4-qwen2.5-7b-inst-40421f3f": (
        "9b06f24658550d528b8e889c4a498e68d8fef6eded61791918fb24145c721882"
    ),
}
R4_FILE_SHA256 = {
    "local/odebf/logs/odebf_s04_p0r4_llama-16593.err": "8c525854b302c0cd4699bd3bf105626453b92a316ca2291936307e07f9f46328",
    "local/odebf/logs/odebf_s04_p0r4_llama-16593.out": "29b6c790847b26adad64b1a6ddb8a606afb018d53e9c9980bd131cd9285ff10f",
    "local/odebf/logs/odebf_s04_p0r4_qwen-16594.err": "7ccb996c827e264d848b03555c2cf30ec5beaf96058d6deb9232f50cf1f628e5",
    "local/odebf/logs/odebf_s04_p0r4_qwen-16594.out": "73cb11f80200c708479a0cf31014593d6d3f4460a184bbc19f5b20c6fe5265a0",
    "local/odebf/state/s04-p0-w64-receipt-field-r4-40421f3f.submission-intent.json": "2b57a42a1212d430099dba1af920f8fe5505ab2396ad91888386dbd88f7dbd40",
    "local/odebf/state/s04-p0-w64-receipt-field-r4-40421f3f.submission-receipt.json": "77d5ba7a2a478ad08ce17a8ac48c1e9eedb6ddcdec7e4a04dbaba1d0c9fac9df",
}

P1_R0_RESULT_SHA256 = {
    "s04-p1-seqb10-native-floor-llama3-8b-inst-v1": (
        "6122f08579ba1f5fb2f634f1eac859781594a5d5dd199e388e42d1449977b246"
    ),
    "s04-p1-seqb10-native-floor-qwen2.5-7b-inst-v1": (
        "6122f08579ba1f5fb2f634f1eac859781594a5d5dd199e388e42d1449977b246"
    ),
}
P1_R0_FILE_SHA256 = {
    "local/odebf/logs/odebf_s04_p1_seqb10_llama-16633.err": (
        "e8cb4e7db7245280f7e120c0921bd4d3d6dbe7e054ef9903557819d6a43b7d65"
    ),
    "local/odebf/logs/odebf_s04_p1_seqb10_llama-16633.out": (
        "2e5292000e3663a5260a3b7f57557ca1da7780de833f6acdc26930241ce3cf9c"
    ),
    "local/odebf/logs/odebf_s04_p1_seqb10_qwen-16634.err": (
        "60265ee2bfbec352d1c424aea8bc9ec7857d36f94dfce9310daaea6ae8070385"
    ),
    "local/odebf/logs/odebf_s04_p1_seqb10_qwen-16634.out": (
        "2e5292000e3663a5260a3b7f57557ca1da7780de833f6acdc26930241ce3cf9c"
    ),
    "local/odebf/state/s04-p1-seqb10-native-floor-v1.submission-intent.json": (
        "081cca0534cd5c0b8e622f139eee2563a7618df39292eb19d657e6f78b5b8464"
    ),
    "local/odebf/state/s04-p1-seqb10-native-floor-v1.submission-receipt.json": (
        "975c8b11b17883040ae7f9194feb3152ffd137f3b705660388c37f3ca6668721"
    ),
}

P1_R1_RESULT_SHA256 = {
    "s04-p1r1-seqb10-native-floor-llama3-8b-inst-v1": (
        "6015561d0803f25149bade061fe2e9010577f81a30fa5a42d1b198cce9f06d23"
    ),
    "s04-p1r1-seqb10-native-floor-qwen2.5-7b-inst-v1": (
        "6c598b687ca3918cfdf6bf5e4f69fd5c7666c9df19c45813a99373436b48ed44"
    ),
}
P1_R1_FILE_SHA256 = {
    "local/odebf/logs/odebf_s04_p1r1_seqb10_llama-16641.err": (
        "7bcdbf07dbdb4e46e3a277e5939fea5eaf42870ebfaca8c350bcb3aed307f5dd"
    ),
    "local/odebf/logs/odebf_s04_p1r1_seqb10_llama-16641.out": (
        "ff218c641536df961ed37b6a88ac2a02ae20ac1b125ace3adb41721c3b03c285"
    ),
    "local/odebf/logs/odebf_s04_p1r1_seqb10_qwen-16642.err": (
        "513aaa629c9ea92be9c141042eb9a08625110c6993f9a6df4f39fa9bf0f0da3e"
    ),
    "local/odebf/logs/odebf_s04_p1r1_seqb10_qwen-16642.out": (
        "423b49a183651c2ae0b7e2c8743048c39afdd1a5261e5f99f55d11a1afa77fa2"
    ),
    "local/odebf/state/s04-p1r1-seqb10-native-floor-v1.submission-intent.json": (
        "89ae7068b4672f41738665ba476b7b53a93fa03ea9c2fcb365334532d76d0a05"
    ),
    "local/odebf/state/s04-p1r1-seqb10-native-floor-v1.submission-receipt.json": (
        "cb0ecd7f935429206594ed83f26f8f7f134e4cacc99252b487c1052ea54f0237"
    ),
}

P1_ALLOWED_CHANGED_PATHS = {
    "project/run_scripts/ode_bf/barriers.py",
    "project/run_scripts/ode_bf/locks/numerical_lock_p1r2.json",
    "project/run_scripts/ode_bf/locks/p1r2_p_population_seal.json",
    "project/run_scripts/ode_bf/locks/p1r2_seqb10_stream_seal.json",
    "project/run_scripts/ode_bf/locks/source_manifest_p1r2.json",
    "project/run_scripts/ode_bf/p1_backend.py",
    "project/run_scripts/ode_bf/p1_controller.py",
    "project/run_scripts/ode_bf/p1_runtime.py",
    "project/run_scripts/ode_bf/p1_selection.py",
    "project/run_scripts/ode_bf/routing.py",
    "project/run_scripts/ode_bf/tests/test_barrier_attribution.py",
    "project/run_scripts/ode_bf/tests/test_p1_backend_controller.py",
    "project/run_scripts/ode_bf/tests/test_p1_runtime_contracts.py",
    "project/run_scripts/ode_bf/tests/test_p1_selection.py",
    "project/run_scripts/ode_bf/tests/test_p1_submission.py",
    "project/run_scripts/ode_bf/tests/test_woodbury_routing.py",
    "project/run_scripts/session04_ode_bf_p1.py",
    "project/run_scripts/session04_ode_bf_p1.sbatch",
    "project/run_scripts/session04_ode_bf_p1_dry_plan.py",
    "project/run_scripts/session04_ode_bf_submit_p1.py",
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


def _r4_immutability_gate() -> dict[str, str]:
    observed: dict[str, str] = {}
    result_parent = REPO_ROOT / "local/odebf/results"
    for name, expected in R4_RESULT_SHA256.items():
        root = result_parent / name
        if root.is_symlink() or not root.is_dir():
            raise ODEBFContractError("R4 result root identity differs")
        digest, count = sha256_regular_tree(root)
        if digest != expected or count != 12:
            raise ODEBFContractError("R4 result root immutability differs")
        observed[name] = digest
    for relative, expected in R4_FILE_SHA256.items():
        path = REPO_ROOT / relative
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
            raise ODEBFContractError("R4 log/state immutability differs")
        observed[relative] = expected
    return observed


def _p1_r0_immutability_gate() -> dict[str, str]:
    observed: dict[str, str] = {}
    result_parent = REPO_ROOT / "local/odebf/results"
    for name, expected in P1_R0_RESULT_SHA256.items():
        root = result_parent / name
        if root.is_symlink() or not root.is_dir():
            raise ODEBFContractError("P1 R0 result root identity differs")
        digest, count = sha256_regular_tree(root)
        if digest != expected or count != 2:
            raise ODEBFContractError("P1 R0 result root immutability differs")
        observed[name] = digest
    for relative, expected in P1_R0_FILE_SHA256.items():
        path = REPO_ROOT / relative
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
            raise ODEBFContractError("P1 R0 log/state immutability differs")
        observed[relative] = expected
    return observed


def _p1_r1_immutability_gate() -> dict[str, str]:
    observed: dict[str, str] = {}
    result_parent = REPO_ROOT / "local/odebf/results"
    for name, expected in P1_R1_RESULT_SHA256.items():
        root = result_parent / name
        if root.is_symlink() or not root.is_dir():
            raise ODEBFContractError("P1 R1 result root identity differs")
        digest, count = sha256_regular_tree(root)
        if digest != expected or count != 13:
            raise ODEBFContractError("P1 R1 result root immutability differs")
        observed[name] = digest
    for relative, expected in P1_R1_FILE_SHA256.items():
        path = REPO_ROOT / relative
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
            raise ODEBFContractError("P1 R1 log/state immutability differs")
        observed[relative] = expected
    return observed


def _cuda_preflight_source_gate() -> str:
    path = PACKAGE_ROOT / "p1_runtime.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    reset_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "reset_peak_memory_stats"
    ]
    if reset_calls:
        raise ODEBFContractError("ODE-BF P1 retains a direct CUDA peak reset")
    allowed_import_count = 0
    forbidden_fragments = (
        "session03",
        "ode_alloc.p1_evaluator",
        "knowledge-revision",
        "knowledge_revision",
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = (node.module or "").lower()
            if module == "project.run_scripts.ode_alloc.p1_runtime":
                if (
                    len(node.names) != 1
                    or node.names[0].name != "_prepare_p1_cuda_runtime"
                    or node.names[0].asname
                    != "_prepare_preserved_one_device_cuda_runtime"
                ):
                    raise ODEBFContractError("P1 CUDA helper import scope differs")
                allowed_import_count += 1
            elif "ode_alloc.p1_runtime" in module or any(
                fragment in module for fragment in forbidden_fragments
            ):
                raise ODEBFContractError("forbidden P1 foreign/session import found")
        elif isinstance(node, ast.Import):
            names = [alias.name.lower() for alias in node.names]
            if any(
                "ode_alloc.p1_runtime" in name
                or any(fragment in name for fragment in forbidden_fragments)
                for name in names
            ):
                raise ODEBFContractError("forbidden P1 foreign/session import found")
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "generate"
        ):
            raise ODEBFContractError("model.generate is forbidden in ODE-BF P1")
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            lowered = node.value.lower()
            if (
                ("/" in lowered or "\\" in lowered or "." in lowered)
                and ("session03" in lowered or "knowledge-revision" in lowered)
            ):
                raise ODEBFContractError("forbidden P1 foreign/session string found")
    if allowed_import_count != 1:
        raise ODEBFContractError("P1 preserved CUDA helper import count differs")
    required_import = (
        "_prepare_p1_cuda_runtime as _prepare_preserved_one_device_cuda_runtime"
    )
    run_start = source.index("def run_p1(")
    run_source = source[run_start:]
    positions = (
        run_source.index("_initialize_p1_cuda_runtime(stages)"),
        run_source.index("seed_all(COMMON_SEED)"),
        run_source.index('measure("model_load")'),
    )
    if required_import not in source or positions != tuple(sorted(positions)):
        raise ODEBFContractError("P1 preserved CUDA preflight source order differs")
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _source_manifest_gate() -> str:
    value, raw_sha = load_rooted_json(
        SOURCE_MANIFEST,
        expected_schema="ode-edit-s04-ode-bf-p1r2-source-manifest/v2",
    )
    entries = value.get("entries")
    if (
        value.get("instruction_id") != INSTRUCTION_ID
        or value.get("expected_base") != BASE_HEAD
        or not isinstance(entries, list)
        or not entries
    ):
        raise ODEBFContractError("P1 source manifest provenance differs")
    locked = {
        entry["path"]: (entry["sha256"], int(entry["size"]))
        for entry in entries
    }
    if len(locked) != len(entries):
        raise ODEBFContractError("P1 source manifest repeats a path")
    observed = _run(["git", "ls-tree", "-r", "--name-only", "HEAD"]).stdout.splitlines()
    expected_paths = {
        path
        for path in observed
        if (
            path.startswith("project/run_scripts/ode_bf/")
            or path.startswith("project/run_scripts/session04_ode_bf_")
        )
        and path != "project/run_scripts/ode_bf/locks/source_manifest_p1r2.json"
    }
    if set(locked) != expected_paths:
        raise ODEBFContractError("P1 source manifest path set differs")
    for relative, (expected, expected_size) in sorted(locked.items()):
        path = REPO_ROOT / relative
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != expected_size
            or sha256_file(path) != expected
        ):
            raise ODEBFContractError("P1 source manifest content differs")
    return raw_sha


def _source_gate() -> str:
    head = _run(["git", "rev-parse", "HEAD"]).stdout.strip()
    parent = _run(["git", "rev-parse", "HEAD^"]).stdout.strip()
    branch = _run(["git", "branch", "--show-current"]).stdout.strip()
    if parent != BASE_HEAD or head == BASE_HEAD or branch != BRANCH:
        raise ODEBFContractError("P1 checkpoint ancestry/branch differs")
    if _run(["git", "status", "--porcelain", "--untracked-files=no"]).stdout:
        raise ODEBFContractError("P1 tracked source is not clean")
    untracked = _run(["git", "ls-files", "--others", "--exclude-standard"]).stdout.splitlines()
    if any(
        not (
            path.startswith("agents/server2/")
            or path.startswith("audits/servers/server2/")
        )
        for path in untracked
    ):
        raise ODEBFContractError("foreign untracked path overlaps the P1 worktree")
    changed = set(
        _run(["git", "diff", "--name-only", f"{BASE_HEAD}..{head}"]).stdout.splitlines()
    )
    if changed != P1_ALLOWED_CHANGED_PATHS:
        raise ODEBFContractError("P1 checkpoint path inventory differs")
    _run(["git", "diff", "--check", f"{BASE_HEAD}..{head}"])
    _source_manifest_gate()
    return head


def _lock_and_seal_gate() -> dict[str, str]:
    for name, expected in {**P1_LOCK_SHA256, **PRESERVED_LOCK_SHA256}.items():
        if sha256_file(LOCK_ROOT / name) != expected:
            raise ODEBFContractError("P1 reviewed lock/seal digest differs")
    stream_value = json.loads(
        (LOCK_ROOT / "p1r2_seqb10_stream_seal.json").read_text(encoding="utf-8")
    )
    stream = verify_p1_stream_seal(stream_value)
    population_value = json.loads(
        (LOCK_ROOT / "p1r2_p_population_seal.json").read_text(encoding="utf-8")
    )
    population = verify_p1_population_seal(population_value, stream=stream)
    numerical, numerical_sha = load_rooted_json(
        LOCK_ROOT / "numerical_lock_p1r2.json",
        expected_schema="ode-edit-s04-ode-bf-p1r2-numerical-lock/v2",
    )
    if (
        numerical["instruction_id"] != SCIENTIFIC_INSTRUCTION_ID
        or numerical["expected_base"] != SCIENTIFIC_BASE_HEAD
        or numerical["edit_batch_size"] != 10
        or numerical["sequential_batch_count"] != 4
        or numerical["logical_edits_per_model"] != 40
        or numerical["arms"] != ["N32_NATIVE", "F_G", "F_BF", "R_BF"]
        or numerical["controller_identity_sha256"] != P1ControllerLock().identity()
        or numerical["controller"].get("rho_accept") != 0.1
        or numerical["controller"].get("minimum_progress") != 1.0e-8
        or numerical["controller"].get("functional_p_decision")
        != "uniform-mean-samplewise-positive-incremental-theta0-kl"
        or numerical["controller"].get("functional_p_rawmax_role")
        != "diagnostic-only"
        or numerical["stream_root_digest"] != stream["root_digest"]
        or numerical["p_population_root_digest"] != population["root_digest"]
    ):
        raise ODEBFContractError("P1 numerical/seal cross identity differs")
    return {
        "numerical_lock_sha256": numerical_sha,
        "stream_root_digest": stream["root_digest"],
        "population_root_digest": population["root_digest"],
    }


def _cpu_static_gate(source_head: str) -> dict[str, Any]:
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get(
        "TRANSFORMERS_OFFLINE"
    ) != "1":
        raise ODEBFContractError("P1 canonical offline environment differs")
    _run(["scripts/check-session-boundary.sh", SESSION_ID])
    _run(["scripts/check-agent-access.sh", "--all-changed"])
    lock_gate = _lock_and_seal_gate()
    cuda_source_sha256 = _cuda_preflight_source_gate()
    r0 = preserved_p0._r0_immutability_gate()
    r1 = preserved_p0._r1_immutability_gate()
    r2 = preserved_p0._r2_immutability_gate()
    r3 = preserved_p0._r3_immutability_gate()
    r4 = _r4_immutability_gate()
    p1_r0 = _p1_r0_immutability_gate()
    p1_r1 = _p1_r1_immutability_gate()
    for relative, expected in preserved_p0.PRESERVED_REPAIR_SHA256.items():
        if sha256_file(REPO_ROOT / relative) != expected:
            raise ODEBFContractError("preserved CUDA repair byte identity differs")

    scientific = [path for path in PACKAGE_ROOT.glob("*.py") if path.name != "__init__.py"]
    cuda_runtime_path = PACKAGE_ROOT / "p1_runtime.py"
    assert_ast_firewall(
        [path for path in scientific if path != cuda_runtime_path]
        + [REPO_ROOT / "project/run_scripts/session04_ode_bf_p1.py"]
    )
    assert_no_alias_specific_controller_branch(
        [
            PACKAGE_ROOT / name
            for name in (
                "p1_backend.py",
                "p1_controller.py",
                "p1_replay.py",
                "p1_runtime.py",
                "p1_state.py",
            )
        ]
    )
    for path in sorted(PACKAGE_ROOT.rglob("*.py")) + [
        REPO_ROOT / "project/run_scripts/session04_ode_bf_p1.py",
        REPO_ROOT / "project/run_scripts/session04_ode_bf_p1_dry_plan.py",
        Path(__file__),
    ]:
        compile(path.read_text(encoding="utf-8"), str(path), "exec", dont_inherit=True)
    _run(["bash", "-n", str(SBATCH)])

    runtime = "/mnt/raid5/janghj/EasyEdit/.venv/bin/python"
    test_env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": f"{REPO_ROOT}:/mnt/raid5/janghj/EasyEdit",
        "MPLCONFIGDIR": str(REPO_ROOT / "local/odebf/matplotlib-p1-cpu-gate"),
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
    count_match = re.search(r"Ran ([0-9]+) tests?", tests.stderr)
    if count_match is None or "\nOK\n" not in tests.stderr:
        raise ODEBFContractError("full P1 warnings-as-errors CPU suite did not pass")
    repair_tests = _run(
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
    repair_count = re.search(r"Ran ([0-9]+) tests?", repair_tests.stderr)
    if repair_count is None or int(repair_count.group(1)) != 17 or "\nOK\n" not in repair_tests.stderr:
        raise ODEBFContractError("preserved CUDA repair 17-test gate did not pass")

    dry_args = [runtime, str(DRY_PLAN), "--source-head", source_head]
    dry_first = _run(dry_args, env=test_env).stdout
    dry_second = _run(dry_args, env=test_env).stdout
    if dry_first != dry_second:
        raise ODEBFContractError("P1 dry plan is not byte-repeatable")
    dry_value = json.loads(dry_first)
    if (
        dry_value.get("edit_batch_size") != 10
        or dry_value.get("sequential_batch_count") != 4
        or dry_value.get("arms") != ["N32_NATIVE", "F_G", "F_BF", "R_BF"]
        or any(
            int(job["memory_forecast"]["forecast_gpu_peak_mib"]) > 60_416
            or int(job["memory_forecast"]["forecast_host_peak_mib"]) > 60_416
            for job in dry_value.get("jobs", [])
        )
    ):
        raise ODEBFContractError("P1 dry resource/panel contract differs")

    artifacts: dict[str, Any] = {}
    forecasts: dict[str, Any] = {}
    for alias in MODEL_ALIASES:
        guard = ODEBFArtifactGuard(REPO_ROOT, ARTIFACT_LOCK, alias)
        receipt = guard.preflight()
        guard.assert_unchanged()
        forecast = forecast_p1_b10_memory(ARTIFACT_LOCK, BASE_ARTIFACT_LOCK, alias)
        artifacts[alias] = {
            "lock_sha256": receipt.lock_sha256,
            "revision": receipt.base_model_revision,
            "projector_sha256": receipt.projector_sha256,
            "held_ode_alloc_tree_sha256": receipt.held_ode_alloc_tree_sha256,
        }
        forecasts[alias] = forecast.raw_free_payload()
    return {
        "source_head": source_head,
        "test_count": int(count_match.group(1)),
        "preserved_repair_test_count": int(repair_count.group(1)),
        "source_manifest_sha256": _source_manifest_gate(),
        "lock_gate": lock_gate,
        "cuda_preflight_source_sha256": cuda_source_sha256,
        "r0_immutability_sha256": r0,
        "r1_immutability_sha256": r1,
        "r2_immutability_sha256": r2,
        "r3_immutability_sha256": r3,
        "r4_immutability": r4,
        "p1_r0_immutability": p1_r0,
        "p1_r1_immutability": p1_r1,
        "dry_plan_sha256": hashlib.sha256(dry_first.encode("utf-8")).hexdigest(),
        "artifacts": artifacts,
        "memory_forecasts": forecasts,
    }


def _output_gate() -> dict[str, Path]:
    result_parent = REPO_ROOT / "local/odebf/results"
    log_parent = REPO_ROOT / "local/odebf/logs"
    state_parent = REPO_ROOT / "local/odebf/state"
    for parent in (result_parent, log_parent, state_parent):
        if parent.exists() and (parent.is_symlink() or not parent.is_dir()):
            raise ODEBFContractError("P1 local output parent is not a real directory")
        parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    roots = {
        alias: result_parent / expected_p1_result_name(alias) for alias in MODEL_ALIASES
    }
    if any(path.exists() or path.is_symlink() for path in roots.values()):
        raise ODEBFContractError("P1 result root already exists")
    for job_name in JOB_NAMES.values():
        if list(log_parent.glob(f"{job_name}-*")):
            raise ODEBFContractError("P1 log namespace already exists")
    intent = state_parent / "s04-p1r2-seqb10-native-floor-v2.submission-intent.json"
    receipt = state_parent / "s04-p1r2-seqb10-native-floor-v2.submission-receipt.json"
    if intent.exists() or intent.is_symlink() or receipt.exists() or receipt.is_symlink():
        raise ODEBFContractError("P1 pair was already attempted")
    roots["__intent__"] = intent
    roots["__receipt__"] = receipt
    roots["__logs__"] = log_parent
    return roots


def _submit_one(
    *,
    alias: str,
    output_root: Path,
    source_head: str,
    log_parent: Path,
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
            RUN_TOKEN,
        ]
    )
    job_id = result.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise ODEBFContractError("sbatch did not return a numeric P1 job ID")
    return job_id


def main() -> int:
    source_head = _source_gate()
    gate = _cpu_static_gate(source_head)
    paths = _output_gate()
    _r4_immutability_gate()
    _p1_r0_immutability_gate()
    _p1_r1_immutability_gate()
    before = preserved_p0._scheduler_jobs()
    if any(record.job_name in JOB_NAMES.values() for record in before):
        raise ODEBFContractError("P1 scheduler job name already exists")
    local_before, cluster_before = assert_node_local_capacity(before, new_gpu_count=2)
    if 2 * 60_416 > 2 * MEMORY_CAP_MIB_PER_GPU:
        raise ODEBFContractError("P1 pair host-memory request exceeds server2 cap")
    intent = {
        "schema": "ode-edit-s04-ode-bf-p1r2-submission-intent/v2",
        "instruction_id": INSTRUCTION_ID,
        "scientific_instruction_id": SCIENTIFIC_INSTRUCTION_ID,
        "authorized_attempt": "FRESH_P1R2_V2",
        "source_head": source_head,
        "jobs": JOB_NAMES,
        "results": {
            alias: str(paths[alias].relative_to(REPO_ROOT)) for alias in MODEL_ALIASES
        },
        "node_local_gpu_before": local_before,
        "cluster_project_gpu_diagnostic": cluster_before,
        "new_gpu_count": 2,
        "cpu_static_gate": gate,
        "retry_or_resubmit": False,
    }
    intent_sha = _write_once(paths["__intent__"], intent)

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
        after_first = preserved_p0._scheduler_jobs()
        assert_node_local_capacity(after_first, new_gpu_count=1)
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
        "schema": "ode-edit-s04-ode-bf-p1r2-submission-receipt/v2",
        "instruction_id": INSTRUCTION_ID,
        "scientific_instruction_id": SCIENTIFIC_INSTRUCTION_ID,
        "authorized_attempt": "FRESH_P1R2_V2",
        "status": "SUBMITTED_PAIR" if failure is None else "PARTIAL_OR_FAILED_NO_RETRY",
        "source_head": source_head,
        "intent_sha256": intent_sha,
        "submitted": submitted,
        "failure": failure,
        "retry_or_resubmit": False,
    }
    receipt_sha = _write_once(paths["__receipt__"], receipt)
    if failure is not None:
        raise ODEBFContractError("P1 one-shot submission did not complete")
    print(
        json.dumps(
            {
                "status": "SUBMITTED_PAIR",
                "source_head": source_head,
                "jobs": submitted,
                "intent_sha256": intent_sha,
                "receipt_sha256": receipt_sha,
                "node": "server2",
                "retry_or_resubmit": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )
    return 0


def _entrypoint() -> None:
    try:
        code = main()
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "PRE_SUBMIT_OR_PARTIAL_HOLD",
                    "exception_class": type(exc).__name__,
                    "exception_message_sha256": hashlib.sha256(
                        str(exc).encode("utf-8")
                    ).hexdigest(),
                    "retry_or_resubmit": False,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
            flush=True,
        )
        raise
    raise SystemExit(code)


if __name__ == "__main__":
    _entrypoint()
