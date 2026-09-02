#!/usr/bin/env python3
"""One-shot submitter for the P1R4 full-residual arm diagnostic pair."""

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
from project.run_scripts.ode_bf.artifacts import (
    ODEBFArtifactGuard,
    load_rooted_json,
    sha256_file,
    sha256_regular_tree,
)
from project.run_scripts.ode_bf.contracts import (
    MODEL_ALIASES,
    ODEBFContractError,
)
from project.run_scripts.ode_bf.firewall import (
    FORBIDDEN_IMPORT_FRAGMENTS,
    assert_ast_firewall,
    assert_no_alias_specific_controller_branch,
)
from project.run_scripts.ode_bf.p1_controller import P1ControllerLock
from project.run_scripts.ode_bf.p1_runtime import (
    expected_p1r4_diagnostic_result_name,
)
from project.run_scripts.ode_bf.p1_selection import (
    verify_p1_population_seal,
    verify_p1_stream_seal,
)
from project.run_scripts.ode_bf.resource import (
    MEMORY_CAP_MIB_PER_GPU,
    assert_node_local_capacity,
    forecast_p1_b10_memory,
)
from project.run_scripts.session04_ode_bf_p1_diag import RUN_TOKEN
from project.run_scripts.session04_ode_bf_p1_diag_dry_plan import JOB_NAMES


SESSION_ID = "019fc5ec-f85b-7770-a73a-1d19be1cd491"
BRANCH = "codex/odeeditsh2-ode-bf-v1"
BASE_HEAD = "0d21dbc719106816991650db6e42f20c06a2e9cb"
INSTRUCTION_ID = "ODEEDIT-S04-ODE-BF-FULL-RESIDUAL-ARMS-P1R4-V1"
PACKAGE_ROOT = REPO_ROOT / "project/run_scripts/ode_bf"
LOCK_ROOT = PACKAGE_ROOT / "locks"
SBATCH = REPO_ROOT / "project/run_scripts/session04_ode_bf_p1_diag.sbatch"
DRY_PLAN = REPO_ROOT / "project/run_scripts/session04_ode_bf_p1_diag_dry_plan.py"
SOURCE_MANIFEST = LOCK_ROOT / "source_manifest_p1r4diag.json"
ARTIFACT_LOCK = LOCK_ROOT / "p0_artifact_lock.json"
BASE_ARTIFACT_LOCK = REPO_ROOT / "project/run_scripts/ode_alloc/p0_artifact_lock_r1.json"

FROZEN_SHA256 = {
    "project/run_scripts/ode_bf/routing.py": "9218a0a0e25098b85658640a6d280d40bdafcd80d9d824f96122d0c19b1809ff",
    "project/run_scripts/ode_bf/barriers.py": "b3d59cf7db3c3cc00e318115228777518ce6731c4797ec589a298319ce18c9f9",
    "project/run_scripts/ode_bf/p1_controller.py": "1015f02a81921abec1208bf892339266af4f415077dbbf7c050b6b5f581d8b3b",
    "project/run_scripts/ode_bf/p1_evaluator.py": "a574b7566b4fc5ee1ff76fdcd19feb829d5eec8b7158b2f91542344c416b0381",
    "project/run_scripts/ode_bf/p1_state.py": "700aa763361748cb1b089d5d0d228832ac405b9fdbb1bdf281e6beeaf174e9a8",
    "project/run_scripts/ode_bf/transaction.py": "95540b9e2df393b8aa0d59973a3a553272b64495b682e11829977d956966db7d",
    "project/run_scripts/ode_bf/locks/numerical_lock_p1r2.json": "0cdb4ff528f0eddea7b40b9d36a103fa372dca8433da8a0a2cf36f77ad2fa903",
    "project/run_scripts/ode_bf/locks/p1r2_seqb10_stream_seal.json": "689dcfbf95b074a946088a31b626f2b64082401a1b525e326ab1e8c46b8a488f",
    "project/run_scripts/ode_bf/locks/p1r2_p_population_seal.json": "6b36d234d676afb7cb22c70b042f0319bb8687cb12ed4e2f45ce98002979a96f",
    "project/run_scripts/ode_bf/locks/source_manifest_p1r2.json": "08447b7d99fd4ae1a5d786b6e9b1b09861759f09fb7dc1c0a255166f9670eabb",
    "project/run_scripts/ode_bf/locks/source_manifest_p1r2diag.json": "5c1ed23086ffebc11b774b9d49b2bc6cee2b7b117fd49e40fc38de5770b37b70",
}

P1R2_RESULT_SHA256 = {
    "s04-p1r2-seqb10-native-floor-llama3-8b-inst-v2": (
        "adeed4fbc06b292350b1c063118e55fc8f2689fe0f06ccbdf3662d7046fcd9e1"
    ),
    "s04-p1r2-seqb10-native-floor-qwen2.5-7b-inst-v2": (
        "f9bb33fcda85513c2f26430052d3c6b708c92aa0a35d2a508f8a80c574ee48d1"
    ),
}
P1R2_FILE_SHA256 = {
    "local/odebf/state/s04-p1r2-seqb10-native-floor-v2.submission-intent.json": (
        "dd5f7ec020a859a33b33854b64dd8d36e4917e53c293e777927d636858841411"
    ),
    "local/odebf/state/s04-p1r2-seqb10-native-floor-v2.submission-receipt.json": (
        "6744ac52e9583e69804aa42fbaf8797ad5d99aecc675d12aeacb5cb990eab609"
    ),
    "local/odebf/logs/odebf_s04_p1r2_seqb10_llama-16669.err": (
        "33c059206e91bd85a893180750fa5d86bcb472884f9e56ac40a54f9397243461"
    ),
    "local/odebf/logs/odebf_s04_p1r2_seqb10_llama-16669.out": (
        "2e5292000e3663a5260a3b7f57557ca1da7780de833f6acdc26930241ce3cf9c"
    ),
    "local/odebf/logs/odebf_s04_p1r2_seqb10_qwen-16670.err": (
        "74f8f7cbc1c03acdf357872fceaf9d3e2d0c1351293fc313aade4f5e093151d2"
    ),
    "local/odebf/logs/odebf_s04_p1r2_seqb10_qwen-16670.out": (
        "2e5292000e3663a5260a3b7f57557ca1da7780de833f6acdc26930241ce3cf9c"
    ),
}

P1R2_DIAG_RESULT_SHA256 = {
    "s04-p1r2-terminal-component-diag-llama3-8b-inst-v1": (
        "fbd7100053dde5f7dd794fdafc43900b0c9adaa1d81db44ee7bc59457db628b3"
    ),
    "s04-p1r2-terminal-component-diag-qwen2.5-7b-inst-v1": (
        "41861ea2c353f14d5577db76f030bd6a2c42bf7ec988f75768981a60486ef9a1"
    ),
}
P1R2_DIAG_FILE_SHA256 = {
    "local/odebf/state/s04-p1r2-terminal-component-diag-v1.submission-intent.json": (
        "a6fb53701799f34aa3e04758611e83309b7cf29be29ac8f0b44b1f083b37614e"
    ),
    "local/odebf/state/s04-p1r2-terminal-component-diag-v1.submission-receipt.json": (
        "b5e6be928968910952b15dcd7b41293911172db9741e8c956d9e89734801af1a"
    ),
    "local/odebf/logs/odebf_s04_p1r2diag_llama-16675.out": (
        "2e5292000e3663a5260a3b7f57557ca1da7780de833f6acdc26930241ce3cf9c"
    ),
    "local/odebf/logs/odebf_s04_p1r2diag_llama-16675.err": (
        "f180533c9a8a104e5144d189c52a4e4518de1b73db3ec11d1e4c4fe8483a1f55"
    ),
    "local/odebf/logs/odebf_s04_p1r2diag_qwen-16676.out": (
        "2e5292000e3663a5260a3b7f57557ca1da7780de833f6acdc26930241ce3cf9c"
    ),
    "local/odebf/logs/odebf_s04_p1r2diag_qwen-16676.err": (
        "57437570169ccd6ae9abeb2593b960a58949ccd672898f7dafab06241b85300a"
    ),
}

P1R3_DIAG_RESULT_SHA256 = {
    "s04-p1r3-fixed-entry-arm-local-llama3-8b-inst-v1": (
        "9424d93b369df4f1373c1e626568151db6c0a8dbe01872010859e2aeaff35ad2"
    ),
    "s04-p1r3-fixed-entry-arm-local-qwen2.5-7b-inst-v1": (
        "dc7850afa3b0906a3ad2b861cae825f2274094664db2b7e1c663fcd5cb64bf90"
    ),
}
P1R3_DIAG_FILE_SHA256 = {
    "local/odebf/state/s04-p1r3-fixed-entry-arm-local-diag-v1.submission-intent.json": (
        "5ce8409a302c7ea1fe2bb1e964aa719e702fef92f49175622597c8bb4783d1e4"
    ),
    "local/odebf/state/s04-p1r3-fixed-entry-arm-local-diag-v1.submission-receipt.json": (
        "2d24ad490418a6298fcbc5e9b3b042631f56af0b6945fd3c329e4cd14aaf2ede"
    ),
    "local/odebf/logs/odebf_s04_p1r3diag_llama-16677.out": (
        "ba2c36db160f5158a4ea99ca82ab83ffea58e1444a7a031555ff07da53b6a471"
    ),
    "local/odebf/logs/odebf_s04_p1r3diag_llama-16677.err": (
        "2282ad8323579767496e718ccf7ce2ebe02023f45ac67283c06a6617ca9b0571"
    ),
    "local/odebf/logs/odebf_s04_p1r3diag_qwen-16678.out": (
        "7d36866bd4a7c42cc7813a6f2d0cbfed5e45c073af6f0b11b13d8a152862e875"
    ),
    "local/odebf/logs/odebf_s04_p1r3diag_qwen-16678.err": (
        "6d09a4b1991a1a30c37097287744e09aa6ff1a53173183170f9d3f02c16591f0"
    ),
}

ALLOWED_CHANGED_PATHS = {
    "project/run_scripts/ode_bf/locks/source_manifest_p1r4diag.json",
    "project/run_scripts/ode_bf/p1_backend.py",
    "project/run_scripts/ode_bf/p1_diagnostics.py",
    "project/run_scripts/ode_bf/p1_runtime.py",
    "project/run_scripts/ode_bf/tests/test_p1_backend_controller.py",
    "project/run_scripts/ode_bf/tests/test_p1_terminal_diagnostics.py",
    "project/run_scripts/session04_ode_bf_p1_diag.py",
    "project/run_scripts/session04_ode_bf_p1_diag.sbatch",
    "project/run_scripts/session04_ode_bf_p1_diag_dry_plan.py",
    "project/run_scripts/session04_ode_bf_submit_p1_diag.py",
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


def _p1r2_immutability_gate() -> dict[str, str]:
    observed: dict[str, str] = {}
    result_parent = REPO_ROOT / "local/odebf/results"
    for name, expected in P1R2_RESULT_SHA256.items():
        root = result_parent / name
        if root.is_symlink() or not root.is_dir():
            raise ODEBFContractError("P1R2 result root identity differs")
        digest, count = sha256_regular_tree(root)
        if digest != expected or count != 6:
            raise ODEBFContractError("P1R2 result root immutability differs")
        observed[name] = digest
    for relative, expected in P1R2_FILE_SHA256.items():
        path = REPO_ROOT / relative
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
            raise ODEBFContractError("P1R2 log/state immutability differs")
        observed[relative] = expected
    for name, expected in P1R2_DIAG_RESULT_SHA256.items():
        root = result_parent / name
        if root.is_symlink() or not root.is_dir():
            raise ODEBFContractError("P1R2 diagnostic result root identity differs")
        digest, count = sha256_regular_tree(root)
        if digest != expected or count != 39:
            raise ODEBFContractError("P1R2 diagnostic result immutability differs")
        observed[name] = digest
    for relative, expected in P1R2_DIAG_FILE_SHA256.items():
        path = REPO_ROOT / relative
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
            raise ODEBFContractError("P1R2 diagnostic log/state immutability differs")
        observed[relative] = expected
    return observed


def _p1r3_immutability_gate() -> dict[str, str]:
    observed: dict[str, str] = {}
    result_parent = REPO_ROOT / "local/odebf/results"
    for name, expected in P1R3_DIAG_RESULT_SHA256.items():
        root = result_parent / name
        if root.is_symlink() or not root.is_dir():
            raise ODEBFContractError("P1R3 diagnostic result root identity differs")
        digest, count = sha256_regular_tree(root)
        if digest != expected or count != 113:
            raise ODEBFContractError("P1R3 diagnostic result immutability differs")
        observed[name] = digest
    for relative, expected in P1R3_DIAG_FILE_SHA256.items():
        path = REPO_ROOT / relative
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
            raise ODEBFContractError("P1R3 diagnostic log/state immutability differs")
        observed[relative] = expected
    return observed


def _frozen_semantics_gate() -> dict[str, str]:
    for relative, expected in FROZEN_SHA256.items():
        path = REPO_ROOT / relative
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
            raise ODEBFContractError("P1R2 scientific semantic bytes changed")
    runtime = (PACKAGE_ROOT / "p1_runtime.py").read_text(encoding="utf-8")
    required = (
        "rho_accept=lock.rho_accept",
        "functional.pretrained.passed",
        "build_outer_entry_pretrained_cache(",
        "outer_entry_p_cache.select(",
        "_arm_local_infeasibility(terminal_feasibility)",
        "diagnostic_stop_at_terminal=True",
    )
    if any(fragment not in runtime for fragment in required):
        raise ODEBFContractError("P1R4 diagnostic/source semantic guard differs")
    positions = (
        runtime.index("diagnostic_recorder.write_terminal("),
        runtime.index("arm_local_infeasibility = _arm_local_infeasibility("),
        runtime.index("if diagnostic_stop_at_terminal:"),
        runtime.index('measure("selected_endpoint_rewrite_verdict")'),
    )
    if positions != tuple(sorted(positions)):
        raise ODEBFContractError("P1R2 terminal diagnostic source order differs")
    helper_start = runtime.index("def _run_terminal_component_diagnostic(")
    helper_end = runtime.index("\ndef run_p1(", helper_start)
    helper = runtime[helper_start:helper_end]
    if any(
        fragment in helper
        for fragment in (
            "_run_native_batch",
            "load_counterfact_cases_after_freeze",
            "transaction.commit",
        )
    ):
        raise ODEBFContractError("P1R4 diagnostic crossed its no-commit boundary")
    for arm in ("P1Arm.F_G", "P1Arm.F_BF", "P1Arm.R_BF"):
        if arm not in helper:
            raise ODEBFContractError("P1R4 diagnostic arm panel differs")
    replay = (PACKAGE_ROOT / "p1_replay.py").read_text(encoding="utf-8")
    if (
        'baseline_kind != "outer_entry"' not in replay
        or "build_outer_entry_pretrained_cache" not in replay
        or "capture_pretrained_entry_kl" not in replay
    ):
        raise ODEBFContractError("P1R4 fixed-entry cache source differs")
    backend = (PACKAGE_ROOT / "p1_backend.py").read_text(encoding="utf-8")
    required_backend = (
        'FULL_CURRENT_RESIDUAL_DEFINITION = "full_current"',
        "FULL_CURRENT_RESIDUAL_DIVISOR = 1",
        "residual = full_residual.clone()",
        "residual = full_current_residual(target_state, current_z)",
    )
    if any(fragment not in backend for fragment in required_backend):
        raise ODEBFContractError("P1R4 full-residual source contract differs")
    dynamic_start = backend.index("def build_p1_dynamic_field(")
    frozen_start = backend.index("def build_p1_frozen_field_from_capture(")
    dynamic = backend[dynamic_start:frozen_start]
    frozen_end = backend.index("\ndef capture_committed_history_key_views(", frozen_start)
    frozen = backend[frozen_start:frozen_end]
    if any("len(layers) - layer_index" in section for section in (dynamic, frozen)):
        raise ODEBFContractError("remaining-layer divisor reaches full-residual arm")
    return {
        **FROZEN_SHA256,
        "full_residual_backend_sha256": sha256_file(PACKAGE_ROOT / "p1_backend.py"),
    }


def _assert_p1_runtime_ast_firewall(path: Path | None = None) -> str:
    """Permit only the pre-authorized CUDA-helper import across namespaces."""

    runtime_path = path or (PACKAGE_ROOT / "p1_runtime.py")
    tree = ast.parse(runtime_path.read_text(encoding="utf-8"), filename=str(runtime_path))
    allowed_module = "project.run_scripts.ode_alloc.p1_runtime"
    allowed_alias = (
        "_prepare_p1_cuda_runtime",
        "_prepare_preserved_one_device_cuda_runtime",
    )
    allowed_import_count = 0
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name.lower() for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            module = (node.module or "").lower()
            if module == allowed_module:
                observed = tuple((alias.name, alias.asname) for alias in node.names)
                if observed != (allowed_alias,):
                    raise ODEBFContractError(
                        "preserved CUDA helper import contract differs"
                    )
                allowed_import_count += 1
            else:
                names = [module]
        if any(
            fragment in name
            for name in names
            for fragment in FORBIDDEN_IMPORT_FRAGMENTS
        ):
            raise ODEBFContractError("forbidden foreign/session import found")
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "generate"
        ):
            raise ODEBFContractError("model.generate is forbidden in ODE-BF")
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            lowered = node.value.lower()
            if (
                ("/" in lowered or "\\" in lowered or "." in lowered)
                and ("session03" in lowered or "knowledge-revision" in lowered)
            ):
                raise ODEBFContractError("forbidden foreign/session string found")
    if allowed_import_count != 1:
        raise ODEBFContractError("preserved CUDA helper import count differs")
    return sha256_file(runtime_path)


def _source_manifest_gate() -> str:
    value, raw_sha = load_rooted_json(
        SOURCE_MANIFEST,
        expected_schema="ode-edit-s04-ode-bf-p1r4diag-source-manifest/v1",
    )
    entries = value.get("entries")
    if (
        value.get("instruction_id") != INSTRUCTION_ID
        or value.get("expected_base") != BASE_HEAD
        or not isinstance(entries, list)
        or not entries
    ):
        raise ODEBFContractError("P1R4 diagnostic source manifest provenance differs")
    locked = {
        entry["path"]: (entry["sha256"], int(entry["size"]))
        for entry in entries
    }
    if len(locked) != len(entries):
        raise ODEBFContractError("P1R4 diagnostic source manifest repeats a path")
    observed = _run(["git", "ls-tree", "-r", "--name-only", "HEAD"]).stdout.splitlines()
    expected_paths = {
        path
        for path in observed
        if (
            path.startswith("project/run_scripts/ode_bf/")
            or path.startswith("project/run_scripts/session04_ode_bf_")
        )
        and path != "project/run_scripts/ode_bf/locks/source_manifest_p1r4diag.json"
    }
    if set(locked) != expected_paths:
        raise ODEBFContractError("P1R4 diagnostic source manifest path set differs")
    for relative, (expected, expected_size) in sorted(locked.items()):
        path = REPO_ROOT / relative
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != expected_size
            or sha256_file(path) != expected
        ):
            raise ODEBFContractError("P1R4 diagnostic source manifest content differs")
    return raw_sha


def _source_gate() -> str:
    head = _run(["git", "rev-parse", "HEAD"]).stdout.strip()
    parent = _run(["git", "rev-parse", "HEAD^"]).stdout.strip()
    branch = _run(["git", "branch", "--show-current"]).stdout.strip()
    if parent != BASE_HEAD or head == BASE_HEAD or branch != BRANCH:
        raise ODEBFContractError("P1R4 diagnostic checkpoint ancestry/branch differs")
    if _run(["git", "status", "--porcelain", "--untracked-files=no"]).stdout:
        raise ODEBFContractError("P1R4 diagnostic tracked source is not clean")
    untracked = _run(["git", "ls-files", "--others", "--exclude-standard"]).stdout.splitlines()
    if any(
        not (
            path.startswith("agents/server2/")
            or path.startswith("audits/servers/server2/")
        )
        for path in untracked
    ):
        raise ODEBFContractError("foreign untracked path overlaps P1R4 diagnostic")
    changed = set(
        _run(["git", "diff", "--name-only", f"{BASE_HEAD}..{head}"]).stdout.splitlines()
    )
    if changed != ALLOWED_CHANGED_PATHS:
        raise ODEBFContractError("P1R4 diagnostic checkpoint path inventory differs")
    _run(["git", "diff", "--check", f"{BASE_HEAD}..{head}"])
    _source_manifest_gate()
    return head


def _lock_and_seal_gate() -> dict[str, str]:
    frozen = _frozen_semantics_gate()
    stream = verify_p1_stream_seal(
        json.loads((LOCK_ROOT / "p1r2_seqb10_stream_seal.json").read_text())
    )
    population = verify_p1_population_seal(
        json.loads((LOCK_ROOT / "p1r2_p_population_seal.json").read_text()),
        stream=stream,
    )
    numerical, numerical_sha = load_rooted_json(
        LOCK_ROOT / "numerical_lock_p1r2.json",
        expected_schema="ode-edit-s04-ode-bf-p1r2-numerical-lock/v2",
    )
    if (
        numerical.get("controller_identity_sha256") != P1ControllerLock().identity()
        or numerical.get("edit_batch_size") != 10
        or numerical.get("controller", {}).get("rho_accept") != 0.1
        or numerical.get("controller", {}).get("functional_p_budget_nats")
        != 0.001
        or numerical.get("stream_root_digest") != stream["root_digest"]
        or numerical.get("p_population_root_digest") != population["root_digest"]
    ):
        raise ODEBFContractError("P1R4 diagnostic lock/seal identity differs")
    return {
        "numerical_lock_sha256": numerical_sha,
        "stream_root_digest": stream["root_digest"],
        "population_root_digest": population["root_digest"],
        "frozen_semantics_sha256": hashlib.sha256(
            json.dumps(frozen, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }


def _cpu_static_gate(source_head: str) -> dict[str, Any]:
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get(
        "TRANSFORMERS_OFFLINE"
    ) != "1":
        raise ODEBFContractError("P1R4 diagnostic canonical offline env differs")
    _run(["scripts/check-session-boundary.sh", SESSION_ID])
    _run(["scripts/check-agent-access.sh", "--all-changed"])
    lock_gate = _lock_and_seal_gate()
    p1r2 = _p1r2_immutability_gate()
    p1r3 = _p1r3_immutability_gate()
    preserved_p0._r0_immutability_gate()
    preserved_p0._r1_immutability_gate()
    preserved_p0._r2_immutability_gate()
    preserved_p0._r3_immutability_gate()
    preserved_p1._r4_immutability_gate()
    preserved_p1._p1_r0_immutability_gate()
    preserved_p1._p1_r1_immutability_gate()

    assert_ast_firewall(
        [
            PACKAGE_ROOT / "p1_diagnostics.py",
            PACKAGE_ROOT / "p1_replay.py",
            REPO_ROOT / "project/run_scripts/session04_ode_bf_p1_diag.py",
        ]
    )
    runtime_firewall_sha256 = _assert_p1_runtime_ast_firewall()
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
    cuda_source_sha256 = preserved_p1._cuda_preflight_source_gate()
    for path in sorted(PACKAGE_ROOT.rglob("*.py")) + [
        REPO_ROOT / "project/run_scripts/session04_ode_bf_p1_diag.py",
        DRY_PLAN,
        Path(__file__),
    ]:
        compile(path.read_text(encoding="utf-8"), str(path), "exec", dont_inherit=True)
    _run(["bash", "-n", str(SBATCH)])

    runtime = "/mnt/raid5/janghj/EasyEdit/.venv/bin/python"
    test_env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": f"{REPO_ROOT}:/mnt/raid5/janghj/EasyEdit",
        "MPLCONFIGDIR": str(REPO_ROOT / "local/odebf/matplotlib-p1r4diag-gate"),
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
    if (
        count_match is None
        or int(count_match.group(1)) < 155
        or "\nOK\n" not in tests.stderr
    ):
        raise ODEBFContractError("P1R4 diagnostic full CPU suite did not pass")
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
    if (
        repair_count is None
        or int(repair_count.group(1)) != 17
        or "\nOK\n" not in repair_tests.stderr
    ):
        raise ODEBFContractError("preserved CUDA repair 17-test gate did not pass")

    dry_args = [runtime, str(DRY_PLAN), "--source-head", source_head]
    dry_first = _run(dry_args, env=test_env).stdout
    dry_second = _run(dry_args, env=test_env).stdout
    if dry_first != dry_second:
        raise ODEBFContractError("P1R4 diagnostic dry plan is not byte-repeatable")
    dry = json.loads(dry_first)
    if (
        dry.get("edit_batch_size") != 10
        or dry.get("sequential_batch_count") != 1
        or dry.get("arms") != ["N32_NATIVE", "F_G", "F_BF", "R_BF"]
        or dry.get("functional_p_baseline_kind") != "outer_entry"
        or dry.get("residual_definition") != "full_current"
        or dry.get("residual_divisor") != 1
        or dry.get("arm_local_infeasibility") is not True
        or dry.get("scientific_promotion_authorized") is not False
        or dry.get("persistent_endpoint_commit_count") != 0
        or dry.get("history_append_count") != 0
        or dry.get("heldout_access_count") != 0
    ):
        raise ODEBFContractError("P1R4 diagnostic dry scope differs")

    artifacts: dict[str, Any] = {}
    forecasts: dict[str, Any] = {}
    for alias in MODEL_ALIASES:
        guard = ODEBFArtifactGuard(REPO_ROOT, ARTIFACT_LOCK, alias)
        receipt = guard.preflight()
        guard.assert_unchanged()
        forecast = forecast_p1_b10_memory(ARTIFACT_LOCK, BASE_ARTIFACT_LOCK, alias)
        if (
            forecast.forecast_gpu_peak_mib > 60_416
            or forecast.forecast_host_peak_mib > 60_416
            or forecast.dense_fp64_full_delta
        ):
            raise ODEBFContractError("P1R4 diagnostic memory forecast differs")
        artifacts[alias] = {
            "lock_sha256": receipt.lock_sha256,
            "revision": receipt.base_model_revision,
            "projector_sha256": receipt.projector_sha256,
        }
        forecasts[alias] = forecast.raw_free_payload()
    return {
        "source_head": source_head,
        "test_count": int(count_match.group(1)),
        "preserved_repair_test_count": int(repair_count.group(1)),
        "source_manifest_sha256": _source_manifest_gate(),
        "lock_gate": lock_gate,
        "cuda_preflight_source_sha256": cuda_source_sha256,
        "runtime_firewall_sha256": runtime_firewall_sha256,
        "p1r2_immutability": p1r2,
        "p1r3_immutability": p1r3,
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
            raise ODEBFContractError("P1R4 diagnostic output parent differs")
        parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    roots = {
        alias: result_parent / expected_p1r4_diagnostic_result_name(alias)
        for alias in MODEL_ALIASES
    }
    if any(path.exists() or path.is_symlink() for path in roots.values()):
        raise ODEBFContractError("P1R4 diagnostic result root exists")
    for job_name in JOB_NAMES.values():
        if list(log_parent.glob(f"{job_name}-*")):
            raise ODEBFContractError("P1R4 diagnostic log namespace exists")
    intent = state_parent / "s04-p1r4-full-residual-arms-diag-v1.submission-intent.json"
    receipt = state_parent / "s04-p1r4-full-residual-arms-diag-v1.submission-receipt.json"
    if intent.exists() or intent.is_symlink() or receipt.exists() or receipt.is_symlink():
        raise ODEBFContractError("P1R4 diagnostic pair was already attempted")
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
        raise ODEBFContractError("sbatch did not return a diagnostic job ID")
    return job_id


def main() -> int:
    source_head = _source_gate()
    gate = _cpu_static_gate(source_head)
    paths = _output_gate()
    _p1r2_immutability_gate()
    _p1r3_immutability_gate()
    before = preserved_p0._scheduler_jobs()
    if any(record.job_name in JOB_NAMES.values() for record in before):
        raise ODEBFContractError("P1R4 diagnostic scheduler name exists")
    local_before, cluster_before = assert_node_local_capacity(before, new_gpu_count=2)
    if 2 * 60_416 > 2 * MEMORY_CAP_MIB_PER_GPU:
        raise ODEBFContractError("P1R4 diagnostic pair host memory exceeds cap")
    intent = {
        "schema": "ode-edit-s04-ode-bf-p1r4diag-submission-intent/v1",
        "instruction_id": INSTRUCTION_ID,
        "authorized_attempt": "DISTINCT_B10_1_FULL_RESIDUAL_ARM_DIAG",
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
        assert_node_local_capacity(
            preserved_p0._scheduler_jobs(), new_gpu_count=1
        )
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
        "schema": "ode-edit-s04-ode-bf-p1r4diag-submission-receipt/v1",
        "instruction_id": INSTRUCTION_ID,
        "authorized_attempt": "DISTINCT_B10_1_FULL_RESIDUAL_ARM_DIAG",
        "status": "SUBMITTED_PAIR" if failure is None else "PARTIAL_OR_FAILED_NO_RETRY",
        "source_head": source_head,
        "intent_sha256": intent_sha256,
        "submitted": submitted,
        "failure": failure,
        "retry_or_resubmit": False,
    }
    receipt_sha256 = _write_once(paths["__receipt__"], receipt)
    if failure is not None:
        raise ODEBFContractError("diagnostic one-shot submission did not complete")
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
        ),
        flush=True,
    )
    return 0


def _entrypoint() -> None:
    try:
        code = main()
    except SystemExit:
        raise
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
