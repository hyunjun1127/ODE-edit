"""Seal owned design manifests and a deterministic GH handoff archive.

Does not implement/launch a runner, send messages, or load a model. Only the
explicit method/experiment manifests and handoff output paths below are written.
"""
from __future__ import annotations
import argparse
import ast
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import tarfile

HERE = Path(__file__).resolve().parent
METHOD = HERE.parent
ROOT = HERE.parents[3]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def member(path):
    data = path.read_bytes()
    return {"path": str(path.relative_to(ROOT)), "bytes": len(data), "sha256": sha(data)}


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def method_checks():
    for result, script in [("math/math-check.json", "math/validate_native_joint.py"),
                           ("math/compute-reuse-check.json", "math/validate_compute_reuse.py")]:
        data = json.loads((METHOD / result).read_text())
        assert data["script_sha256"] == sha((METHOD / script).read_bytes()), "MATH_SOURCE_BINDING"
        ast.parse((METHOD / script).read_text())
    counts = json.loads((METHOD / "compute/native-work-counts.json").read_text())
    assert counts["script"]["sha256"] == sha((METHOD / "compute/count_native_work.py").read_bytes())
    assert counts["aggregate"]["valid_tokens"] == 215075
    assert counts["aggregate"]["dynamic_suffix_after_KL_future_prune"] == 50960
    assert counts["aggregate"]["selected_head_positions"] == 14144
    assert counts["GPU_calls"] == counts["model_inference_calls"] == 0
    ast.parse((METHOD / "compute/count_native_work.py").read_text())
    text = (ROOT / "docs/methods/jlz-native-joint-v4.tex").read_text()
    # Structural validation only; not a substitute for a TeX compiler.
    stack = []
    for kind, name in re.findall(r"\\(begin|end)\{([^}]+)\}", text):
        if kind == "begin":
            stack.append(name)
        else:
            assert stack and stack.pop() == name, "TEX_ENVIRONMENTS"
    assert not stack
    labels = re.findall(r"\\label\{([^}]+)\}", text)
    assert len(labels) == len(set(labels))
    assert set(re.findall(r"\\(?:eqref|ref)\{([^}]+)\}", text)) <= set(labels)
    assert "\\end{document}" in text
    generic = json.loads((METHOD / "portability-contract.json").read_text())
    assert generic["logical_batch"]["fixed_method_B"] is None
    assert generic["geometry_backends"]["off_diagonal_coupling_preserved"]
    return {"math_source_bindings": "PASS", "tokenizer_evidence_binding": "PASS",
            "TeX_static": "PASS", "TeX_compiled": False, "GPU": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--archive-dir", type=Path,
                        default=ROOT / "local/jlz-v4-handoff/20261002-compute-r1")
    args = parser.parse_args()
    checks = method_checks()
    relative = ["method-ko.md", "contract.json", "portability-contract.json", "GH-HANDOFF.md",
                "inputs/exact-native-inputs.json", "inputs/check_exact_native_inputs.py",
                "math/math-check.json", "math/validate_native_joint.py",
                "math/compute-reuse-check.json", "math/validate_compute_reuse.py",
                "compute/native-work-counts.json", "compute/count_native_work.py",
                "compute/reviewed-optimization-proposal.txt"]
    method_files = [METHOD / x for x in relative] + [ROOT / "docs/methods/jlz-native-joint-v4.tex"]
    write_json(METHOD / "artifact-manifest.json", {
        "status": "METHOD_SPEC_COMPUTE_R1_CPU_CHECKS_ONLY", "revision": "compute-r1-generic-adapters",
        "production_runner_implemented": False, "GPU_run": False, "tex_compiled": False,
        "files": [member(p) for p in method_files], "checks": checks})
    plan_path = METHOD / "experiment-2k/build_plan.py"
    spec = importlib.util.spec_from_file_location("jlz_v4_2k_plan", plan_path)
    plan = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plan)
    for name, data in plan.build(args.dataset).items():
        (plan_path.parent / name).write_bytes(data)
    exp_manifest = json.loads((plan_path.parent / "artifact-manifest.json").read_text())
    files = set(method_files + [METHOD / "artifact-manifest.json", plan_path.parent / "artifact-manifest.json",
                               HERE / "verify_package.py", HERE / "seal_package.py"])
    for row in exp_manifest["files"] + exp_manifest["external_sources"]:
        files.add(ROOT / row["path"])
    # This small historical token evidence is needed by the portable recount.
    files.add(ROOT / "plans/global/2026-10-02-jlz-shared-subject-v3/compute/native-prefix-counts.json")
    manifest = {"schema": "JLZ_V4_METHOD_AND_EXPERIMENT_HANDOFF_V1", "date_kst": "2026-10-02",
        "status": "PREPARED_NOT_DELIVERED_NOT_SUBMITTED", "checks": checks,
        "method_entry": str((METHOD / "method-ko.md").relative_to(ROOT)),
        "experiment_entry": str((plan_path.parent / "experiment-ko.md").relative_to(ROOT)),
        "generic_adapter_contract": str((METHOD / "portability-contract.json").relative_to(ROOT)),
        "raw_datasets_models_checkpoints_credentials_included": False,
        "archive_receipt": "external sidecar archive-receipt.json (not self-included)",
        "files": [member(p) for p in sorted(files)]}
    write_json(HERE / "package-manifest.json", manifest)
    files.add(HERE / "package-manifest.json")
    tar_bytes = io.BytesIO()
    with tarfile.open(fileobj=tar_bytes, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for path in sorted(files):
            payload = path.read_bytes()
            info = tarfile.TarInfo(str(path.relative_to(ROOT)))
            info.size = len(payload)
            info.mode = 0o644
            info.uid = info.gid = info.mtime = 0
            tar.addfile(info, io.BytesIO(payload))
    archive = gzip.compress(tar_bytes.getvalue(), mtime=0)
    args.archive_dir.mkdir(parents=True, exist_ok=True)
    target = args.archive_dir / "jlz-v4-method-and-experiment-2k.tar.gz"
    target.write_bytes(archive)
    receipt = {"status": "PREPARED_NOT_DELIVERED_NOT_SUBMITTED", "path": str(target.resolve()),
               "bytes": len(archive), "sha256": sha(archive), "archive_members": len(files),
               "package_manifest": member(HERE / "package-manifest.json"), "checks": checks}
    write_json(HERE / "archive-receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
