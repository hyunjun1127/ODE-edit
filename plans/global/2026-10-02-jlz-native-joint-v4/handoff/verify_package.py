"""Read-only verification of the design handoff; no model/data/network needed."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def main():
    manifest = json.loads((HERE / "package-manifest.json").read_text())
    for row in manifest["files"]:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("UNSAFE_MEMBER_PATH")
        path = ROOT / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError("MISSING_OR_SYMLINK:" + str(relative))
        data = path.read_bytes()
        if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError("MEMBER_MISMATCH:" + str(relative))
    method = ROOT / "plans/global/2026-10-02-jlz-native-joint-v4"
    contract = json.loads((method / "contract.json").read_text())
    experiment = json.loads((method / "experiment-2k/experiment.json").read_text())
    assert contract["revision"] == "compute-r1-generic-adapters"
    assert [x["id"] for x in experiment["new_chains"]] == ["JLZ_A", "JLZ_B"]
    assert not any(experiment["baseline_reuse"][k] for k in ["new_main_runs", "new_pilot_runs", "fallback_runs"])
    assert experiment["batch_size"] * experiment["sequential_commits"] == 2000
    print(json.dumps({"status": "PASS_GH_METHOD_AND_EXPERIMENT_PACKAGE", "files": len(manifest["files"]),
                      "general_method": True, "experiment": "A/B only, BS100x20",
                      "runner_implemented": False, "GPU_tested": False, "submitted": False}))


if __name__ == "__main__":
    main()
