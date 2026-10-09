"""GH compact submission/README validation. No scheduler or model calls."""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LABEL = "MEMIT_FE_HISTORY (FE author hparams)"
EXPECTED = {("Llama3", "CF", "server1"), ("Llama3", "zsRE", "server1"),
            ("Qwen2.5", "CF", "server2"), ("Qwen2.5", "zsRE", "server2")}


def rows(text):
    return [[cell.strip() for cell in line.split("|")[1:-1]]
            for line in text.splitlines() if line.startswith("| ")]


def main():
    receipt = json.loads((Path(__file__).parent / "coordination.json").read_text())
    current = rows((ROOT / "README.md").read_text())
    original = rows(subprocess.check_output(
        ["git", "show", "fbb3a2e3:README.md"], cwd=ROOT, text=True))
    selected = [row for row in current if row[0] == LABEL]
    assert len(selected) == 4 and {tuple(row[1:4]) for row in selected} == EXPECTED
    # Separate subsequent USER authority permits only Qwen SPHERE zsRE status.
    sphere_path = ROOT / "audits/global/qwen-zsre-sphere-oom-rerun-20261010/coordination.json"
    sphere = json.loads(sphere_path.read_text()).get("job") if sphere_path.exists() else None
    protected = [list(r) for r in current if r[0] != LABEL]
    old_protected = [r for r in original if r[0] != LABEL]
    if sphere:
        candidates = [r for r in protected if r[0] == "AlphaEdit+SPHERE" and r[1:5] == ["83.73", "99.40", "97.70", "64.37"]]
        assert len(candidates) == 1
        row = candidates[0]
        old = next(r for r in old_protected if r[:5] == row[:5])
        expected = f'{sphere["initial_state"]}: {sphere["job_name"]} ({sphere["job_id"]})'
        assert row[7:10] == [expected] * 3
        assert sphere["held_inspection_pass"] and sphere["released"]
        row[7:10] = old[7:10]
    assert protected == old_protected, "UNRELATED_TABLE_ROW_CHANGED"
    jobs = receipt["jobs"]
    assert len(jobs) == 4 and len({str(j["job_id"]) for j in jobs}) == 4, "ACTUAL_FOUR_JOBS_REQUIRED"
    for job in jobs:
        key = (job["table_model"], job["table_dataset"], job["server"])
        row = next(r for r in selected if tuple(r[1:4]) == key)
        state = job["initial_state"]
        assert state in {"PENDING", "RUNNING"}
        assert row[4] == f'{state}: {job["job_name"]} ({job["job_id"]})'
        old = next(r for r in original if r[0] == LABEL and tuple(r[1:4]) == key)
        assert row[5:] == old[5:], "NO_FUTURE_METRICS_INVENTED"
        assert job["held_inspection_pass"] and job["released"]
        for field in ["source_commit", "config_sha256", "profile_sha256", "checkpoint_path", "owner_receipt"]:
            assert job[field], field
    print(json.dumps({"status": "PASS", "actual_unique_jobs": 4, "author_rows": 4,
                      "unrelated_table_changes": 0,
                      "separately_authorized_sphere_cells": 3 if sphere else 0, "GPU_calls": 0}))


if __name__ == "__main__":
    main()
