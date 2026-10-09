"""GH의 표/정책/인계 정합 검사. 모델 로드 및 scheduler 호출 없음."""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NONCE = "USER-SH2-QWEN-ZSRE-SPHERE-B9-RESUME-20261010-R1"
PARENT_SHA = "7e44f382ba1f3befcb9f4429f42fd328d55422306f9cef868a920c8bf7ec17b8"


def main():
    record = json.loads((Path(__file__).parent / "coordination.json").read_text())
    assert record["nonce"] == NONCE and record["owner_ack"]
    parent = record["parent"]
    assert parent["job_id"] == "62087" and parent["batch"] == 9 and parent["edits"] == 900
    assert parent["checkpoint_sha256"] == PARENT_SHA and parent["checkpoint_bytes"] == 8535442009
    assert parent["source"] == "7b5097aa447946e35de42229c22b0c0feabd11ae"
    old = record["superseded_cold_job"]
    assert old["job_id"] == "62534" and old["owner_confirmed_final_state"] == "CANCELLED"
    assert old["cold_B1_executed"] is False
    job = record["job"]
    assert job and job["job_id"] not in {"62087", "62534", "62529", "62530", "62531", "62532"}
    assert "resume" in job["job_name"] and "b9" in job["job_name"]
    assert job["held_inspection_pass"] and job["released"]
    assert job["initial_state"] in {"PENDING", "RUNNING"}
    assert job["dependency"] == "afterany:62532"
    for key in ["source_commit", "config_sha256", "owner_receipt", "checkpoint_path"]:
        assert job[key]
    assert (ROOT / job["owner_receipt"]).is_file()
    submission = json.loads((ROOT / job["owner_receipt"]).read_text())
    evidence = json.loads((ROOT / record["owner_evidence"]).read_text())
    binding = json.loads((ROOT / record["owner_CPU_binding"]).read_text())
    cancelled = json.loads((ROOT / old["receipt"]).read_text())
    assert submission["instruction_id"] == NONCE
    for ours, theirs in [("job_id", "job_id"), ("job_name", "job_name"),
                         ("source_commit", "source"), ("config_sha256", "config_sha256"),
                         ("observed_utc", "observed_utc")]:
        assert job[ours] == submission[theirs]
    assert submission["held_inspected"] and submission["released"]
    assert submission["initial_snapshot"]["JobState"] == job["initial_state"]
    assert submission["dependency"] == ["62532"]
    assert job["checkpoint_path"] == submission["output"] + "/checkpoint"
    assert submission["parent_checkpoint"]["sha256"] == PARENT_SHA
    assert submission["parent_checkpoint"]["bytes"] == parent["checkpoint_bytes"]
    assert submission["start_batch"] == 9 and submission["next_batch"] == 10
    assert submission["remaining_batches"] == list(range(10, 21))
    assert job["owner_submission_sha256"] == evidence["submission"]["sha256"]
    assert evidence["CPU_tests"] == job["owner_CPU_tests_pass"] == 17
    assert evidence["parent_edit_applications"] == 900 and evidence["new_edit_applications_planned"] == 1100
    assert cancelled["before"]["JobState"] == "PENDING" and cancelled["after"]["JobState"] == "CANCELLED"
    assert cancelled["before"]["RunTime"] == cancelled["after"]["RunTime"] == "00:00:00"
    assert not cancelled["other_jobs_mutated"]
    original_identity, child_identity = binding["parent_identity"], binding["child_identity"]
    assert original_identity["code_commit"] == parent["source"]
    assert child_identity["code_commit"] == job["source_commit"]
    assert original_identity["code_commit"] != child_identity["code_commit"]
    assert all(original_identity[k] == v for k, v in child_identity.items()
               if k not in {"code_commit", "official_tree_sha256"})
    for ancestor in [parent["source"], "c54779f2873b362497a5ba5d1fa4d9663815a9ec"]:
        subprocess.run(["git", "merge-base", "--is-ancestor", ancestor, job["source_commit"]], cwd=ROOT, check=True)
    contract = record["resume_contract"]
    assert [contract[x] for x in ["start_batch_zero_based", "first_new_batch", "last_batch", "new_batches", "final_edits"]] == [9, 10, 20, 11, 2000]
    assert set(contract["restore"]) == {"editable_weights", "native_history_cache_c", "contexts", "RNG", "batch_sample_cursor"}
    policy = json.loads((ROOT / "control/main-results-policy.json").read_text())
    before_policy = json.loads(subprocess.check_output(["git", "show", "dd171592:control/main-results-policy.json"], cwd=ROOT, text=True))
    exceptions = policy.pop("explicit_resume_exceptions")
    assert policy == before_policy, "UNRELATED_POLICY_CHANGED"
    assert len(exceptions) == 1 and exceptions[0]["instruction_id"] == NONCE
    assert exceptions[0]["parent_checkpoint_sha256"] == PARENT_SHA
    text = (ROOT / "README.md").read_text()
    tables = lambda t: [line for line in t.splitlines() if line.startswith("| ")]
    baseline = tables(subprocess.check_output(["git", "show", "dd171592:README.md"], cwd=ROOT, text=True))
    current = tables(text)
    assert len(current) == len(baseline)
    changes = [(a, b) for a, b in zip(baseline, current) if a != b]
    assert len(changes) == 1, "ONLY_QWEN_SPHERE_ROW_ALLOWED"
    a, b = [[v.strip() for v in line.split("|")[1:-1]] for line in changes[0]]
    assert a[:7] == b[:7] and b[:5] == ["AlphaEdit+SPHERE", "83.73", "99.40", "97.70", "64.37"]
    assert b[7:10] == [f'{job["initial_state"]}: {job["job_name"]} ({job["job_id"]})'] * 3
    assert "B1–B9" in text and "B10–B20" in text and "62534" in text
    print(json.dumps({"status": "PASS", "changed_table_rows": 1, "changed_status_cells": 3,
                      "FE_author_rows_unchanged": 4, "parent_edits": 900,
                      "new_batches": 11, "final_edits": 2000, "GPU_calls": 0}))


if __name__ == "__main__":
    main()
