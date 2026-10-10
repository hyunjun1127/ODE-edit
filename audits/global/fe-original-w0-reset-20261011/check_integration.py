"""FE 영수증·표 상태·비FE 불변 CPU 검산 (scheduler/model 호출 없음)."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
load = lambda p: json.loads((ROOT / p).read_text())
receipt = load(str(HERE.relative_to(ROOT) / "integration.json"))
old = subprocess.check_output(["git", "show", receipt["baseline_main"] + ":README.md"], cwd=ROOT, text=True)
new = (ROOT / "README.md").read_text()
non_fe = lambda text: [s for s in text.splitlines() if s.startswith("|") and not s.startswith("| FE (author repo, W0-fixed z)")]
assert non_fe(old) == non_fe(new), "비FE/PRICE/W0 표 변경"
rows = [s for s in new.splitlines() if s.startswith("| FE (author repo, W0-fixed z)")]
assert len(rows) == 3
submission = load(receipt["submission_receipt"])
assert submission["source"] == receipt["source"]
assert submission["official_tree"] == receipt["official_tree"]
assert submission["held_inspected"] and submission["released"]
assert {j["job_id"] for j in submission["jobs"]} == {j["job_id"] for j in receipt["jobs"]}
for job in receipt["jobs"]:
    original = next(j for j in submission["jobs"] if j["job_id"] == job["job_id"])
    assert original["condition"] == job["condition"]
    idx = 0 if job["condition"].startswith("llama3") else 2
    cells = [s.strip() for s in rows[idx].split("|")[2:-1]]
    selected = cells[:4] if job["condition"].endswith("-cf") else cells[6:]
    prefix = "ING" if job["state"] == "RUNNING" else "PENDING"
    assert all(c == f'{prefix}: {original["job_name"]} ({job["job_id"]})' for c in selected)
    assert cells[4:6] == ["DEFERRED", "DEFERRED"]
assert all(not s.strip() for s in rows[1].split("|")[2:-1]), "미제출 Qwen 셀은 공란"
qwen = load(receipt["qwen"]["source_ready_receipt"])
assert qwen["job_ids"] == []
assert qwen["storage"]["required_free_bytes"] - qwen["storage"]["available_bytes"] == receipt["qwen"]["deficit_bytes"]
assert qwen["storage"]["reserve_bytes"] == receipt["qwen"]["reserve_bytes"]
base = "audits/servers/server1/fe-original-w0-2k-20261011/"
a, b, c = [load(base + name + ".json") for name in ["checkpoint-deleted", "native-checkpoint-deleted", "corrected-qwen-deleted"]]
d = load("audits/servers/server2/fe-original-w0-2k-20261011/delete-after.json")
assert a["all_exact_targets_absent"] and b["all_absent"] and c["deleted"] and d["all_absent"]
assert a["count"] + b["count"] + 1 + d["count"] == receipt["deleted"]["count"] == 17
assert a["logical_bytes"] + b["bytes"] + c["bytes"] + d["deleted_bytes"] == receipt["deleted"]["bytes"] == 77219643881
assert not any([a["backup_created"], b["backup"], c["backup"], d["backup_created"]])
print("PASS: 실제4 jobs / Qwen 미제출2 / FE18 상태셀 / 비FE 표 불변 / 삭제17개 산술. GPU/Slurm 호출0.")
