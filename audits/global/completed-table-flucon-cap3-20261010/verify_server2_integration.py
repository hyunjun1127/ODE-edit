"""소형 owner 증거와 README 셀 대조. 원격/GPU/model/raw load 없음."""
import hashlib
import json
import subprocess
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASELINE = "437f3f21900c6e2ca634804fb36b94cff612b83f"
OWNER = ROOT / "audits/servers/server2/completed-table-flucon-cap3-20261010"


def table(text):
    return [[x.strip() for x in line.split("|")[1:-1]]
            for line in text.splitlines() if line.startswith("| ")]


def main():
    p = OWNER / "table-rows.json"
    assert hashlib.sha256(p.read_bytes()).hexdigest() == "e30daac6d3397afbde9759294ba7b24ccbcd7382a68e159227b974141b60ba0a"
    data = json.loads(p.read_text())
    assert len(data["rows"]) == 26 and data["numeric_eligible"] == 20 and data["issues"] == []
    rows = {r["job_id"]: r for r in data["rows"]}
    before = table(subprocess.check_output(["git", "show", BASELINE + ":README.md"], cwd=ROOT, text=True))
    current = table((ROOT / "README.md").read_text())
    assert len(before) == len(current)
    changes = [(a, b) for a, b in zip(before, current) if a != b]
    assert len(changes) == 3
    mapping = [("61947", "gptj", "AlphaEdit+SPHERE", [5557, 5557, 9694]),
               ("62081", "qwen25", "MEMIT", [6691, 6691, 11476])]
    for jid, model, method, den in mapping:
        r = rows[jid]
        assert r["model"] == model and r["dataset"] == "zsre" and r["numeric_eligible"]
        assert r["observed_state"] == "COMPLETED" and r["requests"] == 2000
        assert [r["denominators"][k] for k in ["rewrite", "paraphrase", "neighborhood"]] == den
        proof = r["zsre_audit"]["query_proof"]
        assert proof["input_mismatches"] == proof["target_mismatches"] == 0
        assert proof["requests"] == 2000 and proof["query_sha256"] == r["query_sha256"]
        assert r["zsre_audit"]["Loc"] == "loc_ans_target_correctness_NOT_W0_agreement"
        a, b = next((a, b) for a, b in changes if a[0] == method and jid in " ".join(a[7:10]))
        expected = [str(Decimal(str(r["metrics"][k])).quantize(Decimal(".01"), rounding=ROUND_HALF_UP))
                    for k in ["Efficacy", "Generalization", "Specificity"]]
        assert b[:7] == a[:7] and b[7:10] == expected
    r = rows["62075"]
    assert r["status"] == "RUNNING" and not r["numeric_eligible"]
    a, b = next((a, b) for a, b in changes if a[0] == "AlphaEdit" and "62075" in a[1])
    assert b[1:5] == [f'ING: {r["job_name"]} (62075)'] * 4
    assert a[0] == b[0] and a[5:] == b[5:]
    cap = json.loads((OWNER / "cap-receipt.json").read_text())
    assert cap["effective_cap"] == 3 and cap["allocated_GPU"] == cap["DAG_width"] == 2
    assert cap["pending_dependency_changes"] == cap["holds"] == cap["cancels"] == cap["new_GPU_jobs"] == 0
    print(json.dumps({"status": "PASS", "new_numeric_cells": 6, "state_cells": 4,
                      "other_table_cells_changed": 0, "cap": 3, "allocated": 2, "DAG_width": 2,
                      "GPU_calls": 0}))


if __name__ == "__main__":
    main()
