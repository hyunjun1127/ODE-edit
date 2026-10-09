"""Read-only compact-receipt to README check; no model/raw/GPU operations."""
import hashlib
import json
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from official.evaluation.generation.paper_display import paper_cell

INPUTS = {
    "server1/table-rows": "d3ea9896d90dc64c798bcd0308f334d309bf92d6b59e543eda836044a6dd7689",
    "server2/audited-final": "387bd9cf995d88a1c52bf08aa74bf74afcea29e264be1ec71d85a4e56479a462",
}
MODELS = {"llama3": "Llama3-8B-Instruct", "qwen25": "Qwen2.5-7B-Instruct", "gptj": "GPT-J-6B"}
METHODS = {"FT": "FT", "MEMIT": "MEMIT", "ALPHAEDIT": "AlphaEdit", "ALPHAEDIT_BLUE": "AlphaEdit-BLUE", "MEMIT_FE": "MEMIT-FE", "SPHERE": "AlphaEdit+SPHERE"}


def display(value, denominator=None):
    if denominator is not None:
        numerator = round(value * denominator / 100)
        exact = Decimal(numerator) * 100 / denominator
        assert abs(float(exact) - value) < 1e-10
    else:
        exact = Decimal(str(value))
    return str(exact.quantize(Decimal(".01"), rounding=ROUND_HALF_UP))


def main():
    tables, history = {}, {}
    section = None
    for line in (ROOT / "README.md").read_text().splitlines():
        if line.startswith("### "):
            section = line[4:]
        if line.startswith("| "):
            cells = [c.strip() for c in line.split("|")]
            if cells[1] == "MEMIT_FE_HISTORY":
                history[cells[2]] = cells
            elif section in MODELS.values():
                tables[section, cells[1]] = cells
    numeric_cells = status_cells = deferred_cells = completed_rows = zsre_rows = 0
    for member, sha in INPUTS.items():
        server, name = member.split("/")
        path = ROOT / f"audits/servers/{server}/baseline-completed-zsre-audit-20261010/{name}.json"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == sha
        data = json.loads(path.read_text())
        for row in data["rows"]:
            variant = row["method"] == "MEMIT_FE_HISTORY"
            if variant:
                label = {"llama3": "Llama3", "qwen25": "Qwen2.5", "gptj": "GPT-J"}[row["model"]]
                cells = history[label]
                offset = 5
                assert f'({row["job_id"]})' in cells[4] and "W20 COMPLETE" in cells[4]
            else:
                cells = tables[MODELS[row["model"]], METHODS[row["method"]]]
                offset = 2 if row["dataset"] == "cf" else 8
            if row.get("numeric_eligible", row.get("complete", False)):
                completed_rows += 1
                metrics = row.get("metrics") or row["summary"]
                cf = row["dataset"] == "cf"
                zsre_rows += not cf
                names = ["Efficacy", "Generalization", "Specificity"]
                values = [display(metrics[n], d if cf else None) for n, d in zip(names, [2000, 4000, 20000])]
                if cf:
                    values.insert(0, display(metrics["Score"]))
                assert cells[offset:offset + len(values)] == values, (row["job_id"], cells, values)
                numeric_cells += len(values)
                for metric, index, unit in [("Flu", 6, "bits"), ("Con", 7, "cosine_0_to_1")]:
                    if metric in row:
                        expected = paper_cell(row[metric], metric=metric, raw_unit=unit)
                        assert cells[index] == expected, (row["job_id"], metric)
                        if expected == "DEFERRED":
                            deferred_cells += 1
                        else:
                            numeric_cells += 1
            else:
                state = "ING" if row["status"] == "RUNNING" else row["status"]
                value = f'{state}: {row["job_name"]} ({row["job_id"]})'
                count = 4 if row["dataset"] == "cf" else 3
                assert cells[offset:offset + count] == [value] * count, row["job_id"]
                status_cells += count
        for job in data.get("generation_jobs", []):
            if job["job_id"] not in {"62259", "62260", "62261"}:
                continue
            method = {"62259": "FT", "62260": "AlphaEdit+SPHERE", "62261": "MEMIT-FE"}[job["job_id"]]
            assert job["state"] == "RUNNING" and not job["completed_receipt_exists"]
            expected = f'ING: {job["job_name"]} ({job["job_id"]})'
            assert tables[MODELS["llama3"], method][6:8] == [expected] * 2
            status_cells += 2
    assert (completed_rows, zsre_rows) == (30, 13)
    print(json.dumps({"status": "PASS_COMPACT_TO_TABLE", "completed_rows": completed_rows, "zsre_public_query_rows": zsre_rows, "numeric_cells": numeric_cells, "deferred_cells": deferred_cells, "status_cells": status_cells, "model_forward_calls": 0}))


if __name__ == "__main__":
    main()
