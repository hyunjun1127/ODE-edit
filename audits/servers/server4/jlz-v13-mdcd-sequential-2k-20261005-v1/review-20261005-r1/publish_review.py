"""Publish compact CPU-review aggregates; never load a model or alter a run.

Inputs are independent review outputs and already tracked historical tables.
This is a publication adapter, not another production evaluator.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

TASK = "jlz-v13-mdcd-sequential-2k-20261005-v1"
AUDIT = f"audits/servers/server4/{TASK}/review-20261005-r1"
REPORT = f"experiment-reports/servers/server4/{TASK}"
HIST = "experiment-reports/servers/server3/jlz-v12-shared-budget-bs100x20-20261004-v1/review-20261005-v1"
ARMS = ("MD", "CD")
KINDS = ("R", "P", "N")
STEPS = (5, 10, 15, 20)


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path, rows):
    assert rows, path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def identity(path, repo=None):
    return {"path": str(path.relative_to(repo)) if repo else str(path),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def build(repo, local):
    r = read(local / "independent-cpu-r1/independent-metrics.json")
    t = read(local / "telemetry-r1/telemetry-summary.json")
    assert r["status"] == "COMPLETE_RAW_CPU_VERIFIED"
    assert r["collector_aggregate_agreement"]
    out = repo / REPORT / "review-20261005-r1"
    out.mkdir(parents=True, exist_ok=True)
    audit = repo / AUDIT
    for src, dst in (
        (local / "independent-cpu-r1/independent-comparison.csv", out / "metrics-all-endpoints.csv"),
        (local / "telemetry-r1/telemetry-batch.csv", out / "telemetry-batch.csv"),
        (local / "telemetry-r1/telemetry-summary.json", out / "telemetry-summary.json"),
        (local / "independent-cpu-r1/raw-read-manifest.json", audit / "raw-read-manifest.json"),
        (local / "binding-check-r2.json", audit / "binding-verification.json"),
        (local / "reader-development-failure-r1.json", audit / "reader-development-failure-r1.json"),
    ):
        shutil.copyfile(src, dst)

    comparison, tails, paired, cohorts, prefix, population, realization, gaps, q = [], [], [], [], [], [], [], [], []
    metrics_w20 = {"scope": r["scope"], "status": r["status"],
                   "MD_to_CD": r["MD_to_CD"], "arms": {}}
    for arm in ARMS:
        a, telemetry = r["arms"][arm], t["arms"][arm]
        assert (a["commits"], a["next_entry_links"], a["history_appends"]) == (20, 19, 100)
        metrics_w20["arms"][arm] = {
            "W0": a["endpoints"]["W0"], "W20": a["endpoints"]["W20"],
            "completion": {k: a[k] for k in ("commits", "next_entry_links", "history_appends", "request_evaluations", "request_updates")},
            "W20_retention": {k: a["milestones"]["W20"][k] for k in ("W0_to_endpoint", "atwrite_to_endpoint")},
        }
        for endpoint in ("W0", "W5", "W10", "W15", "W20"):
            for kind in KINDS:
                s = a["endpoints"][endpoint][kind]
                for measure in ("true_nll", "new_nll", "desired_nll", "margin_true_minus_new"):
                    tails.append({"arm": arm, "endpoint": endpoint, "kind": kind,
                                  "measure": measure, **s[measure + "_quantiles"]})
        for kind in KINDS:
            s = a["endpoints"]["W20"][kind]
            assert s["denominator"] == {"R": 2000, "P": 4000, "N": 20000}[kind]
            comparison.append({"arm": arm, "endpoint": "W20", "kind": kind, **{k: s[k] for k in (
                "numerator", "denominator", "rate", "strict_numerator", "strict_denominator", "strict_rate",
                "desired_token_correct", "desired_token_count", "token_micro", "prompt_macro", "true_nll_mean", "new_nll_mean",
                "desired_nll_mean", "true_minus_new_mean", "new_minus_true_mean")}})
        for step in STEPS:
            endpoint = f"W{step}"
            m = a["milestones"][endpoint]
            for relation in ("W0_to_endpoint", "atwrite_to_endpoint"):
                for kind in KINDS:
                    for measure, s in m[relation][kind].items():
                        paired.append({"arm": arm, "endpoint": endpoint, "relation": relation,
                                       "kind": kind, "measure": measure, **s})
            for birth, c in m["cohorts"].items():
                for kind in KINDS:
                    s = c["summary"][kind]
                    cohorts.append({"arm": arm, "endpoint": endpoint, "birth_batch": int(birth),
                                    "kind": kind, **{k: s[k] for k in ("numerator", "denominator", "rate", "strict_numerator", "strict_rate", "token_micro", "prompt_macro")}})
            for count, c in m["first_prefix"].items():
                for kind in KINDS:
                    s = c["summary"][kind]
                    prefix.append({"arm": arm, "endpoint": endpoint, "first_requests": int(count), "kind": kind,
                                   **{k: s[k] for k in ("numerator", "denominator", "rate", "strict_numerator", "strict_rate", "token_micro", "prompt_macro")}})
            for label in ("active", "superseded"):
                if m[label] is None:  # Empty population: not a measured zero score.
                    continue
                for kind in KINDS:
                    s = m[label][kind]
                    population.append({"arm": arm, "endpoint": endpoint, "population": label, "kind": kind,
                                       **{k: s[k] for k in ("numerator", "denominator", "rate", "strict_numerator", "strict_rate")}})
        for label, scopes in (("actual_local_action", telemetry["actions_overall"]),
                              ("net_entry_displacement", telemetry["net_entry_displacement"])):
            for scope, s in scopes.items():
                realization.append({"arm": arm, "quantity": label, "scope": scope,
                                    **{k: s[k] for k in ("rows", "zero_target_count", "target_energy", "energy_weighted_directional_ratio", "energy_norm_ratio", "energy_cosine", "energy_relative_error")},
                                    "zero_leakage_over_anchor_max": s["zero_target_leakage_over_anchor"]["max"],
                                    "nonzero_directional_min": s["directional_ratio"]["min"],
                                    "nonzero_directional_max": s["directional_ratio"]["max"],
                                    "nonzero_norm_ratio_max": s["norm_ratio"]["max"]})
        for scope, measures in telemetry["gap_norms"].items():
            for measure, s in measures.items():
                gaps.append({"arm": arm, "scope": scope, "measure": measure, **s})
        q.extend({"arm": arm, "quantity": "THIS_BATCH_UPDATE_NOT_CUMULATIVE", **x} for x in telemetry["milestone_batch_update_Q"])
    for endpoint, kinds in r["MD_to_CD"].items():
        for kind in KINDS:
            for measure, s in kinds[kind].items():
                paired.append({"arm": "MD_to_CD", "endpoint": endpoint, "relation": "same_endpoint", "kind": kind, "measure": measure, **s})
    for name, rows in (("comparison-W20.csv", comparison), ("NLL-tails.csv", tails), ("paired-retention.csv", paired),
                       ("birthcohort.csv", cohorts), ("fixed-prefix.csv", prefix), ("active-superseded.csv", population),
                       ("realization-summary.csv", realization), ("gap-decomposition.csv", gaps), ("Q-update-milestones.csv", q)):
        write_csv(out / name, rows)
    dump(out / "metrics-W20.json", metrics_w20)

    hist_root = repo / HIST
    hist_manifest = read(hist_root / "baseline-manifest.json")
    for f in hist_manifest["files"]:
        actual = identity(repo / f["path"], repo)
        assert actual == f, (actual, f)
    with (hist_root / "baseline-comparison.csv").open() as stream:
        hist_rows = list(csv.DictReader(stream))
    matched, historical = [], []
    for arm in ARMS:
        a = r["arms"][arm]
        matched.append({"method": "V13 " + arm, "endpoint": "W20", "requests": 2000,
                        "comparison_scope": "MATCHED_MD_CD", "RS_percent": 100*a["endpoints"]["W20"]["R"]["rate"],
                        "PS_percent": 100*a["endpoints"]["W20"]["P"]["rate"], "NS_percent": 100*a["endpoints"]["W20"]["N"]["rate"],
                        "score_harmonic_percent": 100*a["milestones"]["W20"]["harmonic"]})
    for row in hist_rows:
        if int(row["batch"]) != 20:
            continue
        s = {"method": row["method"], "endpoint": "W20", "requests": 2000,
             "comparison_scope": "HISTORICAL_REFERENCE",
             **{k: float(row[k]) for k in ("RS_percent", "PS_percent", "NS_percent", "score_harmonic_percent")}}
        assert abs(s["score_harmonic_percent"] - 3/sum(1/s[k] for k in ("RS_percent", "PS_percent", "NS_percent"))) < 1e-9
        historical.append(s)
    write_csv(out / "baseline-W20.csv", matched + historical)
    deltas = []
    for ours in matched:
        for other in historical:
            deltas.append({"method": ours["method"], "reference": other["method"], "comparison_scope": "HISTORICAL_AGGREGATE_NOT_PAIRED",
                           **{k + "_difference_pp": ours[k] - other[k] for k in ("RS_percent", "PS_percent", "NS_percent", "score_harmonic_percent")}})
    write_csv(out / "baseline-differences.csv", deltas)
    dump(audit / "historical-baseline-binding.json", {
        "status": "EXISTING_TRACKED_TABLE_SHA_SIZE_VERIFIED", "new_model_calls": 0, "cross_run_paired": "NOT_AVAILABLE",
        "scope": "own first2000 W20, not finalW100; historical runtime/planner/layers/seed differences remain",
        "source": identity(hist_root / "baseline-comparison.csv", repo), "source_manifest": identity(hist_root / "baseline-manifest.json", repo),
        "underlying_tables": hist_manifest["files"],
        "historical_prompt_macro": "NOT_INCLUDED_IN_REUSED_SUMMARY; underlying tables may record it; never substitute token_micro"})
    curve_svg(out / "milestone-PS-NS.svg", r)
    dump(audit / "publication-adapter-checks.json", {
        "passed": True, "input_status": r["status"], "collector_aggregate_agreement": True,
        "W20_exact_denominators": True, "raw_copy": False, "new_GPU": 0, "model_load": 0,
        "baseline_source_SHA_size_verified": 4, "same_endpoint_MD_CD_paired": True,
        "historical_cross_run_paired": False, "times_nested_not_additive": True})


def curve_svg(path, r):
    # A static scientific figure with only four measured allseen points/arm.
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="820" height="300" viewBox="0 0 820 300">',
             '<rect width="820" height="300" fill="white"/>',
             '<style>text{font:13px sans-serif;fill:#333}</style>',
             '<text x="20" y="22">Measured allseen endpoints: growing cohort, not a fixed-cohort forgetting curve</text>']
    for panel, kind, label in ((0, "P", "PS (%)"), (1, "N", "NS (%)")):
        ox = 60 + panel * 390
        parts.append(f'<text x="{ox}" y="48">{label}</text>')
        for pct in (65, 75, 85, 95, 100):
            y = 250 - (pct-65)/35*180
            parts.extend([f'<line x1="{ox}" y1="{y}" x2="{ox+310}" y2="{y}" stroke="#ddd"/>',
                          f'<text x="{ox-30}" y="{y+4}">{pct}</text>'])
        for arm, color in (("MD", "#1469aa"), ("CD", "#c75421")):
            points = []
            for step in STEPS:
                x = ox + (step-5)/15*310
                value = 100*r["arms"][arm]["endpoints"][f"W{step}"][kind]["rate"]
                y = 250-(value-65)/35*180
                points.append(f"{x:.3f},{y:.3f}")
                parts.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="4" fill="{color}"><title>{arm} W{step}: {value:.6f}%</title></circle>')
            parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"/>')
        for step in STEPS:
            x = ox+(step-5)/15*310
            parts.append(f'<text x="{x-14}" y="272">{step*100}</text>')
        parts.extend([f'<text x="{ox+90}" y="294">edited requests</text>',
                      f'<text x="{ox+215}" y="48" style="fill:#1469aa">MD</text>',
                      f'<text x="{ox+255}" y="48" style="fill:#c75421">CD</text>'])
    parts.append('</svg>')
    path.write_text('\n'.join(parts) + '\n')


def seal(repo):
    bases = [repo / REPORT, repo / AUDIT]
    files = [p for base in bases for p in base.rglob('*') if p.is_file() and p.name != 'publication-manifest.json' and '__pycache__' not in p.parts]
    files += [repo / f"project/run_scripts/jlz_realized_writer_sequential/{name}" for name in ("review_completed.py", "test_review_completed.py")]
    files += [repo / path for path in (
        "messages/acks/server4/2026-10-05-jlz-v13-mdcd-review.json",
        "messages/server-heads/server4/2026-10-05-jlz-v13-mdcd-review.json",
        "tasks/status/jlz-v13-mdcd-sequential-bs100x20-s4-20261005-v1/server4.json")]
    records = [identity(p, repo) for p in sorted(files)]
    dump(repo / AUDIT / "publication-manifest.json", {
        "schema": 1, "nonce": "ODEEDIT-GH-SH4-V13-MDCD-REVIEW-20261005-R1",
        "scope": "compact CPU review only; raw/model/tensors/prompts/fullstdout remain ignored local",
        "self_excluded": True, "files": records, "bytes": sum(x['bytes'] for x in records),
        "new_GPU": 0, "Slurm_write": 0, "monitoring_active": False, "automatic_resume": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[5])
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--seal-only", action="store_true")
    args = parser.parse_args()
    if args.seal_only:
        seal(args.repo)
    else:
        build(args.repo, args.local)
