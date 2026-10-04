"""CPU-only publication/review of immutable, already stored V14 B1 scalars.

No model, evaluator, scheduler, or training imports. The execution attempt is
read-only. The output contains compact aggregates and exact small collector
copies; context/per-request raw, prompts, and tensors remain local.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics


FIELDS = ("action_norm", "target_norm", "error_norm", "norm_ratio",
          "directional_ratio", "cosine", "relative_error",
          "zero_target_leakage_over_anchor")
LAYERS = (4, 5, 6, 7, 8)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def stats(values):
    present = [float(v) for v in values if v is not None]
    require(all(math.isfinite(v) for v in present), "nonfinite stored scalar")
    return dict(count=len(present), undefined=len(values) - len(present),
                mean=statistics.mean(present) if present else None,
                median=statistics.median(present) if present else None,
                min=min(present) if present else None,
                max=max(present) if present else None,
                RMS=math.sqrt(statistics.mean(v*v for v in present)) if present else None)


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode()


def csv_bytes(rows):
    require(bool(rows), "empty CSV")
    fields = list(dict.fromkeys(k for r in rows for k in r))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def create_or_reuse(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(not path.is_symlink() and path.is_file() and path.read_bytes() == data,
                "output exists with different bytes: " + str(path))
    else:
        with path.open("xb") as handle:
            handle.write(data)


class StoredAttempt:
    def __init__(self, root):
        self.root = root.resolve(strict=True)
        inventory = json.loads((self.root / "cpu-report/inventory.json").read_bytes())
        require(inventory["no_checkpoint"] and inventory["raw_local_KEEP"], "storage policy")
        self.inventory = {r["path"]: r for r in inventory["files"]}
        self.receipts = {}

    def read(self, relative):
        path = self.root / relative
        require(path.is_file() and not path.is_symlink(), "not a regular input")
        data = path.read_bytes()
        digest = sha(data)
        previous = self.inventory.get(str(path))
        if previous:
            require(len(data) == previous["bytes"] and digest == previous["sha256"],
                    "named stored input differs from collector inventory: " + relative)
        self.receipts[relative] = dict(path=str(path), bytes=len(data), sha256=digest,
                                     verification="CURRENT_NAMED_SMALL_INPUT_FULL_SHA")
        return data

    def load(self, relative):
        return json.loads(self.read(relative))


def raw_rows(attempt, endpoint):
    folder = "B1/W0" if endpoint == "W0" else "B1/V14_RD/evaluation"
    rows = []
    summary = attempt.load(folder + "/summary.json")
    for path in sorted((attempt.root / folder).glob("chunk-*.json")):
        chunk = attempt.load(str(path.relative_to(attempt.root)))
        require(chunk["state"] == summary["state"] and not chunk["optimizer_feedback"],
                "observer chunk state/feedback")
        rows.extend(chunk["rows"])
    require(len(rows) == 1300 == summary["row_count"], "R/P/N row count")
    require(len({r["identity"] for r in rows}) == len(rows), "duplicate observer row")
    require(len({r["case_id"] for r in rows}) == 100, "request count")
    require(summary["no_mutation"] and not summary["optimizer_feedback"], "observer mutation")
    for row in rows:
        require(row["endpoint"] == endpoint, "endpoint identity")
        require(row["margin_true_minus_new"] == row["true_nll"] - row["new_nll"], "margin sign")
        require(all(math.isfinite(row[k]) for k in ("true_nll", "new_nll")), "NLL finite")
        for target in ("true", "new"):
            require(0 <= row[target + "_token_correct"] <= row[target + "_token_count"],
                    "token denominator")
            require(row[target + "_strict"] == (row[target + "_token_correct"] ==
                                                row[target + "_token_count"]), "strict tokens")
    for kind, count in (("R", 100), ("P", 200), ("N", 1000)):
        selected = [r for r in rows if r["kind"] == kind]
        require(len(selected) == count, "panel denominator")
        target = "true" if kind == "N" else "new"
        prefer = lambda r: r["true_nll"] < r["new_nll"] if kind == "N" else r["new_nll"] < r["true_nll"]
        reduced = dict(numerator=sum(prefer(r) for r in selected), denominator=count,
                       strict_numerator=sum(r[target + "_strict"] for r in selected),
                       desired_token_correct=sum(r[target + "_token_correct"] for r in selected),
                       desired_token_count=sum(r[target + "_token_count"] for r in selected))
        reduced["rate"] = reduced["numerator"] / count
        reduced["token_micro"] = reduced["desired_token_correct"] / reduced["desired_token_count"]
        reduced["prompt_macro"] = statistics.mean(r[target + "_token_correct"] / r[target + "_token_count"] for r in selected)
        for label in ("new", "true"):
            reduced[label + "_nll_mean"] = statistics.mean(r[label + "_nll"] for r in selected)
        for key, value in reduced.items():
            expected = summary["summary"][kind][key]
            require(math.isclose(value, expected, rel_tol=1e-13, abs_tol=1e-13),
                    "independent raw reduction mismatch: " + key)
    return rows, summary


def paired(before, after):
    old = {r["identity"]: r for r in before}
    require(set(old) == {r["identity"] for r in after}, "paired row membership")
    result = {}
    for kind in ("R", "P", "N"):
        result[kind] = {}
        for metric in ("preference", "strict"):
            def success(r):
                if metric == "strict":
                    return r[("true" if kind == "N" else "new") + "_strict"]
                return r["true_nll"] < r["new_nll"] if kind == "N" else r["new_nll"] < r["true_nll"]
            counts = Counter()
            for r in (r for r in after if r["kind"] == kind):
                previous = old[r["identity"]]
                require(all(previous[k] == r[k] for k in
                            ("case_id", "kind", "prompt_index", "new_token_identity", "true_token_identity")),
                        "paired token/input identity")
                a, b = success(previous), success(r)
                counts["retained" if a and b else "lost" if a else "gained" if b else "both_failure"] += 1
            result[kind][metric] = {k: counts[k] for k in ("retained", "lost", "gained", "both_failure")}
    return result


def realization(actions):
    rows = actions["rows"]
    require(len(rows) == 3500 and len({(r["layer"], r["row"]) for r in rows}) == 3500,
            "native context/layer identity")
    records = []
    for scope in ("native_mean", "canonical", "rewrite", "KL"):
        if scope == "native_mean":
            scoped = [dict(layer=int(layer), owner=owner, actual=metric)
                      for layer, values in actions["mean"].items() for owner, metric in enumerate(values)]
        else:
            scoped = [r for r in rows if (r["canonical"] if scope == "canonical" else
                                          r["kind"] == ("kl" if scope == "KL" else "rewrite"))]
        require(len(scoped) == (3000 if scope == "rewrite" else 500), "realization scope coverage")
        channels = ("actual",) if scope == "native_mean" else ("ideal", "effective", "actual")
        for layer in ("ALL",) + LAYERS:
            selected = scoped if layer == "ALL" else [r for r in scoped if r["layer"] == layer]
            for channel in channels:
                values = [r[channel] for r in selected]
                require(all((v["norm_ratio"] is None) == (v["target_norm"] == 0) for v in values),
                        "zero-target null semantics")
                require(all((v["directional_ratio"] is None) == (v["target_norm"] == 0) for v in values),
                        "zero-target directional null")
                owners = defaultdict(list)
                for row, value in zip(selected, values):
                    owners[row["owner"]].append(value["error_norm"])
                fields = {key: stats([v[key] for v in values]) for key in FIELDS}
                fields["owner_error_weighted_RMS"] = stats([
                    math.sqrt(statistics.mean(x*x for x in v)) for v in owners.values()])
                fields["owner_error_max"] = stats([max(v) for v in owners.values()])
                records.append(dict(scope=scope, layer=layer, channel=channel,
                                    rows=len(values), zero_target_count=sum(v["target_norm"] == 0 for v in values),
                                    fields=fields))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--v13-attempt", type=Path)
    args = parser.parse_args()
    attempt = StoredAttempt(args.attempt)
    repo = args.repo.resolve(strict=True)
    require(not repo.is_relative_to(attempt.root), "output cannot be execution source")
    report = repo / "experiment-reports/servers/server4/jlz-v14-native-writer-aware-b1"
    audit = repo / "audits/servers/server4/jlz-v14-native-writer-aware-b1/results-review"
    c = attempt.load("config.json")
    lock = attempt.load("execution.lock.json")
    require(sha(attempt.read("config.json")) == lock["config_sha256"], "config lock SHA")
    verified_source = []
    for member in lock["source_members"]:
        path = Path(member["path"])
        require(path.is_relative_to(attempt.root / "source") and path.is_file() and not path.is_symlink(),
                "source closure path")
        data = path.read_bytes()
        require(len(data) == member["bytes"] and sha(data) == member["sha256"], "execution source SHA")
        relative = str(path.relative_to(attempt.root / "source"))
        current_equal = None
        if relative.startswith("project/run_scripts/jlz_native_writer_aware/"):
            current_equal = (repo / relative).read_bytes() == data
            require(current_equal, "publication production bytes differ from execution")
        verified_source.append(dict(path=relative, bytes=len(data), sha256=sha(data),
                                    published_core_equal=current_equal))
    assets = []
    for member in c["assets"]:
        status = Path(member["path"]).stat()
        require(all(getattr(status, field) == member[key] for field, key in
                    (("st_size", "bytes"), ("st_ino", "inode"), ("st_mtime_ns", "mtime_ns"))), "asset current stat")
        assets.append({k: member[k] for k in ("path", "bytes", "sha256")} |
                      dict(verification="PRIOR_FULL_SHA_PLUS_CURRENT_STAT_NOT_REHASHED"))
    for old, new in (("metrics.json", "metrics.json"), ("comparison-B1.csv", "comparison-B1.csv"),
                     ("cost.json", "cost.json"), ("inventory.json", "collector-inventory.json"),
                     ("terminal.json", "collector-terminal.json"), ("report-ko.md", "collector-report-ko.md")):
        create_or_reuse(report / new, attempt.read("cpu-report/" + old))
    before, w0 = raw_rows(attempt, "W0")
    after, endpoint = raw_rows(attempt, "V14_RD")
    identity_member = c["observer_identity"]
    identity_path = Path(identity_member["path"])
    identity_bytes = identity_path.read_bytes()
    require(len(identity_bytes) == identity_member["bytes"] and sha(identity_bytes) == identity_member["sha256"],
            "sealed observer identity SHA")
    identities = json.loads(identity_bytes)["rows"]
    require(len(identities) == len(before) and
            all(all(expected[k] == actual[k] for k in expected) for expected, actual in zip(identities, before)),
            "all observer rows/order/input/token identity")
    require(set(c["packing"]["ids"]) == {r["case_id"] for r in before}, "first100 case membership")
    pairs = paired(before, after)
    collected = attempt.load("cpu-report/metrics.json")
    for kind, metrics in pairs.items():
        for metric, counts in metrics.items():
            original = collected["paired"]["V14_RD"][kind][metric]
            require(all(original[k] == counts[k] for k in ("retained", "lost", "gained")) and
                    original["denominator"] == sum(counts.values()), "collector paired counts")
    fit = attempt.load("B1/fit/fit.json")
    events = [attempt.load("B1/fit/candidate-%02d.json" % i) for i in range(fit["candidates"])]
    require(len(events) == 25 and fit["updates"] == 24 and fit["terminal_candidate"] == 24,
            "fit budget")
    candidate_rows, component_rows = [], []
    for i, event in enumerate(events):
        require(event["candidate"] == i and event["ordinal"] == i+1 and event["B"] == 100 and
                event["updates_before"] == i and event["terminal"] == (i == 24), "candidate sequence")
        require(len(event["J"]) == 100 and len(event["requested_budget"]) == 100,
                "whole-B event coverage")
        require(max(event["requested_budget"]) <= .750001, "requested budget")
        require(math.isclose(statistics.mean(event["J"]), event["J_mean"], abs_tol=1e-12), "J mean")
        require(math.isclose(event["J_mean"], event["nll_mean"] + .0625*event["KL_mean"] + event["norm_mean"],
                             abs_tol=1e-12), "objective components")
        if i == 24:
            require(not event["gradient_measured"] and event["gradient"] is None and
                    event["component_gradient_norms"] is None and event["gradient_status"] == "NO_BACKWARD_TERMINAL",
                    "terminal gradient semantics")
        else:
            require(event["gradient_measured"] and event["reverse"]["bridge_count"] == 1 and
                    event["updates_after"] == i+1 and len(event["projection"]) == 100,
                    "one backward/bridge/update")
            require(all(r["all_columns"] == 100 and r["solve_VJP"] == 1
                        for r in event["reverse"]["layers"]), "causal solve adjoint")
            require(all(r["update"] == i+1 and r["stored_budget"] <= .750001 for r in event["projection"]),
                    "synchronous projection counter")
            require(not all(x < .05 for x in event["J"]), "common early stop ignored")
            for layer in LAYERS:
                grads = event["component_gradient_norms"]
                component_rows.append(dict(candidate=i, layer=layer,
                                           F_gradient_norm=grads["F"][str(layer)],
                                           analytic_norm_gradient_norm=grads["norm"][str(layer)],
                                           total_gradient_norm=grads["total"][str(layer)],
                                           **{k: v for k, v in next(r for r in event["reverse"]["layers"]
                                                                      if r["layer"] == layer).items()
                                              if k != "layer"}))
        row = {k: event.get(k) for k in ("candidate", "ordinal", "updates_before", "updates_after", "terminal",
                                        "gradient_measured", "gradient_status", "J_mean", "nll_mean", "KL_mean", "norm_mean",
                                        "builder_seconds", "masked_seconds", "replay_seconds", "optimizer_seconds")}
        row.update(J_min=min(event["J"]), J_max=max(event["J"]), requests_J_below_005=sum(x < .05 for x in event["J"]),
                   budget_max=max(event["requested_budget"]),
                   solve_residual_max=max(s["relative_residual"] for s in event["solve"].values()),
                   projected_requests=sum(r["tau"] > 0 for r in event.get("projection", [])))
        candidate_rows.append(row)
    writer = attempt.load("B1/V14_RD/writer.json")
    complete = attempt.load("B1/V14_RD/complete.json")
    terminal = attempt.load("B1/terminal.json")
    require(writer["accepted_weight_copy_exact"] and writer["postfit_commits"] == complete["commit_count"] == 1 and
            writer["history_appends"] == complete["history_appends"] == 5, "commit/history count")
    require(writer["before"] == w0["state"] and writer["after"] == endpoint["state"] == complete["state"], "state chain")
    require(fit["weights"] == writer["after"]["W"] and all(fit["R"][l] == r["actual_RHS_sha"]
            for l, r in writer["layers"].items()), "terminal payload identity")
    require(all(h["append_count"] == 1 and h["columns"] == 100 and h["rewrite_only"] and
                h["before"] == writer["before"]["H"][str(h["layer"])] and
                h["after"] == writer["after"]["H"][str(h["layer"])] for h in writer["history"]), "history joins")
    require(terminal["fit_calls"] == terminal["commit_calls"] == 1 and terminal["no_B2"] and
            not terminal["checkpoint_saved"] and not terminal["automatic_retry"], "scope/noCP")
    require(attempt.load("B1/rollback-probe.json")["verified"], "rollback probe")
    qualifications = [attempt.load("B1/qualification/B%d/qualification.json" % b) for b in (1, 2, 3)]
    for q in qualifications:
        require(q["native_loss_pass"] and q["additional_fit"] == q["updates"] == 0, "subset qualification")
        require(all(m["passed"] for kind in ("full_dense_gradient", "full_native_hooks_gradient", "microbatch1_gradient", "action", "solve")
                    for m in q[kind].values()), "actual model parity")
    records = realization(attempt.load("B1/V14_RD/actions.json"))
    create_or_reuse(report / "realization-summary.json", encoded(dict(records=records,
        정의="directional_ratio=<actual,R>/||R||²; norm_ratio=||actual||/||R||; zero target는 null",
        미기록="native_mean은 actual만 저장; net entry displacement/inherited-vector decomposition은 NOT_RECORDED")))
    flattened = [dict(scope=r["scope"], layer=r["layer"], channel=r["channel"], rows=r["rows"],
                      zero_target_count=r["zero_target_count"], metric=k, **value)
                 for r in records for k, value in r["fields"].items()]
    create_or_reuse(report / "realization-summary.csv", csv_bytes(flattened))
    share = attempt.load("B1/V14_RD/shares.json")["rows"]
    require(len(share) == 100, "canonical share coverage")
    share_rows = [dict(scope="canonical", metric="share_L1", **stats([r["share_L1"] for r in share])),
                  dict(scope="canonical", metric="realized_relative_total", **stats([r["realized_relative_total"] for r in share]))]
    for n, layer in enumerate(LAYERS):
        for key in ("planned_share", "direct_share"):
            share_rows.append(dict(scope="canonical", layer=layer, metric=key,
                                   **stats([r[key][n] if r[key] is not None else None for r in share])))
    create_or_reuse(report / "shares-summary.csv", csv_bytes(share_rows))
    create_or_reuse(report / "candidate-summary.csv", csv_bytes(candidate_rows))
    create_or_reuse(report / "component-gradient-summary.csv", csv_bytes(component_rows))
    layer_rows = [dict(layer=int(layer), **{k: v for k, v in row.items() if k != "solve"},
                       solve_relative_residual=row["solve"]["relative_residual"])
                  for layer, row in writer["layers"].items()]
    create_or_reuse(report / "writer-layers.csv", csv_bytes(layer_rows))
    masked = [v for r in writer["masked_native_nll"] for v in r]
    actual = [v for r in writer["actual_native_nll"] for v in r]
    gap = [a-m for a, m in zip(actual, masked)]
    fit_stats = dict(fit=fit, terminal_J=stats(events[-1]["J"]), terminal_J_below_005=sum(x < .05 for x in events[-1]["J"]),
                     projected_request_updates=sum(r["projected_requests"] for r in candidate_rows),
                     max_budget_violation=max(r["budget_max"] for r in candidate_rows)-.75,
                     measured_seconds={k: sum(e.get(k, 0) for e in events) for k in
                                       ("builder_seconds", "masked_seconds", "replay_seconds", "optimizer_seconds")},
                     terminal_native_masked_NLL=stats(masked), terminal_native_actual_NLL=stats(actual),
                     actual_minus_masked_context_NLL=stats(gap),
                     masked_KL=stats(writer["masked_native_KL"]), actual_KL=stats(writer["actual_native_KL"]))
    create_or_reuse(report / "fit-statistics.json", encoded(fit_stats))
    create_or_reuse(audit / "actual-qualification.json", encoded(dict(
        범위="기존 B1 입력 부분집합 B=1/2/3, fixed candidate, 추가fit/update0. 순차 batch2/3가 아니다.",
        checks=qualifications, owner_review=True, independent_reviewer=False)))
    comparison = None
    if args.v13_attempt:
        prior = StoredAttempt(args.v13_attempt)
        old_c = prior.load("config.json")
        old_w0 = []
        for p in sorted((prior.root / "B1/W0").glob("chunk-*.json")):
            old_w0.extend(prior.load(str(p.relative_to(prior.root)))["rows"])
        identity = dict(packing=c["packing"] == old_c["packing"], profile=c["profile"] == old_c["profile"],
                        model_path=c["model"] == old_c["model"],
                        assets=[(r["path"],r["bytes"],r["sha256"]) for r in c["assets"]] ==
                               [(r["path"],r["bytes"],r["sha256"]) for r in old_c["assets"]],
                        observer_rows_sha=c["observer_identity"]["sha256"] == old_c["observer_identity"]["sha256"],
                        runtime_source_root=c["runtime"]["source_root_sha256"] == old_c["runtime"]["source_root_sha256"],
                        W0_raw_exact=before == old_w0)
        require(all(identity.values()), "V13 comparison identity mismatch; do not relabel as matched")
        prior_metrics = prior.load("cpu-report/metrics.json")
        comparison = dict(status="COHORT_RUNTIME_W0_MATCHED_DESCRIPTIVE", identities=identity,
                          planner_identical=False, fit_shared=False,
                          주의="V13 virtual planner/five writers와 V14 writer-aware planner 비교. 동일fit/인과 비교가 아니다.",
                          v13_receipts=prior.receipts)
        comparison_rows = []
        for endpoint_name, metric in list(prior_metrics["endpoints"].items()) + [("V14_RD", endpoint["summary"])]:
            for kind in ("R", "P", "N"):
                row = metric[kind]
                comparison_rows.append(dict(endpoint=endpoint_name, kind=kind, numerator=row["numerator"],
                                            denominator=row["denominator"], rate=row["rate"],
                                            strict_numerator=row["strict_numerator"], token_micro=row["token_micro"],
                                            prompt_macro=row["prompt_macro"]))
        create_or_reuse(report / "comparison-V13-V14-B1.csv", csv_bytes(comparison_rows))
    create_or_reuse(audit / "comparison-identity.json", encoded(comparison))
    source = dict(source_commit=lock["source_commit"], source_tree=lock["source_tree"],
                  execution_lock=attempt.receipts["execution.lock.json"], configuration=attempt.receipts["config.json"],
                  archive={k: lock["archive"][k] for k in ("path", "bytes", "sha256")},
                  source_members=verified_source, assets=assets, runtime=c["runtime"],
                  native_reference=c["native_reference"], launchers=lock["launchers"],
                  packing=c["packing"], observer_identity=c["observer_identity"],
                  설명="model/C0는 prior full SHA + current stat. 이번 raw/code 검산은 지정 소형파일만. runtime GPU pending flag는 준비 시점이다.")
    create_or_reuse(audit / "source-input-manifest.json", encoded(source))
    verification = dict(status="STORED_RAW_REDUCTION_AND_LEDGER_VERIFIED", owner_review=True,
                        independent_reviewer=False, new_GPU=0, new_model_forward=0, new_fit=0,
                        new_evaluator_calls=0, new_submission=0, source_members_verified=len(verified_source),
                        published_original_core_members=sum(r["published_core_equal"] is True for r in verified_source),
                        evaluated_rows=2600, paired_counts=pairs, candidates=25, updates=24,
                        request_evaluations=2500, request_update_participations=2400,
                        fit_calls=1, commit_calls=1, history_appends=5, no_B2=True,
                        noCP=True, rollback_probe=True, observer_no_mutation=True,
                        comparison_status=comparison["status"] if comparison else "NOT_REQUESTED",
                        감사범위="저장 raw의 독립CPU집계. 실기 model 재평가/전체 상태 재현/미기록 gradient 추정은 아니다.")
    create_or_reuse(audit / "verification.json", encoded(verification))
    create_or_reuse(audit / "named-input-inventory.json", encoded(dict(files=attempt.receipts,
        원자료="local KEEP; fullstdout/prompt/tensor Git0. 기존 collector inventory도 exact copy.")))
    print(json.dumps(dict(status=verification["status"], report=str(report), records=len(records),
                          candidate_rows=len(candidate_rows), source_members=len(verified_source)), ensure_ascii=False))


if __name__ == "__main__":
    main()
