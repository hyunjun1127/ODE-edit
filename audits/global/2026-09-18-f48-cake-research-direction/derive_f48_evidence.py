"""Read verified v2 report exports; derive F48 tables and a static research figure.

No model imports, forwards, remote writes, or experiment launches.
"""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "2026-09-18-sequential-local-z-allocation-review/report-snapshot"


def read(name):
    with (SOURCE / name).open() as handle:
        return list(csv.DictReader(handle))


def write_csv(name, rows):
    with (HERE / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


final = read("final-table.csv")
generic = read("generic.csv")
performance = read("performance.csv")
compute = {r["arm"]: r for r in read("compute.csv")}
candidates = read("candidate.csv")
observer = read("candidate-observer.csv")

rows = []
for row in final:
    arm = row["arm"]
    dev = next(x for x in generic if x["arm"] == arm and x["batch"] == "10"
               and x["role"] == "Dev128_OBSERVER")
    base = next(x for x in generic if x["arm"] == arm and x["batch"] == "10"
                and x["role"] == "S64_SELECTED_ONLINE")
    ps = next(x for x in performance if x["arm"] == arm and x["view"] == "W10_ALL1000"
              and x["metric"] == "PS")
    rows.append(dict(arm=arm, RS_percent=float(row["RS_percent"]),
                     PS_percent=float(row["PS_percent"]), NS_percent=float(row["NS_percent"]),
                     P_strict_percent=float(ps["strict_percent"]),
                     P_two_strict_count=int(ps["two_P_strict"]),
                     S64_KL=float(base["D"]), Dev128_KL=float(dev["D"]),
                     program_seconds=float(compute[arm]["program_seconds"]),
                     program_ratio_N4=float(compute[arm]["program_seconds"])
                     / float(compute["N4"]["program_seconds"])))
write_csv("v2-final-frontier.csv", rows)

batch1 = []
for row in candidates:
    if row["arm"] != "G48" or row["batch"] != "1":
        continue
    matches = [r for r in observer if r["arm"] == row["arm"] and r["batch"] == "1"
               and r["candidate"] == row["candidate"]]
    assert {r["metric"] for r in matches} == {"RS", "PS", "NS"}
    metrics = {r["metric"]: r for r in matches}
    batch1.append(dict(gates=row["gates"], E_native=float(row["E"]),
                       S64_KL=float(row["B"]), actual_concat_step_norm=float(row["action_norm"]),
                       RS_percent=float(metrics["RS"]["percent"]),
                       PS_percent=float(metrics["PS"]["percent"]),
                       P_strict_percent=float(metrics["PS"]["strict_percent"]),
                       NS_percent=float(metrics["NS"]["percent"]),
                       feasible=row["feasible"], reasons=row["reasons"]))
assert len(batch1) == 6
write_csv("same-entry-b1-endpoints.csv", batch1)

decomposition = []
for arm in [r["arm"] for r in final]:
    for metric in ["PS", "NS"]:
        atwrite = next(r for r in performance if r["arm"] == arm and r["metric"] == metric
                       and r["view"] == "POOLED_ATWRITE")
        endpoint = next(r for r in performance if r["arm"] == arm and r["metric"] == metric
                       and r["view"] == "W10_ALL1000")
        decomposition.append(dict(arm=arm, metric=metric,
                                  pooled_at_write=float(atwrite["percent"]),
                                  W10=float(endpoint["percent"]),
                                  net_change_pp=float(endpoint["percent"])-float(atwrite["percent"])))
write_csv("arrival-and-retention.csv", decomposition)

step_summary = {}
for arm in [r["arm"] for r in final]:
    selected = [r for r in candidates if r["arm"] == arm and r["selected"] == "True"]
    assert len(selected) == 10
    norms = [float(r["action_norm"]) for r in selected]
    step_summary[arm] = dict(sum_step_concat_fro=sum(norms),
                             sum_step_concat_fro_squared=sum(n*n for n in norms),
                             own_n4_selected=sum(r["own_n4"] == "True" for r in selected))
for arm, values in step_summary.items():
    for key in ["sum_step_concat_fro", "sum_step_concat_fro_squared"]:
        values[key + "_vs_N4_ratio"] = values[key] / step_summary["N4"][key]
summary = {"step_norm_summary": step_summary,
           "note": "Sum of stored actual concat per-step norms/squares; net displacement and "
                   "layer-resolved norms unavailable; association not causal proof."}
n4, f48 = (next(r for r in rows if r["arm"] == a) for a in ["N4", "F48"])
summary["F48_vs_N4"] = {
    "S64_KL_relative_reduction": 1 - f48["S64_KL"] / n4["S64_KL"],
    "Dev128_KL_relative_reduction": 1 - f48["Dev128_KL"] / n4["Dev128_KL"],
    "program_ratio": f48["program_ratio_N4"],
    "NS_delta_pp": f48["NS_percent"] - n4["NS_percent"],
    "PS_delta_pp": f48["PS_percent"] - n4["PS_percent"],
    "P_strict_delta_pp": f48["P_strict_percent"] - n4["P_strict_percent"],
    "P_two_strict_delta_pp": (f48["P_two_strict_count"] - n4["P_two_strict_count"]) / 10,
}
(HERE / "f48-derived-summary.json").write_text(json.dumps(summary, indent=2) + "\n")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), layout="constrained")
colors = {"N4": "#555555", "F48": "#d34b36", "G48": "#167a78", "C4": "#9473a8",
          "C48": "#3e78b4", "C45678": "#b48b24"}
offsets = {"N4": (8, -17), "F48": (10, -5), "G48": (-45, -25), "C4": (-78, 10),
           "C48": (10, 2), "C45678": (-55, 12)}
for r in rows:
    axes[0].scatter(r["PS_percent"], r["NS_percent"], s=65*r["program_ratio_N4"],
                    color=colors[r["arm"]], alpha=.85, edgecolor="white", linewidth=.8)
    axes[0].annotate(f'{r["arm"]} ({r["program_ratio_N4"]:.2f}x)',
                     (r["PS_percent"], r["NS_percent"]), xytext=offsets[r["arm"]],
                     textcoords="offset points", color=colors[r["arm"]], fontsize=9)
axes[0].set(xlabel="Generalization: final PS (%)", ylabel="Locality: final NS (%)",
            title="A. Six actual 1,000-edit chains", xlim=(95.2, 97.2), ylim=(79.95, 83.65))
axes[0].text(.025, .975, "Bubble area = total program time / N4\nF48 was exempt from native-quality guards",
             transform=axes[0].transAxes, va="top", fontsize=9)
axes[0].grid(alpha=.15)

chosen = {(tuple(json.loads(r["gates"]))): r for r in batch1}
sequence = [(0.75, 0.0), (0.75, 0.5), (0.75, 1.0)]
for left, right in zip(sequence[:-1], sequence[1:]):
    a, b = chosen[left], chosen[right]
    axes[1].annotate("", (b["E_native"], b["S64_KL"]*1000),
                     (a["E_native"], a["S64_KL"]*1000),
                     arrowprops=dict(arrowstyle="->", color="#d34b36", lw=1.4))
labels = {(0.75, 0.0): ("L4 .75 only\nPS 91.0%", (-110, 7)),
          (0.75, 0.5): ("F48: L8 .5\nPS 95.0%", (10, -29)),
          (0.75, 1.0): ("L8 1.0\nPS 94.5%", (12, 4)),
          (1.0, 0.0): ("Full L4 (N4)\nPS 97.0%", (12, 7))}
for gates, (label, offset) in labels.items():
    r = chosen[gates]
    color = "#555555" if gates == (1., 0.) else "#d34b36"
    axes[1].scatter(r["E_native"], r["S64_KL"]*1000, s=65, color=color, zorder=4)
    axes[1].annotate(label, (r["E_native"], r["S64_KL"]*1000), xytext=offset,
                     textcoords="offset points", fontsize=9, color=color)
cap = chosen[(1., 0.)]["E_native"] + 1e-4
axes[1].axvspan(.002, cap, color="#167a78", alpha=.09)
axes[1].axvline(cap, color="#167a78", lw=1, ls="--")
axes[1].set(xscale="log", xlabel="Native-context rewrite NLL E (lower is better)",
            ylabel="S64 KL to W0 (x 1,000; lower is better)",
            title="B. Same W0 / same first 100 requests", xlim=(.0028, .29), ylim=(.80, 2.28))
axes[1].text(.035, .05, "L8 supplementation recovers edit quality\nwhile increasing KL from the partial-L4 state.",
             transform=axes[1].transAxes, fontsize=9)
axes[1].grid(alpha=.15)
fig.suptitle("F48: strong locality tradeoff, with distinct strength and target effects", fontsize=14)
fig.savefig(HERE / "f48-frontier-and-mechanism.png", dpi=190)
fig.savefig(HERE / "f48-frontier-and-mechanism.pdf")
plt.close(fig)
print(json.dumps({"final_chains": len(rows), "same_entry_endpoints": len(batch1),
                  "arrival_retention_rows": len(decomposition), "model_forwards": 0}))
