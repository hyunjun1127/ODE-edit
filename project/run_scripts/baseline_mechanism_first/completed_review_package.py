"""CPU-only deterministic figure and raw-free package hashing utilities.

This module never imports a model, evaluator, launcher, or native writer.
Figures are generated solely from the independently reduced CSV tables.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import stat
import subprocess
import tempfile
import sys


def canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_file(path):
    path = Path(path).absolute()
    for component in (path, *path.parents):
        if stat.S_ISLNK(component.lstat().st_mode):
            raise ValueError(f"SYMLINK_REJECTED: {component}")
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError(f"NOT_REGULAR: {path}")
    return path


def member(path, relative_to=None):
    path = safe_file(path)
    s = path.stat()
    return {"path": str(path.relative_to(relative_to)) if relative_to else str(path),
            "bytes": s.st_size, "sha256": sha(path), "mode": oct(stat.S_IMODE(s.st_mode))}


def save_once(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical(obj) + b"\n"
    with path.open("xb") as stream:
        stream.write(data)


def rows(path):
    with safe_file(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def make_figures(root, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "figure.dpi": 120, "savefig.dpi": 140,
                         "axes.spines.top": False, "axes.spines.right": False})
    root, destination = Path(root), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    data = rows(root / "endpoint-performance.csv")
    transitions = rows(root / "paired-transitions.csv")
    # Table schema is the reducer's public API; no silent field guessing.
    endpoints = ["Middle-B060", "Late-B100"]
    metrics = ["RS", "PS", "NS"]
    def get(endpoint, scope, metric):
        found = [r for r in data if r["endpoint"] == endpoint and r["panel"] == scope and r["metric"] == metric]
        if len(found) != 1:
            raise ValueError(("TABLE_KEY", endpoint, scope, metric, len(found)))
        return found[0]
    fig, axs = plt.subplots(2, 3, figsize=(12, 6), constrained_layout=True)
    for i, endpoint in enumerate(endpoints):
        for j, metric in enumerate(metrics):
            r = get(endpoint, "fullseen", metric)
            d = int(r["denominator"])
            n = [int(r["baseline_numerator"]), int(r["replay_numerator"])]
            axs[i, j].bar(["Original", "Replay"], [100*x/d for x in n], color=["#64748b", "#247ba0"])
            axs[i, j].set_ylim(0, 106)
            axs[i, j].set_title(f"{endpoint} full-seen {metric}")
            for k, v in enumerate(n):
                axs[i, j].text(k, 100*v/d+1, f"{v:,}/{d:,}", ha="center", fontsize=9)
            axs[i, j].set_ylabel("Canonical success (%)")
    fig.savefig(destination / "fullseen-rates.png", metadata={"Software": "ODE-edit CPU CSV plotting"})
    plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, endpoint in zip(axs, endpoints):
        x = np.arange(3)
        for shift, scope, label, color in [(-.18,"Current100","Current 100","#d97706"),(.18,"fullseen","Full-seen","#247ba0")]:
            values = [float(get(endpoint, scope, metric)["delta_pp"]) for metric in metrics]
            ax.bar(x+shift, values, width=.35, label=label, color=color)
        ax.set_xticks(x, metrics); ax.axhline(0, color="black", linewidth=.7)
        ax.set_title(endpoint); ax.set_ylabel("Replay - Original (percentage points)"); ax.legend()
    fig.savefig(destination / "current-versus-fullseen-delta.png", metadata={"Software": "ODE-edit CPU CSV plotting"})
    plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, endpoint in zip(axs, endpoints):
        x = np.arange(3)
        for shift, field, label, color in [(-.18,"lost","Lost original successes","#b45309"),(.18,"gained","Gained successes","#247ba0")]:
            values=[int(next(r for r in transitions if r["endpoint"]==endpoint and r["panel"]=="fullseen" and r["metric"]==metric)[field]) for metric in metrics]
            ax.bar(x+shift, values, width=.35, label=label, color=color)
        ax.set_xticks(x, metrics); ax.set_title(endpoint+" full-seen paired transitions")
        ax.set_ylabel("Prompt pairs (not independent runs)"); ax.legend(fontsize=8)
    fig.savefig(destination / "paired-transitions.png", metadata={"Software": "ODE-edit CPU CSV plotting"})
    plt.close(fig)
    return {"python":platform.python_version(),"numpy":np.__version__,"matplotlib":matplotlib.__version__,
            "backend":matplotlib.get_backend(),"font":"DejaVu Sans"}


def plots(root):
    root=Path(root).resolve()
    before=[member(root/name) for name in ("endpoint-performance.csv","paired-transitions.csv")]
    env=make_figures(root,root/"figures")
    first=[member(p,root) for p in sorted((root/"figures").glob("*.png"))]
    with tempfile.TemporaryDirectory(prefix=".plot-reproduction-",dir=root) as tmp:
        env2=make_figures(root,Path(tmp))
        second={p.name:sha(p) for p in Path(tmp).glob("*.png")}
    assert env==env2
    assert all(second[Path(r["path"]).name]==r["sha256"] for r in first), "PNG_BYTE_REPRODUCTION"
    assert before==[member(root/name) for name in ("endpoint-performance.csv","paired-transitions.csv")], "INPUT_MUTATION"
    command=f"/mnt/raid5/janghj/EasyEdit/.venv/bin/python -m project.run_scripts.baseline_mechanism_first.completed_review_package plots --root {root}"
    save_once(root/"plot-reproduction.json",{"status":"BYTE_IDENTICAL_REAL_RERUN_PASS","inputs":before,
        "source":member(__file__),"environment":env,"command":command,"outputs":first,
        "independent_generation_count":2,"manual_image_edit":0,"image_generation_tool":0})


def seal(root):
    root=Path(root).resolve(); repo=Path.cwd().resolve()
    sources=[member(p,repo) for p in sorted((repo/"project/run_scripts/baseline_mechanism_first").rglob("*.py"))]
    save_once(root/"source-manifest.json",{"analysis_source_head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "analysis_source_tree":subprocess.check_output(["git","rev-parse","HEAD^{tree}"],text=True).strip(),
        "members":sources,"members_root":hashlib.sha256(canonical(sources)).hexdigest(),
        "execution_source_head":"58f50a25809779918b22ad0aceded732c097eab4",
        "native_source_head":"b51dcf5ab825608bee81dd13549318d8d267e835"})
    outputs=[member(p,root) for p in sorted(root.rglob("*")) if p.is_file()]
    prohibited={".pt",".pth",".pkl",".npz",".log",".jsonl"}
    assert all(Path(m["path"]).suffix not in prohibited for m in outputs)
    manifest={"schema":"E01_COMPLETED_REVIEW_PACKAGE_V1","outputs":outputs,
        "members_root":hashlib.sha256(canonical(outputs)).hexdigest(),"scientific_promotion":False,
        "scope":"Middle/Late completed observation review; NOT full E01 completion",
        "raw_broadcast":"NO_BROADCAST_NOT_REQUIRED","model_forward":0,"GPU":0,"Slurm_mutation":0}
    save_once(root/"analysis-manifest.json",manifest)
    bound=member(root/"analysis-manifest.json",root)
    identity=hashlib.sha256(canonical(bound)).hexdigest()
    save_once(root/"rooted-receipt.json",{"schema":"E01_COMPLETED_REVIEW_ROOT_V1","status":"REVIEW_COMPLETE_WITH_RECORDED_LIMITATIONS",
        "manifest":bound,"receipt_identity":identity,"members_root":manifest["members_root"],"full_E01_complete":False,
        "scientific_promotion":False,"raw_or_prompt_commit":0})
    verify(root)


def validate(root):
    """Actual second CPU reducer run and focused tests, no model imports."""
    root=Path(root).resolve();repo=Path.cwd().resolve()
    raw="/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-e01-fullseen-schema-repair-r3/local/baseline-mechanism-first-e01/20260912-v1/fullseen-schema-repair-r3"
    dataset="/mnt/raid5/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1/counterfact.json"
    command=[sys.executable,"-m","project.run_scripts.baseline_mechanism_first.completed_review_reducer","--root",raw,"--dataset",dataset]
    results=[]
    with tempfile.TemporaryDirectory(prefix=".reducer-reproduction-",dir=root) as tmp:
        for extra in ([],["--atwrite-only"]):
            completed=subprocess.run(command+["--output",tmp]+extra,capture_output=True,text=True,check=True)
            results.append({"command":command+["--output","<new temporary directory>"]+extra,"exit_code":completed.returncode})
        names=[p.name for p in sorted(Path(tmp).iterdir()) if p.is_file()]
        comparisons=[]
        for name in names:
            comparisons.append({"name":name,"first_sha256":sha(root/name),"second_sha256":sha(Path(tmp)/name)})
        assert all(x["first_sha256"]==x["second_sha256"] for x in comparisons), "REDUCER_REPRODUCTION"
    testcmd=[sys.executable,"-m","unittest","discover","-s","project/run_scripts/baseline_mechanism_first/tests","-p","test_completed_review*.py","-v"]
    tests=subprocess.run(testcmd,capture_output=True,text=True,check=True)
    import re
    count=int(re.search(r"Ran (\d+) tests",tests.stderr).group(1))
    for p in (repo/"project/run_scripts/baseline_mechanism_first").rglob("*.py"):
        compile(p.read_bytes(),str(p),"exec")
    save_once(root/"validation.json",{"status":"FOCUSED_AND_REDUCER_BYTE_REPRODUCTION_PASS",
        "test_command":testcmd,"focused_test_count":count,"test_exit_code":tests.returncode,
        "test_output":tests.stderr,"compile_all_analysis_sources":"PASS","commands":results,
        "comparisons":comparisons,"GPU_model_forward":0,"Slurm_mutation":0})


def inputs(root):
    root=Path(root).resolve();repo=Path.cwd().resolve()
    refs=[]
    for name in ("reduction-checks.json","atwrite-checks.json","evidence-reuse-manifest.json"):
        refs.append(member(root/name))
    audit=repo/"audits/servers/server1/2026-09-13-e01-middle-late-review/source-integrity-review.json"
    refs.append(member(audit))
    policies=[member(repo/p) for p in ["PROTOCOL.md","messages/head/2026-09-13-sh1-completed-task-detailed-review.md",
        "project/proposals/2026-09-12-baseline-mechanism-first-lifelong-editing-design.md"]]
    save_once(root/"input-manifest.json",{"schema":"E01_INPUT_INDEX_V1","evidence_indexes":refs,"contracts_full_read":policies,
        "index_root":hashlib.sha256(canonical(refs)).hexdigest(),
        "index_semantics":"각 index 내부 member에는 실제 absolute path/bytes/SHA/검증 수준이 있다. 대형 HF/P 자산은 기존 봉인 재사용이며 신규 full SHA라고 주장하지 않는다.",
        "new_checkpoints_full_SHA_before_after":4,"fullseen_endpoint_requests":[6000,10000],
        "new_scheduler_query":"46439,46440 only; both COMPLETED0:0; allocation2806/4663 seconds",
        "source_policy_main":"ddb70506d90d2c53642251e4683759a4ee0d9bfd","scientific_promotion":False,
        "remote_new_raw_transfer":0,"raw_broadcast":"NO_BROADCAST_NOT_REQUIRED"})


def verify(root):
    root=Path(root).resolve()
    receipt=json.loads((root/"rooted-receipt.json").read_text())
    mref=receipt["manifest"]
    assert member(root/mref["path"],root)==mref
    assert hashlib.sha256(canonical(mref)).hexdigest()==receipt["receipt_identity"]
    manifest=json.loads((root/mref["path"]).read_text())
    assert hashlib.sha256(canonical(manifest["outputs"])).hexdigest()==manifest["members_root"]
    for ref in manifest["outputs"]:
        p=root/ref["path"]
        assert root in p.absolute().parents and ".." not in Path(ref["path"]).parts
        assert member(p,root)==ref, str(p)
    expected={x["path"] for x in manifest["outputs"]}|{"analysis-manifest.json","rooted-receipt.json"}
    assert {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}==expected
    print(json.dumps({"status":"PACKAGE_REHASH_ACCESS_PASS","members":len(manifest["outputs"]),
        "members_root":manifest["members_root"],"receipt_identity":receipt["receipt_identity"]}))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("command",choices=["plots","seal","verify","validate","inputs"])
    parser.add_argument("--root",required=True);args=parser.parse_args()
    {"plots":plots,"seal":seal,"verify":verify,"validate":validate,"inputs":inputs}[args.command](args.root)
