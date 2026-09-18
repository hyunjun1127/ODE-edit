"""Read-only CPU verification for the frozen server4 SLZ v2 artifacts.

Consolidates the inline commands executed during the 2026-09-18 audit.
The consolidated file has not itself been rerun end-to-end; the independently
observed inline-command results are recorded in artifact-checks.json.

Run on server4 with its existing EasyEdit Python environment, or pipe this file
to SSH's Python stdin. No GPU/model load, scheduler query, file write, network
transfer, or original runtime import occurs inside this script. JSON goes to
stdout. Raw tensors remain on server4.
"""

from collections import Counter, defaultdict
from pathlib import Path
import csv
import hashlib
import json
import math
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch

ROOT = Path("/data/janghj/ODE-edit/local/sequential-local-z-allocation/20260917-v2")
WORKTREE = ROOT / "completed-review-20260918-v1/worktree"
REPORT = WORKTREE / (
    "experiment-reports/servers/server4/"
    "sequential-local-z-allocation-seq1000-2026-09-17-v2/"
    "completed-review-20260918-v1"
)


def read_json(path):
    return json.loads(Path(path).read_text())


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def scalar_finite(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(scalar_finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(scalar_finite(v) for v in value)
    return True


def main():
    started = time.monotonic()
    torch.set_num_threads(4)
    errors = []

    def check(condition, label):
        if not condition:
            errors.append(label)

    lock = read_json(ROOT / "execution.lock.json")
    small_count = small_bytes = prior_stat_count = 0
    for member in lock["members"]:
        path = Path(member["path"])
        stat = path.stat()
        check(stat.st_size == member["bytes"], [str(path), "input size"])
        if member["bytes"] <= 10_000_000:
            small_count += 1
            small_bytes += member["bytes"]
            check(file_sha(path) == member["sha256"], [str(path), "input SHA"])
        if member.get("verification") == "PRIOR_FULL_SHA_STABLE_STAT":
            prior_stat_count += 1
            check([stat.st_dev, stat.st_ino, stat.st_mtime_ns] == member["stat"],
                  [str(path), "prior full SHA stable stat"])
    for key in ("source_archive", "teacher_manifest", "CPU_checks"):
        ref = lock[key]
        check(file_sha(ref["path"]) == ref["sha256"], [key, "SHA"])

    manifest = read_json(REPORT / "analysis-manifest.json")
    publication_members = manifest["source_members"] + manifest["artifacts"]
    for ref in publication_members:
        check(file_sha(WORKTREE / ref["path"]) == ref["sha256"],
              [ref["path"], "publication SHA"])
    ready = read_json(ROOT / "technical/attempt-v1/READY.json")
    for ref in ready["checks"]:
        check(file_sha(ref["path"]) == ref["sha256"], [ref["path"], "technical SHA"])
        check(read_json(ref["path"])["status"] == "PASS", [ref["path"], "technical status"])

    with (REPORT / "raw-inventory.csv").open() as handle:
        inventory = list(csv.DictReader(handle))
    tensor_paths = []
    raw_bytes = 0
    for row in inventory:
        path = Path(row["path"])
        size = int(row["bytes"])
        check(path.stat().st_size == size, [str(path), "raw size"])
        check(file_sha(path) == row["sha256"], [str(path), "raw SHA"])
        raw_bytes += size
        if path.name == "native-evidence.pt":
            tensor_paths.append(path)
    full_sha_seconds = time.monotonic() - started

    by_arm = defaultdict(Counter)
    request_orders = {}
    for path in tensor_paths:
        data = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        observations = data["target_observations"]
        receipt = read_json(path.parent / "receipt.json")
        arm = path.parts[path.parts.index("arms") + 1]
        batch = path.parents[3]
        if batch not in request_orders:
            score = read_json(batch / "episode/scores/000.json")
            request_orders[batch] = [r["case_id"] for r in
                                    score["metrics"]["details"]["training"]["rows"]]
        label = str(path)
        check(set(data) == {"captures", "target", "anchors", "radii", "target_observations"},
              [label, "top keys"])
        check(list(data["target"].shape) == [4096, 100] and
              list(data["anchors"].shape) == [4096, 100] and
              list(data["radii"].shape) == [100], [label, "top shape"])
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                check(value.dtype == torch.float32 and bool(torch.isfinite(value).all()),
                      [label, "top tensor finite/schema", key])
        check(len(observations) == 100, [label, "target count"])
        check(torch.equal(data["target"], torch.stack(data["captures"]["compute_z"], 1)),
              [label, "capture bridge"])
        check(torch.equal(data["target"], torch.stack([r["target"] for r in observations], 1)),
              [label, "observation bridge"])
        check(torch.equal(data["anchors"], torch.stack([r["anchor"] for r in observations], 1)),
              [label, "anchor bridge"])
        for name, values in data["captures"].items():
            check(all(bool(torch.isfinite(t).all()) for t in values), [label, "capture finite", name])
        check([r["case_id"] for r in observations] == request_orders[batch], [label, "request order"])
        check(scalar_finite(observations), [label, "trace scalar finite"])
        for row in observations:
            check(all(bool(torch.isfinite(row[k]).all()) for k in ("target", "anchor", "delta")),
                  [label, "observation tensor finite"])
            check(torch.equal(row["target"], row["anchor"] + row["delta"]),
                  [label, "target equation"])
            check(row["adam_updates"] <= 24 and row["loss_evaluations"] <= 25 and
                  row["loss_evaluations"] == len(row["losses"]) == row["adam_updates"] + 1,
                  [label, "native quota"])
            check(row["layer"] == receipt["layer"], [label, "physical layer"])
        adam = sum(r["adam_updates"] for r in observations)
        losses = sum(r["loss_evaluations"] for r in observations)
        check(adam == receipt["adam_updates"] and losses == receipt["loss_evaluations"],
              [label, "receipt counters"])
        by_arm[arm].update(files=1, targets=len(observations), adam=adam, losses=losses,
                           zero_adam=sum(r["adam_updates"] == 0 for r in observations))

    commits = links = appends = incomplete = 0
    cold_entries = []
    incomplete_reasons = Counter()
    for arm in lock["arms"]:
        out = ROOT / "arms" / arm / "attempt-v1/output"
        terminal = read_json(out / "terminal.json")
        previous = None
        for ref in terminal["commits"]:
            check(file_sha(ref["path"]) == ref["sha256"], [ref["path"], "commit SHA"])
            commit = read_json(ref["path"])
            commits += 1
            history = read_json(commit["history"]["path"])
            appends += history["appends"]
            label = [arm, commit["batch"]]
            check(commit["state"] == history["selected"] and
                  history["entry"]["W"] == history["selected"]["W"] and
                  history["entry"]["M"] == commit["entry"]["M"] and
                  history["candidate_appends"] == 0, label + ["history state"])
            check(sorted(r["layer"] for r in history["rows"]) == [4, 5, 6, 7, 8] and
                  all(r["history_append"] == 1 for r in history["rows"]), label + ["history mapping"])
            if previous is None:
                cold_entries.append(commit["entry"])
            else:
                links += 1
                check(commit["entry"] == previous, label + ["adjacent state"])
            previous = commit["state"]
            for row in read_json(commit["selection"]["path"])["incomplete"]:
                incomplete += 1
                incomplete_reasons[arm + ":" + row["reason"]] += 1
                check(not any(row[k] for k in ("scored", "feasible", "completed")),
                      label + ["incomplete exclusion"])
        check(terminal["terminal_state"] == previous, [arm, "terminal state"])
    check(all(entry == cold_entries[0] for entry in cold_entries), "common cold entry")

    print(json.dumps(dict(
        root=str(ROOT), raw_files=len(inventory), raw_bytes=raw_bytes,
        small_input_members=small_count, small_input_bytes=small_bytes,
        prior_stable_stat_members=prior_stat_count,
        publication_members=len(publication_members), technical_stages=len(ready["checks"]),
        native_files=len(tensor_paths), native_by_arm=dict(by_arm), commits=commits,
        adjacent_links=links, history_appends=appends, incomplete=incomplete,
        incomplete_reasons=dict(incomplete_reasons), errors=errors,
        elapsed_through_full_sha_seconds=full_sha_seconds,
        elapsed_total_seconds=time.monotonic() - started,
        new_gpu=0, original_files_written=0,
    ), ensure_ascii=False, indent=2))
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
