"""One CPU-only server3 setup smoke, not the shared experiment logger.

Run under the documented external 120-second timeout. No science imports.
Only synthetic step/setup_ok scalars and a logical server label are uploaded.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import json
import os
from pathlib import Path
import resource
import uuid

ENTITY = "wkdguswns2256"
PROJECT = "layer allocation"
ROOT = Path("/data/janghj/ODE-edit/local/wandb-setup/20261006-v1")
SETTINGS = dict(
    entity=ENTITY, project=PROJECT, mode="online", console="off",
    save_code=False, disable_code=True, disable_git=True,
    disable_job_creation=True, x_disable_meta=True, x_disable_stats=True,
    x_disable_machine_info=True, init_timeout=30, login_timeout=1,
    finish_timeout=30, finish_timeout_raises=True,
)


def points():
    return [{"step": i, "setup_ok": 1} for i in range(3)]


def validate_readback(rows):
    return len(rows) == 3 and [
        {k: row[k] for k in ("step", "setup_ok")} for row in rows
    ] == points()


def execute(sdk, destination):
    """Only this one-off setup payload is accepted; no arbitrary config/logs."""
    receipt = {
        "time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "sdk": sdk.__version__, "entity": ENTITY, "project": PROJECT,
        "server": "server3", "settings": SETTINGS, "gpu": 0,
        "credential_available": False, "auth_remote_verified": False,
        "online_attempted": False, "readback_points": 0,
        "status": "SETUP_READY_NEEDS_USER_LOGIN",
    }
    # This reads same-user settings/netrc/env. It neither prompts nor writes a key.
    if not sdk.login(prompt=False, verify=False):
        return receipt
    receipt["credential_available"] = True
    marker = ROOT / "online-attempt.json"
    run_id = "s3setup" + uuid.uuid4().hex[:16]
    # Do not create another online run when the same command is repeated.
    with marker.open("x") as f:
        json.dump({"run_id": run_id, "entity": ENTITY, "project": PROJECT}, f)
    receipt.update(online_attempted=True, run_id=run_id, status="LOGGING_BLOCKED")
    run = None
    try:
        run = sdk.init(
            entity=ENTITY, project=PROJECT, id=run_id, resume="never",
            name="server3-setup", group="wandb-realtime-setup",
            config={"server": "server3"}, dir=str(destination),
            settings=sdk.Settings(**SETTINGS),
        )
        if run.settings.mode != "online":
            raise RuntimeError("online mode required")
        receipt["auth_remote_verified"] = True
        receipt["run_url"] = run.url
        for row in points():
            run.log(row, step=row["step"], commit=True)
        run.finish()
        receipt["finish_completed"] = True
        # A single bounded readback, no recurring polling or retry loop.
        remote = sdk.Api(timeout=15).run(f"{ENTITY}/{PROJECT}/{run_id}")
        rows = list(remote.scan_history(keys=["step", "setup_ok"], page_size=3))
        receipt["readback_points"] = len(rows)
        receipt["remote_state"] = remote.state
        if not validate_readback(rows) or remote.state != "finished":
            raise RuntimeError("three-point finished readback not established")
        receipt["status"] = "READY_ONLINE_VERIFIED"
    except Exception as exc:
        # Exception strings/SDK logs can contain private connection information.
        receipt["error_class"] = type(exc).__name__
        if run is not None and not receipt.get("finish_completed"):
            try:
                run.finish(exit_code=1)
            except Exception as finish_exc:
                receipt["finish_error_class"] = type(finish_exc).__name__
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "smoke")
    args = parser.parse_args()
    destination = args.output.resolve()
    if not destination.is_relative_to(ROOT):
        raise ValueError("receipts must remain under this ignored setup root")
    os.umask(0o077)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3,) * 2)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["WANDB_ERROR_REPORTING"] = "false"
    os.environ["WANDB_MODE"] = "online"
    # Scoped to this new CPU smoke; old launchers/jobs remain unchanged.
    os.environ.pop("WANDB_DISABLED", None)
    destination.mkdir(parents=True, exist_ok=True)
    result = None
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        # Cover subprocess/native fd output too, not only Python streams.
        saved_stdout, saved_stderr = os.dup(1), os.dup(2)
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)
        try:
            import wandb
            result = execute(wandb, destination)
        except Exception as exc:
            result = {"status": "LOGGING_BLOCKED", "error_class": type(exc).__name__}
        finally:
            os.dup2(saved_stdout, 1)
            os.dup2(saved_stderr, 2)
            os.close(saved_stdout)
            os.close(saved_stderr)
    receipt_path = destination / ("receipt-" + uuid.uuid4().hex + ".json")
    with receipt_path.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({"status": result["status"], "receipt": str(receipt_path)}))
    return 0 if result["status"] == "READY_ONLINE_VERIFIED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
