"""One CPU-only rent W&B run of exactly three synthetic points, then remote readback."""
import argparse
import json
from pathlib import Path
import re
import sys

from official.tracking import LoggingBlocked, init
from official.tracking.schema import load_env

REPO = Path(__file__).resolve().parents[3]


def git_head(repo=REPO):
    """Resolve HEAD without git (the Job image has none)."""
    git = repo / ".git"
    head = (git / "HEAD").read_text().strip()
    if not head.startswith("ref: "):
        return head
    ref = head[5:]
    if (git / ref).is_file():
        return (git / ref).read_text().strip()
    for line in (git / "packed-refs").read_text().splitlines():
        if line.endswith(" " + ref):
            return line.split()[0]
    raise ValueError("SOURCE_SHA_UNRESOLVED")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-sha", default=None)
    args = parser.parse_args()
    args.source_sha = args.source_sha or git_head()
    settings = load_env(args.env_file)
    args.out.mkdir(parents=True, exist_ok=False, mode=0o700)
    result = dict(server="rent", entity=settings["WANDB_ENTITY"], project=settings["WANDB_PROJECT"],
                  base_url=settings["WANDB_BASE_URL"], sdk_python=settings["ODEEDIT_WANDB_PYTHON"],
                  GPU=0, maximum_points=3, offline_pass=False, credentials_recorded=False)
    tracker = None
    try:
        tracker = init(env_file=args.env_file, spool=args.out / "spool", smoke=True,
                       config=dict(server="rent", task_id="wandb-realtime-setup", arm="cpu-smoke",
                                   attempt=args.out.name, source_sha=args.source_sha))
        result["job_identity"] = tracker.job_identity
        for step in range(3):
            if not tracker.log({"setup_ok": 1, "step": step}, step=step):
                raise RuntimeError("POINT_REJECTED")
        outcome = tracker.finish(timeout=50)
        result.update(outcome)
        verified = outcome.get("status") == "READY_ONLINE_VERIFIED" and not outcome["dropped_points"]
        result["status"] = "READY_ONLINE_VERIFIED" if verified else "LOGGING_BLOCKED_SMOKE"
    except LoggingBlocked as error:
        # LoggingBlocked carries only fixed internal status codes, never an SDK message.
        result["status"] = str(error)
    except ValueError as error:
        # Schema failures are fixed codes; anything else could quote private text.
        code = str(error)
        result["status"] = "LOGGING_BLOCKED_SMOKE"
        result["error_code"] = code if re.fullmatch(r"[A-Z][A-Z0-9_]*(:[a-z_]+)?", code) else "ValueError"
    except Exception as error:
        result["status"] = "LOGGING_BLOCKED_SMOKE"
        result["error_code"] = type(error).__name__
    finally:
        if tracker is not None and not tracker.closed:
            tracker.finish(exit_code=1)
    with (args.out / "result.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result))
    return 0 if result["status"] == "READY_ONLINE_VERIFIED" else 2


if __name__ == "__main__":
    sys.exit(main())
