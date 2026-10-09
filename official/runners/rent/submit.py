"""Submit a rent Kubernetes Job with its KST DDHHMM job number (login node, Python 3.8+).

The template writes ``{{RENT_JOB_ID}}`` in ``metadata.name``, in the
``odeedit-job-id`` label, and as the value of env ``ODEEDIT_RENT_JOB_ID``.
The number is fixed here, before the Job exists, like a Slurm job number.
"""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys

NAMESPACE = "nlp-lab"
STUDENT = "janghj"
GPU_CAP = 2
PLACEHOLDER = "{{RENT_JOB_ID}}"
KST = datetime.timezone(datetime.timedelta(hours=9))


def job_number(now=None):
    return (now or datetime.datetime.now(KST)).astimezone(KST).strftime("%d%H%M")


def render(text, number):
    if PLACEHOLDER not in text:
        raise ValueError("TEMPLATE_PLACEHOLDER_MISSING")
    return text.replace(PLACEHOLDER, number)


def kubectl(*args, stdin=None):
    return subprocess.run(["kubectl", "-n", NAMESPACE, *args], input=stdin, check=True,
                          capture_output=True, text=True).stdout


def gpus(job):
    total = 0
    for container in job["spec"]["template"]["spec"]["containers"]:
        resources = container.get("resources", {})
        total += int(resources.get("limits", {}).get("nvidia.com/gpu", 0))
    return total


def finished(job):
    return any(c.get("type") in ("Complete", "Failed") and c.get("status") == "True"
               for c in job.get("status", {}).get("conditions", []))


def check(job, number, existing):
    meta = job["metadata"]
    env = {e["name"]: e.get("value") for c in job["spec"]["template"]["spec"]["containers"]
           for e in c.get("env", [])}
    if not (meta["name"].startswith(STUDENT + "-") and number in meta["name"]
            and meta.get("labels", {}).get("student") == STUDENT
            and meta.get("labels", {}).get("odeedit-job-id") == number
            and env.get("ODEEDIT_RENT_JOB_ID") == number):
        raise ValueError("RENT_JOB_BINDING")
    if any(j["metadata"].get("labels", {}).get("odeedit-job-id") == number for j in existing):
        raise ValueError("RENT_JOB_ID_IN_USE: retry in the next minute")
    active = sum(gpus(j) for j in existing if not finished(j))
    if active + gpus(job) > GPU_CAP:
        raise ValueError("JANGHJ_GPU_CAP: active %d + requested %d > %d" % (active, gpus(job), GPU_CAP))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("template", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    number = job_number()
    manifest = render(args.template.read_text(), number)
    job = json.loads(kubectl("create", "--dry-run=client", "-o", "json", "-f", "-", stdin=manifest))
    existing = json.loads(kubectl("get", "jobs", "-l", "student=" + STUDENT, "-o", "json"))["items"]
    check(job, number, existing)
    kubectl("create", "--dry-run=server", "-f", "-", stdin=manifest)
    rendered = args.template.with_name("%s.yaml" % job["metadata"]["name"])
    if not args.dry_run:
        rendered.write_text(manifest)
        kubectl("create", "-f", str(rendered))
    print(json.dumps(dict(job=job["metadata"]["name"], rent_job_id=number, gpus=gpus(job),
                          manifest=str(rendered), submitted=not args.dry_run)))


if __name__ == "__main__":
    sys.exit(main())
