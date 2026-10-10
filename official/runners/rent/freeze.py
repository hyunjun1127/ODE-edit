"""Freeze a commit's official/ tree for rent Jobs (login node, Python 3.8, git).

Queued and running Jobs import this frozen copy, so later updates of the clone
never change their source. ``on_main`` records whether the commit is on
origin/main; scientific chains require it, technical qualification does not.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

REPO = Path(__file__).resolve().parents[3]
ROOT = Path.home() / "janghj" / "checkpoints" / "odeedit-official"


def git(*args, binary=False):
    return subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True,
                          text=not binary).stdout


def file_sha(path):
    sha = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def tree(source_root):
    """Same definition as server3's official_tree_sha256, plus the member map."""
    official = Path(source_root) / "official"
    files = [p for p in official.rglob("*") if p.is_file() and not p.is_symlink()
             and "__pycache__" not in p.parts and p.suffix != ".pyc"]
    if not files or any(p.is_symlink() for p in official.rglob("*")):
        raise ValueError("OFFICIAL_TREE_EMPTY_OR_SYMLINK")
    members, hasher = {}, hashlib.sha256()
    for path in sorted(files, key=lambda p: p.relative_to(source_root).as_posix()):
        relative = path.relative_to(source_root).as_posix()
        members[relative] = file_sha(path)
        hasher.update(relative.encode() + b"\0" + members[relative].encode() + b"\n")
    return hasher.hexdigest(), members


def freeze(commit="HEAD", root=ROOT):
    commit = git("rev-parse", commit + "^{commit}").strip()
    git("fetch", "-q", "origin", "main")
    on_main = subprocess.run(["git", "merge-base", "--is-ancestor", commit, "origin/main"],
                             cwd=REPO).returncode == 0
    out = Path(root) / commit[:12]
    receipt = out / "source.json"
    if receipt.exists():
        value = json.loads(receipt.read_text())
        digest, members = tree(out / "source")
        if value["code_commit"] != commit or value["official_tree_sha256"] != digest:
            raise ValueError("FROZEN_SOURCE_CHANGED")
        if on_main and not value["on_main"]:
            value["on_main"] = True
            receipt.write_text(json.dumps(value, indent=2) + "\n")
        return value
    archive = git("archive", "--format=tar", commit, "official", binary=True)
    (out / "source").mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(out / "source")
    digest, members = tree(out / "source")
    value = dict(schema="official-rent-source-v1", code_commit=commit, official_tree_sha256=digest,
                 on_main=on_main, members=members, source_dir=str(out / "source"))
    receipt.write_text(json.dumps(value, indent=2) + "\n")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", default="HEAD")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    value = freeze(args.commit, args.root)
    print(json.dumps({k: value[k] for k in ("code_commit", "official_tree_sha256", "on_main")}))


if __name__ == "__main__":
    main()
