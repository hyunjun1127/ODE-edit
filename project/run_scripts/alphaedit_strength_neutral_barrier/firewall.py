"""Fail-close verification that the experiment hooks an unmodified EasyEdit."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path
from typing import Any


class OfficialSourceBoundary(RuntimeError):
    """The external EasyEdit checkout is not the sealed stock source."""


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True
    ).strip()


def verify_stock_easyedit(root: Path, lock_path: Path) -> dict[str, Any]:
    root = root.absolute()
    if root.is_symlink() or not root.is_dir():
        raise OfficialSourceBoundary(f"EasyEdit root is not a regular directory: {root}")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    head = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    if head != lock["head"] or tree != lock["tree"]:
        raise OfficialSourceBoundary(
            f"EasyEdit identity drift: head={head}, tree={tree}"
        )
    dirty = _git(root, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise OfficialSourceBoundary("EasyEdit tracked source is dirty")

    observed: dict[str, dict[str, Any]] = {}
    for relative, expected_sha in lock["members"].items():
        path = root / relative
        info = os.lstat(path)
        if not stat.S_ISREG(info.st_mode):
            raise OfficialSourceBoundary(f"EasyEdit member is not regular: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected_sha:
            raise OfficialSourceBoundary(f"EasyEdit member drift: {relative}")
        observed[relative] = {
            "sha256": digest,
            "bytes": info.st_size,
            "mode": f"{stat.S_IMODE(info.st_mode):04o}",
        }
    return {
        "head": head,
        "tree": tree,
        "tracked_clean": True,
        "members": observed,
        "implementation_boundary": lock["implementation_boundary"],
    }


__all__ = ["OfficialSourceBoundary", "verify_stock_easyedit"]
