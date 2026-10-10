"""Keep independent, atomic W500/W1000/W1500/W2000 checkpoint directories.

Hard links retain the immutable payload after the rolling checkpoint unlinks it.
Each milestone uses the ordinary checkpoint.load format, including history/RNG.
"""
import fcntl
import json
import os
from pathlib import Path
import shutil
import tempfile

from . import checkpoint

MILESTONES = (5, 10, 15, 20)


def retain(latest, archive):
    latest, archive = Path(latest), Path(archive)
    ref = json.loads((latest / "latest.json").read_text())
    if ref["batch"] not in MILESTONES:
        return None
    if Path(ref["file"]).name != ref["file"]:
        raise ValueError("INVALID_CHECKPOINT_MEMBER")
    source = latest / ref["file"]
    archive.mkdir(parents=True, exist_ok=True)
    target = archive / f"edit-{ref['batch'] * 100:04d}"
    if target.exists():
        if (json.loads((target / "latest.json").read_text()) != ref
                or not os.path.samefile(source, target / ref["file"])):
            raise ValueError("MILESTONE_ALREADY_BOUND_TO_DIFFERENT_STATE")
        return str(target)
    temporary = Path(tempfile.mkdtemp(prefix=".milestone-", dir=archive))
    try:
        os.link(source, temporary / ref["file"])
        with (temporary / "latest.json").open("x") as stream:
            json.dump(ref, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        checkpoint._sync_directory(temporary)
        os.rename(temporary, target)
        checkpoint._sync_directory(archive)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return str(target)


def save(output, *, lock_path, reserve_bytes=0, **kwargs):
    output, lock_path = Path(output), Path(lock_path)
    output.mkdir(parents=True, exist_ok=True)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.is_symlink():
        raise ValueError("CHECKPOINT_LOCK_SYMLINK")
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        # A crash after committing latest but before archiving is recoverable.
        latest = output / "checkpoint"
        if (latest / "latest.json").exists():
            retain(latest, output / "milestones")
        tensors = list(kwargs["weights"].values()) + list(kwargs["cache_c"].values())
        estimated = sum(t.numel() * t.element_size() for t in tensors) + 32 * 2**20
        disk = os.statvfs(output)
        if disk.f_bavail * disk.f_frsize < reserve_bytes + estimated:
            raise ValueError("CHECKPOINT_STORAGE_ADMISSION")
        ref = checkpoint.save(latest, interval=5, **kwargs)
        retained = retain(latest, output / "milestones")
        return dict(latest=ref, milestone=retained)
