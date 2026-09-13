"""E0/E1 identities, create-once publication and nonduplicated execution plan."""
from __future__ import annotations
import dataclasses
import hashlib
import json
import os
import stat
from pathlib import Path

INSTRUCTION = 'ODEEDIT-S06-BASELINE-MECHANISM-FIRST-E01-SH1-V1'
DESIGN_SHA = 'e4dc0b1fc0666775deb6e43cbdf18269c8e8e6d41d70c192e2bd4a2661a6ccbe'
DATASET_SHA = '3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1'
ORDERED_ROOT = '5b01356928ad2e2e46effad5a2c5621c87746b610992b7b92bf3ace16b6f5729'
MODEL_REVISION = '8afb486c1db24fe5011ec46dfbe5b5dccdb575c2'
LAYERS = (4, 5, 6, 7, 8)
ENTRIES = (0, 1000, 5000, 9000)


class ContractBoundary(RuntimeError):
    def __init__(self, code, **receipt):
        self.receipt = dict(code=code, **receipt)
        super().__init__(code)


def canonical(value):
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def member(path, *, expected=None):
    p = Path(path).absolute()
    s = p.lstat()
    if not stat.S_ISREG(s.st_mode):
        raise ContractBoundary('NOT_REGULAR_INPUT', path=str(p))
    sha = file_sha(p)
    if expected is not None and expected != sha:
        raise ContractBoundary('INPUT_SHA_MISMATCH', path=str(p), expected=expected, actual=sha)
    return dict(path=str(p), bytes=s.st_size, sha256=sha,
                mode=oct(stat.S_IMODE(s.st_mode)))


def save(path, value):
    """Publish once. Never overwrite an attempt or follow a symlink namespace."""
    p = Path(path).absolute()
    for component in reversed([p.parent, *p.parent.parents]):
        if component.is_symlink():
            raise ContractBoundary('SYMLINK_OUTPUT_PARENT', path=str(component))
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical(value) + b'\n'
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return member(p)


@dataclasses.dataclass(frozen=True)
class Cell:
    layer: int
    entry_n: int

    def __post_init__(self):
        if self.layer not in LAYERS or self.entry_n not in ENTRIES:
            raise ContractBoundary('OUTSIDE_E01_CELL_SCOPE')

    @property
    def identity(self):
        return f'AlphaEdit-L{self.layer}-n{self.entry_n:05d}'

    @property
    def next_batch(self):
        return self.entry_n // 100 + 1

    @property
    def comparison_batch(self):
        return 1 if self.entry_n == 0 else self.entry_n // 100 + 10

    def plan(self):
        return dict(cell=self.identity, layer=self.layer, entry_n=self.entry_n,
                    first_batch=self.next_batch, comparison_batch=self.comparison_batch,
                    native_batches=list(range(self.next_batch, self.comparison_batch + 1)),
                    E1_batch=self.next_batch, E1_reuses_E0_first_batch=True,
                    native_alpha=1., new_full_10k_chain=False,
                    status='PLANNED_INPUTS_NOT_YET_VERIFIED')


def execution_plan():
    cells = [Cell(layer, n).plan() for layer in LAYERS for n in ENTRIES]
    return dict(instruction_id=INSTRUCTION, scope=['evidence', 'E0', 'E1-A', 'E1-B'],
                cells=cells, E1_cells=20, E1_cell_request_executions=2000,
                E0_including_E1_max_native_batches=sum(len(c['native_batches']) for c in cells),
                first_batch_double_count=0, GPU_hour_cap=None,
                scientific_promotion=False, after_initial_valid='MONITORING_PAUSED_AWAITING_USER')
