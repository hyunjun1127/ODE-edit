"""SH2 host/selected-weight restore adapter for the single shared CF evaluator.

No editing, W0, generation implementation, metric formula or logger fork.
Original checkpoint identity is retained, not relabelled to this evaluation.
"""
import argparse
from contextlib import contextmanager
from pathlib import Path
from official.experiments.prepare import digest
from official.runners.server1 import flucon_eval as shared
from official.runners.server1.common import read, verify
from official.tracking.schema import config as validate_tracking
from .zsre_reeval_restore import load_and_restore

AUTHORITY = 'USER-GH-S1-S2-BASELINE-REFRESH-S2-FLUCON-20261010-R1'
TASK = 'baseline-refresh-s2-flucon-20261010'
LOCAL = Path('/mnt/raid5/janghj/ODE-edit/local') / TASK
_tracking = shared.tracking_values

def tracking_values(config, source):
    value = _tracking(config, source)
    value.update(server='server2', task_id=TASK)
    return validate_tracking(value)

def restore(model, original):
    if original.get('checkpoint_schema') == 'historical-W-method-state':
        from .flucon_historical import restore_historical
        return restore_historical(model,original)
    pointer = read(verify(original['pointer']))
    assert pointer['batch'] == 20 and pointer['final_W20']
    assert pointer['identity_sha256'] == digest(original['identity'])
    assert pointer['sha256'] == original['checkpoint']['sha256']
    assert pointer['file'] == Path(original['checkpoint']['path']).name
    return load_and_restore(model, original, original['hparams'])

@contextmanager
def host_bindings():
    # Process-local dependency injection only; shared source and frozen jobs are
    # never modified. Reset even on exception (also makes CPU tests isolated).
    previous = shared.tracking_values, shared.restore
    shared.tracking_values, shared.restore = tracking_values, restore
    try:
        yield
    finally:
        shared.tracking_values, shared.restore = previous

def run(config, lock):
    c = read(config)
    assert c['registration_authority'] == AUTHORITY
    assert c['asset_schema'] == 'history'  # shared generic FP32 native loader
    with host_bindings():
        shared.run(config, lock)

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','collect'])
    p.add_argument('--config');p.add_argument('--lock',required=True)
    p.add_argument('--preparation');p.add_argument('--output');a=p.parse_args()
    if a.mode=='run':run(a.config,a.lock)
    else:shared.collect(a.preparation,a.output,a.lock)
