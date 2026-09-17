"""Create-once local evidence. No model, teacher, or raw payload in Git."""
from pathlib import Path
from project.run_scripts.local_z_adaptive_allocation.common import encoded, digest, sha, identity, save, tensor_save, Timer

TASK = 'ODEEDIT-S06-L4-PRESERVING-REPAIR-TWOARM-SH4-V1'
ROOT = Path('/data/janghj/ODE-edit/local/l4-preserving-repair/20260917-v1')
ARMS = ('R-GD', 'R-QP')
COLD = Path('/data/janghj/ODE-edit/local/local-z-adaptive-allocation/20260916-v1')

def verify(lock):
    assert lock['instruction_id'] == TASK and lock['arms'] == list(ARMS)
    assert lock['seed'] == 20260916 and lock['gpu_cap'] == 2
    assert lock['batches'] == 10 and lock['batch_size'] == 100
    assert lock['tf32_matmul'] is False and lock['tf32_cudnn'] is False
    for member in lock['members']:
        p = Path(member['path']); st = p.stat()
        assert st.st_size == member['bytes'], ('INPUT_SIZE', str(p))
        if member.get('verification') == 'PRIOR_FULL_SHA_STABLE_STAT':
            assert [st.st_dev, st.st_ino, st.st_mtime_ns] == member['stat'], ('ASSET_DRIFT', str(p))
        else:
            assert sha(p) == member['sha256'], ('INPUT_SHA', str(p))
