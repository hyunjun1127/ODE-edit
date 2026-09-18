"""Fail-closed B1 scope and pre-outcome numerical contract."""
MAX_BATCHES = 1
SEQUENTIAL_AUTHORIZED = False
ARMS = ('N4', 'BPCW512')
NUMERIC = dict(version='BPCW512-B1-integrated-v1', gradient_relative=1e-4,
    FD_scales=12, FD_initial_native_relative=.01, FD_relative=.01, FD_signal_noise=10,
    FD_adjacent=2, FD_direction_count=2, actual_reference_checks=4,
    FD_objective='fixed base-token minus fixed competitor logit, not max switching',
    projector=1e-10, ideal_DK=1e-10, actual_DK=1e-5, actual_leak=1e-5,
    protected_logit_max=1e-3, protected_logit_RMS=1e-4, protected_NLL=1e-4,
    noop_NLL=1e-5, noop_logit=1e-4, ID_changes=0,
    gap_reserve_min=1e-4, gap_repeat_multiplier=10, gap_reserve_max=1e-3,
    pair_rows=(512,1024), pair_gradient_cap=1536, nonlinear_rounds=2,
    candidate_reference_scans=2, final_current_guards=1,
    GPU_Gram_tile=8, host_threads=8)

def require_scope(lock, batch=1):
    if (lock.get('max_batches') != 1 or lock.get('sequential_authorized') is not False or
        lock.get('arms') != list(ARMS) or batch != 1 or lock.get('batch_size') != 100):
        raise ValueError('USER_SEQUENTIAL_APPROVAL_REQUIRED_B1_ONLY')
    if lock.get('numeric') != {**NUMERIC,'pair_rows':list(NUMERIC['pair_rows'])}:
        raise ValueError('NUMERIC_CONTRACT_DRIFT')
    if lock.get('native_policy')!='FRESH_SAME_HOST_SHARED_ONCE' or lock.get('native_fit_reuse'):
        raise ValueError('FRESH_MATCHED_NATIVE_REQUIRED')
    if lock.get('task_gpu_cap')!=1 or lock.get('project_gpu_cap')!=2:
        raise ValueError('GPU_CAP_CONTRACT')
