"""Prospectively frozen authority, stage and numerical boundaries."""
from pathlib import Path

ROOT = Path('/data/janghj/ODE-edit/local/single-layer-mechanism-first/20260919-v1')
INSTRUCTION = 'ODEEDIT-S06-SL-MECHANISM-FIRST-ZHOOK-SH4-V1'
AUTHORITY = 'be71bfba2454aa4645e28d960d495dfc25a65993'
REVISION = '8afb486c1db24fe5011ec46dfbe5b5dccdb575c2'
B1_ARMS = ('N4', 'EN_KL_Q', 'DEC_LINE', 'DEC_MODES_CUM')
CHAIN_ARMS = ('N4', 'DEC_MODES_STEP', 'DEC_MODES_CUM')
SCALES = (1., .5, .25, .125)
SEED = 20260916
BASIS_SEED = 20260919
NUMERIC = dict(current_nll=1e-4, projector_fp64=1e-10, actual_DK=1e-5,
    actual_leak=1e-5, logit_max=1e-3, logit_rms=1e-4,
    noop_nll=1e-5, noop_logits=1e-4, gradient_relative=1e-4,
    FD_relative=.01, FD_signal_to_noise=10.,
    reference_deficit_allowance=1e-4, risk_abs=1e-10, risk_relative=1e-6,
    repeat_spread_fraction=.1, zero_projected_gradient_relative=1e-12,
    phase2_absolute=1e-12, phase2_relative=1e-8, linear_row_scaled=1e-8)


def require_stage(stage, arm, batch, gate=None):
    if stage == 'T0' and arm == 'TECHNICAL' and batch == 0:
        return
    if stage == 'B1' and arm in B1_ARMS and batch == 1:
        return
    required = {'S3': ('B1_TO_S3', range(2, 4)), 'S10': ('S3_TO_S10', range(4, 11))}
    if stage not in required or arm not in CHAIN_ARMS:
        raise ValueError('UNAUTHORIZED_STAGE_ARM')
    name, batches = required[stage]
    if batch not in batches or not gate or gate.get('name') != name or gate.get('pass') is not True:
        raise ValueError('PRIOR_STAGE_GATE_REQUIRED')


def risk_tolerance(psi_native, phi_reference_native):
    return NUMERIC['risk_abs'] + NUMERIC['risk_relative'] * max(psi_native, phi_reference_native)


def check_lock(lock):
    from project.run_scripts.single_layer_edit_preserving_correction.common import digest
    if lock.get('lock_identity') != digest({k:v for k,v in lock.items() if k != 'lock_identity'}):
        raise ValueError('LOCK_IDENTITY')
    if lock['instruction'] != INSTRUCTION or lock['model_revision'] != REVISION:
        raise ValueError('AUTHORITY_OR_MODEL')
    if lock['numerical'] != NUMERIC or lock['project_gpu_cap'] != 2 or lock['task_gpu_cap'] != 2:
        raise ValueError('NUMERICAL_RESOURCE_DRIFT')
    if lock['resources']['GPU'] != 1 or lock['resources']['CPU'] != 8 or lock['resources']['mem_MiB'] > 60416:
        raise ValueError('RESOURCE_CEILING')
    if lock.get('prior_skip_or_storage_waiver_inherited') is not False:
        raise ValueError('HISTORICAL_WAIVER_FORBIDDEN')
    if not Path(lock['output']).resolve().is_relative_to(ROOT):
        raise ValueError('TASK_OUTPUT_BOUNDARY')
