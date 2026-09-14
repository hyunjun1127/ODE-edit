"""Fail-closed scientific admission and typed calibration readiness.

This is NOT an executable BG model runner. Mathematical core and preparation
exist; actual model adapter/persistent controller have not passed GPU validation.
No branch here submits a job or repairs missing native checkpoints by editing.
"""
import math
from .control import verify_dispatch


class CalibrationMissing(RuntimeError):
    pass


def calibrated_budget(dispatch_path, calibration):
    verify_dispatch(dispatch_path)
    expected = set(range(1,11))
    points = calibration.get('endpoints',[])
    if len(points) != 10 or {p.get('batch') for p in points} != expected:
        raise CalibrationMissing('CALIBRATION_MISSING: exactly B1..B10 required, no partial maximum')
    if calibration.get('status') != 'CALIBRATION10_FORWARD_VERIFIED':
        raise CalibrationMissing('CALIBRATION_MISSING: saved-state forward receipt not complete')
    assert calibration['initial_model'] == 'pre-edit W0'
    assert calibration['policy'] == 'N4_L4_L2_1' and calibration['editing_reruns'] == 0
    assert calibration['reference_role'] == 'S64' and calibration['documents'] == 64
    assert calibration['teacher_dtype'] == 'float32'
    assert calibration['sample_prefix'] == [0,1000]
    reference=calibration['reference_identity_sha256'];teacher=calibration['teacher_sha256']
    assert len(reference)==64 and len(teacher)==64
    for p in points:
        assert p['reference_identity_sha256']==reference and p['teacher_sha256']==teacher
        assert len(p['actual_weight_sha256'])==64 and p['state_restored_and_forward_only']
        assert p['ordinal_end']==100*p['batch'] and math.isfinite(p['D64'])
    b_num=calibration['numerical_floor'];assert math.isfinite(b_num) and b_num>0
    b=max(b_num,.9*max(p['D64'] for p in points))
    assert calibration['fixed_budget']==b
    return b


def scientific_admission(dispatch_path, readiness):
    """Necessary guards, not a model-level correctness PASS."""
    d=verify_dispatch(dispatch_path)
    assert readiness['policies']==d['new_scientific_policies'] and readiness['chains']==1
    assert readiness['initial_model']=='pre-edit W0' and readiness['ordinals']==[0,1000]
    b=calibrated_budget(dispatch_path,readiness['calibration'])
    for key in ['reference768_sealed','teacher192_sealed','model_adapter_validated',
                'native_relation_gpu_checked','persistent_runner_validated','resource_admitted']:
        assert readiness.get(key) is True, key
    return dict(necessary_preconditions=True,calibration_budget=b,G0_PASS=False,
                note='Actual firstB100 finite/denominator/finalization/resume evidence still required')
