"""Create-once failure binding and prospective same-method/noCP repair plan."""
import argparse
import json
from pathlib import Path
from .config import ROOT
from .technical_repair import LENIENT_GATE_ID
from project.run_scripts.single_layer_edit_preserving_correction.common import member,write


def build(destination):
    old=ROOT/'T0/hook-attempt-v1';parent=ROOT/'PROGRAM/dependency-attempt-v1/execution.lock.json'
    value=json.loads(parent.read_text());value.pop('execution',None);value.pop('lock_identity',None)
    original=old/'execution.lock.json';loaded=json.loads(original.read_text())
    failure=old/'output/failure.json'
    if json.loads(failure.read_text())['error']!="RuntimeError('T0_NATIVE_Z_HOOK_PARITY_FAILED')":raise ValueError('EXACT_FAILURE_REQUIRED')
    files={name:member(old/rel) for name,rel in dict(
        original_lock='execution.lock.json',failure='output/failure.json',original_runtime='output/runtime-load.json',
        panel='output/technical-hook/panel.json',native_trace='output/technical-hook/z-comparison-tensors.pt',
        native_write='output/technical-hook/actual-write-native.pt',
        failed_hook_summary='output/technical-hook/hook-summary.json').items()}
    value.pop('hook_dependency',None)
    value.update(hook_repair=dict(kind='T0_NATIVE_Z_HOOK_PARITY_REPAIR',
        original_execution=loaded['execution']['commit'],**files),
        status='FROZEN_PLAN_REPAIR_NOT_ACTUAL_PARITY_PASS',
        output=str(ROOT/'PROGRAM/hook-repair-r1/output'),
        user_repair_authority='2026-09-20 USER: 실험 끝난거 fail repair 바람',
        hook_gate_policy=dict(id=LENIENT_GATE_ID,
            user_quote='저런 부분은 너무 strict한 gate인데 어느정도 lenient하게 파악해도 돼.',
            scope='independent native/cache z trajectory relative-gradient discrepancy only',
            original_gradient_ceiling=1e-4,new_role='DIAGNOSTIC_WARN_NOT_SOLE_STOP',
            actual_write_NLL_ceiling=1e-4,strict_pair_ids='EXACT',
            loss_stop_finite_and_full_T0='REMAIN_REQUIRED',same_point_gradient_PASS=False),
        old_dependent_job='50983',old_dependent_job_mutation=False,
        hook_execution='INLINE_IN_SAME_PROCESS; NO_NEW_WEIGHT_CHECKPOINT',
        monitoring_boundary='REPAIR_SUBMISSION_HANDOFF; NO_AUTOMATIC_MONITOR_RESUME',
        reuse_scope='valid fixed-four unhooked native targets/traces/write only; failed hook is not PASS',
        new_technical_z_request_calls=8,new_unhooked_native_reference_calls=0,
        method_threshold_changes=False,technical_gate_role_changes=True,
        prior_failed_allocation_GPU_seconds=136,
        scientific_batches_completed_before_repair=0)
    return write(destination,value)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    print(json.dumps(build(p.parse_args().output)))
