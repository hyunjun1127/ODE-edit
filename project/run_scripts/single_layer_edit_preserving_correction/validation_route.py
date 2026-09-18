"""Admission status only; never changes method acceptance or numerical constants."""
import json
from pathlib import Path
from .common import member

SKIP_NONCE = 'ODEEDIT-GH-SH4-ENFC-SKIP-T-ALL-M-20260918-R1'
SKIPPED = 'SKIPPED_USER_DIRECTED'


def check_waiver(value):
    required = dict(nonce=SKIP_NONCE, gpu_cap=2, technical_phase=SKIPPED,
        full_numerical_validation='NOT_ESTABLISHED', new_T_submit=False,
        M_requires_T_READY=False, new_M_old_T_failcancel=False,
        M_independent_episodes=10, M_final_L4_endpoints=80,
        M_native_fit_B1='REUSE', M_new_native_fits_max=9, reuse_first=True,
        method_acceptance_guards_unchanged=True, S_R_L_submit=False)
    if any(value.get(k) != v for k, v in required.items()):
        raise ValueError('SKIP_T_AUTHORITY_MISMATCH')


def validation_binding(lock, episode):
    if lock['stage'] != 'M' or episode not in range(10):
        raise ValueError('M_ONLY_ALLOWLIST')
    ref = lock['technical_evidence']
    if member(ref['path']) != ref:
        raise ValueError('VALIDATION_EVIDENCE_IDENTITY')
    evidence = json.loads(Path(ref['path']).read_text())
    skipped = evidence['status'] == SKIPPED
    if skipped:
        waiver = lock['skip_T_override']
        if member(waiver['path']) != waiver:
            raise ValueError('SKIP_T_WAIVER_IDENTITY')
        check_waiver(json.loads(Path(waiver['path']).read_text()))
        if (lock['allowed_stages'] != ['M'] or lock.get('T_job') is not None
                or lock.get('T_failure_path') is not None or lock.get('parallel_override') is not None
                or lock.get('old_T_failcancel') is not False
                or lock.get('full_numerical_validation') != 'NOT_ESTABLISHED'):
            raise ValueError('SKIP_T_ROUTE_MUST_BE_INDEPENDENT')
        return evidence, False, True
    if lock['allowed_stages'] != ['T', 'M']:
        raise ValueError('LEGACY_M_ALLOWLIST')
    provisional = evidence['status'] == 'PROVISIONAL_T_UNRESOLVED'
    if provisional and (not lock.get('parallel_override') or lock.get('T_job') != 49928):
        raise ValueError('PROVISIONAL_AUTHORITY_REQUIRED')
    if not provisional and evidence['status'] != 'T_READY':
        raise ValueError('T_NOT_READY')
    if Path(lock['T_failure_path']).exists():
        raise ValueError('LINKED_T_ALREADY_FAILED')
    return evidence, provisional, False
