"""Explicit reserve-only admission waiver; actual file operations still raise."""
import json
from pathlib import Path
from .common import member

STATUS='USER_WAIVED_RESERVE_PENDING_USER_SPACE_CLEANUP'
NONCE='ODEEDIT-GH-SH4-ENFC-STORAGE-WAIVER-SUBMIT-20260918-R1'


def verify(value):
    required=dict(nonce=NONCE,reserve_bytes_original=72*(1<<30),reserve_is_submission_block=False,
        storage_status=STATUS,space_cleanup_owner='USER',SH4_delete_move_authorized=False,
        actual_IO_error_detection=True,atomic_endpoint_integrity=True,M_final_endpoints_retained=80,
        M_episodes=10,GPU_cap=2,T='SKIPPED_USER_DIRECTED',full_numerical_validation='NOT_ESTABLISHED',
        M_waits_for_T=False,S_R_L=False)
    if any(value.get(k)!=v for k,v in required.items()):raise ValueError('STORAGE_WAIVER_AUTHORITY_MISMATCH')


def admission(free,reserve,waiver=None):
    if waiver is None:
        if free<reserve:raise ValueError('M_STORAGE_RESERVE_UNAVAILABLE')
        return dict(status='RESERVE_ESTIMATE_MET_NOT_EXCLUSIVE',free_bytes=free,reserve_bytes=reserve)
    if member(waiver['path'])!=waiver:raise ValueError('STORAGE_WAIVER_BYTES_CHANGED')
    value=json.loads(Path(waiver['path']).read_text());verify(value)
    if reserve!=value['reserve_bytes_original']:raise ValueError('ORIGINAL_RESERVE_MUST_BE_PRESERVED')
    return dict(status=STATUS,free_bytes=free,reserve_bytes_original=reserve,waiver=waiver,
        reserve_shortfall_bytes=max(0,reserve-free),exclusive_or_reserved=False,
        user_cleanup='PLANNED_NOT_VERIFIED',SH4_deletion_or_move=False,actual_IO_errors_remain_fatal=True)


def bootstrap(lock):
    if lock.get('storage_override') is not None:
        # Identity only; no bootstrap disk reserve gate or cleanup watcher.
        admission(lock['disk']['available'],lock['disk']['reserve_bytes'],lock['storage_override'])
