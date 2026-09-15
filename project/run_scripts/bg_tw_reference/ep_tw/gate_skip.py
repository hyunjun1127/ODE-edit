"""Explicit user waiver routing; not a numerical validation or policy change."""
from pathlib import Path
from .control import identity, sha

TASK='ODEEDIT-S06-EP-TW1-DIAGNOSTIC-GATES-SKIP-RUN-SH4-V1'
MODE='SKIPPED_USER_DIRECTED'
SWEEP_TASK='ODEEDIT-S06-EP-TW1-ALPHA-CAP-SWEEP-SH4-V1'
SKIPPED=['saved_episode_repair_prerequisite','repair_pass_receipt','conditional_E_D_science_gate',
    'FD_grid_convergence_resolution_jitter','direct_weight_gradient_comparison',
    'independent_direction_probes','ULP_statistics','W0_self_KL_threshold',
    'functional_materialized_numerical_parity','synthetic_zero_nonzero_checks',
    'validate_episode','scientific_checks','native_map_redundant_RHS_comparison',
    'canonical_method_NLL_byte_equality_gate']

def skip_enabled(lock):
    if lock.get('validation_mode')!=MODE:return False
    assert lock['instruction_id'] in (TASK,SWEEP_TASK) and lock['numerical_validation']=='NOT_ESTABLISHED'
    assert 'repair_pass_path' not in lock and 'repair_dispatch' not in lock
    assert lock['validation_override']['instruction_id']==TASK
    if lock['instruction_id']==SWEEP_TASK:
        assert lock['validation_override']['inherited_by']==SWEEP_TASK
    return True

def diagnostic(lock,name,callback=None):
    """Do not execute callback at all under the explicit waiver."""
    if skip_enabled(lock):
        return dict(status=MODE,diagnostic=name,numerical_validation='NOT_ESTABLISHED',
                    calls=0,reason='EXPLICIT_USER_OVERRIDE_NOT_A_PASS')
    assert callback is not None
    return callback()

def parity_record(lock,rows,observation,expected_ids,check_strict=True):
    # Identity/order is executable integrity. Floating point byte comparison is
    # only a warning in skip mode, using observations already computed.
    assert [r['case_id'] for r in rows]==expected_ids,'CANONICAL_CASE_ID_ORDER'
    assert [r['case_id'] for r in observation['rows']]==expected_ids,'METHOD_CASE_ID_ORDER'
    nll_equal=[r['new_nll'] for r in rows]==[r['nll'] for r in observation['rows']]
    strict_equal=[r['case_id'] for r in rows if r['new_strict']]==observation['strict_ids']
    if not skip_enabled(lock):
        assert nll_equal,'CANONICAL_CURRENT_NLL_PARITY'
        if check_strict:assert strict_equal,'CANONICAL_CURRENT_STRICT_PARITY'
    return dict(status='WARNING' if not nll_equal or (check_strict and not strict_equal) else 'EXISTING_ROWS_EQUAL',
        nll_exact_equal=nll_equal,strict_ids_equal=strict_equal,additional_forwards=0,
        diagnostic_gate=MODE if skip_enabled(lock) else 'ORIGINAL',
        method_quality_screen_unchanged=True)

def verify_skip_members(lock):
    assert skip_enabled(lock)
    stats={m['path']:m for m in lock['member_stats']};fresh=reused=0
    for m in lock['members']:
        p=Path(m['path']);s=p.stat()
        assert dict(path=str(p),dev=s.st_dev,inode=s.st_ino,mtime_ns=s.st_mtime_ns,bytes=s.st_size)==stats[str(p)],('MEMBER_STAT_DRIFT',str(p))
        assert s.st_size==m['bytes']
        if p.is_relative_to(lock['source_root']) or s.st_size<8*(1<<20):
            assert sha(p)==m['sha256'],('MEMBER_SHA_DRIFT',str(p));fresh+=1
        else:reused+=1
    assert identity(lock['source_archive']['path'])==lock['source_archive']
    return dict(new_full_sha_members=fresh,prior_full_sha_stable_stat_reuse=reused,
                numerical_tests_executed=0)
