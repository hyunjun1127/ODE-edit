"""Portable content seals complement, not replace, process-local pointer guards."""
from project.run_scripts.ordered_response_barrier_ode.fp32_overlay import tensor_sha256
from project.run_scripts.ordered_response_barrier_ode.preflight import canonical_hash


def content_identity(values):
    return canonical_hash([dict(key=repr(key),shape=list(value.shape),dtype=str(value.dtype),
        content_sha256=tensor_sha256(value)) for key,value in sorted(values.items(),key=lambda item:repr(item[0]))])


def verify_warm_state(family,warm,static_cov_identity):
    if family.family=='AlphaEdit':
        expected=content_identity({'cache_c':warm['alpha_cache']})
        actual=content_identity({'cache_c':family.module.cache_c})
        reference='SHA_SEALED_PRIMARY_WARM_CACHE_TENSOR'
    else:
        expected=static_cov_identity
        actual=content_identity(family.module.COV_CACHE)
        reference='PINNED_STATIC_COV_SOURCE_AND_UNCHANGED_RELOAD_ENTRY'
    if actual!=expected:raise RuntimeError('PORTABLE_WARM_CONTENT_BOUNDARY')
    return dict(status='PASS',reference=reference,expected_content_identity=expected,actual_content_identity=actual,
        original_process_bound_identity=warm['commit']['committed_method_state_sha256'],
        new_process_bound_identity=family.method_state_identity(),
        cross_process_pointer_equality_required=False,within_process_pointer_version_guards_preserved=True,
        equation_change_count=0,tolerance_change_count=0)
