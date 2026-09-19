"""Pure execution-dedup exactness: never substitute method protection tolerances.

Small scalar/identity receipts only; tensors are compared using shape/dtype and
their exact header+bytes digests written by the execution route. Matching hashes
is not an independent tensor reconstruction or GPU continuation test.
"""
import math

REQUIRED = ('data_id', 'input_sha256', 'native_sha256', 'geometry_sha256',
            'current_binding_sha256', 'teacher_sha256', 'model_epoch',
            'loss', 'gradient_sha256', 'projected_gradient_sha256', 'chi', 'eta0',
            'reference_document_ids', 'reference_position_counts', 'trials',
            'selected_sha256', 'stop_reason', 'history_appends', 'full_sweep_rows')
TRIAL_FIELDS = ('trial', 'weight_sha256', 'loss', 'actual_p', 'armijo',
                'guard', 'invariant', 'decision')


def compare_exact(left, right):
    mismatches = []

    def compare(a, b, path):
        if isinstance(a, float) and not math.isfinite(a):
            mismatches.append(dict(path=path, reason='NONFINITE_LEFT'))
            return
        if isinstance(b, float) and not math.isfinite(b):
            mismatches.append(dict(path=path, reason='NONFINITE_RIGHT'))
            return
        if type(a) is not type(b):
            mismatches.append(dict(path=path, reason='TYPE_MISMATCH'))
        elif isinstance(a, dict):
            if a.keys() != b.keys():
                mismatches.append(dict(path=path, reason='KEY_SET_MISMATCH'))
            for key in sorted(a.keys() & b.keys()):
                compare(a[key], b[key], path+'.'+key)
        elif isinstance(a, (list, tuple)):
            if len(a) != len(b):
                mismatches.append(dict(path=path, reason='CARDINALITY_MISMATCH'))
            for i, (x, y) in enumerate(zip(a, b)):
                compare(x, y, path+f'[{i}]')
        elif isinstance(a, float):
            # hex retains signed zero; scalar arithmetic routes must match too.
            if a.hex() != b.hex():
                mismatches.append(dict(path=path, reason='EXACT_FLOAT_MISMATCH', left=a, right=b))
        elif a != b:
            mismatches.append(dict(path=path, reason='VALUE_MISMATCH'))

    for name, result in (('legacy', left), ('reuse', right)):
        for key in REQUIRED:
            if key not in result:
                mismatches.append(dict(path=name+'.'+key, reason='NOT_RECORDED'))
        for i, trial in enumerate(result.get('trials', [])):
            for key in TRIAL_FIELDS:
                if key not in trial:
                    mismatches.append(dict(path=f'{name}.trials[{i}].{key}', reason='NOT_RECORDED'))
        normal_no_direction=(result.get('stop_reason') in ('REPAIR_SPACE_EMPTY','RANK_UNRESOLVED') and
            result.get('gradient_sweeps')==0 and result.get('gradient_sha256') is None and
            result.get('selected_sha256')==result.get('native_sha256') and not result.get('trials') and
            not result.get('full_sweep_rows'))
        if result.get('gradient_sweeps') != 1 and not normal_no_direction:
            mismatches.append(dict(path=name+'.gradient_sweeps', reason='FULL_SWEEP_NOT_EXACTLY_ONE'))
        if result.get('method_gradient_shared') is not False:
            mismatches.append(dict(path=name+'.method_gradient_shared', reason='INDEPENDENCE_NOT_ESTABLISHED'))
        if result.get('history_appends') != 1:
            mismatches.append(dict(path=name+'.history_appends', reason='HISTORY_NOT_EXACTLY_ONCE'))
        docs = result.get('reference_document_ids', [])
        sizes = result.get('reference_position_counts', [])
        if len(docs) != 512 or len(set(docs)) != 512 or len(sizes) != 512 or any(
                type(t) is not int or not 1 <= t <= 256 for t in sizes):
            mismatches.append(dict(path=name+'.coverage', reason='FULL_R512_ACTUAL_POSITION_COVERAGE_INVALID'))
        if len(result.get('trials', [])) > 8:
            mismatches.append(dict(path=name+'.trials', reason='TRIAL_CAP_EXCEEDED'))
    for key in REQUIRED:
        if key in left and key in right:
            compare(left[key], right[key], key)
    return dict(status='EXACT_RECEIPT_MATCH' if not mismatches else 'EXACTNESS_NOT_ESTABLISHED',
                mismatches=mismatches, numerical_tolerance=0,
                protection_tolerance_used_as_parity_tolerance=False,
                gradient_coverage='NOT_RUN_NORMAL_EMPTY_OR_UNRESOLVED_SPACE' if left.get('gradient_sweeps')==right.get('gradient_sweeps')==0 else 'SEE_FULL_SWEEP_ROWS',
                method_efficacy_claim=False,
                tensor_verification='CALLER_SUPPLIED_EXACT_DIGESTS_NOT_RECONSTRUCTION')
