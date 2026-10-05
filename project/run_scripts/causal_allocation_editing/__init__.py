"""Causal Allocation Editing: frozen receipts, one RAM-only 20-batch chain."""
import json
import math
from pathlib import Path
from project.run_scripts.jlz_realization.common import require, sha, digest, member, write, state, tensor_sha
from project.run_scripts.jlz_shared_budget.common import verify

TASK = 'causal-allocation-editing'
NONCE = 'USER-GH-SH4-CAUSAL-ALLOCATION-EDITING'
ROOT = Path(__file__).resolve().parents[3]
LOCAL = Path('/data/janghj/ODE-edit/local/causal-allocation-editing')
DESIGN = 'plans/global/2026-10-06-jlz-causal-joint-writer'
ENVELOPE = 'messages/head/causal-allocation-editing.json'
SCHEDULE = 'plans/global/2026-10-04-jlz-v12-marginal-allocation/experiment-2k/case-schedule-first2000.csv'
MILESTONES = (5, 10, 15, 20)
ORDERED_SHA = '0b912d11659eb087ee71a391b7bea1e02ecc9d999a48254eb8559434965640f4'

def batches(records, size=100):
    require(type(size) is int and size > 0, 'INVALID_LOGICAL_B')
    for start in range(0, len(records), size):
        yield start // size + 1, records[start:start + size], records[:start + size]

def rows_from(folder, expected_state=None):
    rows = []
    for path in sorted(Path(folder).glob('chunk-*.json')):
        require(path.is_file() and not path.is_symlink(), 'UNSAFE_OBSERVER_CHUNK')
        obj = json.loads(path.read_text())
        require(obj['optimizer_feedback'] is False, 'OBSERVATION_FEEDBACK')
        if expected_state is not None:
            require(obj['state'] == expected_state, 'OBSERVER_STATE_IDENTITY')
        rows.extend(obj['rows'])
    return rows

def expected_rows(identities, ids):
    by_case = {}
    for row in identities:
        by_case.setdefault(row['case_id'], []).append(row)
    require(len(set(ids)) == len(ids), 'DUPLICATE_CASE_OCCURRENCE')
    require(all(case in by_case for case in ids), 'MISSING_OBSERVER_CASE')
    return [row for case in ids for row in by_case[case]]

def validate_rows(rows, identities, ids, endpoint=None):
    """Exact ordered token/occurrence binding, independent arithmetic elsewhere."""
    expected = expected_rows(identities, ids)
    require(len(rows) == len(expected), 'OBSERVER_CARDINALITY')
    require(len({r['identity'] for r in rows}) == len(rows), 'DUPLICATE_OBSERVER_IDENTITY')
    for row, ref in zip(rows, expected):
        for key in ('identity', 'case_id', 'kind', 'prompt_index', 'new_token_identity', 'true_token_identity'):
            require(row[key] == ref[key], 'OBSERVER_IDENTITY:' + key)
        if endpoint is not None:
            require(row['endpoint'] == endpoint, 'OBSERVER_ENDPOINT')
        for label in ('new', 'true'):
            require(type(row[label + '_nll']) in (int, float) and math.isfinite(row[label + '_nll']), 'NONFINITE_OBSERVER')
            n, k = row[label + '_token_count'], row[label + '_token_correct']
            require(type(n) is int and type(k) is int and n > 0 and 0 <= k <= n, 'OBSERVER_TOKEN_COUNTS')
            require(type(row[label + '_strict']) is bool and row[label + '_strict'] == (k == n), 'OBSERVER_STRICT')
        require(row['margin_true_minus_new'] == row['true_nll'] - row['new_nll'], 'OBSERVER_MARGIN_SIGN')
    from project.run_scripts.jlz_realization.observe import reduce_rows
    return reduce_rows(rows)
