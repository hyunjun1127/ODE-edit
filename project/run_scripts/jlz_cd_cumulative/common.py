"""Task identities and exact observer binding; no model import at module scope."""
import json
import math
from pathlib import Path
from project.run_scripts.jlz_realization.common import require, sha, digest, member, write, state, tensor_sha
from project.run_scripts.jlz_shared_budget.common import verify

ROOT = Path(__file__).resolve().parents[3]
LOCAL = Path('/data/janghj/ODE-edit/local/jlz-cd-cumulative-allocation/20261005-r1')
TASK = 'jlz-cd-cumulative-allocation-bs100x20-s4-20261005-r1'
NONCE = 'ODEEDIT-USER-GH-SH4-CD-CUMULATIVE-2K-20261005-R1'
ENVELOPE = 'messages/head/2026-10-05-sh4-cd-cumulative-2k.json'
EXCEPTION = 'control/experiment-exceptions/jlz-cd-cumulative-2k-sh4-20261005.json'
DESIGN = 'plans/global/2026-10-05-jlz-cd-cumulative-allocation'
CONTRACT = DESIGN + '/method-contract.json'
SCHEDULE = 'plans/global/2026-10-04-jlz-v12-marginal-allocation/experiment-2k/case-schedule-first2000.csv'
PRIOR = Path('/data/janghj/ODE-edit/local/jlz-v13-mdcd-sequential-2k/20261005-v1/attempt-r1')
FROZEN = '2ff0ecc63dda2c830fbd68e6a2d76f3317a446f7'
ARMS = ('CD_Q', 'CD_C')
ALPHA = {'CD_Q': 0, 'CD_C': 1}
MILESTONES = (5, 10, 15, 20)


def batches(records, size):
    require(type(size) is int and size > 0, 'INVALID_LOGICAL_B')
    for start in range(0, len(records), size):
        yield start // size + 1, records[start:start + size], records[:start + size]


def selected_for_post(current, seen, number):
    return seen if number in MILESTONES else current


def expected_rows(identities, ids):
    by_case = {}
    for row in identities:
        by_case.setdefault(row['case_id'], []).append(row)
    require(len(set(ids)) == len(ids), 'DUPLICATE_OCCURRENCE_CASE_ID')
    require(all(case in by_case for case in ids), 'MISSING_OBSERVER_CASE')
    return [row for case in ids for row in by_case[case]]


def validate_rows(rows, identities, ids, endpoint=None):
    """Runtime identity guard; collector uses a separate stdlib arithmetic reducer."""
    from project.run_scripts.jlz_realization.observe import reduce_rows
    from project.run_scripts.jlz_realized_writer_sequential.review_completed import validate_rows as independent_validate
    expected = expected_rows(identities, ids)
    independent_validate(rows, expected, endpoint)
    return reduce_rows(rows)


def rows_from(folder, expected_state=None):
    rows = []
    for path in sorted(Path(folder).glob('chunk-*.json')):
        require(path.is_file() and not path.is_symlink(), 'UNSAFE_OBSERVER_CHUNK')
        data = json.loads(path.read_text())
        require(data['optimizer_feedback'] is False, 'OBSERVATION_FEEDBACK')
        if expected_state is not None:
            require(data['state'] == expected_state, 'CHUNK_STATE_IDENTITY')
        rows.extend(data['rows'])
    return rows


def finite_tree(value, label='SCALAR_NONFINITE'):
    if isinstance(value, float):
        require(math.isfinite(value), label)
    elif isinstance(value, dict):
        for v in value.values():
            finite_tree(v, label)
    elif isinstance(value, (list, tuple)):
        for v in value:
            finite_tree(v, label)
