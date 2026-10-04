from pathlib import Path
from project.run_scripts.jlz_realization.common import require, sha, digest, member, write, state, tensor_sha
from project.run_scripts.jlz_shared_budget.common import verify

ROOT = Path(__file__).resolve().parents[3]
LOCAL = Path('/data/janghj/ODE-edit/local/jlz-v13-mdcd-sequential-2k/20261005-v1')
TASK = 'jlz-v13-mdcd-sequential-bs100x20-s4-20261005-v1'
NONCE = 'ODEEDIT-USER-GH-SH4-JLZ-V13-MDCD-2K-CAP2-20261005-R1'
AUTHORITY = '89ad31eeb624c170225fd06e81da8b6014e71380'
ENVELOPE = 'messages/head/2026-10-05-jlz-v13-mdcd-2k-sh4.json'
CONTRACT = 'plans/global/2026-10-05-jlz-v13-mdcd-sequential-2k-sh4/execution-command.json'
EXCEPTION = 'control/experiment-exceptions/jlz-v13-mdcd-sequential-2k-sh4-20261005.json'
DESIGN = 'plans/global/2026-10-05-jlz-v13-realized-writer'
SCHEDULE = 'plans/global/2026-10-04-jlz-v12-marginal-allocation/experiment-2k/case-schedule-first2000.csv'
FROZEN = '082300955e21a2c29218d66d98c5d2c37bc53a20'
PRIOR = Path('/data/janghj/ODE-edit/local/jlz-v13-realized-writer-b1/20261005-v1/attempt-r1')
ARMS = ('MD', 'CD')
MILESTONES = (5, 10, 15, 20)

def batches(records, size):
    require(type(size) is int and size > 0, 'INVALID_LOGICAL_B')
    for start in range(0, len(records), size):
        yield start // size + 1, records[start:start + size], records[:start + size]

def selected_for_post(current, seen, number):
    return seen if number in MILESTONES else current

def expected_rows(identities, ids):
    wanted = set(ids)
    return {r['identity']: r for r in identities if r['case_id'] in wanted}

def validate_rows(rows, identities, ids, endpoint=None):
    from project.run_scripts.jlz_realization.observe import reduce_rows
    expected = expected_rows(identities, ids)
    require(len(rows) == len(expected) and {r['identity'] for r in rows} == set(expected), 'EXACT_EVAL_IDENTITY_SET')
    for r in rows:
        ref = expected[r['identity']]
        require((r['case_id'], r['kind'], r['prompt_index']) == (ref['case_id'], ref['kind'], ref['prompt_index']), 'EVAL_ROLE')
        require(all(r[label + '_token_identity'] == ref[label + '_token_identity'] for label in ('new', 'true')), 'TOKEN_IDENTITY')
        require(abs(r['margin_true_minus_new'] - (r['true_nll'] - r['new_nll'])) <= 1e-12, 'MARGIN_SIGN')
        if endpoint is not None:
            require(r['endpoint'] == endpoint, 'ENDPOINT_IDENTITY')
    return reduce_rows(rows)

def rows_from(folder):
    import json
    rows = []
    for p in sorted(Path(folder).glob('chunk-*.json')):
        rows.extend(json.loads(p.read_text())['rows'])
    return rows
