"""Check design artifacts and identities, without loading tensors or running experiments."""
from pathlib import Path
import csv
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parent


def rows(name):
    return list(csv.DictReader((ROOT / name).open()))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()


def main():
    checks = []
    errors = []

    def check(label, ok):
        checks.append(dict(check=label, passed=bool(ok)))
        if not ok:
            errors.append(label)

    documents = {}
    for p in ROOT.glob('*.json'):
        if p.name not in ['design-validation.json', 'design-manifest.json']:
            documents[p.name] = json.loads(p.read_text(), parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
    contract = documents['experiment-contract.json']
    summary = documents['plan-summary.json']
    checkpoints = contract['checkpoints']
    facts = rows('fact-ledger.csv')
    cells = rows('main-cells.csv')
    pilots = rows('pilot-cells.csv')
    pairs = rows('pair-cells.csv')
    states = {r['state_id']: r for r in rows('state-bank.csv')}
    tasks = rows('score-tasks.csv')
    check('10k unique case ids and ordinals', len(facts) == len({r['case_id'] for r in facts}) == 10000 and {int(r['ordinal']) for r in facts} == set(range(10000)))
    check('156 unique logical cells', len(cells) == len({r['cell_id'] for r in cells}) == 156)
    check('32 pilot cells are main subset', len(pilots) == 32 and {r['cell_id'] for r in pilots} <= {r['cell_id'] for r in cells})
    check('300 identical-family pilot case ids', len(rows('pilot-panel.csv')) == 300)
    check('16 pair cells, distinct nonoverlapping earlier/later updates', len(pairs) == 16 and all(int(r['past_cohort_index']) < int(r['future_cohort_index']) < 11 for r in pairs))
    check('all reported checkpoints bound', len(rows('checkpoint-bindings.csv')) == 24 and len(rows('checkpoint-tensor-hashes.csv')) == 120)
    check('scientific negative never gates continuation', contract['analysis']['scientific_effect_size_gate'] is False)

    def actual_mask(t):
        return (1 << checkpoints.index(t)) - 1

    def mask(sid):
        return int(states[sid]['retained_interval_mask'], 16)

    cell_math = []
    for c in cells:
        ci, t, a, b = (int(c[k]) for k in ['cohort_index', 'eval_t', 'update_start', 'anchor'])
        good = (checkpoints[ci], checkpoints[ci + 1]) == (a, b) and t >= b
        good &= mask(c['M_t_state']) == actual_mask(t)
        good &= mask(c['B_t_state']) == actual_mask(t) & ~(1 << ci)
        good &= mask(c['M_b_state']) == actual_mask(b)
        good &= mask(c['B_b_state']) == actual_mask(a)
        cell_math.append(good)
    check('all main cell state expressions and anchors', all(cell_math))
    pair_math = []
    for c in pairs:
        ui, vi = int(c['past_cohort_index']), int(c['future_cohort_index'])
        full = actual_mask(100)
        pair_math.append(mask(c['M_state']) == full and mask(c['minus_U_state']) == full & ~(1 << ui)
            and mask(c['minus_V_state']) == full & ~(1 << vi)
            and mask(c['minus_UV_state']) == full & ~(1 << ui) & ~(1 << vi))
    check('all pair four-state expressions', all(pair_math))

    main_states = [r for r in states.values() if r['used_in_main'] == 'True']
    check('25 actual and 132 removed states', sum(r['kind'] == 'ACTUAL' for r in main_states) == 25 and sum(r['kind'] == 'COUNTERFACTUAL' for r in main_states) == 132)
    check('16 new pair states beyond main', sum(r['used_in_pairs'] == 'True' and r['used_in_main'] != 'True' for r in states.values()) == 16)
    check('full prompt row count', 3 * sum(int(c['request_count']) for c in cells) == summary['main_prompt_time_rows'] == 333600)

    # Verify task panels from their case IDs, including pre-anchor evaluations.
    task_lookup = {}
    for r in tasks:
        panel = [f for f in facts if f['cohort_id'] == r['cohort_id']
                 and (r['case_selector'] == 'whole_cohort' or f['pilot_selected'] == 'True')]
        check('task panel ' + r['task_id'], len(panel) == int(r['request_count']) and digest([int(f['case_id']) for f in panel]) == r['case_order_sha256'])
        task_lookup[(r['state_id'], r['cohort_id'], r['case_selector'])] = r
    check('score task IDs unique', len(tasks) == len({r['task_id'] for r in tasks}))
    check('all main states have full cohort score tasks', all((c[slot], c['cohort_id'], 'whole_cohort') in task_lookup for c in cells for slot in ['M_t_state','B_t_state','M_b_state','B_b_state']))
    check('all pair states have correct past cohort tasks', all((c[slot], c['past_cohort'], 'whole_cohort') in task_lookup for c in pairs for slot in ['M_state','minus_U_state','minus_V_state','minus_UV_state']))
    check('unique task sequence total', sum(int(r['target_sequences']) for r in tasks if r['main'] == 'True') == summary['main_target_sequences_after_state_panel_dedup'] == 1333800)

    nodes = documents['execution-dag.json']['nodes']
    ids = {r['id'] for r in nodes}
    visited = set()
    pending = list(nodes)
    while pending:
        ready = [r for r in pending if set(r['depends_on']) <= visited]
        if not ready:
            break
        for r in ready:
            visited.add(r['id'])
            pending.remove(r)
    check('DAG no missing dependencies or cycle', not pending and visited == ids and len(ids) == len(nodes))
    check('T3A and T3B share T2F only', all(r['depends_on'] == ['T2F'] for r in nodes if r['id'] in ['T3A','T3B']))

    for r in documents['source-receipts.json']['evaluator']:
        p = ROOT / r['snapshot_path']
        check('pinned evaluator ' + p.name, hashlib.sha256(p.read_bytes()).hexdigest() == r['sha256'])
    for p in ROOT.glob('*.md'):
        for dest in re.findall(r'\]\(([^)]+)\)', p.read_text()):
            if not dest.startswith(('https://', 'http://')):
                target = (p.parent / dest.split('#')[0]).resolve()
                check('document link ' + p.name + ' -> ' + dest, target.exists() or target.name == 'design-validation.json')
    schema_status = 'NOT_CHECKED_LIBRARY_UNAVAILABLE'
    try:
        import jsonschema
        jsonschema.Draft7Validator.check_schema(documents['measurement.schema.json'])
        schema_status = 'SCHEMA_VALID; NO_MODEL_ROWS_EXIST_TO_VALIDATE'
        check('JSON Schema definition', True)
    except ImportError:
        pass

    report = dict(status='PASS_DESIGN_CONSISTENCY_ONLY' if not errors else 'FAIL',
        checks=len(checks), errors=errors, json_schema_status=schema_status,
        model_forwards=0, tensor_loads=0, cpu_toy_gates=0, scheduler_submissions=0,
        local_dataset_rehash='completed by build_plan.py; receipts preserved',
        unvalidated=['checkpoint payloads on execution host', 'full model shard binding', 'token manifest', 'GPU fidelity', 'measured effects'],
        results=checks)
    (ROOT / 'design-validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    manifest = {str(p.relative_to(ROOT)): dict(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name != 'design-manifest.json' and '__pycache__' not in p.parts}
    (ROOT / 'design-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ['status','checks','errors','json_schema_status','unvalidated']}, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
