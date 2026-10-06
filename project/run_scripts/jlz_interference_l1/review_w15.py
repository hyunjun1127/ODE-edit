"""CPU-only immutable B1..B15 review; never inspect mutable B16+ or wait.

Reuse source-bound collector arithmetic and independent stdlib metric reducer.
Scope-only AST adaptation caps collector to 15 and excludes live terminal/error
and unfinished directories. No production file/controller is modified.
"""
import ast
import copy
import csv
import inspect
import json
import statistics
import sys
from pathlib import Path
from . import collect as C

ATTEMPT = Path('/data/janghj/ODE-edit/local/jlz-interference-priced-l1-2k/repair-59721')
SOURCE = '0415aba3c160170d306be8196792f198dad4d122'
CONFIG = '26096236ba0fe1a683c98d954904dbf0a048d4611f03cd62b1aef77f7c00091f'
LOCK = '78f4095399905ceb73864507ebee4a00a34b0457507ee67b5dc896013a20e385'


def capped_review():
    node = ast.parse(inspect.getsource(C._arm_review)).body[0]
    changes = []
    for stmt in node.body:
        if isinstance(stmt, ast.For) and ast.unparse(stmt.iter) == 'range(1, 21)':
            stmt.iter.args[1] = ast.Constant(16)
            changes.append('batch_loop_1_to_15')
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            name = ast.unparse(stmt.targets[0])
            if name in ('terminal', 'firsterror'):
                stmt.value = ast.Constant(None)
                changes.append('do_not_read_live_' + name)
        if isinstance(stmt, ast.For) and ast.unparse(stmt.target) == 'folder':
            C.require("out.glob('batch-*')" in ast.unparse(stmt.iter), 'EXPECTED_UNFINISHED_LOOP')
            stmt.iter = ast.List(elts=[], ctx=ast.Load())
            changes.append('do_not_read_uncommitted_or_B16plus')
    C.require(len(changes) == 4, 'EXACT_SCOPE_ADAPTATION')
    scope = dict(C.__dict__)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 '<bounded-immutable-W15-review>', 'exec'), scope)
    return scope['_arm_review'], changes


def compact_stats(values):
    # mean/canonical use column arrays, rewrite/KL use owner-row objects.
    def scalars(items):
        for value in items:
            if isinstance(value, list):
                yield from scalars(value)
            elif value is not None:
                yield float(value)
    values = list(scalars(values))
    return dict(count=len(values), mean=statistics.fmean(values) if values else None,
                min=min(values) if values else None, max=max(values) if values else None)


def main():
    out = Path(sys.argv[1]).resolve()
    C.require(not out.exists(), 'CREATE_ONCE_REVIEW')
    reader = C.Reader()
    config = reader.json(ATTEMPT/'config.json')
    lock = reader.json(ATTEMPT/'execution.lock.json')
    C.require(C.sha(ATTEMPT/'config.json') == CONFIG and C.sha(ATTEMPT/'execution.lock.json') == LOCK
              and lock['source_commit'] == SOURCE and config['execution_arms'] == ['PRICE'], 'EXECUTION_BINDING')
    # Small source files only; no repeated model hash/load or scheduler calls.
    for item in lock['source_members']:
        data = reader.bytes(item['path'])
        C.require(len(data) == item['bytes'] and C.sha(item['path']) == item['sha256'], 'FROZEN_SOURCE_CHANGED')
    for rel in ('project/run_scripts/jlz_interference_l1/collect.py',
                'project/run_scripts/jlz_interference_l1/w0_reuse.py',
                'project/run_scripts/jlz_realized_writer_sequential/review_completed.py'):
        C.require(C.sha(C.ROOT/rel) == C.sha(ATTEMPT/'source'/rel), 'REUSED_REDUCER_SOURCE_IDENTITY')
    records = C.load_prefix(Path(config['stream']).parent, 2000)
    C.require(C.digest([r['case_id'] for r in records]) == C.ORDERED_SHA, 'ORDERED_FIRST2000')
    identities = reader.bound(config['observer_identity'])['rows']
    review, changes = capped_review()
    result, prefix = review(reader, ATTEMPT, config, lock, 'PRICE', records, identities, {})
    C.require(result['commits'] == 15 and result['actual'] == dict(joins=14, history_appends=75)
              and set(prefix) == {5, 10, 15}, 'SEALED_W15_COVERAGE')
    C.require({k: v['denominator'] for k, v in C.reduce_rows(prefix[15]).items()} ==
              dict(R=1500, P=3000, N=15000), 'W15_DENOMINATORS')
    C.require('torch' not in sys.modules and 'transformers' not in sys.modules, 'CPU_STDLIB_NO_MODEL')
    # Read-set safety: all observed batches are immutable committed <=15.
    for path in reader.files:
        for part in Path(path).parts:
            if part.startswith('batch-') and part[6:].isdigit():
                C.require(int(part[6:]) <= 15, 'REVIEW_HORIZON_15')
    # Detect changes to any evidence between read and final review seal.
    for item in reader.files.values():
        C.require(C.sha(item['path']) == item['sha256'], 'EVIDENCE_CHANGED_DURING_REVIEW')
    realization = []
    for batch in result['realization']:
        for layer, data in batch['realization']['layers'].items():
            for role in ('mean', 'canonical', 'rewrite', 'KL'):
                rows = data[role]
                if isinstance(rows, dict): rows = [rows]
                realization.append(dict(batch=batch['batch'], layer=int(layer), role=role,
                    **{key: compact_stats(r[key] for r in rows) for key in
                       ('normratio', 'directionalratio', 'cosine', 'relative_error', 'zero_target_leakage')},
                    ideal_Q=data['ideal_Q'], effective_Q=data['effective_Q'],
                    Q_C0=data['Q_C0'], Q_H=data['Q_H'],
                    effective_update_norm=data['effective_update_norm']))
    summary = dict(task=C.TASK, status='W15_INTERMEDIATE_CPU_VERIFIED', execution_source=SOURCE,
        config_sha256=CONFIG, lock_sha256=LOCK, fixed_review_edits=1500, commits=15,
        joins=14, history_appends=75, W20='NOT_REVIEWED_NOT_CLAIMED',
        counters=result['counters'], metrics=result['metrics'], paired=result['paired'],
        cost=result['cost'], W0_observation_cost=result['W0_observation_cost'],
        realization_summary=realization, controls='USER_CANCELLED_NOT_RESTARTED',
        GPU=0, model_load=0, new_evaluation=0, scheduler_writes=0,
        checkpoint_saved=False, exact_resume='NOT_AVAILABLE',
        review_level='OWNER_AUDIT_USING_INDEPENDENT_STDLIB_ROW_REDUCER',
        separate_reviewer=False, scope_adaptation=changes, input_files=len(reader.files),
        raw_rehashed_immutable=True, no_missing_as_zero=True)
    out.mkdir(parents=True)
    (out/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    (out/'input-manifest.json').write_text(json.dumps(dict(files=list(reader.files.values())),indent=2)+'\n')
    rows=[]
    for p in result['metrics']:
        for kind, m in p['metrics'].items():
            rows.append(dict(endpoint=p['endpoint'], requests=p['requests'],kind=kind,
                **{k:m[k] for k in ('numerator','denominator','rate','strict_numerator','strict_denominator',
                                  'strict_rate','token_micro','prompt_macro','true_nll_mean','new_nll_mean')},
                harmonic=C.harmonic(p['metrics'])))
    with (out/'metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(status=summary['status'], commits=15, joins=14, H=75,
                         counters=summary['counters'], files=len(reader.files),
                         milestone=[dict(endpoint=p['endpoint'],counts={k:[v['numerator'],v['denominator']]
                             for k,v in p['metrics'].items()}) for p in summary['metrics']
                             if p['endpoint'].endswith('ALL_SEEN')]),ensure_ascii=False))


if __name__ == '__main__':
    main()
