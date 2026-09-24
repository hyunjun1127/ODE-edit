"""Bounded CPU review of the immutable r2 failure; no runtime or scheduler calls."""
import ast
import argparse
import csv
import math
from .common import *

OLD = ROOT / 'attempt-r2-routing'
LOCAL = ROOT / 'rerun-20260925-r1'
DEST = REPO / 'audits/servers/server4/historical-update-timeaxis-20260924-v1/rerun-20260925-r1'


def compare_rows(left, right):
    assert left.keys() == right.keys()
    max_nll = max_margin = 0.
    nrows = 0
    flips = []
    identity = ('case_id', 'kind', 'prompt_index', 'prompt', 'target', 'target_token_ids')
    for category, a in left.items():
        b = right[category]
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            assert all(x[k] == y[k] for k in identity), 'ROW_IDENTITY'
            assert math.isfinite(x['nll']) and math.isfinite(y['nll']), 'NONFINITE_NLL'
            for row in (x, y):
                labels, predictions = row['target_token_ids'], row['token_predictions']
                assert len(labels) == len(predictions) > 0
                correct = [v == w for v, w in zip(predictions, labels, strict=True)]
                assert correct == row['token_correct'] and all(correct) == row['all_tokens_correct']
            max_nll = max(max_nll, abs(x['nll'] - y['nll']))
            nrows += 1
    for kind in ('rewrite', 'rephrase'):
        for n, t, nn, tt in zip(left[kind+'_target_new'], left[kind+'_target_true'],
                               right[kind+'_target_new'], right[kind+'_target_true'], strict=True):
            assert (n['case_id'], n['prompt_index'], n['prompt']) == (t['case_id'], t['prompt_index'], t['prompt'])
            m, mm = t['nll']-n['nll'], tt['nll']-nn['nll']
            max_margin = max(max_margin, abs(m-mm))
            if (m > 0) != (mm > 0):
                flips.append(dict(case_id=n['case_id'], panel=kind, prompt_index=n['prompt_index'], left=m, right=mm))
    return dict(answer_rows=nrows, max_nll=max_nll, max_margin=max_margin, boundary_flips=flips)


def write_csv(path, values):
    with path.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(values[0]))
        writer.writeheader(); writer.writerows(values)


def main(verify_only=False):
    DEST.mkdir(parents=True, exist_ok=True)
    lockpath = OLD / 'execution.lock.json'
    assert sha(lockpath) == '54abf9a7572eceda008db9c001ed8d1fc9de8f5a7397990a4bed26f313bc1699'
    lock = read(lockpath)
    assert lock['source_commit'] == '730a4a9768e5650e01fd9afdc4e0f7895c86ea92'
    for m in lock['source_members']:
        assert sha(m['path']) == m['sha256']
    binding = read(lock['T0']['path'])
    assert sha(lock['T0']['path']) == lock['T0']['sha256']
    for family in FAMILIES:
        for cp in binding['checkpoints'][family].values():
            st = Path(cp['path']).stat()
            assert (st.st_size, st.st_ino, st.st_mtime_ns) == (cp['bytes'], cp['inode'], cp['mtime_ns'])
    for m in binding['model_shards']:
        st = Path(m['path']).stat()
        assert (st.st_size, st.st_mtime_ns) == (m['bytes'], m['mtime_ns'])
    assert sha(binding['token_manifest']['path']) == binding['token_manifest']['sha256']
    tokens = {(r['case_id'],r['panel'],r['prompt_index']):r for r in read(binding['token_manifest']['path'])}
    for m in binding['design_members']:
        # Received member receipt carries absolute path and original file bytes.
        assert sha(m['path']) == m['sha256']
        relative = Path(m['path']).relative_to(DESIGN)
        assert sha(REPO/'plans/global/2026-09-24-historical-update-timeaxis-v1'/relative) == m['sha256']

    identity = {k:lock[k] for k in ('instruction_id','attempt','source_sha256','contract_sha256','T0_sha256','token_sha256')}
    summary = []
    for family in FAMILIES:
        failure = read(OLD/'output'/family/'FAILURE.json')
        assert failure['identity'] == identity and failure['stage'] == 'T1'
        for p in sorted((OLD/'output'/family/'fidelity').glob('endpoint-*.json')):
            x = read(p); t = x['t']
            expected = binding['w0_selected'] if t == 0 else binding['checkpoints'][family][str(t)]['weights']
            for state in (x['state'],x['restored']):
                assert state['restore'] == 'EXACT_BYTES' and state['weight_hashes'] == expected
                assert state['state_weight_hash'] == digest(expected) and state['removed'] == []
            assert len(x['ids']) == len(set(x['ids'])) == 16
            for variant in ('raw','repeat_raw','mb1_raw','restored_raw'):
                raw = x[variant]
                for kind, panel in (('rewrite','rewrite'),('rephrase','paraphrase')):
                    order = [(i,j) for i in x['ids'] for j in (range(1) if kind=='rewrite' else range(2))]
                    for side, field in (('new','target_ids'),('true','competitor_ids')):
                        rs = raw[kind+'_target_'+side]
                        assert [(r['case_id'],r['prompt_index']) for r in rs] == order
                        for r in rs:
                            assert r['target_token_ids'] == tokens[r['case_id'],panel,r['prompt_index']][field]
                    for n,trow in zip(raw[kind+'_target_new'],raw[kind+'_target_true'],strict=True):
                        token = tokens[n['case_id'],panel,n['prompt_index']]
                        assert historical_digest([n['case_id'],n['prompt_index'],n['prompt'],n['target'],trow['target']]) == token['pair_identity']
            for index, (label, field) in enumerate((('repeat','repeat_raw'),('MB16_vs_MB1','mb1_raw'),('restore','restored_raw'))):
                d = compare_rows(x['raw'], x[field]); old = x['checks'][index]
                assert all(d[k] == old[k] for k in ('max_nll','max_margin','boundary_flips'))
                assert d['max_nll'] <= .00025 and d['max_margin'] <= .0005
                assert all(abs(f['left']) <= .0005 and abs(f['right']) <= .0005 for f in d['boundary_flips'])
                summary.append(dict(family=family, endpoint=t, comparison=label, answer_rows=d['answer_rows'],
                                    max_nll=d['max_nll'],max_margin=d['max_margin'],flips=len(d['boundary_flips']),
                                    source_sha256=sha(p),status='RECOMPUTED_FROM_SAVED_ROWS_PASS'))
    assert len(summary) == 27
    alpha = read(OLD/'output/BASE_ALPHAEDIT/T1/PASS.json')
    assert alpha['identity'] == identity and alpha['status'] == 'PASS'
    diagonals = alpha['detail']['diagonals']; assert len(diagonals) == 12
    for d in diagonals:
        expected = binding['w0_selected'] if d['start']==0 else binding['checkpoints'][FAMILIES[0]][str(d['start'])]['weights']
        assert d['arithmetic']['weight_hashes'] == d['direct']['weight_hashes'] == expected
        assert d['arithmetic']['restore'] == d['direct']['restore'] == 'EXACT_BYTES'
        assert d['check']['max_nll'] <= .00025 and d['check']['max_margin'] <= .0005
    error = ast.literal_eval(read(OLD/'output/BASE_MEMIT/FAILURE.json')['error'])
    assert error[0] == 'NLL_FIDELITY' and error[1] > .00025
    collector = read(OLD/'output/T4/terminal.json')
    assert collector['status'] == 'TECHNICAL_BLOCKED' and collector['science_gate_bypassed'] is False
    tasks = list((OLD/'output').glob('*/tasks/*/PASS.json')); assert not tasks
    inventory = [record(p) for p in sorted((OLD/'output').rglob('*')) if p.is_file()]
    inventory += [record(OLD/n) for n in ('53182.err','53183.err','53184.err','53182.out','53183.out','53184.out','submission.json','release.json','execution.lock.json')]
    if not verify_only:
        save(DEST/'artifact-index.json',inventory)
        write_csv(DEST/'saved-fidelity-independent.csv',summary)
    allocation = read(LOCAL/'accounting-snapshot.json')
    assert sum(j['elapsed_seconds']*j['gpus'] for j in allocation['jobs']) == 1569
    if not verify_only: save(DEST/'accounting.json',allocation)
    result = dict(status='BLOCKED_NUMERICAL_CONTRACT', first_cause=dict(family='BASE_MEMIT',stage='T1',
                  comparison='MB16_VS_MB1',endpoint=100,endpoint_basis='source call order + saved endpoints 0/1/20/50 + stdout; failed W100 raw was not saved',
                  nll_abs=error[1],ceiling=.00025,ratio=error[1]/.00025,excess=error[1]-.00025,
                  category_case_raw='NOT_RECORDED', low_level_cause='NOT_ESTABLISHED'),
                  peer_failure='BASE_ALPHAEDIT already passed T1, then failed at matching-family join',
                  saved_endpoint_receipts=9, independent_row_comparisons=sum(v['answer_rows'] for v in summary),
                  alpha_diagonal_receipts=12,scientific_score_tasks=0,new_job_ids=[],
                  allocated_GPU_seconds=1569,CPU_collector_seconds=0,
                  CP24_verification='PRIOR_FULL_SHA_PLUS_UNCHANGED_STAT',model4_verification='PRIOR_FULL_SHA_PLUS_UNCHANGED_STAT',
                  design29='EXACT_FULL_SHA_MATCH_LOCAL_AND_MAIN_PUBLICATION',source=lock['source_commit'],
                  runtime_modified=False,threshold_changed=False,new_GPU=0,scheduler_mutations=0,
                  monitoring_active=False,automatic_resume=False,independent_agent_red=False)
    if verify_only:
        assert read(DEST/'diagnosis.json') == result
        assert read(DEST/'artifact-index.json') == inventory
    else:
        save(DEST/'diagnosis.json',result)
    print(json.dumps(result,ensure_ascii=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--verify-only',action='store_true')
    main(parser.parse_args().verify_only)
