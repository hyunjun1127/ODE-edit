"""저장된 W0 원자료의 CPU 재집계. 모델 실행/원자료 수정 없음.

사용법: python3 collect.py cf|zsre|generation|cf_chunks FILE_OR_DIR
원격 서버에서는 이 파일을 python3 - 의 stdin으로 전달한다.
"""
import collections
import datetime
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import socket
import sys

mode, target = sys.argv[1:]
inventory = []
def read(path):
    p = Path(path).resolve()
    b = p.read_bytes()
    inventory.append(dict(path=str(p), sha256=hashlib.sha256(b).hexdigest(), bytes=len(b)))
    return json.loads(b)

def compact_identity(d):
    return {k:v for k,v in d.items() if k not in ('input_signature', 'ordered_occurrences', 'ordered_case_ids') and not isinstance(v,list)}

def check_ids(cases, dataset):
    ids = [r['case_id'] for r in cases]
    assert len(ids) == len(set(ids)) == 2000
    expected = {'cf':'0b912d11659eb087ee71a391b7bea1e02ecc9d999a48254eb8559434965640f4', 'zsre':'494b15107deb77d9741606b7bddfecb8b350a486bb5a751655b294d4ebc6f1a4'}[dataset]
    actual = hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
    assert actual == expected, (actual,expected)
    return actual

out = dict(host=socket.gethostname(), observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), mode=mode, endpoint='W0', model_forward_calls=0, inventory=inventory)
if mode == 'cf_chunks':
    root = Path(target)
    lock = read(root.parent.parent/'execution.lock.json')
    prep = read(lock['preparation']['path'])
    assert inventory[-1]['sha256']==lock['preparation']['sha256']
    stream = read(prep['baseline_stream']['path'])
    assert inventory[-1]['sha256']==prep['baseline_stream']['sha256']=='66edc483a8d4bcadedd479e4c36759a686ad61a38741d8870a9052b795710e37'
    check_ids(stream,'cf')
    observer = read(prep['observer_identity']['path'])
    assert inventory[-1]['sha256']==prep['observer_identity']['sha256']
    expected = {r['identity']:r for r in observer['rows']}
    initial = read(root.parent/'initial.json')
    result = read(root.parent/'W0-result.json')
    assert result['status'] == 'W0_COMPLETE' and result['state'] == initial['cold']
    summary = read(root/'summary.json')
    assert summary['endpoint']=='W0' and summary['requests']==2000 and summary['no_mutation'] is True
    assert summary['state']==initial['cold']
    rows = []
    for p in sorted(root.glob('chunk-*.json')):
        d = read(p)
        assert d['state'] == initial['cold'] and d['optimizer_feedback'] is False
        rows.extend(d['rows'])
    by = collections.OrderedDict()
    for r in rows:
        assert r['endpoint'] == 'W0'
        assert all(r[k]==v for k,v in expected[r['identity']].items())
        assert math.isfinite(r['new_nll']) and math.isfinite(r['true_nll'])
        by.setdefault(r['case_id'], dict(case_id=r['case_id'], rewrite_prompts_probs=[], paraphrase_prompts_probs=[], neighborhood_prompts_probs=[]))
        kind = {'R':'rewrite','P':'paraphrase','N':'neighborhood'}[r['kind']]
        by[r['case_id']][kind+'_prompts_probs'].append(dict(target_new=r['new_nll'], target_true=r['true_nll']))
    cases = list(by.values())
    out['cold_state_verified'] = True
    out['identity'] = {k:initial[k] for k in ('job_id','source','profile_sha')}
    out['identity'].update(model_snapshot=prep['model'],stream_sha256=prep['baseline_stream']['sha256'],observer_identity_sha256=prep['observer_identity']['sha256'])
    stored = {label:100*result['summary'][key]['rate'] for key,label in [('R','Eff'),('P','Gen'),('N','Loc')]}
elif mode == 'generation':
    d = read(target)
    ready = read(Path(target).with_name('READY.json'))
    assert ready['actual_model_edits']==0 and ready['cf_generation']['sha256']==inventory[0]['sha256']
    cases = d['rows']
    out['ordered_case_ids_sha256'] = check_ids(cases,'cf')
    out['identity'] = compact_identity(d['identity'])
    metrics = {}
    for key,flag in [('ngram_entropy','fluency_valid'),('reference_score','consistency_valid')]:
        assert all(r['metrics'][flag] and math.isfinite(r['metrics'][key]) for r in cases)
        metrics[key] = math.fsum(r['metrics'][key] for r in cases)/2000
        assert math.isclose(metrics[key],d['summary'][key],abs_tol=1e-12)
    out.update(metrics=metrics, requests=2000, units={'ngram_entropy':'bits','reference_score':'cosine_0_to_1'})
else:
    d = read(target)
    if isinstance(d, list):
        cases = d
        receipt = read(Path(target).with_name('w0-receipt.json'))
        assert receipt['endpoint']=='W0' and receipt['observed_requests']==2000
        assert receipt['factual']['cases_sha256'] == inventory[0]['sha256']
        out['identity'] = {k:receipt[k] for k in ['model','dataset','code_commit','stream_sha256','tokenizer_sha256','official_tree_sha256']}
        stored = receipt['factual']['summary']
    else:
        evaluation = d.get('evaluation',d)
        cases = evaluation['cases']
        stored = evaluation['summary']
        out['identity'] = compact_identity(evaluation['identity'])
        assert evaluation['model_no_mutation'] is True
        out['model_no_mutation'] = True
        if Path(target).name == 'W0.json':
            ready = read(Path(target).parent.parent/'READY.json')
            assert d['endpoint']=='W0' and ready['status']=='READY_COLD_W0_COMPLETE'
            assert ready['factual']['sha256']==inventory[0]['sha256']
        else:
            ready = read(Path(target).with_name('READY.json'))
            assert ready['actual_model_edits']==0
            if 'evaluation' in d:
                assert ready['reference']['sha256']==inventory[0]['sha256']
            else:
                assert ready['cf_external_identity']==evaluation['identity']['external_identity']
        out['cold_W0_receipt_verified'] = True
    stored = {name:stored[key] for name,key in [('Eff','Efficacy'),('Gen','Generalization'),('Loc','Specificity_loc_ans' if mode=='zsre' else 'Specificity')]}

if mode != 'generation':
    dataset = 'cf' if mode in ('cf','cf_chunks') else 'zsre'
    out['ordered_case_ids_sha256'] = check_ids(cases,dataset)
    metrics, denominators = {}, {}
    for kind,label,count in [('rewrite','Eff',1),('paraphrase','Gen',2),('neighborhood','Loc',10)]:
        rates, numerator, denominator = [], 0, 0
        for case in cases:
            if dataset == 'cf':
                probs = case[kind+'_prompts_probs']
                assert len(probs)==count
                assert all(math.isfinite(x[k]) for x in probs for k in ('target_new','target_true'))
                bits = [x['target_true'] < x['target_new'] if kind=='neighborhood' else x['target_new'] < x['target_true'] for x in probs]
            else:
                obs = case[kind+'_observations']
                bits = []
                for row in obs:
                    assert len(row['predicted_token_ids']) == len(row['target_token_ids']) > 0
                    actual = [p==t for p,t in zip(row['predicted_token_ids'],row['target_token_ids'])]
                    assert actual == row['token_correct']
                    bits.extend(actual)
                assert bits == case[kind+'_prompts_correct']
            rates.append(Fraction(sum(bits),len(bits)))
            numerator += sum(bits)
            denominator += len(bits)
        metrics[label] = float(100*sum(rates)/2000)
        assert math.isclose(metrics[label],stored[label],abs_tol=1e-10), (label,metrics[label],stored[label])
        denominators[label] = dict(correct=numerator, observations=denominator, request_macro=True)
    if dataset == 'cf':
        metrics['Score'] = 3/math.fsum(1/metrics[k] for k in ('Eff','Gen','Loc')) if all(metrics.values()) else 0.
    out.update(metrics=metrics, requests=2000, denominators=denominators, stored_summary_match=True)
print(json.dumps(out,ensure_ascii=False,indent=2))
