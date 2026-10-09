"""Independent saved-token CPU reduction; no evaluator/model imports or raw writes."""
import csv
import hashlib
import json
import math
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = Path('/mnt/raid5/janghj/ODE-edit/local/official-baselines/server1')

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()

def load(member):
    data = Path(member['path']).read_bytes()
    assert len(data) == member['bytes']
    assert hashlib.sha256(data).hexdigest() == member['sha256']
    return json.loads(data)

def aggregate(groups):
    assert groups and all(g and all(type(b) is bool for b in g) for g in groups)
    return dict(requests=len(groups), token_count=sum(map(len, groups)),
                correct_tokens=sum(sum(g) for g in groups),
                request_macro_pct=100*math.fsum(sum(g)/len(g) for g in groups)/len(groups))

def reduce_cases(cases, kind):
    groups = []
    for i, case in enumerate(cases):
        derived = []
        for obs in case[kind+'_observations']:
            assert obs['kind'] == kind and obs['case_index'] == i
            assert obs['occurrence_index'] == case['occurrence_index']
            pred, target = obs['predicted_token_ids'], obs['target_token_ids']
            assert len(pred) == len(target) > 0
            assert all(type(v) is int and v >= 0 for v in pred+target)
            bits = [p == t for p,t in zip(pred,target)]
            assert obs['token_correct'] == bits and all(type(b) is bool for b in obs['token_correct'])
            assert obs['token_count'] == len(bits) and obs['token_correct_count'] == sum(bits)
            assert obs['strict_correct'] == all(bits)
            assert len(obs['nll_by_token']) == len(bits)
            assert all(math.isfinite(n) and n >= 0 for n in obs['nll_by_token'])
            derived.extend(bits)
        assert case[kind+'_prompts_correct'] == derived
        groups.append(derived)
    return aggregate(groups)

def main():
    # Unequal-length control: macro=50%, token micro=25%; do not substitute micro.
    assert aggregate([[True], [False,False,False]])['request_macro_pct'] == 50
    prior = json.loads((BASE.parent/'results-review-20261009/results.json').read_text())
    approved = [r for r in prior['rows'] if r['dataset']=='zsre' and r['complete']]
    terminals = sorted(str(p) for p in ROOT.rglob('COMPLETE.json') if 'zsre' in str(p))
    assert set(terminals) == {r['terminal']['path'] for r in approved}, 'NEW_OR_UNREVIEWED_COMPLETION'
    rows = []
    for old in approved:
        raw, terminal, cfg = load(old['endpoint']), load(old['terminal']), load(old['config'])
        assert terminal['requests']==2000 and terminal['native_apply_calls']==20
        assert terminal['identity'] == old['identity']
        assert raw['identity']['dataset']=='zsre' and raw['summary']['requests']==2000
        assert digest(raw['identity']) == raw['identity_sha256']
        cases=raw['cases']; assert len(cases)==2000
        assert [c['occurrence_index'] for c in cases] == raw['identity']['ordered_occurrences']
        assert len(set(c['occurrence_index'] for c in cases))==2000
        assert digest([c['case_id'] for c in cases])==old['ordered_case_ids_sha256']
        scores={label:reduce_cases(cases,kind) for label,kind in
                [('Eff','rewrite'),('Gen','paraphrase'),('Loc','neighborhood')]}
        for label,summarykey in [('Eff','Efficacy'),('Gen','Generalization'),('Loc','Specificity_loc_ans')]:
            assert abs(scores[label]['request_macro_pct']-raw['summary'][summarykey]) < 1e-10
        agreement=aggregate([c['neighborhood_W0_agreement'] for c in cases])
        assert abs(agreement['request_macro_pct']-raw['summary']['Specificity']) < 1e-10
        row={k:old[k] for k in ('model','dataset','method','job_id','source','config','endpoint','terminal','identity','ordered_case_ids_sha256')}
        row.update(status='W20_SAVED_TOKEN_IDS_AND_BITS_CPU_REDUCED',counts=scores,
                   before_Loc_W0_agreement=agreement['request_macro_pct'],after_Loc=scores['Loc']['request_macro_pct'],
                   Eff=scores['Eff']['request_macro_pct'],Gen=scores['Gen']['request_macro_pct'],
                   auxiliary_W0_agreement=agreement,stored_loc_ans_crosscheck_only=True)
        rows.append(row)
    out=dict(instruction='USER-GH-ZSRE-LOC-RECALCULATE-MAIN-20261009-R1',server='server1',
             reducer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             inventory=dict(root=str(ROOT),completed_zsre_endpoints=len(terminals),approved_baseline_completed=6,
                            approved_ours_completed=0,other_unreviewed_completions=0),
             claims=dict(new_forward=0,GPU=0,raw_modified=False,job_modified=False,
                         tokenization_equivalence=False,paper_reproduction_PASS=False),rows=rows)
    (BASE/'results.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
    with (BASE/'table-rows.csv').open('w') as f:
        keys=['model','dataset','method','job_id','Eff','Gen','before_Loc_W0_agreement','after_Loc']
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:r[k] for k in keys} for r in rows)
    print(json.dumps([{k:r[k] for k in ('method','job_id','Eff','Gen','before_Loc_W0_agreement','after_Loc')} for r in rows],indent=2))

if __name__=='__main__': main()
