"""CPU-only independent request-macro loc_ans reduction; no raw export."""
import hashlib
import json
import math
from pathlib import Path

ROOT=Path('/mnt/raid5/janghj/ODE-edit/local/official-baselines-server2/20261008-r1/registration-no-gpu-qual-r1')
JOBS={'FT':'61726','MEMIT':'61728','ALPHAEDIT':'61730','ALPHAEDIT_BLUE':'61732','MEMIT_FE':'61734','SPHERE':'61735'}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()
def mean(xs):
    assert xs and all(math.isfinite(x) for x in xs)
    return math.fsum(xs)/len(xs)
def macro(vectors):return 100*mean([mean([float(b) for b in bits]) for bits in vectors])

def measure(cases,kind,expected_target):
    vectors=[]
    for c in cases:
        bits=c[kind+'_prompts_correct']
        assert bits and all(type(b) in (bool,int) and b in (0,1) for b in bits)
        independently=[]
        for obs in c[kind+'_observations']:
            assert obs['target_kind']==expected_target
            predicted,target=obs['predicted_token_ids'],obs['target_token_ids']
            assert len(predicted)==len(target)>0
            assert all(type(x) is int for x in [*predicted,*target])
            actual=[p==t for p,t in zip(predicted,target)]
            assert actual==obs['token_correct']
            assert sum(actual)==obs['token_correct_count'] and len(actual)==obs['token_count']
            independently+=actual
        assert independently==bits
        vectors.append(independently)
    return dict(request_macro_pct=macro(vectors),cases=len(vectors),
        tokens=sum(map(len,vectors)),correct_tokens=sum(sum(v) for v in vectors),
        independent_prediction_target_match=True,
        aggregation='100 * mean_requests(mean_target_token_correct); NOT token micro')

def main():
    # Fixed CPU controls distinguish macro from micro and answer accuracy from agreement.
    assert macro([[True],[False,False,False]])==50.0
    assert 100*1/4==25.0
    rows=[]
    for method,job in JOBS.items():
        folder=ROOT/('ZSRE_'+method);result=json.loads((folder/'result.json').read_text())
        assert result['status']=='SCIENTIFIC_COMPLETE' and result['batches']==20
        assert result['method']==method and result['dataset']=='zsre' and len(result['commits'])==20
        m=result['factual_endpoints']['W20'];p=Path(m['path'])
        assert p.stat().st_size==m['bytes'] and sha(p)==m['sha256']
        raw=json.loads(p.read_text());cases=raw['cases']
        assert raw['identity_sha256']==digest(raw['identity'])
        assert raw['endpoint']=='W20' and raw['dataset']=='zsre' and len(cases)==2000
        assert [c['occurrence_index'] for c in cases]==list(range(1,2001))
        assert raw['identity']['ordered_occurrences']==list(range(1,2001))
        measures={name:measure(cases,kind,target) for name,kind,target in
            [('Eff','rewrite','new'),('Gen','paraphrase','new'),('Loc','neighborhood','loc_ans')]}
        for key,label in [('Eff','Efficacy'),('Gen','Generalization'),('Loc','Specificity_loc_ans')]:
            assert abs(measures[key]['request_macro_pct']-raw['summary'][label])<=1e-10
        auxiliary=macro([c['neighborhood_W0_agreement'] for c in cases])
        assert abs(auxiliary-raw['summary']['Specificity'])<=1e-10
        scores=[measures[k]['request_macro_pct'] for k in ('Eff','Gen','Loc')]
        harmonic=0.0 if any(x==0 for x in scores) else 3/math.fsum(1/x for x in scores)
        rows.append(dict(model='gptj',method=method,dataset='zsre',job_id=job,
            status='W20_2000_CPU_REDUCED_PREDICTION_TARGET_VERIFIED',
            source=result['code_commit'],checkpoint_identity=result['checkpoint_identity'],raw=m,
            raw_identity_sha256=raw['identity_sha256'],ordered_cohort_sha256=digest([(c['occurrence_index'],c['case_id']) for c in cases]),
            before_table_Loc_W0_agreement_pct=auxiliary,
            after_table_Loc_answer_accuracy_pct=measures['Loc']['request_macro_pct'],
            auxiliary_W0_prediction_agreement_pct=auxiliary,metrics=measures,
            derived_Eff_Gen_Loc_harmonic_pct=harmonic,
            summary_loc_ans_crosscheck='MATCH; independent reduction used predicted and target token IDs, not summary rename'))
    print(json.dumps(dict(instruction_id='USER-GH-ZSRE-LOC-RECALCULATE-MAIN-20261009-R1',rows=rows,
        reducer=dict(path='official/runners/server2/zsre_loc_review_20261009.py',sha256=sha(__file__)),
        inventory_scope='Existing approved server2 official GPTJ six completed zsRE main cells; Qwen migration is newly released, not a completed result',
        observations_exported=False,tokenization_equivalence_claim=False,paper_reproduction_PASS=False,
        GPU=0,model_loads=0,new_forward=0,job_mutations=0,CF_changed=False,online_history_rewrite=False,
        CPU_macro_micro_control='PASS (50 request-macro != 25 token-micro)'),indent=2))

if __name__=='__main__':main()
