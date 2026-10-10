"""CPU/raw-only refresh; single saved scheduler snapshot, no CP deserialization."""
import csv,math
from pathlib import Path
from datetime import datetime,timezone
from official.runners.server1.common import read,verify,member
from official.experiments.prepare import write_new,digest
from official.runners.server1 import flucon_eval as e
from official.evaluation.generation.assets import load_assets
from official.evaluation.generation.native_observer import read_observed

O=Path(__file__).parent;B=e.BASE
NONCE='USER-GH-S1-S2-COMPLETED-TABLE-S1-FLUCON-CAP3-20261010-R1'
snap=Path('/mnt/raid5/janghj/ODE-edit/local/completed-table-flucon-cap3-20261010/server1')
acct=read(snap/'accounting.json');states={s.split('|')[0]:s.split('|')[2] for s in acct['rows'].splitlines() if s}
prior=Path('audits/servers/server1/baseline-completed-zsre-audit-20261010/table-rows.json')
rows=[]
for old in read(prior)['rows']:
    r=dict(old);verify(r['endpoint']);verify(r['terminal']);verify(r['config'])
    assert states[r['job_id']]=='COMPLETED'
    r.update(scheduler_state=states[r['job_id']],reused_audit=member(prior),raw_terminal_SHA_unchanged=True)
    if r['dataset']=='zsre':
        proof=read(verify(r['query_proof']));raw=read(verify(r['endpoint']));p=proof['proof']
        for key in ('frozen_evaluator','frozen_query_module','frozen_public_lock'):verify(proof[key])
        verify(proof['stream']);assert p['requests']==2000 and p['input_mismatches']==p['target_mismatches']==0
        assert len(raw['cases'])==2000 and raw['query_sha256']==p['query_sha256']
        metrics={};counts={}
        for g,label in [('rewrite','Efficacy'),('paraphrase','Generalization'),('neighborhood','Specificity')]:
            rates=[];n=correct=0
            for case in raw['cases']:
                obs=case[g+'_observations'];bits=[x['predicted_token_id']==x['target_token_id'] for x in obs]
                assert bits and bits==case[g+'_prompts_correct'] and bits==[x['correct'] for x in obs]
                rates.append(sum(bits)/len(bits));n+=len(bits);correct+=sum(bits)
            v=100*math.fsum(rates)/2000;assert math.isfinite(v) and abs(v-raw['summary'][label])<1e-10
            assert n==p['token_denominators'][g]==raw['token_denominators'][g]
            metrics[label]=v;counts[g]=dict(requests=2000,tokens=n,correct=correct,missing_requests=0)
        assert metrics==r['metrics'];r.update(metrics=metrics,counts=counts,CPU_independent_reduction='RECOMPUTED_UNCHANGED')
    rows.append(r)
gs=read(B/'flucon-paper-scale-20261010/registration-r1/submission.json')
collector_path=B/'flucon-paper-scale-20261010/registration-r1/collector-results.json'
collector=read(collector_path);assert collector['instruction']==e.INSTRUCTION
collector_by={r['key']:r for r in collector['rows']}
glock=read(verify(gs['execution_lock']))
for m in glock['source_members']:verify(m)
verify(gs['jobs']['collector']['script'])
# The now-completed frozen collector already independently rescored every raw
# case with exact assets. Bind that proof, then recheck original raw hashes,
# completeness, work, counts and aggregation here; do not repeat TF-IDF rescore.
generation=[]
for key,j in gs['jobs'].items():
    if key=='collector':continue
    c=read(verify(j['config']));out=Path(c['output'])
    if not (out/'COMPLETE.json').exists():
        generation.append(dict(job_id=j['job_id'],state=states[j['job_id']],status='NO_COMPLETE_RAW',key=key));continue
    done=read(out/'COMPLETE.json');assert done['source']==gs['source'] and done['config']==j['config']
    checked=collector_by[key]
    assert checked['status']=='CPU_RAW_VERIFIED' and checked['endpoint']==done['endpoint'] and checked['summary']==done['summary']
    verify(c['reference'])
    raw=read_observed(verify(done['endpoint']))
    assert raw['summary']==done['summary'] and raw['summary']['planned_count']==len(raw['rows'])==2000
    assert done['original_checkpoint']==c['original']['checkpoint'] and done['weights_unchanged'] and done['RNG_unchanged']
    gen=dict(job_id=j['job_id'],job_name=j['name'],source=gs['source'],config=j['config'],endpoint=done['endpoint'],terminal=member(out/'COMPLETE.json'),
        status='W20_GENERATION_CPU_RAW_VERIFIED',summary=raw['summary'],collector_verification=member(collector_path),
        collector_source=gs['source'],score_recomputation='REUSED_EXACT_COMPLETED_FROZEN_COLLECTOR_ASSET_BOUND_PROOF',
        Flu=e.display(raw['summary']['ngram_entropy'],'bits'),Con=e.display(raw['summary']['reference_score'],'cosine'))
    matches=[r for r in rows if r['job_id']==c['original']['job_id']];assert len(matches)==1
    matches[0]['generation']=gen;generation.append(gen)
    print('GENERATION_CPU_VERIFIED',j['job_id'],gen['Flu']['paper_display_x100'],gen['Con']['paper_display_x100'],flush=True)
author=read(B/'fe-author-hparams-2k-20261010/registration-r1/submission.json')
for dataset,j in author['jobs'].items():
    c=read(verify(j['config']));out=Path(c['output']);done=out/'COMPLETE.json'
    # This bounded snapshot is not permission to use partially edited checkpoint.
    assert not done.exists(),'NEW_AUTHOR_COMPLETION_REQUIRES_SEPARATE_CPU_VALIDATION'
    commits=sorted((out/'commits').glob('batch-*.json'))
    rows.append(dict(model='llama3',method='MEMIT_FE_HISTORY (FE author hparams)',dataset=dataset,job_id=j['job_id'],job_name=j['name'],
        source=author['source'],config=j['config'],status='INCOMPLETE_NO_W20_CHECKPOINT',scheduler_state=states[j['job_id']],
        complete=False,observed_commits=len(commits),generation='NO_VALID_W20_CP' if dataset=='cf' else 'NOT_APPLICABLE'))
value=dict(nonce=NONCE,server='server1',recorded_at=datetime.now(timezone.utc).isoformat(),rows=rows,generation=generation,
           scheduler_snapshot=member(snap/'inventory.json'),accounting=member(snap/'accounting.json'),
           reducer=member(__file__),query_note='CPU native public-query parity, not pretrained forward parity',
           raw_WB_unchanged=True,checkpoint_loads=0,forward_calls=0,README_owner='GH',broadcast='NO_BROADCAST_NOT_REQUIRED')
write_new(O/'table-rows.json',value)
with (O/'table-rows.csv').open('x',newline='') as f:
    w=csv.writer(f);w.writerow(['model','method','dataset','job_id','state','Eff','Gen','Loc','Score','Flu_x100','Con_x100'])
    for r in rows:
        m=r.get('metrics',r.get('summary',{}));g=r.get('generation');g=g if isinstance(g,dict) else {}
        w.writerow([r['model'],r['method'],r['dataset'],r['job_id'],r['status'],m.get('Efficacy'),m.get('Generalization'),m.get('Specificity'),m.get('Score'),g.get('Flu',{}).get('paper_display_x100'),g.get('Con',{}).get('paper_display_x100')])
print('ROWS',len(rows),flush=True)
