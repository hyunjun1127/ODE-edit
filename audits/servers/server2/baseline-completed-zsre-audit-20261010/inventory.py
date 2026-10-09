"""Bounded main-table inputs from completed raw; no GPU or CP deserialize."""
import csv,json,math,subprocess
from datetime import datetime,timezone
from pathlib import Path
from official.experiments.prepare import read,file_sha,digest,write_new
from official.evaluation.reduce import counterfact
from official.runners.server2.zsre_reeval import validate_result
from official.runners.server2.qwen_mask_ft_eval import COUNTS
BASE=Path('/mnt/raid5/janghj/ODE-edit/local')
FIRST=BASE/'qwen-baselines-server2-20261009/registration-r1'
OLD=FIRST.parent/'registration-native-eval-r2'
NEW=BASE/'qwen-baseline-mask-cold-rerun-20261010/registration-r1'
GJ=BASE/'official-baselines/server2/zsre-2k-reeval-20261009/registration-r1'
OUT=Path(__file__).resolve().parent
def check(x,c):
    if not x:raise ValueError(c)
def member(p):return dict(path=str(p),sha256=file_sha(p),bytes=Path(p).stat().st_size)
def verified(m):check(file_sha(m['path'])==m['sha256'],'RAW_SHA');return read(m['path'])
def order(cases,stream):
    check(len(cases)==2000 and [(c['occurrence_index'],c['case_id']) for c in cases]==[(r['occurrence_index'],r['case_id']) for r in stream],'ORDERED_2000')
    return digest([(c['occurrence_index'],c['case_id']) for c in cases])
def public(cases):
    values={};counts={}
    for group,label in [('rewrite','Efficacy'),('paraphrase','Generalization'),('neighborhood','Specificity')]:
        means=[];n=0
        for c in cases:
            obs=c[group+'_observations'];bits=[r['predicted_token_id']==r['target_token_id'] for r in obs]
            check(bits and bits==c[group+'_prompts_correct'] and bits==[r['correct'] for r in obs],'TOKEN_BITS')
            means.append(math.fsum(bits)/len(bits));n+=len(bits)
        values[label]=100*math.fsum(means)/2000;counts[group]=n
    return values,counts
def close(a,b):
    for k,v in a.items():check(math.isfinite(v) and abs(v-b[k])<1e-10,'SUMMARY_'+k)

ids=['61898','61962','61964','62072']+[str(n) for n in range(61942,61948)]+[str(n) for n in range(62073,62088,2)]
accounting=subprocess.check_output(['sacct','-X','-n','-P','-j',','.join(ids),'--format=JobIDRaw,JobName%100,User,State,NodeList,WorkDir%400'],text=True)
states={p[0]:p for p in (l.split('|') for l in accounting.splitlines())}
stamp=datetime.now(timezone.utc).isoformat();rows=[]
def initial(job,model,ds,method):
    s=states[job];check(s[2]=='janghj','OWNER')
    return dict(job_id=job,job_name=s[1],observed_state=s[3],model=model,dataset=ds,method=method,
        server='server2',observed_at=stamp,numeric_eligible=False,metrics={},status=s[3],online_delivery='NOT_REQUERIED')

for root,job,cell in [(FIRST,'61898','qwen25-cf-ft'),(OLD,'61962','qwen25-cf-alphaedit_blue'),(OLD,'61964','qwen25-zsre-alphaedit_blue')]:
    config=read(root/'configs'/f'{cell}.json');row=initial(job,'qwen25',config['dataset'],config['method'])
    row.update(source_commit=read(root/'source-lock.json')['code_commit'],config_sha256=config['config_sha256'],rerun_attempt=str(root))
    p=root/'runs'/cell/'terminal.json'
    if p.exists():
        try:
            t=read(p);check(t['actual_job_id']==job and t['completed_edits']==2000 and t['status']=='W20_COMPLETE','TERMINAL')
            check(t['config_sha256']==config['config_sha256'] and len(t['commits'])==20,'CONFIG_COMMITS')
            for i,m in enumerate(t['commits'],1):
                c=verified(m);check(c['completed_batch']==i and c['code_commit']==row['source_commit'] and c['checkpoint_identity']==t['checkpoint_identity'],'COMMIT')
            ref=t['calculation_evidence']['factual'];cases=verified(ref);stream=read(root/'streams'/f"{config['dataset']}-stream.json")
            cohort=order(cases,stream)
            if config['dataset']=='cf':
                values=counterfact(cases);den={g:sum(len(c[g+'_prompts_probs']) for c in cases) for g in ('rewrite','paraphrase','neighborhood')}
                genref=t['calculation_evidence']['generation'];gen=verified(genref)
                check(len(gen['rows'])==2000 and [r['occurrence'] for r in gen['rows']]==list(range(1,2001)),'GENERATION_COHORT')
                check([r['case_id'] for r in gen['rows']]==[r['case_id'] for r in stream],'GENERATION_ORDER')
                check(gen['identity_sha256']==c['generation']['identity_sha256'],'GENERATION_IDENTITY')
                scores={}
                for field,valid,name in [('ngram_entropy','fluency_valid','Flu'),('reference_score','consistency_valid','Con')]:
                    measured=[r['metrics'][field] for r in gen['rows'] if r['metrics'][valid]]
                    check(measured and all(math.isfinite(v) for v in measured),'GENERATION_FINITE')
                    scores[name]=math.fsum(measured)/len(measured)
                    check(abs(scores[name]-gen['summary'][field])<1e-10,'GENERATION_REDUCER')
                    den[name]=len(measured)
                row.update(**scores,generation_raw=genref,generation_requests=2000,
                    generation_prompts=sum(r['metrics']['generation_prompt_count'] for r in gen['rows']),
                    generation_tokens=sum(r['metrics']['generated_token_count'] for r in gen['rows']))
            else:
                values,den=public(cases);check(den==c['factual']['work']['token_denominators'],'QUERY_DENOMINATOR')
                check(c['factual']['work']['query_sha256']==read(root/'query-parity.json')['query_sha256'],'QUERY_IDENTITY')
            close(values,c['factual']['summary'])
            row.update(metrics=values,denominators=den,numeric_eligible=True,status='W20_RAW_CPU_VERIFIED',requests=2000,
                raw=ref,terminal=member(p),ordered_cohort_sha256=cohort,cold_state_identity=t['checkpoint_identity'],
                stream_sha256=t['checkpoint_identity']['stream_sha256'],ordered_sample_sha256=read(root/'streams'/f"{config['dataset']}-stream.lock.json")['ordered_case_ids_sha256'])
        except Exception as e:row.update(status='RAW_VALIDATION_FAILED',error=str(e))
    rows.append(row)

for root,method,job,model in [(NEW,'FT','62072','qwen25')]+[(GJ,m,str(j),'gptj') for m,j in zip(('FT','MEMIT','ALPHAEDIT','ALPHAEDIT_BLUE','MEMIT_FE','SPHERE'),range(61942,61948))]:
    row=initial(job,model,'zsre',method);p=root/'ft-eval/result.json' if root==NEW else root/'outputs'/method/'result.json'
    if p.exists():
        try:
            r=read(p);raw=verified(r['raw']);inputs=read(root/('ft-eval-inputs.json' if root==NEW else 'inputs.json'))
            values=validate_result(raw,inputs,model_family=model,counts=COUNTS if model=='qwen25' else None)
            check(r['job_id']==job and r['status']=='EVAL_COMPLETE_W20_2000','EVAL_FINAL')
            cohort=order(raw['cases'],verified(inputs['stream']))
            row.update(metrics=values,denominators=raw['token_denominators'],numeric_eligible=True,status='W20_PUBLIC_QUERY_CPU_VERIFIED',
                source_commit=r['source'],config_sha256=r['config_sha256'],raw=r['raw'],terminal=member(p),
                ordered_cohort_sha256=cohort,stream_sha256=inputs['stream']['sha256'],query_sha256=raw['query_sha256'],requests=2000,
                rerun_attempt=str(root),checkpoint_provenance=r.get('original_checkpoint',r.get('checkpoint')))
        except Exception as e:row.update(status='RAW_VALIDATION_FAILED',error=str(e))
    rows.append(row)
for j in read(NEW/'released.json')['jobs']:
    if j['kind']!='gpu' or j['cell']=='qwen25-zsre-ft-eval':continue
    cfg=read(NEW/'configs'/f"{j['cell']}.json");row=initial(j['job_id'],'qwen25',cfg['dataset'],cfg['method'])
    row.update(source_commit=j['source'],config_sha256=cfg['config_sha256'],generation_schedule=cfg.get('generation_schedule','NOT_APPLICABLE'))
    rows.append(row)
write_new(OUT/'table-rows.json',dict(at=stamp,rows=rows,accounting=accounting,reducer_sha256=file_sha(__file__),
    GPU=0,checkpoint_loads=0,online_history_changes=0,GH_sole_README=True))
with (OUT/'table-rows.csv').open('x',newline='') as f:
    w=csv.writer(f,lineterminator='\n');w.writerow(['model','dataset','method','job_id','job_name','state','eligible','Eff','Gen','Loc','Score','Flu','Con'])
    for r in rows:
        m=r['metrics'];w.writerow([r['model'],r['dataset'],r['method'],r['job_id'],r['job_name'],r['status'],r['numeric_eligible'],m.get('Efficacy',''),m.get('Generalization',''),m.get('Specificity',''),m.get('Score',''),r.get('Flu',''),r.get('Con','')])
print(json.dumps([(r['job_id'],r['status']) for r in rows]))
