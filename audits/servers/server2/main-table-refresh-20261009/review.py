"""One bounded own-job snapshot + read-only independent W20 reductions."""
from pathlib import Path
import json,csv,math,subprocess,hashlib
from datetime import datetime,timezone
from official.experiments.prepare import read,file_sha,digest,write_new
from official.evaluation.reduce import counterfact,zsre

BASE=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
OLD=Path('/mnt/raid5/janghj/ODE-edit/local/qwen-baselines-server2-20261009/registration-r1')
NEW=OLD.parent/'registration-native-eval-r2'
EVAL=Path('/mnt/raid5/janghj/ODE-edit/local/official-baselines/server2/zsre-2k-reeval-20261009/registration-r1')
NONCE='USER-GH-ALL-SH-COMPLETED-MAIN-TABLE-REFRESH-20261009-R1-SERVER2'
def check(x,c):
    if not x:raise ValueError(c)
def member(p):
    p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=file_sha(p))
def verify(m):
    p=Path(m['path']);check(file_sha(p)==m['sha256'],'MEMBER_SHA')
    if 'bytes' in m:check(p.stat().st_size==m['bytes'],'MEMBER_BYTES')
    return read(p)
def order(cases):
    check(len(cases)==2000 and [c['occurrence_index'] for c in cases]==list(range(1,2001)),'ORDER_COUNT')
    return digest([(c['occurrence_index'],c['case_id']) for c in cases])
def public_reduce(cases):
    values={};den={}
    for g,label in [('rewrite','Efficacy'),('paraphrase','Generalization'),('neighborhood','Specificity')]:
        means=[];n=0
        for c in cases:
            o=c[g+'_observations'];bits=[x['predicted_token_id']==x['target_token_id'] for x in o]
            check(bool(bits) and bits==c[g+'_prompts_correct'] and bits==[x['correct'] for x in o],'TOKEN_BITS')
            n+=len(bits);means.append(math.fsum(bits)/len(bits))
        values[label]=100*math.fsum(means)/2000;den[g]=n
    return values,den
def close(values,summary):
    for k,v in values.items():check(math.isfinite(v) and abs(v-summary[k])<1e-10,'SUMMARY_'+k)

prior=read(BASE/'audits/servers/server2/qwen-migration-results-20261009/gptj-results.json')['rows']
qjobs=[(OLD,j) for j in read(OLD/'released.json')['jobs'] if j['job_id'] in ('61898','61900')]
qjobs += [(NEW,j) for j in read(NEW/'released.json')['jobs'] if j['kind']=='gpu']
ejobs=[j for j in read(EVAL/'released.json')['jobs'] if j.get('method')!='collector']
ids=sorted({r['job_id'] for r in prior}|{j['job_id'] for _,j in qjobs}|{j['job_id'] for j in ejobs},key=int)
check(not (OUT/'snapshot.json').exists(),'BOUNDED_SNAPSHOT_ALREADY_EXISTS')
stamp=datetime.now(timezone.utc).isoformat()
account=subprocess.check_output(['sacct','-X','-P','-n','-j',','.join(ids),'--format=JobIDRaw,JobName%100,User,State,ExitCode,NodeList,WorkDir%500,Elapsed'],text=True,timeout=60)
queue=subprocess.check_output(['squeue','-j',','.join(ids),'-h','-o','%i|%j|%T|%E'],text=True,timeout=30)
states={}
for line in account.splitlines():
    parts=line.split('|');jid=parts[0]
    if jid in ids:
        check(parts[2]=='janghj','OWNER')
        states[jid]=dict(job_name=parts[1],observed_state=parts[3],exit_code=parts[4],node=parts[5],workdir=parts[6],elapsed=parts[7])
write_new(OUT/'snapshot.json',dict(at=stamp,accounting=account,queue=queue,job_ids=ids,queries=2,mutations=0))
rows=[];history=[]
def initial(model,ds,method,job):
    check(job in states,'ACCOUNTING_MISSING_'+job)
    return dict(server='server2',model=model,dataset=ds,method=method,job_id=job,observed_at=stamp,
        **states[job],numeric_eligible=False,metrics={},status=states[job]['observed_state'],
        online_delivery='NOT_QUERIED',checkpoint_payload_loaded=False)

for old in prior:
    row=initial('gptj',old['dataset'],old['method'],old['job_id'])
    try:
        result=verify(old['result']);raw=verify(old['W20_raw'])
        check(result['status']=='SCIENTIFIC_COMPLETE' and result['batches']==20 and len(result['commits'])==20,'W20_TERMINAL')
        check(result['ownstate_links']==19,'STATE_LINKS')
        for n,m in enumerate(result['commits'],1):
            c=verify(m);check(c['batch']==n and c['requests']==100 and c['checkpoint_identity']==result['checkpoint_identity'],'COMMIT')
        check(raw['endpoint']=='W20' and raw['requests']==2000 and raw['identity_sha256']==digest(raw['identity']),'RAW_IDENTITY')
        cohort=order(raw['cases']);check(cohort==old['ordered_cohort_sha256'],'COHORT')
        identity=result['checkpoint_identity']
        row.update(source_commit=result['code_commit'],config_sha256=identity['config_sha256'],
            ordered_cohort_sha256=cohort,stream_sha256=identity['stream_sha256'],cold_state_identity=identity,
            rerun_attempt=str(Path(old['result']['path']).parent),raw=old['W20_raw'],terminal=old['result'],commits=20)
        if old['dataset']=='cf':
            values=counterfact(raw['cases']);close(values,raw['summary'])
            row.update(status='W20_FACTUAL_CPU_VERIFIED',numeric_eligible=True,metrics=values,
                old_metrics=old['summary'],delta={k:values[k]-old['summary'][k] for k in values},Flu='DEFERRED',Con='DEFERRED',
                denominators={g:sum(len(c[g+'_prompts_probs']) for c in raw['cases']) for g in ('rewrite','paraphrase','neighborhood')})
        else:row.update(status='OLD_TOKEN_PREFIX_W20_REEVALUATION_REQUIRED',reason='Not public-query final metrics; no relabel')
    except Exception as e:row.update(status='RAW_VALIDATION_FAILED',error=str(e))
    (rows if old['dataset']=='cf' else history).append(row)

for j in ejobs:
    method=j['method'];row=initial('gptj','zsre',method,j['job_id'])
    row.update(source_commit=j['source'],config_sha256=j['config_sha256'],rerun_attempt=str(EVAL),
        status='REEVALUATION_'+row['observed_state'])
    p=EVAL/'outputs'/method/'result.json'
    if p.exists():
        try:
            r=read(p);raw=verify(r['raw']);values,den=public_reduce(raw['cases']);close(values,raw['summary'])
            check(r['status']=='EVAL_COMPLETE_W20_2000' and r['job_id']==j['job_id'] and r['source']==j['source'],'EVAL_TERMINAL')
            check(raw['query_sha256']==read(EVAL/'inputs.json')['query_proof']['query_sha256'],'QUERY')
            row.update(status='W20_PUBLIC_QUERY_CPU_VERIFIED',numeric_eligible=True,metrics=values,denominators=den,
                raw=r['raw'],terminal=member(p),ordered_cohort_sha256=order(raw['cases']),cold_state_identity=r['identity'])
        except Exception as e:row.update(status='RAW_VALIDATION_FAILED',error=str(e))
    rows.append(row)

for root,j in qjobs:
    cell=j['cell'];config=read(root/'configs'/(cell+'.json'));ds=config['dataset']
    row=initial('qwen25',ds,config['method'],j['job_id'])
    check(row['job_name']==j['name'] and row['workdir']==str(root),'QWEN_SCHEDULER_IDENTITY')
    sl=read(root/'streams'/(ds+'-stream.lock.json'))
    row.update(source_commit=j['source'],config_sha256=config['config_sha256'],rerun_attempt=str(root),
        stream_sha256=sl['stream_sha256'],ordered_sample_sha256=sl['ordered_case_ids_sha256'],
        generation_schedule='W0_AND_W20_FIRST2000' if ds=='cf' else 'NOT_APPLICABLE')
    p=root/'runs'/cell/'terminal.json'
    if p.exists():
        try:
            terminal=read(p)
            check(terminal['status']=='W20_COMPLETE' and terminal['completed_edits']==2000 and len(terminal['commits'])==20,'W20_TERMINAL')
            check(terminal['actual_job_id']==j['job_id'] and terminal['config_sha256']==config['config_sha256'],'JOB_CONFIG')
            for n,m in enumerate(terminal['commits'],1):
                c=verify(m);check(c['completed_batch']==n and c['code_commit']==j['source'] and c['checkpoint_identity']==terminal['checkpoint_identity'],'COMMIT_IDENTITY')
            last=c;ref=terminal['calculation_evidence']['factual'];cases=verify(ref)
            stream=read(root/'streams'/(ds+'-stream.json'))
            if isinstance(stream,dict):stream=stream['records']
            check([(c['occurrence_index'],c['case_id']) for c in cases]==[(c['occurrence_index'],c['case_id']) for c in stream],'STREAM_ORDER')
            row.update(raw=ref,terminal=member(p),commits=20,ordered_cohort_sha256=order(cases),cold_state_identity=terminal['checkpoint_identity'])
            if ds=='zsre' and root==NEW:
                work=last['factual']['work'];check(work['evaluation_profile']=='zsre-public-query-v1','PROFILE')
                check(work['query_sha256']==read(root/'query-parity.json')['query_sha256'],'QUERY')
                values,den=public_reduce(cases);check(den==work['token_denominators'],'DENOMINATORS');close(values,last['factual']['summary'])
                row.update(numeric_eligible=True,status='W20_PUBLIC_QUERY_CPU_VERIFIED',metrics=values,denominators=den,
                    query_sha256=work['query_sha256'],old_metrics='PENDING',delta='NEWLY_OBSERVED_NO_PREVIOUS_PUBLIC_SCORE')
            elif ds=='zsre':row.update(status='OLD_TOKEN_PREFIX_W20_REEVALUATION_REQUIRED',reason='FT protected original source; do not promote old query metrics')
            else:row.update(status='W20_REQUIRES_GENERATION_RAW_REDUCTION',reason='No partial generation promotion')
        except Exception as e:row.update(status='RAW_VALIDATION_FAILED',error=str(e))
    rows.append(row)

value=dict(nonce=NONCE,observed_at=stamp,policy_sha256=file_sha(BASE/'control/main-results-policy.json'),
    rows=rows,historical_edit_completion_inventory=history,reducer_sha256=file_sha(__file__),
    numeric_eligible=sum(r['numeric_eligible'] for r in rows),new_Qwen_public=sum(r['numeric_eligible'] and r['model']=='qwen25' for r in rows),
    GPU=0,model_loads=0,job_mutations=0,online_history_mutations=0,raw_CP_KEEP=True,
    exclusions='PRICE/tuning/history variants not promoted; other-server replicas excluded',broadcast='NO_BROADCAST_NOT_REQUIRED')
write_new(OUT/'table-rows.json',value)
with (OUT/'table-rows.csv').open('x',newline='') as f:
    w=csv.writer(f);w.writerow(['model','dataset','method','job_id','job_name','scheduler_state','result_status','eligible','Eff','Gen','Loc','Score','Flu','Con','source','config_sha256'])
    for r in rows:
        m=r['metrics'];w.writerow([r['model'],r['dataset'],r['method'],r['job_id'],r['job_name'],r['observed_state'],r['status'],r['numeric_eligible'],m.get('Efficacy',''),m.get('Generalization',''),m.get('Specificity',''),m.get('Score',''),r.get('Flu',''),r.get('Con',''),r.get('source_commit',''),r.get('config_sha256','')])
print(json.dumps(dict(at=stamp,eligible=value['numeric_eligible'],new_Qwen=value['new_Qwen_public'],rows=[{k:r[k] for k in ('model','dataset','method','job_id','status','metrics')} for r in rows]),indent=2))
