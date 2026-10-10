"""One bounded CPU/raw and admission audit. No Slurm mutation or GPU execution."""
import csv,json,math,subprocess,copy
from datetime import datetime,timezone
from pathlib import Path
from official.experiments.prepare import read,file_sha,digest,write_new
from official.runners.server2 import submit
from official.runners.server2.fe_author_submit import graph
from official.evaluation.generation.paper_display import paper_cell

OUT=Path(__file__).resolve().parent
REPO=OUT.parents[3]
BASE=Path('/mnt/raid5/janghj/ODE-edit/local')
NEW=BASE/'qwen-baseline-mask-cold-rerun-20261010/registration-r1'
GJ=BASE/'official-baselines/server2/zsre-2k-reeval-20261009/registration-r1'
NONCE='USER-GH-S1-S2-COMPLETED-TABLE-S1-FLUCON-CAP3-20261010-R1'
# Reuse only the already reviewed independent reducer/source-proof definitions;
# its top-level old snapshot/publication code must not execute.
definition=REPO/'audits/servers/server2/baseline-completed-zsre-audit-20261010/audit.py'
ns={'__file__':str(definition)}
exec(compile(definition.read_text().split('\ndata=read(')[0],str(definition),'exec'),ns)
verified,member,check,order,independent,close,source_proof=(ns[k] for k in ('verified','member','check','order','independent','close','source_proof'))
stamp=datetime.now(timezone.utc).isoformat()
previous=read(definition.parent/'audited-final.json');rows=copy.deepcopy(previous['rows'])
rows=[r for r in rows if r['job_id']!='62087'] # Failed ancestor is not a separate main result.
for job,ds,method in [('62531','cf','MEMIT_FE_HISTORY (FE author hparams)'),('62532','zsre','MEMIT_FE_HISTORY (FE author hparams)'),('62538','zsre','SPHERE')]:
    rows.append(dict(job_id=job,model='qwen25',dataset=ds,method=method,server='server2',numeric_eligible=False,metrics={}))
accounting=subprocess.check_output(['sacct','-X','-nP','-j',','.join(r['job_id'] for r in rows),
    '--format=JobIDRaw,JobName%100,User,State,NodeList,WorkDir%400'],text=True)
states={p[0]:p for p in (line.split('|') for line in accounting.splitlines())}
issues=[]
for row in rows:
    s=states[row['job_id']];check(s[2]=='janghj' and s[4] in ('server2','None assigned',''),'OWNER_NODE')
    row.update(job_name=s[1],observed_state=s[3],observed_at=stamp,report_path='experiment-reports/servers/server2/completed-table-flucon-cap3-20261010/report-ko.md')
    if not row['numeric_eligible']:
        row.update(status=s[3],metrics={})
        continue
    try:
        verified(row['raw']);verified(row['terminal'])
        row.update(reused_prior_CPU_audit=member(definition.parent/'audited-final.json'),raw_and_terminal_SHA_rechecked=True)
        # zsRE recomputation from stored prediction/target IDs, never W0 agreement.
        if row['dataset']=='zsre':
            raw=verified(row['raw']);cases=raw['cases'] if isinstance(raw,dict) else raw
            values,den,micro=independent(cases);close(values,row['metrics'])
            check(den==row['denominators'] and len(cases)==2000,'REUSED_DENOMINATORS')
            row['independent_CPU_recomputed']=values
    except Exception as error:
        row.update(numeric_eligible=False,status='UNVERIFIED',error=str(error));issues.append(dict(job=row['job_id'],error=str(error)))

def checkpoint_stat(path,sha):
    p=Path(path);s=p.stat()
    check(p.is_file() and not p.is_symlink() and s.st_size>0,'CHECKPOINT_FILE')
    return dict(path=str(p),bytes=s.st_size,device=s.st_dev,inode=s.st_ino,mtime_ns=str(s.st_mtime_ns),
                manifest_sha256=sha,full_payload_rehashed_this_audit=False,checkpoint_deserialized=False)

for row in rows:
    if row['job_id'] not in ('61947','62081'):continue
    try:
        if row['job_id']=='61947':
            root=GJ;p=root/'outputs/SPHERE/result.json';r=read(p);full=verified(r['raw']);cases=full['cases']
            inputs=read(root/'inputs.json');proof=inputs['query_proof'];stream=verified(inputs['stream'])
            check(r['job_id']=='61947' and r['status']=='EVAL_COMPLETE_W20_2000','EVAL_TERMINAL')
            check(full['model_no_mutation'] and full['RNG_restored'] and r['original_CP_unchanged'],'EVAL_GUARDS')
            identity=full['identity'];tok=identity['original_identity']['tokenizer_sha256']
            query=full['query_sha256'];summary=full['summary'];raw_member=r['raw']
            check(identity['evaluator_sha256']==ns['EVAL'],'EVAL_SOURCE_BINDING')
            row.update(source_commit=r['source'],config_sha256=r['config_sha256'],original_edit_job_id=identity['original_job_id'],
                       checkpoint_provenance=r['checkpoint'],checkpoint_stat=checkpoint_stat(r['checkpoint']['path'],r['checkpoint']['sha256']),
                       eval_only_not_new_chain=True)
        else:
            root=NEW;out=root/'runs/qwen25-zsre-memit';p=out/'terminal.json';r=read(p)
            check(r['actual_job_id']=='62081' and r['status']=='W20_COMPLETE' and r['completed_edits']==2000,'EDIT_TERMINAL')
            check(len(r['commits'])==20,'20_COMMITS')
            for n,m in enumerate(r['commits'],1):
                c=verified(m);check(c['completed_batch']==n and c['checkpoint_identity']==r['checkpoint_identity'],'COMMIT_CHAIN')
            pointer=read(out/'checkpoint/latest.json');check(pointer==r['checkpoint'] and pointer['batch']==20 and pointer['final_W20'],'W20_POINTER')
            check(c['checkpoint_sha256']==pointer['sha256'],'CP_COMMIT_BINDING')
            raw_member=r['calculation_evidence']['factual'];cases=verified(raw_member)
            stream=read(root/'streams/zsre-stream.json');proof=read(root/'query-parity.json')
            identity=r['checkpoint_identity'];tok=identity['tokenizer_sha256'];query=c['factual']['work']['query_sha256'];summary=c['factual']['summary']
            check(c['factual']['work']['evaluation_profile']=='zsre-public-query-v1','PUBLIC_QUERY_PROFILE')
            check(file_sha(root/'configs/qwen25-zsre-memit.json')==file_sha(Path(r['commits'][0]['path']).parents[2].parent/'configs/qwen25-zsre-memit.json'),'CONFIG_PATH')
            row.update(source_commit=identity['code_commit'],config_sha256=identity['config_sha256'],commits=20,
                       cold_state_identity=identity,checkpoint_receipt=pointer,
                       checkpoint_stat=checkpoint_stat(out/'checkpoint'/pointer['file'],pointer['sha256']))
        order(cases,stream);check(query==proof['query_sha256'],'QUERY_PROOF_BINDING')
        values,den,micro=independent(cases);close(values,summary);check(den==proof['token_denominators'],'PUBLIC_DENOMINATOR')
        if 'Specificity_loc_ans' in summary:check(abs(summary['Specificity_loc_ans']-values['Specificity'])<1e-10,'LOC_ALIAS')
        evidence=source_proof(root,proof,stream,tok)
        row.update(metrics=values,denominators=den,numeric_eligible=True,status='W20_PUBLIC_QUERY_REQUEST_MACRO_VERIFIED',
                   raw=raw_member,terminal=member(p),requests=2000,zsre_audit=evidence,
                   query_sha256=query,ordered_cohort_sha256=digest([(c['occurrence_index'],c['case_id']) for c in cases]),
                   diagnostic_token_micro_NOT_TABLE=micro,new_since_previous_audit=True,rerun_attempt=str(root))
    except Exception as error:
        row.update(numeric_eligible=False,status='UNVERIFIED',error=str(error));issues.append(dict(job=row['job_id'],error=str(error)))

for row in rows:
    for metric,unit in [('Flu','bits'),('Con','cosine_0_to_1')]:
        value=row.get(metric)
        if isinstance(value,(int,float)):
            row[metric+'_raw_unit']=unit;row[metric+'_paper_x100']=paper_cell(value,metric=metric,raw_unit=unit)
        elif row['dataset']=='cf':row.setdefault(metric,'DEFERRED')
    if row['job_id']=='62538':row.update(resume_parent_job='62087',parent_batches=list(range(1,10)),child_batches=list(range(10,21)),
        ancestry_status='W20_NOT_YET_OBSERVED',parent_SHA='7e44f382ba1f3befcb9f4429f42fd328d55422306f9cef868a920c8bf7ec17b8')

live=submit.inventory();check(not live['ambiguous'],'UNCLASSIFIED_ADMISSION')
frontier,width=graph(live);allocated=sum(r['allocated_GPUs'] for r in live['project'])
check(width<=3 and allocated<=3,'CAP3_PENDING_RESOURCE_ACTION_REQUIRED')
caps=[Path('/mnt/raid5/janghj/ODE-edit/servers/local/gpu-caps.tsv'),REPO/'servers/local/gpu-caps.tsv']
for p in caps:check(p.read_text()=='server2\tserver2\t3\t60416\todeedit_*,odealloc_*\n','LOCAL_CAP_OTHER_FIELDS_PRESERVED')
write_new(OUT/'cap-receipt.json',dict(nonce=NONCE,at=stamp,accepted_turn='01a1236d-9585-7183-88d1-d29261b81fcc',
    prior_effective_cap=4,effective_cap=3,local_caps=[member(p) for p in caps],allocated_GPU=allocated,DAG_width=width,
    frontier=frontier,inventory=live,pending_dependency_changes=0,holds=0,cancels=0,new_GPU_jobs=0,
    reason='Existing exact dependency DAG already satisfies cap3; source/config untouched.',other_servers_changed=False))
data=dict(nonce=NONCE,observed_at=stamp,rows=rows,issues=issues,numeric_eligible=sum(r['numeric_eligible'] for r in rows),
    accounting=accounting,reducer=member(__file__),reused_reducer=member(definition),
    excluded_broken_context=[61956,61960,61968],old_failed_ancestor_not_separate_main=62087,
    source_model_GPU_checks='CPU_RAW_ONLY_NO_FORWARD',checkpoint_loads=0,GH_sole_README=True,
    broadcast='NO_BROADCAST_NOT_REQUIRED',monitoring_active=False)
write_new(OUT/'table-rows.json',data)
with (OUT/'table-rows.csv').open('x',newline='') as f:
    w=csv.writer(f);w.writerow(['model','dataset','method','job_id','name','state','eligible','Eff','Gen','Loc','Score','Flu_x100','Con_x100'])
    for r in rows:
        m=r['metrics'] if r['numeric_eligible'] else {}
        w.writerow([r['model'],r['dataset'],r['method'],r['job_id'],r['job_name'],r['status'],r['numeric_eligible'],m.get('Efficacy',''),m.get('Generalization',''),m.get('Specificity',''),m.get('Score',''),r.get('Flu_paper_x100',r.get('Flu','NOT_APPLICABLE')),r.get('Con_paper_x100',r.get('Con','NOT_APPLICABLE'))])
print(json.dumps(dict(eligible=data['numeric_eligible'],issues=issues,allocated=allocated,width=width,
    new=[{k:r.get(k) for k in ('job_id','status','metrics','denominators')} for r in rows if r.get('new_since_previous_audit')]),indent=2))
