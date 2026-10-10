"""Bounded accounting plus saved observation CPU reduction, no model/CP load."""
import csv,copy,json,subprocess
from datetime import datetime,timezone
from pathlib import Path
from official.experiments.prepare import read,write_new,digest,file_sha
from official.evaluation.reduce import counterfact
OUT=Path(__file__).resolve().parent;REPO=OUT.parents[3]
definition=REPO/'audits/servers/server2/baseline-completed-zsre-audit-20261010/audit.py'
ns={'__file__':str(definition)}
exec(compile(definition.read_text().split('\ndata=read(')[0],str(definition),'exec'),ns)
verified,member,check,order,independent,close,source_proof=(ns[k] for k in ('verified','member','check','order','independent','close','source_proof'))
previous=REPO/'audits/servers/server2/completed-table-flucon-cap3-20261010/table-rows.json'
rows=copy.deepcopy(read(previous)['rows']);stamp=datetime.now(timezone.utc).isoformat();issues=[]
accounting=subprocess.check_output(['sacct','-X','-nP','-j',','.join(r['job_id'] for r in rows),
 '--format=JobIDRaw,JobName%100,User,State,NodeList,WorkDir%400'],text=True)
states={p[0]:p for p in (line.split('|') for line in accounting.splitlines())}
root=Path('/mnt/raid5/janghj/ODE-edit/local/qwen-baseline-mask-cold-rerun-20261010/registration-r1')
for r in rows:
    s=states[r['job_id']];check(s[2]=='janghj' and s[4] in ('server2','None assigned',''),'OWNER_NODE')
    r.update(job_name=s[1],observed_state=s[3],observed_at=stamp)
    try:
        if r['numeric_eligible']:
            raw=verified(r['raw']);verified(r['terminal'])
            cases=raw['cases'] if isinstance(raw,dict) else raw
            if r['dataset']=='zsre':
                values,den,micro=independent(cases);close(values,r['metrics'])
                r.update(independent_CPU_recomputed=values,denominators=den)
            else:
                values=counterfact(cases);close(values,r['metrics'])
            r.update(raw_terminal_SHA_rechecked=True,previous_audit=member(previous))
        elif r['job_id'] in ('62075','62083','62085'):
            folder=root/'runs'/f"qwen25-{r['dataset']}-{r['method'].lower()}"
            terminal=folder/'terminal.json'
            if not terminal.exists():r.update(status=s[3]);continue
            t=read(terminal);check(t['actual_job_id']==r['job_id'] and t['status']=='W20_COMPLETE' and t['completed_edits']==2000,'FINAL_TERMINAL')
            check(len(t['commits'])==20,'COMMIT_COUNT')
            for n,m in enumerate(t['commits'],1):
                c=verified(m);check(c['completed_batch']==n and c['checkpoint_identity']==t['checkpoint_identity'],'COMMIT_IDENTITY')
            pointer=read(folder/'checkpoint/latest.json')
            check(pointer==t['checkpoint'] and pointer['final_W20'] and pointer['batch']==20,'FINAL_POINTER')
            check(pointer['sha256']==c['checkpoint_sha256'],'FINAL_COMMIT_CP')
            check((folder/'checkpoint'/pointer['file']).is_file(),'FINAL_PAYLOAD_PRESENT')
            raw=verified(t['calculation_evidence']['factual']);stream=read(root/'streams'/f"{r['dataset']}-stream.json")
            order(raw,stream)
            if r['dataset']=='zsre':
                values,den,micro=independent(raw);proof=read(root/'query-parity.json')
                check(c['factual']['work']['query_sha256']==proof['query_sha256'],'ACTUAL_QUERY_BINDING')
                check(den==proof['token_denominators'],'TOKEN_DENOMINATORS')
                evidence=source_proof(root,proof,stream,t['checkpoint_identity']['tokenizer_sha256'])
                r.update(zsre_audit=evidence,diagnostic_token_micro_NOT_TABLE=micro)
            else:
                values=counterfact(raw);den={g:sum(len(x[g+'_prompts_probs']) for x in raw) for g in ('rewrite','paraphrase','neighborhood')}
            close(values,c['factual']['summary'])
            r.update(numeric_eligible=True,status='W20_CPU_RAW_VERIFIED',metrics=values,denominators=den,
                raw=t['calculation_evidence']['factual'],terminal=member(terminal),requests=2000,commits=20,
                checkpoint_receipt=pointer,source_commit=t['checkpoint_identity']['code_commit'],
                config_sha256=t['config_sha256'],ordered_cohort_sha256=digest([(x['occurrence_index'],x['case_id']) for x in raw]),
                new_since_previous_audit=True)
        else:r.update(status=s[3])
    except Exception as e:r.update(numeric_eligible=False,status='UNVERIFIED',error=str(e));issues.append(dict(job=r['job_id'],error=str(e)))
    r['report_path']='experiment-reports/servers/server2/baseline-refresh-s2-flucon-20261010/report-ko.md'
data=dict(nonce='USER-GH-S1-S2-BASELINE-REFRESH-S2-FLUCON-20261010-R1',accepted_turn='01a124db-95bd-7231-981d-429bbc98305c',
    at=stamp,rows=rows,numeric_eligible=sum(r['numeric_eligible'] for r in rows),issues=issues,accounting=accounting,
    reducer=member(__file__),reused_reducer=member(definition),GPU=0,checkpoint_loads=0,GH_sole_README=True,
    broadcast='NO_BROADCAST_NOT_REQUIRED')
write_new(OUT/'table-rows.json',data)
with (OUT/'table-rows.csv').open('x',newline='') as f:
    w=csv.writer(f,lineterminator='\n');w.writerow(['model','dataset','method','job_id','job_name','state','eligible','Eff','Gen','Loc','Score','Flu_x100','Con_x100'])
    for r in rows:
        m=r['metrics'] if r['numeric_eligible'] else {}
        w.writerow([r['model'],r['dataset'],r['method'],r['job_id'],r['job_name'],r['status'],r['numeric_eligible'],m.get('Efficacy',''),m.get('Generalization',''),m.get('Specificity',''),m.get('Score',''),r.get('Flu_paper_x100',r.get('Flu','NOT_APPLICABLE')),r.get('Con_paper_x100',r.get('Con','NOT_APPLICABLE'))])
print(json.dumps(dict(eligible=data['numeric_eligible'],issues=issues,new=[r['job_id'] for r in rows if r.get('new_since_previous_audit')]),indent=2))
