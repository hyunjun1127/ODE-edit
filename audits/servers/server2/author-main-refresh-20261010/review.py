"""Bounded saved-raw audit; never loads a model/checkpoint or mutates jobs."""
import csv,copy,json,math,subprocess
from datetime import datetime,timezone
from pathlib import Path
from official.experiments.prepare import read,write_new,digest,file_sha
from official.evaluation.reduce import counterfact
from official.evaluation.generation.native_observer import read_observed
from official.evaluation.generation.paper_display import paper_cell

OUT=Path(__file__).parent;REPO=OUT.parents[3];BASE=Path('/mnt/raid5/janghj/ODE-edit/local')
definition=REPO/'audits/servers/server2/baseline-completed-zsre-audit-20261010/audit.py'
ns={'__file__':str(definition)}
exec(compile(definition.read_text().split('\ndata=read(')[0],str(definition),'exec'),ns)
verified,member,check,order,independent,close=(ns[k] for k in ('verified','member','check','order','independent','close'))
previous=REPO/'audits/servers/server2/baseline-refresh-s2-flucon-20261010/table-rows.json'
rows=copy.deepcopy(read(previous)['rows']);stamp=datetime.now(timezone.utc).isoformat();issues=[]
author=read(BASE/'fe-author-hparams-2k-20261010/registration-r1/submission.json')
gen=read(BASE/'baseline-refresh-s2-flucon-20261010/registration-r1/submission.json')
ids=sorted({r['job_id'] for r in rows}|{str(r['job_id']) for r in gen['jobs'].values()})
accounting=subprocess.check_output(['sacct','-X','-nP','-j',','.join(ids),'--format=JobIDRaw,JobName%100,User,State,NodeList,WorkDir%400'],text=True)
states={p[0]:p for p in (s.split('|') for s in accounting.splitlines())}
for r in rows:
    s=states[r['job_id']];check(s[2]=='janghj' and s[4] in ('server2','None assigned',''),'OWNER_NODE')
    r.update(observed_state=s[3],observed_at=stamp,job_name=s[1],prior_audit=member(previous),report_path='experiment-reports/servers/server2/author-main-refresh-20261010/report-ko.md')
    if r['method']=='MEMIT_FE':
        r['table_placement']='LEGACY_NATIVE_FE_NOT_AUTHOR' if r['model']=='qwen25' else 'GPTJ_NATIVE_LEGACY_NO_AUTHOR_EXPERIMENT'
    try:
        if r['numeric_eligible']:
            raw=verified(r['raw']);verified(r['terminal']);cases=raw['cases'] if isinstance(raw,dict) else raw
            if r['dataset']=='zsre':
                values,den,micro=independent(cases);check(len(cases)==2000 and den==r['denominators'],'DENOMINATORS')
                # Previously source-bound full-stream proof remains explicit, not renamed forward parity.
                evidence=r['zsre_audit'];check(evidence['CPU_query_parity_only'],'PUBLIC_QUERY_EVIDENCE')
                for m in evidence['frozen_members']:check(file_sha(m['path'])==m['sha256'],'FROZEN_EVALUATOR')
            else:values=counterfact(cases)
            close(values,r['metrics']);r['raw_terminal_SHA_rechecked']=True
        else:r['status']=s[3]
        if r['job_id'] in ('62531','62532'):
            reg=author['jobs'][r['dataset']];cfg=verified(reg['config']);folder=Path(reg['output'])
            r.update(profile='FE_AUTHOR_HPARAMS_HISTORY',method='MEMIT-FE (FE author hparams + history)',
                source_commit=author['source'],config_sha256=cfg['config_sha256'],author_profile_sha256=cfg['author_profile_sha256'],
                table_placement='MODEL_MAIN_AUTHOR_REPLACES_NATIVE_FE_WITHOUT_RELABEL',output=str(folder),
                committed_batches=len(list((folder/'commits').glob('batch-*.json'))),
                Flu='DEFERRED' if r['dataset']=='cf' else 'NOT_APPLICABLE',Con='DEFERRED' if r['dataset']=='cf' else 'NOT_APPLICABLE')
            if not (folder/'COMPLETE.json').exists():continue
            t=read(folder/'COMPLETE.json');check(t['requests']==2000 and t['native_apply_calls']==20 and t['history_appends_per_layer']==20,'AUTHOR_COMPLETE')
            check(t['identity']['config_sha256']==cfg['config_sha256'] and t['identity']['code_commit']==author['source'],'AUTHOR_IDENTITY')
            for n in range(1,21):
                c=read(folder/'commits'/f'batch-{n:02d}.json');check(c['batch']==n and c['identity']==t['identity'],'AUTHOR_COMMIT')
                check(c['cursor']['edit']['before']['successful_calls']==n-1 and c['cursor']['edit']['after']['successful_calls']==n,'HISTORY_CONTINUITY')
            pointer=read(folder/'checkpoint/latest.json');check(pointer==c['checkpoint'] and pointer['batch']==20 and pointer['final_W20'],'W20_CP')
            cp=folder/'checkpoint'/pointer['file'];cp_member=member(cp);check(cp_member['sha256']==pointer['sha256'],'CP_FULL_SHA')
            full=verified(t['final_cursor']['factual']);cases=full['cases'];stream=verified(cfg['stream_member']);order(cases,stream)
            check(full['model_no_mutation'] and full['RNG_restored'],'FACTUAL_GUARDS')
            if r['dataset']=='cf':
                values=counterfact(cases);den={g:sum(len(x[g+'_prompts_probs']) for x in cases) for g in ('rewrite','paraphrase','neighborhood')}
            else:
                raise ValueError('NEW_AUTHOR_ZSRE_REQUIRES_FROZEN_PUBLIC_QUERY_PROOF_BINDING')
            close(values,full['summary'])
            r.update(numeric_eligible=True,status='W20_AUTHOR_CPU_RAW_VERIFIED',metrics=values,denominators=den,requests=2000,commits=20,
                raw=t['final_cursor']['factual'],terminal=member(folder/'COMPLETE.json'),checkpoint=cp_member,checkpoint_receipt=pointer,
                ordered_cohort_sha256=digest([(x['occurrence_index'],x['case_id']) for x in cases]),stream=cfg['stream_member'],
                future_flucon_consumer_pending=t['future_flucon_consumer_pending'],new_author_result=True)
    except Exception as e:
        r.update(numeric_eligible=False,status='UNVERIFIED',error=str(e));issues.append(dict(job=r['job_id'],error=str(e)))

generation=[]
for key,reg in gen['jobs'].items():
    if key=='collector':continue
    s=states[str(reg['job_id'])];check(s[2]=='janghj','EVAL_OWNER');cfg=verified(reg['config']);folder=Path(reg['output'])
    g=dict(key=key,job_id=str(reg['job_id']),job_name=s[1],observed_state=s[3],observed_at=stamp,
           status='NOT_COMPLETE',numeric_eligible=False,config=reg['config'],source=gen['source'],original_checkpoint=cfg['original']['checkpoint'])
    try:
        if (folder/'COMPLETE.json').exists():
            t=read(folder/'COMPLETE.json');check(t['config']==reg['config'] and t['source']==gen['source'],'GEN_SOURCE_CONFIG')
            check(t['original_checkpoint']==cfg['original']['checkpoint'] and t['checkpoint_unchanged'] and t['weights_unchanged'] and t['RNG_unchanged'],'GEN_CP_GUARDS')
            check(file_sha(t['endpoint']['path'])==t['endpoint']['sha256'],'ENDPOINT_SHA')
            obs=read_observed(t['endpoint']['path']);check(obs['summary']==t['summary'] and len(obs['rows'])==2000,'GEN_FULLCOUNT')
            stream=verified(cfg['stream']);stream=stream['records'] if isinstance(stream,dict) else stream
            check([(x['occurrence'],x['case_id']) for x in obs['rows']]==[(x['occurrence_index'],x['case_id']) for x in stream],'GEN_ORDER')
            g.update(status='W20_GENERATION_SAVED_RAW_VERIFIED',numeric_eligible=True,terminal=member(folder/'COMPLETE.json'),raw=t['endpoint'],
                summary=t['summary'],execution_receipt=obs['native_execution_member'],raw_metric_verification='native reader SHA/identity/row metrics; no TFIDF refit or new generation',
                ordered_cohort_sha256=digest([(x['occurrence'],x['case_id']) for x in obs['rows']]))
            for metric,k,u in [('Flu','ngram_entropy','bits'),('Con','reference_score','cosine_0_to_1')]:
                v=t['summary'][k];g[metric]=dict(raw_value=v,raw_unit=u,paper_display_x100=paper_cell(v,metric=metric,raw_unit=u))
            target=key.rsplit('-',1)[1]
            for r in rows:
                if r['dataset']=='cf' and r['job_id']==target:
                    r.update(Flu=g['Flu']['raw_value'],Con=g['Con']['raw_value'],Flu_paper_x100=g['Flu']['paper_display_x100'],Con_paper_x100=g['Con']['paper_display_x100'],generation=g)
    except Exception as e:g.update(status='UNVERIFIED',error=str(e));issues.append(dict(job=g['job_id'],error=str(e)))
    generation.append(g)

data=dict(nonce='USER-GH-S1-S2-FE-AUTHOR-MAIN-REFRESH-20261010-R1',accepted_turn='01a12655-1e13-7380-b3d7-83f4a30b1105',observed_at=stamp,
    rows=rows,generation=generation,issues=issues,reducer=member(__file__),reused_reducer=member(definition),
    GPU=0,checkpoint_loads=0,job_mutations=0,GH_sole_README=True,broadcast='NO_BROADCAST_NOT_REQUIRED')
write_new(OUT/'table-rows.json',data)
write_new(OUT/'inventory.json',dict(observed_at=stamp,accounting=accounting,prior_audit=member(previous),issues=issues,
    author_submission=member(BASE/'fe-author-hparams-2k-20261010/registration-r1/submission.json'),generation_submission=member(BASE/'baseline-refresh-s2-flucon-20261010/registration-r1/submission.json')))
with (OUT/'table-rows.csv').open('x',newline='') as f:
    w=csv.writer(f);w.writerow(['model','dataset','method','job_id','job_name','state','eligible','Eff','Gen','Loc','Score','Flu_x100','Con_x100'])
    for r in rows:
        m=r['metrics'] if r['numeric_eligible'] else {}
        w.writerow([r['model'],r['dataset'],r['method'],r['job_id'],r['job_name'],r['status'],r['numeric_eligible'],*[m.get(k,'') for k in ('Efficacy','Generalization','Specificity','Score')],r.get('Flu_paper_x100',r.get('Flu','NOT_APPLICABLE')),r.get('Con_paper_x100',r.get('Con','NOT_APPLICABLE'))])
print(json.dumps(dict(eligible=sum(r['numeric_eligible'] for r in rows),generation=sum(g['numeric_eligible'] for g in generation),issues=issues,author=[r for r in rows if r.get('new_author_result')]),default=str)[:2500])
