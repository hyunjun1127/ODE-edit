"""Read-only completion and independent zsRE request-macro audit."""
import csv,json,math
from pathlib import Path
from official.experiments.prepare import read,file_sha,digest,write_new
from official.evaluation.reduce import counterfact
from official.evaluation.generation.paper_display import paper_cell

OUT=Path(__file__).resolve().parent
REPO=OUT.parents[3]
BASE=Path('/mnt/raid5/janghj/ODE-edit/local')
NEW=BASE/'qwen-baseline-mask-cold-rerun-20261010/registration-r1'
OLD=BASE/'qwen-baselines-server2-20261009/registration-native-eval-r2'
GJ=BASE/'official-baselines/server2/zsre-2k-reeval-20261009/registration-r1'
EVAL='d6a5b34eafd27660a2dee4632c638b6bf4c3614246071711cf5159a002415a45'
def check(ok,reason):
    if not ok: raise ValueError(reason)
def member(p):return dict(path=str(p),bytes=Path(p).stat().st_size,sha256=file_sha(p))
def verified(m):
    check(file_sha(m['path'])==m['sha256'],'SHA_MISMATCH')
    return read(m['path'])
def close(a,b):
    for k,v in a.items():check(math.isfinite(v) and abs(v-b[k])<1e-10,'SUMMARY_'+k)
def order(cases,stream):
    check(len(cases)==len(stream)==2000,'FULL_2000')
    check([(r['occurrence_index'],r['case_id']) for r in cases]==[(r['occurrence_index'],r['case_id']) for r in stream],'COHORT_ORDER')
def independent(cases):
    values={};den={};micro={}
    for g,k in [('rewrite','Efficacy'),('paraphrase','Generalization'),('neighborhood','Specificity')]:
        means=[];correct=0;n=0
        for r in cases:
            obs=r[g+'_observations']
            bits=[o['predicted_token_id']==o['target_token_id'] for o in obs]
            check(bits and bits==r[g+'_prompts_correct'] and bits==[o['correct'] for o in obs],'PREDICTED_TARGET_BITS')
            means.append(math.fsum(bits)/len(bits));correct+=sum(bits);n+=len(bits)
        values[k]=100*math.fsum(means)/len(cases);den[g]=n;micro[k]=100*correct/n
    return values,den,micro
def source_proof(root,proof,stream,tokenizer_sha):
    lock=read(root/'source-lock.json');entries={e.get('relative',e.get('path')):e for e in lock['members']}
    paths=['official/evaluation/zsre_paper.py','official/evaluation/zsre_query_parity.py','official/evaluation/zsre_public_sources/lock.json']
    members=[]
    for rel in paths:
        p=root/'source'/rel;check(file_sha(p)==entries[rel]['sha256'],'FROZEN_SOURCE_SHA');members.append(member(p))
    check(members[0]['sha256']==EVAL,'EVALUATOR_SOURCE')
    check(proof['oracle_lock_sha256']==digest(read(members[2]['path'])),'ORIGINAL_AST_LOCK_CANONICAL_JSON')
    check(proof['requests']==2000 and proof['input_mismatches']==proof['target_mismatches']==0,'QUERY_PARITY')
    check(proof['status']=='PASS_CPU_QUERY_ONLY' and proof['stream_sha256']==digest(stream),'CANONICAL_STREAM_PARITY')
    return dict(frozen_members=members,query_proof=proof,tokenizer_sha256=tokenizer_sha,
                byte_stream_vs_canonical_JSON_distinct=True,CPU_query_parity_only=True,
                pretrained_numeric_parity='NOT_MEASURED',metric='100*mean_requests(mean_target_token_correct)',
                Loc='loc_ans_target_correctness_NOT_W0_agreement',missing_requests=0)

data=read(OUT/'table-rows.json');rows=data['rows'];issues=[]
# New completed corrected CF: no generation endpoint is required when explicitly deferred.
for row in rows:
    if row['job_id'] not in ('62073','62075','62077','62079'):continue
    cell='qwen25-cf-'+row['method'].lower();root=NEW/'runs'/cell;p=root/'terminal.json'
    if not p.exists():continue
    try:
        t=read(p);check(t['actual_job_id']==row['job_id'] and t['status']=='W20_COMPLETE' and t['completed_edits']==2000,'TERMINAL')
        check(len(t['commits'])==20,'TWENTY_COMMITS')
        for i,m in enumerate(t['commits'],1):
            c=verified(m);check(c['completed_batch']==i and c['checkpoint_identity']==t['checkpoint_identity'],'COMMIT_IDENTITY')
        check(t['checkpoint']['final_W20'] and t['checkpoint']['batch']==20,'CHECKPOINT_RECEIPT')
        raw=verified(t['calculation_evidence']['factual']);stream=read(NEW/'streams/cf-stream.json');order(raw,stream)
        values=counterfact(raw);close(values,c['factual']['summary'])
        row.update(metrics=values,numeric_eligible=True,status='W20_CORRECTED_COLD_CF_RAW_VERIFIED',requests=2000,
            raw=t['calculation_evidence']['factual'],terminal=member(p),commits=20,checkpoint_receipt=t['checkpoint'],
            cold_state_identity=t['checkpoint_identity'],ordered_cohort_sha256=digest([(r['occurrence_index'],r['case_id']) for r in raw]),
            stream_sha256=file_sha(NEW/'streams/cf-stream.json'),denominators={g:sum(len(r[g+'_prompts_probs']) for r in raw) for g in ('rewrite','paraphrase','neighborhood')},
            Flu='DEFERRED',Con='DEFERRED')
    except Exception as e:row.update(numeric_eligible=False,status='UNVERIFIED',error=str(e))
# Recent GPTJ CF raw audit reused only after exact raw/terminal SHA recheck.
prior=read(REPO/'audits/servers/server2/main-table-refresh-20261009/table-rows.json')
for old in prior['rows']:
    if old['model']=='gptj' and old['dataset']=='cf' and old['numeric_eligible']:
        verified(old['raw']);verified(old['terminal'])
        rows.append(dict(old,raw_terminal_rechecked_at=data['at'],recent_CPU_audit_reused=True,
                         scheduler_snapshot_reused=True,Flu='DEFERRED',Con='DEFERRED'))
for row in rows:
    if row['dataset']!='zsre' or not row['numeric_eligible']:continue
    try:
        job=row['job_id'];isblue=job=='61964';root=OLD if isblue else NEW if job=='62072' else GJ
        if isblue:
            raw=verified(row['raw']);proof=read(root/'query-parity.json');stream=read(root/'streams/zsre-stream.json')
            last=read(root/'runs/qwen25-zsre-alphaedit_blue/commits/b20.json')
            actual_query=last['factual']['work']['query_sha256'];tok=row['cold_state_identity']['tokenizer_sha256']
            original_job=job;summary=last['factual']['summary'];identity=row['cold_state_identity']
        else:
            result=verified(row['terminal']);full=verified(row['raw']);raw=full['cases'];identity=full['identity']
            inputs=read(root/('ft-eval-inputs.json' if job=='62072' else 'inputs.json'))
            proof=inputs['query_proof'];stream=verified(inputs['stream']);actual_query=full['query_sha256']
            tok=identity.get('tokenizer_sha256',identity['original_identity']['tokenizer_sha256']);original_job=identity.get('original_job_id',61900 if job=='62072' else None)
            if job=='62072':
                cfg=read(root/'ft-eval/config.json')
                check(cfg['config_sha']==identity['config_sha256']==result['config_sha256'],'FT_CONFIG_IDENTITY')
                check(cfg['evaluator_sha256']==EVAL and cfg['tokenizer_sha256']==tok and cfg['evaluation_profile']=='zsre-public-query-W20-only-v1','FT_EVALUATOR_CONFIG_BINDING')
            else:check(identity['evaluator_sha256']==EVAL,'ACTUAL_EVALUATOR_BINDING')
            summary=full['summary']
            check(full['model_no_mutation'] and full['RNG_restored'],'ACTUAL_STATE_GUARDS')
        order(raw,stream);check(actual_query==proof['query_sha256'],'ACTUAL_QUERIES_VS_FULL_STREAM_PROOF')
        values,den,micro=independent(raw);close(values,summary);close(values,row['metrics'])
        check(den==proof['token_denominators'],'TOKEN_COUNTS')
        if 'Specificity_loc_ans' in summary:check(abs(summary['Specificity_loc_ans']-values['Specificity'])<1e-10,'LOC_ALIAS')
        evidence=source_proof(root,proof,stream,tok)
        evidence.update(actual_query_sha256=actual_query,token_denominators=den,requests=2000,
            independently_recomputed=values,diagnostic_token_micro_NOT_TABLE=micro,original_edit_job_id=original_job,
            eval_only_not_new_edit_experiment=not isblue,identity=identity)
        row.update(zsre_audit=evidence,original_edit_job_id=original_job,status='W20_PUBLIC_QUERY_REQUEST_MACRO_VERIFIED')
    except Exception as e:
        row.update(numeric_eligible=False,status='UNVERIFIED',error=str(e));issues.append(dict(job=row['job_id'],error=str(e)))
for row in rows:
    for metric,unit in [('Flu','bits'),('Con','cosine_0_to_1')]:
        value=row.get(metric)
        if isinstance(value,(int,float)):row[metric+'_paper_x100']=paper_cell(value,metric=metric,raw_unit=unit)
data.update(nonce='USER-GH-ALL-SH-BASELINE-COMPLETED-ZSRE-AUDIT-20261010-R1',accepted_turn='01a122d0-699d-77a2-9764-77789dcbfa24',
    rows=rows,issues=issues,numeric_eligible=sum(r['numeric_eligible'] for r in rows),reducer=member(__file__),
    exclusions={'broken_context_old_jobs':[61956,61960,61968],
      'original_GPTJ_zsre':[dict(job=j,status='PUBLIC_QUERY_REEVALUATION_REQUIRED_UNLESS_REPLACED_BY_LINKED_EVAL_ONLY') for j in [61726,61728,61730,61732,61734,61735]],
      'Qwen_original_FT_61900':'PUBLIC_QUERY_REEVALUATION_REQUIRED; superseded for table by eval62072',
      'other':'OURS/PRICE/tuning/heldout/historical/replicas not promoted'},
    GPU=0,checkpoint_loads=0,job_mutations=0,online_history_changes=0,broadcast='NO_BROADCAST_NOT_REQUIRED')
write_new(OUT/'audited-final.json',data)
with (OUT/'audited-final.csv').open('x',newline='') as f:
    w=csv.writer(f,lineterminator='\n');w.writerow(['model','dataset','method','job_id','job_name','state','eligible','Eff','Gen','Loc','Score','Flu_x100','Con_x100'])
    for r in rows:
        m=r['metrics'] if r['numeric_eligible'] else {}
        w.writerow([r['model'],r['dataset'],r['method'],r['job_id'],r['job_name'],r['status'],r['numeric_eligible'],m.get('Efficacy',''),m.get('Generalization',''),m.get('Specificity',''),m.get('Score',''),r.get('Flu_paper_x100',r.get('Flu','NOT_MEASURED')),r.get('Con_paper_x100',r.get('Con','NOT_MEASURED'))])
print(json.dumps(dict(numeric_eligible=data['numeric_eligible'],issues=issues,rows=[(r['job_id'],r['status'],r['metrics'] if r['numeric_eligible'] else {}) for r in rows])))
