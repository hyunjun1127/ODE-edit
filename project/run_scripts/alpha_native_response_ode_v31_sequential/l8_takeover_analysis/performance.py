"""CPU scalar aggregation of completed native sequential endpoints."""
import argparse,csv,hashlib,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np
from ..reporting import metrics,bits,csvwrite,jwrite,sha

MISSING='NOT_RECORDED_PUBLISHED_PROMPT_PAIRS'
MODELS=('llama3-8b-inst','qwen2.5-7b-inst')
ARMS=('O_NATIVE','JV_NATIVE','L8_ONLY_NATIVE')
KINDS=('rewrite','rephrase','locality')


def read(path):return json.loads(Path(path).read_text())
def rows(path):return list(csv.DictReader(Path(path).open()))
def numeric(row):
    result={}
    for k,v in row.items():
        try:result[k]=float(v) if '.' in v or 'e' in v.lower() else int(v)
        except (ValueError,TypeError):result[k]=v
    return result
def stats(values):
    x=np.asarray(values,dtype=np.float64)
    if not len(x) or not np.isfinite(x).all():raise ValueError('FINITE_NONEMPTY_DISTRIBUTION')
    return dict(n=len(x),mean=float(x.mean()),median=float(np.median(x)),p90=float(np.quantile(x,.9)),max=float(x.max()))
def hash_json(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def enrich(public,meta):
    """Recompute canonical pairs and distributions; never substitute accuracy."""
    out=metrics(public,meta);rr=public['rows']
    index={(r['case_id'],r['prompt_index'],r['kind']):r for r in rr}
    if len(index)!=len(rr):raise ValueError('DUPLICATE_PUBLIC_ROW')
    for prefix,key,mult in [('rewrite','RS',1),('rephrase','PS',2),('locality','NS',10)]:
        pairs=bits(public,prefix)
        if len(pairs)!=public['request_count']*mult:raise ValueError('PROMPT_CARDINALITY')
        if sum(v['success'] for v in pairs.values())!=out[key+'_num']:raise ValueError('CANONICAL_PREFERENCE_COUNT')
        bycase=defaultdict(list)
        for (case,prompt),value in pairs.items():bycase[case].append(value['success'])
        if sum(all(v) for v in bycase.values())!=out[key+'_strict_num']:raise ValueError('STRICT_PREFERENCE_COUNT')
        for label,val in stats([v['margin'] for v in pairs.values()]).items():out[prefix+'_margin_'+label]=val
        for target in ('new','true'):
            kind=prefix+'_target_'+target;subset=[r for r in rr if r['kind']==kind]
            distribution=stats([r['nll'] for r in subset])
            for label,value in distribution.items():
                if label!='n' and not math.isclose(value,out[kind+'_nll_'+label],rel_tol=1e-12,abs_tol=1e-12):
                    raise ValueError('NLL_SUMMARY_MISMATCH')
            out[kind+'_correct_token_count']=sum(r['correct_token_count'] for r in subset)
            out[kind+'_target_token_denominator']=sum(r['target_token_count'] for r in subset)
            flags=defaultdict(list)
            for r in subset:flags[r['case_id']].append(r['all_tokens_correct'])
            out[kind+'_strict_teacher_forced_correct_num']=sum(all(v) for v in flags.values())
            out[kind+'_strict_teacher_forced_correct_den']=len(flags)
    out['evidence']='SERVER4_REHASHED_PUBLIC_ROWS'
    return out


def published_margins(row):
    """Paired means can be subtracted; marginal quantiles cannot."""
    out=dict(row)
    for k in KINDS:
        delta=float(row[k+'_target_true_nll_mean'])-float(row[k+'_target_new_nll_mean'])
        out[k+'_margin_mean']=-delta if k=='locality' else delta
        out[k+'_margin_n']=row[k+'_target_new_row_count']
        for s in ('median','p90','max'):out[k+'_margin_'+s]=MISSING
    out['evidence']='SERVER2_GIT_PUBLICATION_REHASH_NOT_REMOTE_RAW'
    return out


def build(root,sh1,output):
    root,sh1,output=map(Path,(root,sh1,output));output.mkdir(parents=True,exist_ok=False)
    ext=root/'external-main';sample=read(root/'sample.lock.json');record={r['case_id']:r for r in sample['records']}
    inputs=set()
    def load(path):inputs.add(path);return read(path)
    def old(name):
        p=ext/(name+'.csv');inputs.add(p);return [numeric(r) for r in rows(p)]
    final=[published_margins(r) for r in old('final_metrics')]
    current=[published_margins(r) for r in old('current_batch_metrics')]
    seen=[published_margins(r) for r in old('seen_prefix_metrics')]
    cohorts=old('retention_cohort_metrics');detailed=[];age=[]
    partitions=[numeric(r) for r in rows(sh1/'endpoint-partitions.csv')];inputs.add(sh1/'endpoint-partitions.csv')
    prompt_rows=[]
    for index,alias in [(4,MODELS[0]),(5,MODELS[1])]:
        chain=root/f'chain-{index}-{alias}-L8_ONLY_NATIVE';terminal=load(chain/'terminal-receipt.json')
        assert terminal['status']=='TERMINAL_VALID' and terminal['requested']==1000 and terminal['completed_batches']==10
        at_write={};previous={}
        for k in range(1,11):
            bd=chain/f'batch-{k:02}';commit=load(bd/'commit.json');writer=load(bd/'writer.json')
            meta=dict(alias=alias,arm='L8_ONLY_NATIVE',batch=k,W_sha256=commit['committed_weight_sha256'])
            pub=writer['endpoint']['evaluation'];current.append(enrich(pub,dict(meta,scope='CURRENT_B100')))
            if k in (1,5,10):
                full=load(bd/'seen-full.json');row=enrich(full,dict(meta,scope='single_W_seen_prefix'));seen.append(row)
                assert full['request_count']==k*100
                if k==10:final.append(dict(row,scope='single_final_W10_full1000'))
                # Raw-free scalar rows are publication-safe: no text/token/logit.
                for r in full['rows']:
                    prompt_rows.append(dict(meta,case_hash=record[r['case_id']]['raw_record_sha256'],
                        request_sha256=r['request_sha256'],kind=r['kind'],prompt_index=r['prompt_index'],
                        nll=r['nll'],all_tokens_correct=r['all_tokens_correct'],
                        correct_token_count=r['correct_token_count'],target_token_count=r['target_token_count'],
                        input_identity_sha256=r['input_identity_sha256']))
                for label,lo,hi in [('early',0,int(k*100*.2)),('middle',int(k*100*.2),int(k*100*.8)),('recent',int(k*100*.8),k*100)]:
                    ids={r['case_id'] for r in sample['records'][lo:hi]};a=dict(meta,stratum=label,request_denominator=len(ids))
                    for prefix,key in zip(KINDS,('RS','PS','NS')):
                        b=[v for (case,_),v in bits(full,prefix).items() if case in ids]
                        a[key+'_num']=sum(v['success'] for v in b);a[key+'_den']=len(b);a[key+'_rate']=a[key+'_num']/len(b)
                    age.append(a)
            retention=load(bd/'rewrite-retention.json');assert len(retention)==100*k
            grouped=defaultdict(list)
            for r in retention:
                case=r['case_id'];rec=record[case];cohort=rec['batch_index'];now=r['success']
                assert r['state_sha256']==meta['W_sha256'] and r['at_batch']==k and r['controller_influence_count']==0
                assert now==(r['new_nll']<r['true_nll']) and r['margin']==r['true_nll']-r['new_nll']
                if cohort==k:at_write[case]=now
                overwrite=any(x['subject_relation_group']==rec['subject_relation_group'] and cohort<x['batch_index']<=k
                    and x['target_new_sha256']!=rec['target_new_sha256'] for x in sample['records'])
                d=dict(meta,case_hash=rec['raw_record_sha256'],request_sha256=rec['request_sha256'],
                    cohort=cohort,age=k-cohort,success=now,margin=r['margin'],new_nll=r['new_nll'],true_nll=r['true_nll'],
                    at_write_success=at_write[case],initially_failed=not at_write[case],
                    at_write_success_now_failure=at_write[case] and not now,
                    prior_failure=case in previous and not previous[case],
                    prior_failure_now_recovery=case in previous and not previous[case] and now,
                    overwrite_candidate=overwrite)
                detailed.append(d);grouped[cohort].append(d);previous[case]=now
            for cohort,rr in grouped.items():
                cohorts.append(dict(meta,cohort=cohort,age=k-cohort,canonical_denominator=len(rr),
                    current_success=sum(r['success'] for r in rr),at_write_success=sum(r['at_write_success'] for r in rr),
                    at_write_success_now_failure=sum(r['at_write_success_now_failure'] for r in rr),
                    forgetting_conditional_denominator=sum(r['at_write_success'] for r in rr),
                    initially_failed=sum(r['initially_failed'] for r in rr),
                    prior_failure_denominator=sum(r['prior_failure'] for r in rr),
                    prior_failure_now_recovery=sum(r['prior_failure_now_recovery'] for r in rr),
                    overwrite_candidate_count=sum(r['overwrite_candidate'] for r in rr),
                    nonoverwrite_forgetting_num=sum(r['at_write_success_now_failure'] and not r['overwrite_candidate'] for r in rr),
                    nonoverwrite_forgetting_den=sum(r['at_write_success'] and not r['overwrite_candidate'] for r in rr),
                    margin_mean=float(np.mean([r['margin'] for r in rr]))))
        rr=[r for r in detailed if r['alias']==alias and r['batch']==10]
        partitions.append(dict(alias=alias,arm='L8_ONLY_NATIVE',all_denominator=len(rr),at_write_success=sum(r['at_write_success'] for r in rr),
            initially_failed=sum(r['initially_failed'] for r in rr),at_write_success_to_final_failure=sum(r['at_write_success_now_failure'] for r in rr),
            initial_failure_denominator=sum(r['initially_failed'] for r in rr),
            previous_checkpoint_failure_to_recovery=sum(r['prior_failure_now_recovery'] for r in rr),
            conditional_failure_denominator=sum(r['at_write_success'] for r in rr),
            initially_failed_to_final_recovery=sum(r['initially_failed'] and r['success'] for r in rr),
            final_RS=sum(r['success'] for r in rr),nonoverwrite_failure=sum(r['at_write_success_now_failure'] and not r['overwrite_candidate'] for r in rr),
            nonoverwrite_success_den=sum(r['at_write_success'] and not r['overwrite_candidate'] for r in rr),
            overwrite_candidates=sum(r['overwrite_candidate'] for r in rr)))
    final.sort(key=lambda r:(MODELS.index(r['alias']),(('PRE_EDIT_ORIGINAL_W0',)+ARMS).index(r['arm'])))
    main=[r for r in final if r['arm'] in ARMS];assert len(main)==6 and len(current)==60 and len(seen)==18 and len(cohorts)==330
    deltas=[]
    for alias in MODELS:
        rr={r['arm']:r for r in final if r['alias']==alias}
        for ref in ('PRE_EDIT_ORIGINAL_W0','O_NATIVE','JV_NATIVE'):
            l8=rr['L8_ONLY_NATIVE'];base=rr[ref];d=dict(alias=alias,reference=ref,comparison='L8_ONLY_NATIVE',scope='FINAL_W10_FULL1000',
                paired_prompt_transitions='NOT_AVAILABLE_SERVER2_PROMPT_ROWS_NOT_PUBLISHED',causal_claim=False)
            for key in ('RS','PS','NS'):
                assert l8[key+'_den']==base[key+'_den'];d[key+'_den']=l8[key+'_den'];d[key+'_delta_num']=l8[key+'_num']-base[key+'_num'];d[key+'_delta_pp']=100*(l8[key+'_rate']-base[key+'_rate'])
            for kind in KINDS:
                for target in ('new','true'):
                    for s in ('mean','median','p90','max'):d[f'{kind}_{target}_nll_delta_{s}']=l8[f'{kind}_target_{target}_nll_{s}']-base[f'{kind}_target_{target}_nll_{s}']
                d[kind+'_margin_delta_mean']=l8[kind+'_margin_mean']-base[kind+'_margin_mean']
            deltas.append(d)
    tables={'final_metrics':final,'main_six_arm_table':main,'current_batch_metrics':current,'seen_prefix_metrics':seen,
        'retention_cohort_metrics':cohorts,'l8_rewrite_retention_records':detailed,'endpoint_partitions':partitions,
        'l8_age_strata':age,'final_deltas':deltas,'l8_seen_scalar_rows':prompt_rows}
    for name,rs in tables.items():csvwrite(output/(name+'.csv'),rs)
    summary=dict(status='PERFORMANCE_RECOMPUTE_PASS',tables={k:len(v) for k,v in tables.items()},
        final=[{k:v for k,v in r.items() if k in ('alias','arm','RS_num','RS_den','PS_num','PS_den','PS_strict_num','NS_num','NS_den')} for r in main],
        sources=[dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(inputs)],
        no_model_no_evaluator=True,unavailable='L8 vs W0/O/JV exact prompt transitions: server2 prompt pairs absent; no reconstruction')
    jwrite(output/'performance-summary.json',summary)
    lines=['# Final W10 전체 1,000 요청: O / JV / L8-only','',
        '각 arm의 최종 materialized W10에서 동일 sealed 1,000 requests 전체를 평가했다. Current B100/online 합계가 아니다. RS/PS는 new NLL < true NLL, NS는 true NLL < new NLL; tie 실패. Cross-host end-to-end descriptive 비교이며 동일 W/M/z 대조가 아니다.','',
        '|Model|Arm|RS|PS|strict PS|NS|','|---|---|---:|---:|---:|---:|']
    for r in main:
        cells=[f"{r[key+'_num']}/{r[key+'_den']} ({r[key+'_rate']*100:.2f}%)" for key in ('RS','PS','NS')]
        lines.append(f"|{r['alias']}|{r['arm']}|{cells[0]}|{cells[1]}|{r['PS_strict_num']}/{r['PS_strict_den']}|{cells[2]}|")
    lines+=['','O/JV: sealed Server2 Git publication 재해시. L8: completed Server4 public/raw scalar를 재계산. Remote Server2 raw/checkpoint 재해시와 paired prompt transition은 미수행. 세부 무결성 결과는 별도 integrity receipt에 결속한다.']
    (output/'first-six-arm-table-ko.md').write_text('\n'.join(lines)+'\n')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--sh1',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(build(a.root,a.sh1,a.output)['final'],ensure_ascii=False))
