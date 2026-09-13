"""CPU-only, source-independent reduction of sealed sequential NLL pairs.

Imports only the prior independent CPU reducer, never the execution evaluator.
No prompt text, individual cases, or individual evaluation rows are published.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from statistics import fmean
from .review_metrics import (aggregate, check_rows, desired, digest, distribution,
    expected_rows, file_sha, key, mmlu_prediction, outcome, pairs, panel_digest,
    require, write_csv, quality, DATA_SHA, MULT)

ARMS = ('N4', 'RES8', 'S875', 'S75', 'FULL8', 'REFIT4')

def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')

def validate(doc, records, endpoint):
    require(doc['requests']==len(records), 'REQUEST_CARDINALITY')
    require(doc['request_order']==digest([r['case_id'] for r in records]), 'REQUEST_ORDER')
    require(doc['endpoint_state_sha256']==endpoint, 'EVALUATION_ENDPOINT')
    result={}
    for metric,m in MULT.items():
        s=doc['metrics'][metric]; rr=s['rows']
        require(len(rr)==len(records)*m,'PROMPT_CARDINALITY')
        check_rows(rr,metric,expected_rows(records,metric))
        a=aggregate(rr,metric)
        for x,y in [('numerator','numerator'),('denominator','prompt_denominator'),('rate','rate'),('bit_order_sha256','bit_order_sha256')]:
            require(s[x]==a[y],'INDEPENDENT_REDUCTION_'+x)
        result[metric]=a
    return result

def load_eval(path, batch):
    d=json.loads(path.read_text())
    require(d['batch']==batch and d['evaluation_nonmutation'] is True,'ENDPOINT_NONMUTATION')
    require(d['endpoint_state_sha256']==panel_digest(d['endpoint_state']),'ENDPOINT_STATE_HASH')
    require(d['optimizer_access']==d['audit_evaluations']==d['imputation']==0,'FORBIDDEN_ACCESS_OR_IMPUTATION')
    return d

def annotation(records, end):
    last={}
    for i,r in enumerate(records[:end]):
        w=r['requested_rewrite']
        if w.get('relation_id') is not None: last[(w['subject'],w['relation_id'])]=i
    out={}
    for i,r in enumerate(records[:end]):
        w=r['requested_rewrite']; j=last.get((w['subject'],w.get('relation_id')))
        out[r['case_id']]='UNKNOWN_RELATION' if j is None else ('ACTIVE_TARGET' if records[j]['requested_rewrite']['target_new']['str']==w['target_new']['str'] else 'SUPERSEDED')
    return out

def subset(rows, ids):
    allowed=set(ids)
    return [r for r in rows if r['case_id'] in allowed]

def first(root,out,records):
    rows=[]
    for cell,arm in enumerate(ARMS):
        p=root/f'cell-{cell}/B060/evaluation.json'; d=load_eval(p,60)
        a=validate(d['fullseen'],records[:6000],d['endpoint_state_sha256'])
        row=dict(arm=arm,batch=60,population='FULL_SEEN_6000',request_denominator=6000,
                 evaluation_sha256=file_sha(p),validation='INDEPENDENT_NLL_PAIR_IDENTITY_COUNTS_PASS; STATE_CPU_AUDIT_SEPARATE')
        for m,v in a.items(): row.update({m+'_n':v['numerator'],m+'_d':v['prompt_denominator'],m+'_rate':v['rate'],m+'_percent':100*v['rate']})
        rows.append(row)
    for row in rows:
        for m in MULT:
            row[m+'_delta_n_vs_N4']=row[m+'_n']-rows[0][m+'_n']
            row[m+'_delta_pp_vs_N4']=100*(row[m+'_rate']-rows[0][m+'_rate'])
    write_csv(out/'first-final-table.csv',rows);dump(out/'first-final-table.json',rows)
    return rows

def reduce(root,out,records,core):
    panel=json.loads((core/'panel-lock.json').read_text())
    inputs=json.loads((core/'inputs.json').read_text())
    hist=[records[r['ordinal']] for r in panel['Historical']['rows']]
    wiki=json.loads(Path(inputs['wiki_panel']).read_text())['rows']
    wiki_ids=[(r['ordinal'],panel_digest(r['input_ids']),len(r['input_ids'])-1) for r in wiki]
    mmlu=json.loads(Path(inputs['mmlu100']).read_text())
    dev=[mmlu[i] for i in panel['MMLU']['groups']['development']['indices']]
    require([panel_digest(r) for r in dev]==panel['MMLU']['groups']['development']['row_hashes'],'DEV32_BINDING')
    all_docs={}; rates=[]; nll=[]; general=[]; checks=[]; files=[]; cohort=[]; transitions=[]; static=[]; general_pairs=[]
    def summarize(rr,arm,batch,pop,metric,group='ALL'):
        base=dict(arm=arm,batch=batch,population=pop,metric=metric,group=group)
        rates.append(dict(base,**aggregate(rr,metric)))
        clusters={}
        for r in rr: clusters.setdefault(r['case_id'],[]).append(r)
        for field in ('new_nll','true_nll','desired_margin'):
            fn=lambda r:desired(r,metric) if field=='desired_margin' else r[field]
            for unit,values in [('PROMPT',[fn(r) for r in rr]),('REQUEST_CLUSTER_MEAN',[fmean(fn(r) for r in gr) for gr in clusters.values()])]:
                nll.append(dict(base,field=field,aggregation_unit=unit,**distribution(values)))
    for cell,arm in enumerate(ARMS):
        docs={};all_docs[arm]=docs
        for b in range(51,61):
            path=root/f'cell-{cell}/B{b:03d}/evaluation.json';d=load_eval(path,b);docs[b]=d
            files.append(dict(arm=arm,batch=b,path=str(path),bytes=path.stat().st_size,sha256=file_sha(path)))
            pops={'current':records[(b-1)*100:b*100],'historical':hist}
            if b in (55,60):pops['suffix']=records[5000:b*100]
            if b==60:pops.update(fullseen=records[:6000],entry_old=records[:5000])
            ann=annotation(records,b*100)
            for pop,selected in pops.items():
                validate(d[pop],selected,d['endpoint_state_sha256'])
                for m in MULT:
                    rr=d[pop]['metrics'][m]['rows']
                    if pop=='historical':require(all(r['historical_status']['status']==ann[r['case_id']] for r in rr),'HISTORICAL_STATUS')
                    summarize(rr,arm,b,pop,m)
                    if pop in ('historical','suffix','fullseen','entry_old'):
                        for group in ('ACTIVE_TARGET','SUPERSEDED','UNKNOWN_RELATION'):
                            summarize([r for r in rr if ann[r['case_id']]==group],arm,b,pop,m,group)
                    checks.append(dict(arm=arm,batch=b,population=pop,metric=m,requests=len(selected),prompts=len(rr),identity_order='PASS',independent_nll='PASS',duplicate=0,nonfinite=0,imputation=0))
                    if pop=='fullseen':
                        for cb in range(1,61):
                            sub=subset(rr,[r['case_id'] for r in records[(cb-1)*100:cb*100]])
                            cohort.append(dict(arm=arm,batch=b,cohort_batch=cb,metric=m,**aggregate(sub,m)))
                        for group,lo,hi in [('FIRST100',0,100),('FIRST500',0,500),('FIRST1000',0,1000),('OLD_EARLY',0,1000),('OLD_MIDDLE',1000,4000),('OLD_RECENT',4000,5000),('NEW_EARLY',5000,5200),('NEW_MIDDLE',5200,5800),('NEW_RECENT',5800,6000)]:
                            summarize(subset(rr,[r['case_id'] for r in records[lo:hi]]),arm,b,group,m)
            for reuse in d['reuse']:
                require(reuse['added_forwards']==0 and reuse['source_endpoint_state_sha256']==d['endpoint_state_sha256'],'REUSE_ENDPOINT')
            source='fullseen' if b==60 else ('suffix' if b==55 else None)
            if source:
                for pop in (['current','suffix','entry_old','historical'] if b==60 else ['current']):
                    for m in MULT:
                        src={key(r):r for r in d[source]['metrics'][m]['rows']}
                        for r in d[pop]['metrics'][m]['rows']:
                            for f in ('new_nll','true_nll','success','new_strict','true_strict','new_token_correct','true_token_correct','new_token_count','true_token_count'):
                                require(r[f]==src[key(r)][f],'EXACT_REUSE_ROW')
            wr=d['wiki']['rows'];require([(r['ordinal'],r['input_sha256'],r['predicted_tokens']) for r in wr]==wiki_ids,'WIKI_IDENTITY_MASK')
            require(len(wr)==128 and all(math.isfinite(r['nll']) for r in wr),'WIKI_FINITE')
            wn=fmean(r['nll'] for r in wr);require(math.isclose(wn,d['wiki']['mean_nll'],abs_tol=1e-12),'WIKI_MEAN')
            mr=d['mmlu']['rows'];require(len(mr)==32 and [r['row_sha256'] for r in mr]==[panel_digest(r) for r in dev],'MMLU_IDENTITY')
            for r,source_row in zip(mr,dev):
                require(r['gold']==source_row['answer'],'MMLU_GOLD_BINDING')
                require(len(r['alternative_nll'])==4 and all(math.isfinite(v) for v in r['alternative_nll']),'MMLU_NLL_FINITE')
                require(all(math.isclose(p,math.exp(-v),rel_tol=1e-6,abs_tol=1e-40) for p,v in zip(r['alternative_probability'],r['alternative_nll'])),'MMLU_PROBABILITY_NLL')
                pred=mmlu_prediction(r['alternative_probability']);require(pred==r['prediction'] and r['correct']==(pred==r['gold']) and r['tie_or_underflow_invalid']==(pred==-1),'MMLU_INTEGER_ALTERNATIVE')
            correct=sum(r['correct'] for r in mr);invalid=sum(r['prediction']==-1 for r in mr)
            require(correct==d['mmlu']['correct'] and invalid==d['mmlu']['invalid'] and correct/32==d['mmlu']['accuracy'],'MMLU_AGGREGATE')
            general.append(dict(arm=arm,batch=b,wiki_nll=wn,wiki_sequences=128,wiki_tokens=sum(r['predicted_tokens'] for r in wr),mmlu_correct=correct,mmlu_denominator=32,mmlu_invalid=invalid,mmlu_accuracy=correct/32,**{'wiki_'+k:v for k,v in distribution(r['nll'] for r in wr).items()}))
        for m in MULT:
            summarize([r for b in range(51,61) for r in docs[b]['current']['metrics'][m]['rows']],arm,60,'ONLINE_AT_WRITE_1000_DIFFERENT_STATES',m)
        for b in (55,60):
            ann=annotation(records,b*100)
            for m in MULT:
                before=[r for cb in range(51,b+1) for r in docs[cb]['current']['metrics'][m]['rows']]
                after=docs[b]['suffix']['metrics'][m]['rows']
                for g in ('ALL','ACTIVE_TARGET','SUPERSEDED','UNKNOWN_RELATION'):
                    aa=before if g=='ALL' else [r for r in before if ann[r['case_id']]==g]
                    bb=after if g=='ALL' else [r for r in after if ann[r['case_id']]==g]
                    transitions.append(dict(contrast='AT_WRITE_TO_SUFFIX',before=arm,after=arm,batch=b,population='suffix',metric=m,group=g,**pairs(aa,bb,m)))
                for cb in range(51,b+1):
                    a=docs[cb]['current']['metrics'][m]['rows'];z=subset(after,[r['case_id'] for r in records[(cb-1)*100:cb*100]])
                    transitions.append(dict(contrast='COHORT_AT_WRITE_TO_SUFFIX',before=arm,after=arm,batch=b,cohort_batch=cb,population='suffix',metric=m,group='ALL',**pairs(a,z,m)))
        for m in MULT:
            before=docs[55]['suffix']['metrics'][m]['rows'];after=subset(docs[60]['suffix']['metrics'][m]['rows'],[r['case_id'] for r in records[5000:5500]])
            transitions.append(dict(contrast='W55_TO_W60_FIXED_FIRST_SUFFIX500',before=arm,after=arm,batch=60,population='suffix500',metric=m,group='ALL',**pairs(before,after,m)))
        old=json.loads((core/'output'/arm/'evaluation.json').read_text())
        for pop in ('current','historical'):
            for m in MULT:
                a=old[pop]['metrics'][m]['rows'];z=docs[51][pop]['metrics'][m]['rows']
                static.append(dict(arm=arm,population=pop,metric=m,**pairs(a,z,m),old_source='46451',new_source='46475'))
    contrasts=[('N4',a) for a in ARMS if a!='N4']+[('S75','RES8'),('REFIT4','RES8'),('FULL8','RES8')]
    for before,after in contrasts:
        for b in range(51,61):
            for pop in ('current','historical')+ (('suffix',) if b==55 else ()) + (('suffix','entry_old','fullseen') if b==60 else ()):
                for m in MULT:
                    a=all_docs[before][b][pop]['metrics'][m]['rows'];z=all_docs[after][b][pop]['metrics'][m]['rows']
                    transitions.append(dict(contrast='CROSS_ARM_SAME_PROMPTS_DIFFERENT_STATE',before=before,after=after,batch=b,population=pop,metric=m,group='ALL',**pairs(a,z,m)))
            a=all_docs[before][b]['wiki']['rows'];z=all_docs[after][b]['wiki']['rows'];dd=[y['nll']-x['nll'] for x,y in zip(a,z)]
            require([(r['ordinal'],r['input_sha256']) for r in a]==[(r['ordinal'],r['input_sha256']) for r in z],'WIKI_PAIRED_ORDER')
            general_pairs.append(dict(before=before,after=after,batch=b,population='wiki',denominator=128,worse=sum(v>0 for v in dd),better=sum(v<0 for v in dd),unchanged=sum(v==0 for v in dd),**distribution(dd)))
            a=all_docs[before][b]['mmlu']['rows'];z=all_docs[after][b]['mmlu']['rows']
            require([r['row_sha256'] for r in a]==[r['row_sha256'] for r in z],'MMLU_PAIRED_ORDER')
            general_pairs.append(dict(before=before,after=after,batch=b,population='mmlu',denominator=32,lost=sum(x['correct'] and not y['correct'] for x,y in zip(a,z)),gained=sum(not x['correct'] and y['correct'] for x,y in zip(a,z)),retained=sum(x['correct'] and y['correct'] for x,y in zip(a,z)),failed_both=sum(not x['correct'] and not y['correct'] for x,y in zip(a,z)),prediction_changed=sum(x['prediction']!=y['prediction'] for x,y in zip(a,z))))
    contract=json.loads((Path(__file__).resolve().parents[3]/'plans/global/2026-09-13-low-cost-write-donor-pilot-contract.json').read_text())
    frontier=[]
    for b in range(51,61):
        for row in quality({arm:all_docs[arm][b] for arm in ARMS},contract):
            frontier.append(dict(row,batch=b))
    tables={'batchmetrics':rates,'nll-distributions':nll,'general-panels':general,'general-paired-transitions':general_pairs,'cohort-final':cohort,'paired-transitions':transitions,'static-core-comparison':static,'metric-validation':checks,'metric-input-inventory':files,'quality-frontier':frontier}
    tables['finalpopulationmetrics']=[r for r in rates if r['batch']==60]
    for name,rows in tables.items():write_csv(out/(name+'.csv'),rows)
    supporting=[Path(__file__),Path(__file__).with_name('review_metrics.py'),core/'panel-lock.json',core/'inputs.json',Path(inputs['wiki_panel']),Path(inputs['mmlu100'])]
    receipt=dict(status='INDEPENDENT_METRIC_REDUCTION_PASS',arms=list(ARMS),batch_evaluations=len(files),checks=len(checks),dataset_sha256=DATA_SHA,imputation=0,nonfinite=0,duplicate=0,new_forward=0,raw_broadcast='NO_BROADCAST_NOT_REQUIRED',table_rows={k:len(v) for k,v in tables.items()},scope='METRIC_IDENTITY_COUNTS_ONLY; CPU_STATE_AND_GPU_PARITY_SEPARATE',inputs=files,supporting_inputs=[dict(path=str(p),bytes=p.stat().st_size,sha256=file_sha(p)) for p in supporting],quantiles='LINEAR_(N-1)*P',reused_subsets='SAME_ENDPOINT_EXACT_ROWS_NOT_ADDITIONAL_EVALUATIONS',overwrite='EXACT_KNOWN_SUBJECT_RELATION_TARGET_ONLY_NOT_SEMANTIC_EQUIVALENCE',table_hashes={name:file_sha(out/(name+'.csv')) for name in tables})
    dump(out/'metric-reduction-receipt.json',receipt)
    return receipt

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--core',type=Path,required=True);p.add_argument('--first-only',action='store_true');a=p.parse_args()
    require(file_sha(a.dataset)==DATA_SHA,'FIXED10K_SHA');records=json.loads(a.dataset.read_text());require(len(records)==10000 and len({r['case_id'] for r in records})==10000,'FIXED10K_UNIQUE')
    first(a.root,a.out,records)
    if not a.first_only:reduce(a.root,a.out,records,a.core)
    print(json.dumps({'out':str(a.out),'first_table_sha256':file_sha(a.out/'first-final-table.csv')}))

if __name__=='__main__':main()
