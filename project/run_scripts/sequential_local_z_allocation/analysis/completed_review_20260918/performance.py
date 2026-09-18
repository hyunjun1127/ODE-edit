"""Read-only independent paired/per-population analysis. No model imports."""
import collections, math
import numpy as np
from reducer import *

def stats(values,prefix):
    a=np.asarray(values,dtype=float)
    return {prefix+'_'+k:float(v) for k,v in zip(['mean','p00','p50','p95','p99','p100'],[a.mean(),*np.quantile(a,[0,.5,.95,.99,1])])} if len(a) else {prefix+'_mean':None}
def summary(rows,tag,**ctx):
    n=len(rows)
    if not n:return dict(**ctx,metric=tag,n=0)
    target='true' if tag=='NS' else 'new';bits=[success(r,tag) for r in rows]
    out=dict(**ctx,metric=tag,n=n,count=sum(bits),percent=100*sum(bits)/n,strict_count=sum(r[target+'_strict'] for r in rows),strict_percent=100*sum(r[target+'_strict'] for r in rows)/n,token_correct=sum(r[target+'_token_correct'] for r in rows),token_count=sum(r[target+'_token_count'] for r in rows),ties=sum(r['new_nll']==r['true_nll'] for r in rows))
    for field in ['new_nll','true_nll']:out.update(stats([r[field] for r in rows],field))
    out.update(stats([(r['new_nll']-r['true_nll']) if tag=='NS' else (r['true_nll']-r['new_nll']) for r in rows],'desired_margin'))
    if tag=='PS':
        groups=collections.defaultdict(list)
        for r in rows:groups[r['case_id']].append(r)
        out.update(two_P_n=len(groups),two_P_strict=sum(len(g)==2 and all(r['new_strict'] for r in g) for g in groups.values()))
    return out
def transition(before,after,tag,**ctx):
    a={r['identity']:r for r in before};b={r['identity']:r for r in after};assert set(a)==set(b)
    target='true' if tag=='NS' else 'new';details=[]
    for ident,x in a.items():
        y=b[ident];assert (x['case_id'],x['prompt_index'])==(y['case_id'],y['prompt_index'])
        bs,ys=success(x,tag),success(y,tag)
        details.append(dict(case_id=x['case_id'],identity=ident,before_success=bs,after_success=ys,delta_new_nll=y['new_nll']-x['new_nll'],delta_true_nll=y['true_nll']-x['true_nll'],delta_desired_nll=y[target+'_nll']-x[target+'_nll'],strict_lost=x[target+'_strict'] and not y[target+'_strict'],strict_gained=not x[target+'_strict'] and y[target+'_strict']))
    out=dict(**ctx,metric=tag,n=len(details),before=sum(x['before_success'] for x in details),after=sum(x['after_success'] for x in details),lost=sum(x['before_success'] and not x['after_success'] for x in details),gained=sum(not x['before_success'] and x['after_success'] for x in details),retained=sum(x['before_success'] and x['after_success'] for x in details),both_failure=sum(not x['before_success'] and not x['after_success'] for x in details),strict_lost=sum(x['strict_lost'] for x in details),strict_gained=sum(x['strict_gained'] for x in details))
    out['delta_pp']=100*(out['after']-out['before'])/out['n']
    for field in ['delta_new_nll','delta_true_nll','delta_desired_nll']:out.update(stats([x[field] for x in details],field))
    out['desired_nll_worse_count']=sum(x['delta_desired_nll']>0 for x in details)
    return out,details
def main():
    records=load('/data/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1/counterfact.json')[:1000];ordmap={r['case_id']:i for i,r in enumerate(records)}
    fact=lambda r:(r['requested_rewrite']['subject'],r['requested_rewrite']['relation_id'])
    latest={fact(r):i for i,r in enumerate(records)};active={records[i]['case_id'] for i in latest.values()}
    w0=load(load(ROOT/'execution.lock.json')['W0_observation']['path']);validate(w0,records)
    final={};allmetrics=[];trans=[];paired=[];generic=[];candidateobs=[];w0n=[];rowchanges=[]
    for arm in ARMS:
        out=ROOT/'arms'/arm/'attempt-v1/output';full=load(out/'B010/seen-full.json');validate(full,records);final[arm]=full
        w5=load(out/'B005/seen-full.json');validate(w5,records[:500]);atwrite={k:[] for k in MULT};entryall={k:[] for k in MULT}
        for b in range(1,11):
            bd=out/f'B{b:03d}';entry=load(bd/'entry-current.json');selected=load(bd/'selected-current.json');cur=records[(b-1)*100:b*100]
            validate(entry,cur);validate(selected,cur)
            for tag in MULT:
                er=entry['metrics'][tag]['rows'];sr=selected['metrics'][tag]['rows'];atwrite[tag]+=sr;entryall[tag]+=er
                allmetrics += [summary(rs,tag,arm=arm,view=view,batch=b,population='CURRENT100') for rs,view in [(er,'ENTRY'),(sr,'AT_WRITE')]]
                t,d=transition(er,sr,tag,arm=arm,comparison='ENTRY_TO_ATWRITE',batch=b);trans.append(t)
            if b in [5,10]:
                dev=load(bd/'Dev128.json');generic.append(dict(arm=arm,batch=b,role='Dev128_OBSERVER',D=dev['D'],documents=dev['denominator']))
            selection=load(bd/'selection.json')
            generic.append(dict(arm=arm,batch=b,role='S64_SELECTED_ONLINE',D=selection['selected']['scores']['base_kl'],documents=64))
            generic.append(dict(arm=arm,batch=b,role='S64_OWN_N4_ONLINE',D=selection['candidates'][0]['scores']['base_kl'],documents=64))
            for binding in sorted((bd/'candidate-observer').glob('*-binding.json')):
                br=load(binding);d=load(br['rows']['path']);validate(d,cur)
                for tag in MULT:candidateobs.append(summary(d['metrics'][tag]['rows'],tag,arm=arm,batch=b,candidate=int(binding.name.split('-')[0]),gates=json.dumps(br['gates']),state=br['state_token'],selected=br['state_token']==selection['selected']['state_token'],scope='ACTUAL_COMPLETED_SAME_ENTRY_NOT_CHAIN',score_reused=br['exact_endpoint_reuse']))
        for tag,mult in MULT.items():
            fr=full['metrics'][tag]['rows'];wr=w5['metrics'][tag]['rows'];fs=fr[:500*mult]
            for rs,view in [(fr,'W10_ALL1000'),(fs,'W10_FIRST500'),(fr[500*mult:],'W10_LAST500'),(wr,'W5_FIRST500'),(atwrite[tag],'POOLED_ATWRITE'),(entryall[tag],'POOLED_ENTRY')]:allmetrics.append(summary(rs,tag,arm=arm,view=view,batch=0,population='ALL'))
            for pop in ['ACTIVE','SUPERSEDED']:
                rs=[r for r in fr if (r['case_id'] in active)==(pop=='ACTIVE')];allmetrics.append(summary(rs,tag,arm=arm,view='W10',batch=0,population=pop))
            for comparison,a,b in [('ATWRITE_TO_W10',atwrite[tag],fr),('ATWRITE_TO_W5_FIRST500',atwrite[tag][:500*mult],wr),('W5_TO_W10_FIRST500',wr,fs),('W0_TO_W10',w0['metrics'][tag]['rows'],fr)]:
                t,ds=transition(a,b,tag,arm=arm,comparison=comparison,batch=0);trans.append(t)
                rowchanges.extend(dict(arm=arm,comparison=comparison,metric=tag,**x) for x in ds)
            for cohort in range(10):
                cr=fr[cohort*100*mult:(cohort+1)*100*mult];aw=atwrite[tag][cohort*100*mult:(cohort+1)*100*mult]
                allmetrics.append(summary(cr,tag,arm=arm,view='W10_COHORT',batch=cohort+1,population='ALL'))
                t,ds=transition(aw,cr,tag,arm=arm,comparison='COHORT_ATWRITE_TO_W10',batch=cohort+1);trans.append(t)
            if tag=='NS':
                good={r['identity'] for r in w0['metrics'][tag]['rows'] if success(r,tag)}
                w0n.append(summary([r for r in fr if r['identity'] in good],tag,arm=arm,view='W10_GIVEN_W0_SUCCESS'))
        print('METRICS',arm,flush=True)
    pairs=[('C4','N4'),('C48','C4'),('C48','F48'),('C48','G48'),('C45678','C48')]
    pairs += [(a,'N4') for a in ARMS if a not in ['N4','C4']]
    for a,b in pairs:
        for tag in MULT:
            t,ds=transition(final[b]['metrics'][tag]['rows'],final[a]['metrics'][tag]['rows'],tag,comparison=a+'-'+b,after_arm=a,before_arm=b)
            # Request-cluster bootstrap: average within each case, then resample 1000 cases.
            bycase=collections.defaultdict(list)
            for x in ds:bycase[x['case_id']].append(int(x['after_success'])-int(x['before_success']))
            values=np.array([np.mean(bycase[c]) for c in sorted(bycase)]);rng=np.random.default_rng(20260918)
            boot=np.mean(values[rng.integers(0,len(values),size=(2000,len(values)))],axis=1)*100
            t.update(ci95_low=float(np.quantile(boot,.025)),ci95_high=float(np.quantile(boot,.975)),bootstrap_repetitions=2000,bootstrap_seed=20260918,cluster_unit='CASE_NOT_PROMPT',cases=len(values));paired.append(t)
            rowchanges.extend(dict(arm=a,comparison=a+'-'+b,metric=tag,**x) for x in ds)
    csvout('performance.csv',allmetrics);csvout('retention.csv',trans);csvout('paired.csv',paired);csvout('generic.csv',generic);csvout('candidate-observer.csv',candidateobs);csvout('w0-conditioned-ns.csv',w0n)
    # Hash-only IDs plus numeric values; local-only per-row changes are not published.
    writejson(LOCAL/'paired-row-changes.json',rowchanges)
    overwrites=[dict(prior_ordinal=i,latest_ordinal=latest[fact(r)],prior_case=r['case_id'],latest_case=records[latest[fact(r)]]['case_id'],target_changed=r['requested_rewrite']['target_new']['str']!=records[latest[fact(r)]]['requested_rewrite']['target_new']['str']) for i,r in enumerate(records) if latest[fact(r)]!=i]
    writejson(REPORT/'population-summary.json',dict(requests=1000,distinct_case=len(ordmap),active=len(active),superseded=1000-len(active),overwrite_events=overwrites,fact_rule='EXACT_RAW_SUBJECT_RELATION_LATEST_RECEIVED',paired_local_path=str(LOCAL/'paired-row-changes.json'),paired_rows_sha256=sha(LOCAL/'paired-row-changes.json'),new_gpu=0,unobserved_failure_times='NOT_INFERRED'))
if __name__=='__main__':main()
