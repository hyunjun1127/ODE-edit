"""Canonical prompt-pair metrics from existing current and all-seen records."""
import argparse,json
from pathlib import Path
from .common import *

def evaluate(e,order,meta,endpoint=False):
    deep_identity(e)
    with canonical_v2_reducer():rq,_,pairs=evaluation(e,order,meta,endpoint)
    raw=pd.DataFrame(e['rows'])
    enriched=[]
    for cat in ('rewrite','rephrase','locality'):
        selected=pairs[pairs.category==cat].copy()
        for target in ('new','true'):
            ids=raw[raw.kind==cat+'_target_'+target][['request_sha256','prompt_index','input_identity_sha256','all_tokens_correct','correct_token_count','target_token_count']]
            keys=['request_sha256','prompt_index']
            ids=ids.rename(columns={k:k+'_'+target for k in ids.columns if k not in keys})
            selected=selected.merge(ids,on=keys,validate='one_to_one',sort=False)
        enriched.append(selected)
    return rq,pd.concat(enriched,ignore_index=True)

def distribution(pairs,meta):
    out=[]
    for category,p in pairs.groupby('category',sort=False):
        for target in ('new','true'):
            for unit in ('prompt','request_cluster'):
                g=p if unit=='prompt' else p.groupby('request_sha256',sort=False)[['nll_new','nll_true','margin_true_minus_new']].mean()
                out.append({**meta,'category':category,'target':target,'unit':unit,'prompt_den':len(p),
                    'strict_num':int(p['all_tokens_correct_'+target].sum()),
                    'correct_token_num':int(p['correct_token_count_'+target].sum()),'target_token_den':int(p['target_token_count_'+target].sum()),
                    **{'nll_'+k:v for k,v in strict_stats(g['nll_'+target]).items()},
                    **{'margin_true_minus_new_'+k:v for k,v in strict_stats(g.margin_true_minus_new).items()}})
    return out

def build(root,out):
    root,out=Path(root),Path(out);require(not out.exists(),'create-once output');out.mkdir(parents=True)
    requests=[];pairs_all=[];rates=[];distributions=[];age_rates=[];age_dist=[];cohorts=[];partitions=[];inputs=[]
    for cid,cell in enumerate(CELLS):
        cr=root/'results'/f'task-{cid}'
        for arm in ARMS:
            ar=cr/f'arm-{arm}';journals=[read(ar/f'batch-{b:02d}.json') for b in range(1,11)]
            order=[h for b in journals for h in b['request_sha256']];require(canonical_hash(order)==ORDER_ROOT,'common order')
            ordinal={h:i for i,h in enumerate(order)};initial=[]
            for b,j in enumerate(journals,1):
                for stage,e in [('ENTRY_CURRENT',j['entry_evaluation']),('CURRENT_B100',j['endpoint']['evaluation'])]:
                    meta=dict(cell=cell,arm=arm,batch=b,stage=stage)
                    rq,ps=evaluate(e,j['request_sha256'],meta,stage=='CURRENT_B100')
                    rq['cohort']=b;requests.append(rq);pairs_all.append(ps.assign(cohort=b))
                    rates.append({**meta,**summarize_requests(rq)});distributions.extend(distribution(ps,meta))
                    if stage=='CURRENT_B100':initial.append(rq)
                directory=ar/'cumulative'/f'W-after-B{b:02d}';cp=read(directory/'receipt.json');rqs=[];pss=[]
                for ci,em in enumerate(cp['evaluation_members'],1):
                    p=directory/em['path'];require(sha256_file(p)==em['sha256'],'evaluation file SHA');e=read(p)
                    require(e['W_sha256']==j['committed_weight_sha256'],'allseen W binding')
                    meta=dict(cell=cell,arm=arm,batch=b,stage='CHECKPOINT_W_ON_ALL_SEEN_REQUESTS')
                    rq,ps=evaluate(e['evaluation'],journals[ci-1]['request_sha256'],meta)
                    rqs.append(rq.assign(cohort=ci));pss.append(ps.assign(cohort=ci))
                    inputs.append(dict(path=str(p),sha256=em['sha256'],identity=em['identity_sha256']))
                rq=pd.concat(rqs,ignore_index=True);ps=pd.concat(pss,ignore_index=True)
                require(len(rq)==b*100 and not rq.request_sha256.duplicated().any(),'seen request count')
                requests.append(rq);pairs_all.append(ps);rates.append({**meta,**summarize_requests(rq)});distributions.extend(distribution(ps,meta))
                # Age definitions are relative to each checkpoint seen prefix.
                for name,lo,hi in [('early',0,b*20),('middle',b*20,b*80),('recent',b*80,b*100)]:
                    ids=set(order[lo:hi]);ra=rq[rq.request_sha256.isin(ids)];pa=ps[ps.request_sha256.isin(ids)]
                    am={**meta,'age':name,'start_ordinal':lo,'end_ordinal_exclusive':hi}
                    age_rates.append({**am,**summarize_requests(ra)});age_dist.extend(distribution(pa,am))
                initial_frame=pd.concat(initial,ignore_index=True)
                merged=rq.merge(initial_frame[['request_sha256','rewrite_success']],on='request_sha256',suffixes=('','_at_write'),validate='one_to_one')
                for cohort,g in merged.groupby('cohort'):
                    before=g.rewrite_success_at_write.astype(bool);now=g.rewrite_success.astype(bool)
                    cohorts.append({**meta,'cohort':int(cohort),'requests':len(g),'at_write_success':int(before.sum()),'now_success':int(now.sum()),
                        'initial_failure':int((~before).sum()),'success_to_loss':int((before&~now).sum()),'initial_failure_to_recovery':int((~before&now).sum()),
                        'loss_conditional_den':int(before.sum()),'order_hash':canonical_hash(g.request_sha256.tolist())})
                if b==10:
                    before=merged.rewrite_success_at_write.astype(bool);now=merged.rewrite_success.astype(bool)
                    partitions.append(dict(cell=cell,arm=arm,requests=len(merged),at_write_success=int(before.sum()),initial_failure=int((~before).sum()),success_to_loss=int((before&~now).sum()),recovery=int((~before&now).sum()),final_RS=int(now.sum()),overwrite='UNAVAILABLE_NO_RELATION_TARGET_CONFLICT_METADATA_IN_RUN_PUBLIC_ROWS'))
            print(f'PERFORMANCE {cell}/{arm}: current1000 + seen5500 canonical prompt pairs PASS',flush=True)
    req=pd.concat(requests,ignore_index=True);pairs=pd.concat(pairs_all,ignore_index=True);core=pd.DataFrame(rates)
    seen=core[core.stage=='CHECKPOINT_W_ON_ALL_SEEN_REQUESTS'];final=seen[seen.batch==10]
    require(len(seen)==200 and len(final)==20,'20arm 200seen checkpoint table')
    require(len(req[req.stage=='CHECKPOINT_W_ON_ALL_SEEN_REQUESTS'])==110000,'110000seen records')
    require(not req.duplicated(['cell','arm','batch','stage','request_sha256']).any(),'request state duplicate')
    require(not pairs.duplicated(['cell','arm','batch','stage','request_sha256','category','prompt_index']).any(),'prompt state duplicate')
    compare=[];paired=[];transitions=[]
    for cell in CELLS:
        for b in range(1,11):
            ps=pairs[(pairs.cell==cell)&(pairs.batch==b)&(pairs.stage=='CHECKPOINT_W_ON_ALL_SEEN_REQUESTS')]
            for arm in ARMS[1:]:
                q=ps[ps.arm==arm].merge(ps[ps.arm=='O'],on=['request_sha256','category','prompt_index'],suffixes=('_arm','_O'),validate='one_to_one')
                require(len(q)==b*100*13,'paired denominator')
                for target in ('new','true'):require((q['input_identity_sha256_'+target+'_arm']==q['input_identity_sha256_'+target+'_O']).all(),'paired prompt input identity')
                for category,g in q.groupby('category',sort=False):
                    a=g.success_arm.astype(bool);o=g.success_O.astype(bool)
                    paired.append(dict(cell=cell,arm=arm,reference='O',batch=b,category=category,prompt_n=len(g),arm_success=int(a.sum()),O_success=int(o.sum()),loss=int((o&~a).sum()),recovery=int((~o&a).sum()),both_success=int((o&a).sum()),both_failure=int((~o&~a).sum()),delta_rate=float(a.mean()-o.mean()),order_hash=canonical_hash(list(zip(g.request_sha256,g.prompt_index.astype(int)))),
                        **{key+'_delta_'+k:v for key in ['nll_new','nll_true','margin_true_minus_new'] for k,v in strict_stats(g[key+'_arm']-g[key+'_O']).items()}))
            cg=seen[(seen.cell==cell)&(seen.batch==b)].set_index('arm')
            for arm,reference in [(a,'O') for a in ARMS[1:]]+[('NQFIX','QCL'),('ORBFH','NQFIX'),('ORBFH','JAC')]:
                compare.append(dict(cell=cell,batch=b,arm=arm,reference=reference,**{m+'_delta_pp':100*(cg.loc[arm,m+'_rate']-cg.loc[reference,m+'_rate']) for m in ('RS','PS','PS_strict','NS','rewrite_acc','rephrase_acc')}))
    for (cell,arm),g in seen.groupby(['cell','arm'],sort=False):
        g=g.sort_values('batch')
        for before,after in zip(g.to_dict('records'),g.to_dict('records')[1:]):
            transitions.append(dict(cell=cell,arm=arm,previous_batch=before['batch'],batch=after['batch'],**{m+'_delta_pp':100*(after[m+'_rate']-before[m+'_rate']) for m in ('RS','PS','PS_strict','NS')}))
    tables={'performance-all-scopes.csv':core,'cumulative-core.csv':seen,'final-20-arm.csv':final,'current-B100.csv':core[core.stage=='CURRENT_B100'],
        'category-distributions.csv':pd.DataFrame(distributions),'age-rates.csv':pd.DataFrame(age_rates),'age-distributions.csv':pd.DataFrame(age_dist),
        'retention-matrix.csv':pd.DataFrame(cohorts),'atwrite-final-partitions.csv':pd.DataFrame(partitions),'paired-prompt-transitions.csv':pd.DataFrame(paired),
        'paired-arm-deltas.csv':pd.DataFrame(compare),'checkpoint-transitions.csv':pd.DataFrame(transitions),
        'request-state-records.csv.gz':req,'prompt-pair-records.csv.gz':pairs}
    for name,f in tables.items():frame_write(out/name,f)
    first=['# ORBODE cumulative rerun — final W_B10 전체1,000 요청','',
        'RS=new<true, PS=prompt별 new<true, NS=true<new; ties fail. 20 independent arm chains, 각 B100×10 sequential. 아래는 각 final W10에서 실제 전체1,000 평가이며 current B100/online pooling이 아니다. W1…W10 실제 seen-prefix 200행은 cumulative-core.csv. 무결성 full checkpoint CPU 재해시는 별도 진행/receipt로 결속한다.','',
        '|Cell|Arm|RS /1000|PS /2000|strict PS /1000|NS /10000|rewrite acc /1000|rephrase acc /2000|','|---|---|---:|---:|---:|---:|---:|---:|']
    for row in final.to_dict('records'):
        first.append('|'+row['cell']+'|'+row['arm']+'|'+'|'.join(f"{int(row[k+'_num'])}/{int(row[k+'_den'])} ({row[k+'_rate']*100:.2f}%)" for k in ('RS','PS','PS_strict','NS','rewrite_acc','rephrase_acc'))+'|')
    with (out/'first-final-cumulative-table-ko.md').open('x') as f:f.write('\n'.join(first)+'\n')
    audit=dict(status='CANONICAL_CURRENT_CUMULATIVE_REDUCER_PASS',cells=4,arms=20,checkpoints=200,cumulative_request_states=110000,final_request_states=20000,final_is_subset=True,rows={k:len(v) for k,v in tables.items()},inputs=inputs,source=SOURCE,new_evaluator=0,model_GPU=0)
    audit['identity_sha256']=canonical_hash(audit);write_json_once(out/'performance-receipt.json',audit);return audit

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=RAW);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(build(a.root,a.output)['rows']))
