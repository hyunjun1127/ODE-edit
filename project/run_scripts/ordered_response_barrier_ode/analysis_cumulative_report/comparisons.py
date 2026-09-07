"""Matched identities, metadata-only overwrite audit and descriptive associations."""
import argparse,json
from pathlib import Path
from .common import *
from .performance import evaluate

def conflict_ledger(ordered):
    """Exact subject+relation, different requested target; never outcome-selected."""
    previous={};out=[]
    for ordinal,record,request_hash in ordered:
        r=record['requested_rewrite'];key=(r['subject'],r['relation_id']);target=canonical_hash(r['target_new'])
        for old_ordinal,old_id,old_hash,old_target in previous.get(key,[]):
            if old_target!=target:
                out.append(dict(previous_ordinal=old_ordinal,next_ordinal=ordinal,previous_case_id=old_id,next_case_id=record['case_id'],
                    previous_request_sha256=old_hash,next_request_sha256=request_hash,subject_relation_sha256=canonical_hash(key),
                    previous_target_sha256=old_target,next_target_sha256=target,classification='EXACT_SUBJECT_RELATION_DIFFERENT_TARGET_METADATA_ONLY'))
        previous.setdefault(key,[]).append((ordinal,record['case_id'],request_hash,target))
    return out

def build(perf,mech,root,out):
    perf,mech,root,out=map(Path,(perf,mech,root,out));require(not out.exists(),'create-once');out.mkdir(parents=True)
    request=pd.read_csv(perf/'request-state-records.csv.gz');pairs=pd.read_csv(perf/'prompt-pair-records.csv.gz')
    current=request[request.stage=='CURRENT_B100'];seen=request[request.stage=='CHECKPOINT_W_ON_ALL_SEEN_REQUESTS']
    pre=read(root/'preflight.json');ds=pre['stream']['dataset'];require(sha256_file(Path(ds['absolute_path']))==ds['sha256'],'dataset SHA')
    records={r['case_id']:r for r in read(ds['absolute_path'])};ordered=[]
    for b in range(1,11):
        j=read(root/'results/task-0/arm-O'/f'batch-{b:02d}.json')
        ordered += [((b-1)*100+i,records[int(cid)],h) for i,(cid,h) in enumerate(zip(j['case_ids'],j['request_sha256']))]
    conflicts=conflict_ledger(ordered);affected={r['previous_request_sha256'] for r in conflicts}
    frame_write(out/'overwrite-conflicts.csv',conflicts or [dict(classification='NO_EXACT_METADATA_CONFLICT_FOUND')])
    rows=[];online=[];gaps=[];prompt_retention=[];derived=[]
    for cell in CELLS:
        for arm in ARMS:
            c=current[(current.cell==cell)&(current.arm==arm)]
            s=seen[(seen.cell==cell)&(seen.arm==arm)]
            final=s[s.batch==10].merge(c[['request_sha256','rewrite_success']],on='request_sha256',suffixes=('','_atwrite'),validate='one_to_one')
            for label,ids in [('ALL',set(final.request_sha256)),('OVERWRITE_METADATA',affected),('NO_OVERWRITE_METADATA',set(final.request_sha256)-affected)]:
                g=final[final.request_sha256.isin(ids)];before=g.rewrite_success_atwrite.astype(bool);now=g.rewrite_success.astype(bool)
                rows.append(dict(cell=cell,arm=arm,group=label,requests=len(g),initial_success=int(before.sum()),initial_failure=int((~before).sum()),loss=int((before&~now).sum()),recovery=int((~before&now).sum()),final_RS=int(now.sum()),conditional_loss_den=int(before.sum()),primary_exclusions=0))
            for b in range(1,11):
                own=c[c.batch<=b];cum=s[s.batch==b];o=summarize_requests(own);cu=summarize_requests(cum)
                online.append(dict(cell=cell,arm=arm,batch=b,evaluation_type='ONLINE_AT_OWN_WRITE_NOT_ONE_WEIGHT',**o))
                gaps.append(dict(cell=cell,arm=arm,batch=b,**{k+'_cumulative_minus_online_pp':100*(cu[k+'_rate']-o[k+'_rate']) for k in ('RS','PS','PS_strict','NS')}))
            p=pairs[(pairs.cell==cell)&(pairs.arm==arm)]
            fp=p[(p.stage=='CHECKPOINT_W_ON_ALL_SEEN_REQUESTS')&(p.batch==10)]
            ip=p[p.stage=='CURRENT_B100'];keys=['request_sha256','category','prompt_index']
            m=fp.merge(ip,on=keys,suffixes=('_final','_own'),validate='one_to_one')
            for target in ('new','true'):require((m['input_identity_sha256_'+target+'_final']==m['input_identity_sha256_'+target+'_own']).all(),'own vs final prompt input identity')
            for category,g in m.groupby('category'):
                a=g.success_final.astype(bool);o=g.success_own.astype(bool)
                prompt_retention.append(dict(cell=cell,arm=arm,category=category,prompts=len(g),atwrite_success=int(o.sum()),final_success=int(a.sum()),loss=int((o&~a).sum()),recovery=int((~o&a).sum()),conditional_loss_den=int(o.sum()),paired_order_hash=canonical_hash(list(zip(g.request_sha256,g.prompt_index))),
                    **{k+'_change_'+sk:sv for k in ('nll_new','nll_true','margin_true_minus_new') for sk,sv in strict_stats(g[k+'_final']-g[k+'_own']).items()}))
        for b in range(1,11):
            j=read(root/f'results/task-{CELLS.index(cell)}'/f'arm-ORBFH/batch-{b:02d}.json');d=j['derived_endpoint']
            row=dict(cell=cell,batch=b,arm='ORBHit',primary_arm_denominator=0,hypothetical_sequential_chain=False)
            if d is None:row['status']='NOT_RECORDED'
            else:
                row.update(status=d.get('status'),state_version=d.get('state_version'))
                if d.get('evaluation') is not None:
                    rq,_=evaluate(d['evaluation'],j['request_sha256'],{},True);row.update(summarize_requests(rq))
            derived.append(row)
    frame_write(out/'atwrite-final-conflict-strata.csv',rows);frame_write(out/'online-own-write.csv',online)
    frame_write(out/'cumulative-vs-online-gaps.csv',gaps);frame_write(out/'own-write-final-prompt-transitions.csv',prompt_retention);frame_write(out/'derived-prefix-inventory.csv',derived)
    h=pd.read_csv(mech/'historical-checkpoint-residual.csv');a=pd.read_csv(mech/'layer-actual-action.csv');b=pd.read_csv(mech/'batch-mechanism.csv');core=pd.read_csv(perf/'cumulative-core.csv')
    assoc=[];joined=[]
    for cell in CELLS:
        for arm in ARMS:
            q=h[(h.cell==cell)&(h.arm==arm)].groupby('batch')[['q_after_mean','q_worsened','distance_from_own_immediate_activation_mean']].mean()
            f=core[(core.cell==cell)&(core.arm==arm)].set_index('batch')
            z=b[(b.cell==cell)&(b.arm==arm)].set_index('batch')
            l8=a[(a.cell==cell)&(a.arm==arm)&(a.layer==8)].set_index('batch')
            g=f[['RS_rate','PS_rate','NS_rate']].join(q).join(z[['batch_net_actual_frobenius','W0_net_actual_frobenius']]).join(l8[['batch_squared_share']])
            joined.append(g.reset_index().assign(cell=cell,arm=arm))
            for mechanism in ['q_after_mean','distance_from_own_immediate_activation_mean','batch_net_actual_frobenius','W0_net_actual_frobenius','batch_squared_share']:
                for metric in ['RS_rate','PS_rate','NS_rate']:
                    valid=g[[mechanism,metric]].dropna();coef=valid.corr(method='spearman').iloc[0,1] if len(valid)>1 and valid.nunique().min()>1 else None
                    assoc.append(dict(cell=cell,arm=arm,mechanism=mechanism,metric=metric,checkpoint_n=len(valid),spearman=coef,status='DESCRIPTIVE_REPEATED_CHECKPOINT_NO_CAUSAL_CLAIM' if coef is not None else 'NOT_IDENTIFIABLE_CONSTANT_OR_MISSING'))
    frame_write(out/'mechanism-cumulative-associations.csv',assoc);frame_write(out/'mechanism-performance-checkpoints.csv',pd.concat(joined,ignore_index=True))
    receipt=dict(status='MATCHED_COMPARISONS_COMPLETE',overwrite_conflicts=len(conflicts),overwritten_requests=len(affected),dataset_sha256=ds['sha256'],dataset_path=ds['absolute_path'],relation_key='EXACT_SUBJECT_AND_RELATION_DIFFERENT_TARGET_HASH',causal_claim=False,rows={p.name:len(pd.read_csv(p)) for p in out.glob('*.csv')})
    write_json_once(out/'comparisons-receipt.json',receipt);return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=RAW)
    for k in ('performance','mechanism','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();print(json.dumps(build(a.performance,a.mechanism,a.root,a.output)))
