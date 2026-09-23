"""CPU-only independent review. No runtime/model/evaluator/scheduler imports.

Input paths are restricted to terminal attempts established by the accounting
snapshot. Raw prompt/target/token rows stay local; Git receives aggregates/IDs.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import os
import time
import numpy as np

ROOT=Path('/data/janghj/ODE-edit/local/alpha-key-concentration-causal/20260923-r1')
REPO=Path(__file__).resolve().parents[3]
PACKAGE=REPO/'experiment-reports/servers/server4/completed-jobs-review-2026-09-24-v1'
SCRATCH=Path('/data/janghj/ODE-edit/local/server4-completed-jobs-review/20260924-v1')
WRITERS=ROOT/'execution/attempt-r4/writers'
ENTRIES=(50,70,80,90)
BRANCHES=('NATIVE','SHAM','H5','H6','H56','MASS56')


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def read(p):return json.loads(Path(p).read_text())


def write_json(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n')


def table(name,rows,fields=None):
    p=PACKAGE/name;p.parent.mkdir(parents=True,exist_ok=True)
    cols=fields or list(dict.fromkeys(k for r in rows for k in r)) or ['status']
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols,lineterminator='\n');w.writeheader();w.writerows(rows)
    return p


def binding(a,b):
    for key in ('case_id','prompt_index','prompt','endpoint_id'):
        assert a[key]==b[key],('PAIR_IDENTITY',key)
    for key in ('prompt_token_ids_sha256','vocab_size','prompt_token_count'):
        assert a['full_vocab'][key]==b['full_vocab'][key],('PAIR_TOKEN_IDENTITY',key)
    key=[a['case_id'],a['prompt_index'],a['prompt'],a['target'],b['target'],a['target_token_ids'],b['target_token_ids']]
    return hashlib.sha256(json.dumps(key,ensure_ascii=False).encode()).hexdigest()


def token_stats(row):
    truth=row['target_token_ids'];pred=row['token_predictions'];flags=row['token_correct']
    assert len(truth)==len(pred)==len(flags)>0
    calculated=[a==b for a,b in zip(truth,pred)]
    assert calculated==flags and all(calculated)==row['all_tokens_correct']
    f=row['full_vocab'];assert f['full_vocab_checked'] and f['vocab_size']==128256
    assert f['true_token_ids']==truth and f['argmax_token_ids']==pred
    assert len(f['scoring_positions_unpadded'])==len(truth)
    unique=[ok and t==1 for ok,t in zip(calculated,f['argmax_tie_counts'])]
    assert all(unique)==f['all_unique_argmax_correct']
    return sum(calculated),len(truth),all(calculated),all(unique)


def reduce_raw(path):
    obj=read(path);raw=obj['raw_local_only'];result={}
    if 'true' in raw and 'new' in raw:
        raw={'locality_target_true':raw['true'],'locality_target_new':raw['new']}
    for metric,prefix,desired in [('RS','rewrite','new'),('PS','rephrase','new'),('NS','locality','true')]:
        if prefix+'_target_new' not in raw:continue
        news=raw[prefix+'_target_new'];trues=raw[prefix+'_target_true'];assert len(news)==len(trues)>0
        rows=[];seen=set()
        for a,b in zip(news,trues):
            identity=binding(a,b);assert identity not in seen;seen.add(identity)
            new,true=float(a['nll']),float(b['nll']);assert math.isfinite(new) and math.isfinite(true)
            n=token_stats(a);t=token_stats(b);want=n if desired=='new' else t
            rows.append(dict(identity=identity,case_id=a['case_id'],prompt_index=a['prompt_index'],
                endpoint=a['endpoint_id'],new_nll=new,true_nll=true,desired_nll=new if desired=='new' else true,
                desired_margin=true-new if desired=='new' else new-true,success=new<true if desired=='new' else true<new,
                tie=new==true,correct=want[0],tokens=want[1],strict=want[2],unique_strict=want[3],
                new_correct=n[0],new_tokens=n[1],new_strict=n[2],true_correct=t[0],true_tokens=t[1],true_strict=t[2]))
        result[metric]=rows
        if metric in obj['compact'].get('metrics',{}):
            stored=obj['compact']['metrics'][metric]
            assert (sum(r['success'] for r in rows),len(rows))==(stored['numerator'],stored['denominator'])
    return result


def paired(before,after,bootstrap=False):
    assert [r['identity'] for r in before]==[r['identity'] for r in after]
    n=len(before);lost=[];gained=[];tf_lost=[];tf_gained=[];d=[];by_case={}
    for a,b in zip(before,after):
        if a['success'] and not b['success']:lost.append(a['identity'])
        if not a['success'] and b['success']:gained.append(a['identity'])
        if a['strict'] and not b['strict']:tf_lost.append(a['identity'])
        if not a['strict'] and b['strict']:tf_gained.append(a['identity'])
        d.append(b['desired_nll']-a['desired_nll'])
        by_case.setdefault(a['case_id'],[]).append(int(b['success'])-int(a['success']))
    result=dict(n=n,lost=len(lost),gained=len(gained),delta_pp=100*(len(gained)-len(lost))/n,
        tf_lost=len(tf_lost),tf_gained=len(tf_gained),desired_nll_delta_mean=float(np.mean(d)),
        desired_nll_delta_p95=float(np.quantile(d,.95)),desired_nll_delta_p99=float(np.quantile(d,.99)),
        desired_nll_delta_max=max(d),lost_ids=';'.join(lost),gained_ids=';'.join(gained))
    if bootstrap:
        # Same case sampled as one cluster; variable cluster denominators preserved.
        values=list(by_case.values());s=np.array([sum(x) for x in values]);counts=np.array([len(x) for x in values])
        idx=np.random.default_rng(20260924).integers(0,len(values),(2000,len(values)))
        draws=100*s[idx].sum(1)/counts[idx].sum(1)
        result.update(cluster_cases=len(values),bootstrap_seed=20260924,bootstrap_repeats=2000,
                      descriptive_ci_low=float(np.quantile(draws,.025)),descriptive_ci_high=float(np.quantile(draws,.975)))
    return result


def metrics():
    rows=[];pairs=[];joint_rows=[];retention=[];checks=[];kr={}
    for e in ENTRIES:
        native={}
        for b in BRANCHES:
            paths=sorted((WRITERS/f'W{e:03d}'/b/'stages').glob('*/current.json'))
            for p in paths:
                stage=p.parent.name;data=reduce_raw(p)
                for k,v in data.items():rows.append(dict(entry=e,branch=b,stage=stage,panel='Current',metric=k,**aggregate(v)))
                joint_rows.append(dict(entry=e,branch=b,stage=stage,**joint(data)))
                if stage=='history':
                    if b=='NATIVE':native=data
                    else:
                        for k,v in data.items():pairs.append(dict(entry=e,branch=b,comparison='same-entry NATIVE',panel='Current',metric=k,**paired(native[k],v,True)))
            for p in sorted((WRITERS/f'W{e:03d}'/b/'stages').glob('*/N512.json')):
                data=reduce_raw(p)['NS'];obj=read(p)['compact'];assert len(data)==512
                rows.append(dict(entry=e,branch=b,stage=p.parent.name,panel='N512',metric='NS',**aggregate(data)))
                assert sum(r['success'] for r in data)==obj['pairwise_NS_numerator']
                baseline=[bool(r['w0_true_argmax_correct']) for r in obj['rows']]
                now=[r['strict'] for r in data]
                retention.append(dict(entry=e,branch=b,stage=p.parent.name,panel='N512',w0_correct=sum(baseline),
                    retained=sum(a and z for a,z in zip(baseline,now)),gross_lost=sum(a and not z for a,z in zip(baseline,now)),
                    recovered_from_w0_failure=sum(not a and z for a,z in zip(baseline,now))))
            for hp in sorted((WRITERS/f'W{e:03d}'/b/'stages').glob('*/H512.json')):
                if hp.parent.name=='history':continue
                for k,v in reduce_raw(hp).items():rows.append(dict(entry=e,branch=b,stage=hp.parent.name,panel='H512_ALL',metric=k,**aggregate(v)))
            p=WRITERS/f'W{e:03d}'/b/'stages/history/H512.json';data=reduce_raw(p);obj=read(p)['compact']
            masks=obj['masks'];assert masks['statistics_denominator']==masks['statistics_weight_sum']==512
            assert masks['selection_uses_success'] is False and obj['controller_influence']==0
            checks.append(dict(entry=e,branch=b,mask=masks))
            for k,v in data.items():rows.append(dict(entry=e,branch=b,stage='history',panel='H512_ALL',metric=k,**aggregate(v)))
            if b=='NATIVE':nativeH=data
            else:
                for k,v in data.items():pairs.append(dict(entry=e,branch=b,comparison='same-entry NATIVE',panel='H512_ALL',metric=k,**paired(nativeH[k],v)))
        for p in sorted((WRITERS/f'W{e:03d}/components').glob('L*/*/*/observer/current.json')):
            rel=p.relative_to(WRITERS/f'W{e:03d}/components');layer,family,variant=rel.parts[:3];data=reduce_raw(p)
            for k,v in data.items():rows.append(dict(entry=e,branch=f'{layer}/{family}/{variant}',stage='component',panel='Current',metric=k,**aggregate(v)))
            if family=='kr':kr[(e,layer,variant)]=data
            q=p.parent/'N512.json';v=reduce_raw(q)['NS'];rows.append(dict(entry=e,branch=f'{layer}/{family}/{variant}',stage='component',panel='N512',metric='NS',**aggregate(v)))
            q=p.parent/'H512.json'
            for k,v in reduce_raw(q).items():rows.append(dict(entry=e,branch=f'{layer}/{family}/{variant}',stage='component',panel='H512_ALL',metric=k,**aggregate(v)))
    table('all-endpoint-metrics.csv',rows);table('paired-transitions.csv',pairs);table('current-joint.csv',joint_rows)
    table('N512-w0-retention.csv',retention)
    # Masks contain IDs but never raw prompts; keep detailed manifest local.
    write_json(SCRATCH/'history-masks.json',checks)
    table('history-mask-summary.csv',[dict(entry=x['entry'],branch=x['branch'],**{k:v for k,v in x['mask'].items() if not isinstance(v,list)}) for x in checks])
    write_json(PACKAGE/'metrics-receipt.json',dict(rows=len(rows),paired_rows=len(pairs),joint_rows=len(joint_rows),
        retention_rows=len(retention),GPU_calls=0,independent_raw_reduction=True,bootstrap_unit='case cluster; descriptive fixed-trajectory only',kr_keys=[list(k) for k in kr]))
    print('METRICS_COMPLETE',len(rows),len(pairs),flush=True)


def inventory():
    start=time.time();rows=[];seen={};total=0
    for attempt in ('attempt-r1','attempt-r3','attempt-r4'):
        root=ROOT/'execution'/attempt
        for p in sorted(root.rglob('*')):
            if not p.is_file():continue
            real=p.resolve();assert real.is_relative_to(ROOT/'execution')
            before=real.stat();key=(before.st_dev,before.st_ino)
            if key not in seen:
                h=sha(real);after=real.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
                seen[key]=h;total+=before.st_size
            else:h=seen[key]
            rows.append(dict(path=str(p),realpath=str(real),bytes=before.st_size,sha256=h,
                             symlink=p.is_symlink(),device=before.st_dev,inode=before.st_ino,mtime_ns=before.st_mtime_ns))
            if len(rows)%500==0:print('HASH_PROGRESS',len(rows),total,round(time.time()-start,1),flush=True)
    table('output-inventory.csv',rows)
    write_json(PACKAGE/'output-inventory-receipt.json',dict(members=len(rows),unique_inodes=len(seen),
        unique_bytes=total,logical_bytes=sum(r['bytes'] for r in rows),sha256=sha(PACKAGE/'output-inventory.csv'),
        seconds=time.time()-start,full_output_hash_new=True,input_CP12_model_rehashed=False))
    print('HASH_COMPLETE',len(rows),total,flush=True)


def aggregate(rows):
    margins=np.array([r['desired_margin'] for r in rows]);n=len(rows)
    return dict(numerator=sum(r['success'] for r in rows),denominator=n,percent=100*sum(r['success'] for r in rows)/n,
        ties=sum(r['tie'] for r in rows),token_correct=sum(r['correct'] for r in rows),target_tokens=sum(r['tokens'] for r in rows),
        token_micro=sum(r['correct'] for r in rows)/sum(r['tokens'] for r in rows),
        prompt_macro=float(np.mean([r['correct']/r['tokens'] for r in rows])),
        strict_count=sum(r['strict'] for r in rows),unique_strict_count=sum(r['unique_strict'] for r in rows),
        new_nll_mean=float(np.mean([r['new_nll'] for r in rows])),true_nll_mean=float(np.mean([r['true_nll'] for r in rows])),
        desired_nll_mean=float(np.mean([r['desired_nll'] for r in rows])),desired_margin_mean=float(margins.mean()),
        desired_margin_p01=float(np.quantile(margins,.01)),desired_margin_p05=float(np.quantile(margins,.05)),
        desired_margin_median=float(np.median(margins)),desired_nll_p95=float(np.quantile([r['desired_nll'] for r in rows],.95)),
        desired_nll_p99=float(np.quantile([r['desired_nll'] for r in rows],.99)))


def joint(data):
    ids=list(dict.fromkeys(r['case_id'] for r in data['RS']));assert len(ids)==100
    counts=dict(pair_R_twoP_joint=0,TF_R_twoP_joint=0,TF_unique_R_twoP_joint=0)
    for i in ids:
        rows=[r for k in ('RS','PS') for r in data[k] if r['case_id']==i];assert len(rows)==3
        counts['pair_R_twoP_joint']+=all(r['success'] for r in rows)
        counts['TF_R_twoP_joint']+=all(r['strict'] for r in rows)
        counts['TF_unique_R_twoP_joint']+=all(r['unique_strict'] for r in rows)
    return dict(requests=100,**counts)


def first():
    metrics=[];joints=[];identities={}
    for entry in ENTRIES:
        expected=[r['case_id'] for r in read(WRITERS/f'W{entry:03d}/NATIVE/write/terminal.json')['target_receipts']]
        assert len(expected)==len(set(expected))==100
        for branch in BRANCHES:
            p=WRITERS/f'W{entry:03d}'/branch/'stages/history/current.json';data=reduce_raw(p)
            assert [r['case_id'] for r in data['RS']]==expected
            assert {k:len(v) for k,v in data.items()}=={'RS':100,'PS':200,'NS':1000}
            ids={k:[r['identity'] for r in v] for k,v in data.items()}
            if entry in identities:assert ids==identities[entry]
            else:identities[entry]=ids
            metrics += [dict(entry=entry,branch=branch,metric=k,**aggregate(v),raw_sha256=sha(p),raw_path=str(p)) for k,v in data.items()]
            joints.append(dict(entry=entry,branch=branch,**joint(data)))
    table('first-actual-table.csv',metrics);table('first-joint-table.csv',joints)
    print(json.dumps([dict(entry=e,branch=b,**{m:next(r['numerator'] for r in metrics if (r['entry'],r['branch'],r['metric'])==(e,b,m)) for m in ('RS','PS','NS')}) for e in ENTRIES for b in BRANCHES],indent=2))
    write_json(PACKAGE/'first-table-receipt.json',dict(status='INDEPENDENT_RAW_NLL_AND_TF_REDUCED',rows=len(metrics),
        table_sha256=sha(PACKAGE/'first-actual-table.csv'),joint_sha256=sha(PACKAGE/'first-joint-table.csv'),
        full_source_state_cost_review='IN_PROGRESS',model_calls=0,scheduler_completion_not_scientific_PASS=True))


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['first','metrics','inventory']);a=p.parse_args()
    if a.phase=='first':first()
    if a.phase=='metrics':metrics()
    if a.phase=='inventory':inventory()


if __name__=='__main__':main()
