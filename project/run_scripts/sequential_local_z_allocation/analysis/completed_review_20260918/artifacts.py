"""CPU-only full output inventory and weights_only native-evidence checks."""
import collections, csv, os, time
import numpy as np
import torch
from reducer import *
from performance import stats

def main():
    torch.set_num_threads(4);started=time.monotonic();inventory=[];checks=[];targets=[];schemas=[];bygroup=collections.defaultdict(list)
    fitrows=list(csv.DictReader((REPORT/'fit-cost.csv').open()));expected={r['evidence']:r for r in fitrows}
    records=load('/data/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1/counterfact.json')[:1000]
    for arm in ARMS:
        out=ROOT/'arms'/arm/'attempt-v1/output'
        for p in sorted(out.rglob('*')):
            if not p.is_file():continue
            s=p.stat();h=sha(p);inventory.append(dict(arm=arm,path=str(p),relative=str(p.relative_to(out)),bytes=s.st_size,sha256=h,mtime_ns=s.st_mtime_ns))
            if p.suffix!='.pt':continue
            assert str(p) in expected,'UNEXPECTED_TENSOR_FILE'
            f=expected[str(p)];assert h==f['evidence_sha'] and s.st_size==int(f['evidence_bytes'])
            d=torch.load(p,map_location='cpu',weights_only=True,mmap=True);b=int(f['batch']);layer=int(f['layer']);ctx=dict(arm=arm,batch=b,layer=layer,fit=p.parent.name)
            assert set(d)=={'captures','target','anchors','radii','target_observations'}
            assert d['target'].shape==d['anchors'].shape==(4096,100) and d['radii'].shape==(100,)
            assert all(t.dtype==torch.float32 and t.device.type=='cpu' and torch.isfinite(t).all() for t in [d['target'],d['anchors'],d['radii']])
            rows=d['target_observations'];assert len(rows)==100
            assert [r['case_id'] for r in rows]==[r['case_id'] for r in records[(b-1)*100:b*100]]
            assert torch.equal(d['target'],torch.stack(d['captures']['compute_z'],dim=1))
            assert torch.equal(d['target'],torch.stack([r['target'] for r in rows],dim=1))
            assert torch.equal(d['anchors'],torch.stack([r['anchor'] for r in rows],dim=1))
            for key,ts in d['captures'].items():
                assert all(t.device.type=='cpu' and t.dtype==torch.float32 and torch.isfinite(t).all() for t in ts)
            for key in ['target','anchors','radii']:
                t=d[key];schemas.append(dict(**ctx,name=key,shape=str(tuple(t.shape)),dtype=str(t.dtype),sha256=hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest(),finite=True))
            assert sum(r['adam_updates'] for r in rows)==int(f['Adam']) and sum(r['loss_evaluations'] for r in rows)==int(f['loss']) and sum(r['clamp_hits'] for r in rows)==int(f['clamp'])
            for r in rows:
                assert 0<=r['adam_updates']<=24 and 1<=r['loss_evaluations']<=25 and len(r['losses'])==r['loss_evaluations']
                assert r['clamp_hits']==len(r['clamp_events']) and r['layer']==layer
                assert all(math.isfinite(x) for x in [r['final_loss'],r['final_nll'],r['final_kl'],r['final_decay'],r['radius']])
                assert r['stop_reason']==('LOSS_LT_0.05' if r['final_loss']<.05 else 'LOSS_BUDGET_25')
                row=dict(**ctx,case_id=r['case_id'],adam=r['adam_updates'],loss_evaluations=r['loss_evaluations'],stop=r['stop_reason'],clamp_hits=r['clamp_hits'],loss=r['final_loss'],nll=r['final_nll'],kl=r['final_kl'],decay=r['final_decay'],radius=r['radius'],delta_norm=float(r['delta'].double().norm()),target_norm=float(r['target'].double().norm()),loss_component_residual=r['final_loss']-r['final_nll']-r['final_kl']-r['final_decay'])
                targets.append(row);bygroup[(arm,b,layer)].append(row)
            checks.append(dict(**ctx,path=str(p),status='CPU_WEIGHTS_ONLY_SCHEMA_FINITE_CAPTURE_COUNTERS_PASS',W_M_present=False))
            del d,rows
        print('ARTIFACTS',arm,flush=True)
    aggregates=[]
    for (arm,b,l),rs in bygroup.items():
        row=dict(arm=arm,batch=b,layer=l,targets=len(rs),zero_adam=sum(x['adam']==0 for x in rs),cap_stop=sum(x['stop']=='LOSS_BUDGET_25' for x in rs),loss_stop=sum(x['stop']=='LOSS_LT_0.05' for x in rs),Adam=sum(x['adam'] for x in rs),loss_evaluations=sum(x['loss_evaluations'] for x in rs),clamp_hits=sum(x['clamp_hits'] for x in rs))
        for k in ['loss','nll','kl','decay','delta_norm','radius','loss_component_residual']:row.update(stats([x[k] for x in rs],k))
        aggregates.append(row)
    csvout('raw-inventory.csv',inventory);csvout('tensor-schema.csv',schemas);csvout('native-target-summary.csv',aggregates)
    writejson(LOCAL/'native-target-scalars.json',targets);writejson(LOCAL/'tensor-checks.json',checks)
    writejson(REPORT/'artifact-summary.json',dict(files=len(inventory),bytes=sum(x['bytes'] for x in inventory),per_arm={a:{'files':sum(x['arm']==a for x in inventory),'bytes':sum(x['bytes'] for x in inventory if x['arm']==a)} for a in ARMS},native_files=len(checks),native_targets=len(targets),disk_W_M=0,full_selected_state_reconstruction='NOT_AVAILABLE',GPU_continuation='NOT_TESTED',CPU_seconds=time.monotonic()-started,scalar_local_sha=sha(LOCAL/'native-target-scalars.json'),schema_full_SHA=True))
if __name__=='__main__':main()
