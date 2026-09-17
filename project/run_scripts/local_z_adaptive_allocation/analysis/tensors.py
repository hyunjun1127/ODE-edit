"""CPU-only saved tensor/schema/hash audit; no model or evaluator imports."""
import gc
import hashlib
import json
import time
from pathlib import Path
import torch
from .bootstrap import ROOT,REVIEW,identity,save
from .metrics import ARMS,read,table,digest

def tensor_hash(t):
    # Original runtime hashes raw contiguous bytes, not dtype/shape headers.
    a=t.detach().cpu().contiguous().numpy();return hashlib.sha256(memoryview(a).cast('B')).hexdigest()

def describe(t):
    assert t.dtype==torch.float32 and t.device.type=='cpu' and torch.isfinite(t).all()
    raw=tensor_hash(t)
    header=hashlib.sha256((str(t.dtype)+str(list(t.shape))).encode());header.update(memoryview(t.contiguous().numpy()).cast('B'))
    return dict(dtype=str(t.dtype),shape=list(t.shape),finite=True,raw_sha256=raw,
        header_bytes_sha256=header.hexdigest(),
        norm=float(torch.linalg.vector_norm(t)),bytes=t.numel()*t.element_size())

def materialize(u,v,g):return u.clone() if g==0 else v.clone() if g==1 else u+g*(v-u)

def run(attempt='tensor-audit',reuse_inventory=None):
    torch.set_num_threads(4);assert not torch.cuda.is_initialized()
    out=REVIEW/attempt;out.mkdir(exist_ok=False);started=time.monotonic()
    inventory=[];cps=[];recon=[];fits_meta=[];target_rows=[];mismatch=[]
    # One fullSHA inventory of this task's sealed scientific/preparation output.
    roots=[ROOT/'technical/attempt-v1']+[ROOT/'arms'/a/'attempt-v1' for a in ARMS]
    if reuse_inventory:
        inventory=read(reuse_inventory)
        assert all(Path(m['path']).stat().st_size==m['bytes'] for m in inventory)
    else:
        for r in roots:
            for p in sorted(r.rglob('*')):
                if p.is_file():inventory.append(dict(identity(p),relative=str(p.relative_to(ROOT))))
    save(out/'raw-inventory.json',inventory)
    known={m['path']:m for m in inventory}
    for arm in ARMS:
        root=ROOT/'arms'/arm/'attempt-v1/output';previous=None;last_diag=None
        for b in range(1,11):
            batch=root/f'B{b:03d}';entry=read(batch/'entry.json')['state'];commit=read(batch/'commit.json')
            checkpoint=batch/'checkpoint.pt'
            cp=None
            if checkpoint.exists():
                cp=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
                assert cp['next_batch']==b+1 and cp['next_ordinal']==b*100 and cp['received']==list(range(b*100))
                assert cp['state']==commit['state'] and digest(cp['rng'])==commit['state']['rng'] and digest(cp['contexts'])==commit['state']['contexts']
                assert cp['source']=='32a92ad6f3fff2f258d8778f3936d152e975ac1b'
                current_diag={}
                for family in ('W','M'):
                    for l,t in cp[family].items():
                        desc=describe(t);assert desc['raw_sha256']==commit['state'][family][str(l)]
                        if family=='W':assert list(t.shape)==[4096,14336]
                        else:
                            assert list(t.shape)==[1,14336,14336]
                            current_diag[l]=t[0].diagonal().clone()
                            if last_diag is not None:assert torch.all(current_diag[l]>=last_diag[l])
                        cps.append(dict(arm=arm,batch=b,tensor=family+str(l),**desc,checkpoint_sha256=known[str(checkpoint)]['sha256']))
                last_diag=current_diag
            # Enumerate saved native target payload counters, independent of aggregate plan.
            fits={}
            for path in sorted((batch/'proposals').glob('*.pt')):
                f=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
                if 'weight' in f:
                    desc=describe(f['weight']);r=f['receipt']
                    fits[path.stem]=f
                    if 'endpoint_weight_sha256' in r:assert r['endpoint_weight_sha256']==desc['raw_sha256']
                    fits_meta.append(dict(arm=arm,batch=b,fit=path.stem,**desc,layer=r['layer'],readout_layer=r.get('readout_layer',r['layer']),
                        compute_z=r.get('compute_z',0),solve=r.get('solve',0),history_append=r['history_append']))
                    observations=f.get('target_observations',[])
                else:
                    assert path.stem=='terminal-Z8';assert torch.isfinite(f['Z8']).all();observations=f['observations']
                for obs in observations:
                    assert 0<=obs['adam_updates']<=24 and 1<=obs['loss_evaluations']<=25
                    assert all(torch.isfinite(obs[k]).all() for k in ('anchor','delta'))
                    target_rows.append(dict(arm=arm,batch=b,fit=path.stem,case_id=obs['case_id'],Adam=obs['adam_updates'],loss=obs['loss_evaluations'],
                        early_stop=obs['adam_updates']<24,zero_step=obs['adam_updates']==0,loss_final=obs['loss_final'],
                        final_delta_norm=float(obs['delta'].norm()),radius=obs['radius'],clamp_hits='NOT_RECORDED'))
            generation=read(batch/'proposals/generation.json')
            if b==1:
                # W0 tensor was not separately stored. Do not infer it by inverse FP32 deltas.
                assert cp is not None;previous={l:t.clone() for l,t in cp['W'].items()}
                recon.append(dict(arm=arm,batch=b,candidate=commit['selected'],status='SELECTED_CP_HASH_VERIFIED; ALL_CANDIDATE_W0_RECONSTRUCTION_NOT_TESTED'))
            else:
                assert {str(l):tensor_hash(t) for l,t in previous.items()}==entry['W']
                selected=None
                for m in generation['declared_candidates']:
                    name=m['candidate_id'];g4=m['a4'];g8=m['a8']
                    if name=='REFIT4':w={4:fits['refit4']['weight'],8:previous[8]}
                    elif name=='N4':w={4:fits['own-N4']['weight'],8:previous[8]}
                    else:
                        terminal=m['target_mode']=='TERMINAL';prefix='terminal8' if terminal else 'local8'
                        first= fits['terminal4' if terminal else 'own-N4']['weight']
                        w={4:materialize(previous[4],first,g4),8:previous[8] if g8==0 else materialize(previous[8],fits[f'{prefix}-a4-{g4:g}']['weight'],g8)}
                    actual={str(l):tensor_hash(t) for l,t in w.items()}
                    ok=actual==m['weights']
                    if not ok:mismatch.append(dict(arm=arm,batch=b,candidate=name,expected=m['weights'],actual=actual))
                    norm=(sum(float((w[l].double()-previous[l].double()).square().sum()) for l in (4,8)))**.5
                    recon.append(dict(arm=arm,batch=b,candidate=name,status='CPU_EXACT_STORED_WEIGHT_HASH' if ok else 'MISMATCH',
                        action_norm_cpu=norm,action_norm_stored=m['action_norm'],action_norm_abs_diff=abs(norm-m['action_norm'])))
                    if name==commit['selected']:selected={l:t.clone() for l,t in w.items()}
                assert selected is not None and {str(l):tensor_hash(t) for l,t in selected.items()}==commit['state']['W']
                if cp is not None:assert all(torch.equal(selected[l],cp['W'][l]) for l in (4,8))
                previous=selected
            del fits,cp;gc.collect()
        print('CPU_TENSOR_ARM_DONE',arm,flush=True)
    table(out/'checkpoint-tensors.csv',cps);table(out/'candidate-reconstruction.csv',recon)
    table(out/'native-fit-tensors.csv',fits_meta);table(out/'target-counters.csv',target_rows)
    save(out/'tensor-audit.json',dict(status='PASS' if not mismatch else 'HOLD_MISMATCH',mismatches=mismatch,CP_count=len(cps)//4,
        checkpoint_tensor_count=len(cps),reconstructed_candidate_count=sum(r['status']=='CPU_EXACT_STORED_WEIGHT_HASH' for r in recon),
        target_calls=len(target_rows),Adam=sum(r['Adam'] for r in target_rows),loss=sum(r['loss'] for r in target_rows),
        source_scope='SAVED_TARGET_AND_SELECTED_STATE_CPU_ONLY',W0_candidate_reconstruction='NOT_TESTED_B1',
        final_history_keys='NOT_SAVED; cannot independently recompute M append Gram',GPU_off_on='NOT_TESTED',
        bytes=sum(r['bytes'] for r in inventory),members=len(inventory),seconds=time.monotonic()-started,GPU_calls=0))
    assert not torch.cuda.is_initialized()

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--attempt',default='tensor-audit');p.add_argument('--reuse-inventory')
    a=p.parse_args();run(a.attempt,a.reuse_inventory)
