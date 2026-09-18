"""Read-only stored tensors, CPU only. No model construction or evaluator calls."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import torch
from .enf_sequential_review import ROOT, ARM, REPORT, read, member, sha, write, csvout

def rdigest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def thash(x,header=False):
    x=x.detach().cpu().contiguous();h=hashlib.sha256()
    if header:h.update(f'{tuple(x.shape)}|{x.dtype}|'.encode('ascii'))
    h.update(memoryview(x.numpy()).cast('B'));return h.hexdigest()

def geometry_hash(x):
    a=x.numpy();h=hashlib.sha256(json.dumps({'shape':list(a.shape),'dtype':a.dtype.str},sort_keys=True,separators=(',',':')).encode()+b'\n')
    h.update(memoryview(a.copy(order='C')).cast('B'));return h.hexdigest()

def load(p):
    return torch.load(p,weights_only=True,map_location='cpu',mmap=True)

def finite_tree(x):
    if isinstance(x,torch.Tensor):
        assert x.device.type=='cpu'
        if x.is_floating_point():assert bool(torch.isfinite(x).all())
        return 1
    if isinstance(x,dict):return sum(finite_tree(v) for v in x.values())
    if isinstance(x,(list,tuple)):return sum(finite_tree(v) for v in x)
    return 0

def fact(x):
    r=x['requested_rewrite'];return (r['subject'],r['relation_id'])

def past_ids(ledger,current):
    latest={fact(r):r for r in ledger};skip={fact(r) for r in current}
    a=[r for f,r in latest.items() if f not in skip]
    a.sort(key=lambda r:(hashlib.sha256(f'ENFC-v1|past|{r["case_id"]}'.encode()).hexdigest(),str(r['case_id'])))
    return [r['case_id'] for r in a[:64]]

def audit(repo,scratch):
    torch.set_num_threads(8);repo=Path(repo);scratch=Path(scratch);out=repo/REPORT
    lock=read(ROOT/'execution.lock.json');state=[];targets=[];ledgerrows=[];previous=None;prevledger=[];inventory=[];tensorcount=0
    for n in range(1,11):
        b=ARM/f'B{n:03d}';e=read(b/'ENTRY.json');c=read(b/'COMMIT.json');s=read(b/'selection-ledger.json')
        cp=load(b/'checkpoint.pt');sel=load(b/'selected-L4.pt');ide=load(b/'selected-ideal-delta.pt')['ideal_delta']
        binding=read(b/'native/native-binding.json');native=load(binding['source']['path']);WN=native['weight']
        tensorcount+=finite_tree(cp)+finite_tree(sel)+finite_tree(ide)+finite_tree(native)
        assert list(cp['weight'].shape)==[4096,14336] and list(cp['M4'].shape)==[1,14336,14336]
        assert cp['weight'].dtype==cp['M4'].dtype==torch.float32 and ide.dtype==torch.float64
        assert cp['next_batch']==n+1 and cp['arm']=='EN-F' and cp['source']==lock['execution']['head'] and cp['lock_identity']==lock['lock_identity']
        assert cp['sample_order']==lock['sample_order'] and len(cp['received_ledger'])==n*100
        assert [r['case_id'] for r in cp['received_ledger']]==lock['sample_order'][:n*100]
        ids=dict(W=thash(cp['weight']),M=thash(cp['M4']),rng=rdigest(cp['RNG']),ledger=rdigest(cp['received_ledger']),context=rdigest(cp['contexts']))
        assert ids==c['state']==cp['state_identity']
        assert torch.equal(cp['weight'],sel['weight'])
        assert thash(sel['weight'],True)==s['selected_weight_sha256'] and thash(ide,True)==s['ideal_delta_sha256']
        assert thash(WN)==binding['endpoint']==native['receipt']['endpoint_weight_sha256']
        assert binding['entry']['W']==e['identity']['W'] and binding['entry']['M']==e['identity']['M']
        assert native['receipt']['history_append']==0 and native['receipt']['projector_sha256']==read(ARM/'runtime-load.json')['identity']['P4']['selected_tensor_sha256']
        assert past_ids(prevledger,cp['received_ledger'][-100:])==read(b/'past-provenance.json')['case_ids']
        assert cp['received_ledger'][:-100]==prevledger
        currentledger=cp['received_ledger'];latest={fact(r):r['case_id'] for r in currentledger}
        registry=[dict(case_id=r['case_id'],ordinal=r['ordinal'],status='ACTIVE' if latest[fact(r)]==r['case_id'] else 'SUPERSEDED') for r in currentledger]
        assert registry==cp['active_registry']
        for r in registry[-100:]:assert r['ordinal'] in range((n-1)*100,n*100)
        if n==10:ledgerrows=registry
        assert torch.equal((WN.double()+ide).float(),sel['weight'])
        D=sel['weight'].double()-WN.double();assert thash(D,True)==s['actual_delta_sha256']
        norm=float(D.norm());assert abs(norm-s['actual_delta_norm'])<1e-10
        accepted=next(t for t in s['trials'] if t['accepted'])
        evs=[read(p) for p in sorted((b/'events').glob('*.json'))];gev=next(v for v in evs if v['record']['event']=='direction')
        gp=gev['saved']['gradient']['path'];G=load(gp)['gradient'];tensorcount+=finite_tree(G)
        assert thash(G,True)==gev['tensors']['gradient']['sha256']
        dot=float((G.double()*D).sum());pred=accepted['p_actual']
        assert abs(dot-pred)<1e-10
        inferred_gq=ide/(-accepted['eta']);chi=float(inferred_gq.square().sum())
        assert abs(chi-gev['record']['chi'])<1e-10
        f=load(b/'geometry/factors.pt');tensorcount+=finite_tree(f)
        assert geometry_hash(f['blocked'])==read(b/'geometry/space.json')['blocked_sha256']
        native_norm=float(native['receipt']['actual_delta_norm'])
        extra=dict(native_norm=native_norm,correction_norm=norm,correction_native_ratio=norm/native_norm,
            ideal_actual_rounding_norm=float((D-ide).norm()),p_actual_cpu=dot,p_actual_record=pred,p_actual_abs_gap=abs(dot-pred),chi_cpu_from_saved_ideal=chi,
            selected_cpu_materialization_exact=True,cumulative_W_minus_W0='NOT_AVAILABLE_NO_RELOAD_OF_PRETRAINED_MODEL')
        if previous is not None:
            assert thash(previous)==e['identity']['W']
            native_delta=WN.double()-previous.double();actualnative=float(native_delta.norm())
            assert abs(actualnative-native_norm)<1e-5
            extra.update(native_norm_cpu=actualnative,net_step_norm_cpu=float((sel['weight'].double()-previous.double()).norm()),
                correction_native_cos=float((D*native_delta).sum())/(norm*actualnative))
        state.append(dict(batch=n,checkpoint_shape_W=list(cp['weight'].shape),checkpoint_shape_M=list(cp['M4'].shape),
            dtype='torch.float32',finite=True,hashes_match=True,received=n*100,active=len(latest),superseded=n*100-len(latest),
            history_append=c['history_append'],next_batch=cp['next_batch'],Past64_count=len(past_ids(prevledger,currentledger[-100:])),**extra))
        for r in native['target_observations']:
            targets.append(dict(batch=n,case_id=r['case_id'],reuse=n==1,adam=r['adam_updates'],loss_evaluations=r['loss_evaluations'],
                loss_final=r['loss_final'],radius=float(r['radius']),clamp_hits='NOT_RECORDED'))
        previous=sel['weight'].clone();prevledger=currentledger
        del cp,sel,ide,native,WN,D,G,f,inferred_gq,evs;gc.collect()
        print('CPU_TENSOR_BATCH_OK',n,flush=True)
    # Full raw inventory within exact EN-F subtree only; no sibling traversal.
    for p in sorted(ARM.rglob('*')):
        if p.is_file():inventory.append(member(p))
    known={r['path']:r for r in inventory}
    errors=[]
    # Validate every nested file reference whose target is an inventoried EN-F file.
    def refs(x):
        if isinstance(x,dict):
            if all(k in x for k in ('path','sha256','bytes')) and x['path'] in known:
                if any(x[k]!=known[x['path']][k] for k in ('bytes','sha256')):errors.append(x['path'])
            for v in x.values():refs(v)
        elif isinstance(x,list):
            for v in x:refs(v)
    for r in inventory:
        if r['path'].endswith('.json'):refs(read(r['path']))
    assert not errors,errors
    csvout(out/'checkpoint-state.csv',state);csvout(out/'raw-inventory.csv',inventory)
    csvout(scratch/'native-target-observations.csv',targets);csvout(scratch/'active-registry.csv',ledgerrows)
    targetsummary=[]
    for n in range(1,11):
        a=[r for r in targets if r['batch']==n]
        targetsummary.append(dict(batch=n,reuse=n==1,targets=len(a),Adam=sum(r['adam'] for r in a),loss_evaluations=sum(r['loss_evaluations'] for r in a),
            max24_count=sum(r['adam']==24 for r in a),zero_step_count=sum(r['adam']==0 for r in a),early_stop_count=sum(r['adam']<24 for r in a),
            loss_final_mean=sum(r['loss_final'] for r in a)/len(a),clamp_hits='NOT_RECORDED'))
    csvout(out/'native-fit-counts.csv',targetsummary)
    write(scratch/'tensor-summary.json',dict(status='CPU_STORED_TENSOR_AND_ARITHMETIC_CHECKS_PASS',batches=state,target_summary=targetsummary,
        checked_tensor_objects=tensorcount,file_count=len(inventory),bytes=sum(r['bytes'] for r in inventory),
        current_CP=10,adjacent_links=9,source_or_model_execution=False,GPU_continuation='NOT_TESTED',
        full_K_E_tensor='NOT_RETAINED_HASH_AND_ALIASES_ONLY',DK_recheck='RUNTIME_RECORDED_SCALAR_CHECK_ONLY',
        history_equation='SOURCE_AND_BEFORE_AFTER_HASH_NOT_INDEPENDENT_SELECTED_KEY_RECOMPUTATION'))
    print('CPU_TENSOR_AND_INVENTORY_COMPLETE',len(inventory),sum(r['bytes'] for r in inventory),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,default=Path.cwd());p.add_argument('--scratch',required=True,type=Path)
    a=p.parse_args();audit(a.repo,a.scratch)
