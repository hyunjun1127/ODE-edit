"""Sequential weights_only CPU audit; no model/native/evaluator imports."""
import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import stat
import time
import torch

ROOT=Path('/data/janghj/ODE-edit/local/ep-tw1-c4/20260915-v1/gate-skip-r1')
def read(p):return json.loads(Path(p).read_text())
def canon(x):return json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
def digest(x):return hashlib.sha256(canon(x)).hexdigest()
def filehash(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def hashes(t):
    t=t.detach().contiguous();assert t.device.type=='cpu'
    data=t.reshape(-1).view(torch.uint8).numpy()
    hs=[hashlib.sha256(),hashlib.sha256(str((str(t.dtype),list(t.shape))).encode()),
        hashlib.sha256(canon(dict(dtype=str(t.dtype),shape=list(t.shape)))+b'\n')]
    finite=True;sumsq=0.
    for i in range(0,t.numel(),1<<20):
        x=t.reshape(-1)[i:i+(1<<20)];finite=finite and bool(torch.isfinite(x).all())
        sumsq+=float(x.double().square().sum())
    for i in range(0,data.size,8<<20):
        for h in hs:h.update(memoryview(data[i:i+(8<<20)]))
    return dict(raw_sha=hs[0].hexdigest(),fixture_header_sha=hs[1].hexdigest(),policy_header_sha=hs[2].hexdigest(),
        shape=list(t.shape),dtype=str(t.dtype),finite=finite,frobenius=sumsq**.5)
def scalar_norm(t):
    return sum(float(x.double().square().sum()) for x in t.reshape(-1).split(1<<20))**.5
def tensors(value,prefix=''):
    if isinstance(value,torch.Tensor):yield prefix,value
    elif isinstance(value,dict):
        for k,v in value.items():yield from tensors(v,prefix+'/'+str(k))
    elif isinstance(value,(tuple,list)):
        for i,v in enumerate(value):yield from tensors(v,prefix+'/'+str(i))
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def csvfile(p,rows):
    import csv
    with p.open('x',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));wr.writeheader();wr.writerows(rows)

def run(out,*,root=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    source_root=Path(root) if root is not None else ROOT
    raw=source_root/'scientific-v1';lock=read(source_root/'execution.lock.json');inventory=[];byfile={}
    for p in sorted(raw.rglob('*')):
        if p.is_dir():continue
        s=p.lstat();assert stat.S_ISREG(s.st_mode) and not p.is_symlink()
        h=filehash(p);z=p.stat();assert (s.st_ino,s.st_size,s.st_mtime_ns)==(z.st_ino,z.st_size,z.st_mtime_ns)
        row=dict(path=str(p),relative_path=str(p.relative_to(raw)),bytes=s.st_size,sha256=h,inode=s.st_ino,mtime_ns=s.st_mtime_ns)
        inventory.append(row);byfile[str(p)]=row
    csvfile(out/'raw-member-inventory.csv',inventory)
    print('RAW_FULLSHA',len(inventory),sum(x['bytes'] for x in inventory),flush=True)
    term=read(raw/'terminal.json');entry0=read(raw/'entry.json');last=entry0['state'];previous_W=None
    cps=[];links=[];tensorrows=[];actions=[];targets=[];refcount=0
    def verify_refs(value):
        nonlocal refcount
        if isinstance(value,dict):
            if all(k in value for k in ('path','bytes','sha256')) and str(value['path']).startswith(str(raw)+'/'):
                actual=byfile[value['path']];assert all(value[k]==actual[k] for k in ('bytes','sha256'));refcount+=1
            for v in value.values():verify_refs(v)
        elif isinstance(value,list):
            for v in value:verify_refs(v)
    for p in raw.rglob('*.json'):verify_refs(read(p))
    for b in range(1,11):
        p=raw/f'B{b:03d}';c=read(p/'commit.json');e=read(p/'entry.json');pol=read(p/'policy.json')
        assert c['entry']==e['state']==last and e['history_count']==b-1
        assert e['order']==lock['batches'][b-1] and c['source_lock']['sha256']==filehash(source_root/'execution.lock.json')
        if b>1:links.append(dict(from_batch=b-1,to_batch=b,W=True,M=True,P=True,context=True,RNG=True,ledger=True,exact=True))
        assert c['native']['history_append']==0 and len(c['history'])==1 and c['history'][0]['history_append']==1
        assert c['native']['compute_z']==100 and c['native']['solve']==1 and c['history_count']==b
        cp=torch.load(p/'checkpoint.pt',weights_only=True,map_location='cpu',mmap=True)
        assert list(cp['weights'])==['model.layers.4.mlp.down_proj.weight']
        W=cp['weights']['model.layers.4.mlp.down_proj.weight'];M=cp['M4'];wh=hashes(W);mh=hashes(M)
        assert wh['shape']==[4096,14336] and mh['shape']==[1,14336,14336] and wh['finite'] and mh['finite']
        assert wh['dtype']==mh['dtype']=='torch.float32'
        assert wh['fixture_header_sha']==c['endpoint']['W4'] and mh['fixture_header_sha']==c['endpoint']['M4']
        assert mh['raw_sha']==c['history'][0]['after_sha256'] and wh['raw_sha']==c['history'][0]['weight_sha256']
        assert digest(cp['contexts'])==c['endpoint']['contexts'] and digest(cp['rng'])==c['endpoint']['rng']
        assert digest(cp['accepted_ledger'])==c['endpoint']['ledger']
        md=cp['metadata'];assert md['endpoint']==c['endpoint'] and md['next_ordinal']==b*100 and md['next_batch']==b+1
        assert md['seen_ids']==sum([x['case_ids'] for x in lock['batches'][:b]],[])
        assert md['P_mapping']==lock['projector_mapping'] and md['model_revision']==lock['model_revision']
        for name,h in [('W4',wh),('M4',mh)]:tensorrows.append(dict(batch=b,key=name,**h))
        native=torch.load(p/'native-targets-map.pt',weights_only=True,map_location='cpu',mmap=True)
        route=torch.load(p/'route.pt',weights_only=True,map_location='cpu',mmap=True)
        for bundle,content in [('native',native),('route',route)]:
            for key,t in tensors(content):
                assert t.device.type=='cpu' and bool(torch.isfinite(t).all()),(b,bundle,key)
        vp=hashes(native['native_proposal']);aa=hashes(native['A']);a2=hashes(route['map_A'])
        assert aa==a2 and route['raw_weight_sha256']==vp['fixture_header_sha'] and route['selected_weight_sha256']==wh['fixture_header_sha']
        assert vp['raw_sha']==c['native']['endpoint_weight_sha256']==pol['sweeps']['raw_sha256']
        cand={r['id']:r for r in pol['selection']['candidate_receipts']}
        assert vp['policy_header_sha']==cand['RAW']['sha256'] and wh['policy_header_sha']==cand[c['selected']]['sha256']
        tensorrows.extend([dict(batch=b,key='RAW_Vp',**vp),dict(batch=b,key='A',**aa)])
        delta=W-native['native_proposal'];actual_norm=scalar_norm(delta)
        recnorm=cand[c['selected']]['actual_correction_norm']
        assert abs(actual_norm-recnorm)<1e-10*max(1.,recnorm)
        assert actual_norm<=pol['correction']['native_trust_limit']*(1+lock['numerical_policy']['trust_rtol'])
        ge=route['gE'].double();gd=route['gD'].double();q=float((ge*gd).sum());e2=float(ge.square().sum())
        recordq=pol['correction']['projection']['q_ge_gd']
        actions.append(dict(batch=b,selected=c['selected'],actual_raw_to_selected_norm=actual_norm,recorded_norm=recnorm,
            selected_W_norm=wh['frobenius'],M_norm=mh['frobenius'],
            selected_increment_norm=scalar_norm(W-previous_W) if previous_W is not None else 'W0_TENSOR_NOT_LOADED_RECORDED_ONLY',
            native_action_recorded=c['native']['actual_delta_norm'],q_cpu=q,q_recorded=recordq,q_abs_difference=abs(q-recordq),
            ge_norm=e2**.5,gd_norm=float(gd.square().sum())**.5,
            header_bridge='RAW_FIXTURE_POLICY_ALL_SAME_STORED_BYTES',GPU_model_parity='NOT_ESTABLISHED'))
        obs=native['target_observations'];assert len(obs)==100 and [x['case_id'] for x in obs]==e['order']['case_ids']
        assert sum(x['adam_updates'] for x in obs)==c['native']['anchor_capture']['adam_updates']
        assert sum(x['loss_evaluations'] for x in obs)==c['native']['anchor_capture']['loss_evaluations']
        counts=Counter(x['adam_updates'] for x in obs)
        targets.append(dict(batch=b,requests=len(obs),Adam=sum(x['adam_updates'] for x in obs),loss=sum(x['loss_evaluations'] for x in obs),
            zero_steps=counts[0],early_stop=sum(n for k,n in counts.items() if k<24),steps_histogram=json.dumps(counts,sort_keys=True),
            teacher_ID_count=len({x['teacher_sha256'] for x in obs}),saved_local_teacher_payload=False))
        led=cp['accepted_ledger'];assert led['next_ordinal']==b*100 and led['processed_batches']==b
        assert len(led['requested_events'])==b*100 and led['batch_receipts'][-1]==c['ledger']
        cps.append(dict(batch=b,checkpoint=str(p/'checkpoint.pt'),bytes=(p/'checkpoint.pt').stat().st_size,
            file_sha256=byfile[str(p/'checkpoint.pt')]['sha256'],selected_keys='model.layers.4.mlp.down_proj.weight',
            W_shape='4096x14336',M_shape='1x14336x14336',dtype='float32',finite=True,
            W_M_hash_bridge=True,context_RNG_ledger=True,history_count=b,next_ordinal=b*100,
            selected_restore='RUNTIME_RECORDED_PLUS_CPU_RELOAD',full_model_GPU_continuation='NOT_TESTED'))
        last=c['endpoint'];previous_W=W.clone()
        del cp,W,M,native,route,ge,gd,delta;gc.collect()
        print('CP_CPU_VERIFIED',b,flush=True)
    assert last==term['state'] and len(term['commits'])==10 and term['history_count']==10
    assert not (raw/'G0_PASS.json').exists() and (raw/'INITIAL_EXECUTION_OBSERVED_WITH_VALIDATION_SKIPPED.json').exists()
    marker=read(raw/'INITIAL_EXECUTION_OBSERVED_WITH_VALIDATION_SKIPPED.json')
    assert marker['completed_requests']==marker['next_ordinal']==100 and marker['history_finalizations']==1
    assert marker['B2_entry']['sha256']==byfile[str(raw/'B002/entry.json')]['sha256']
    for name,data in [('checkpoint-inventory',cps),('checkpoint-tensor-hashes',tensorrows),('state-links',links),('write-actions',actions),('target-counters',targets)]:csvfile(out/(name+'.csv'),data)
    summary=dict(status='CPU_FILE_TENSOR_STATE_LINK_AND_HASH_BRIDGE_VERIFIED',raw_files=len(inventory),raw_bytes=sum(x['bytes'] for x in inventory),
        raw_file_refs_verified=refcount,checkpoints=10,checkpoint_bytes=sum(x['bytes'] for x in cps),
        adjacent_links=9,inner_history_appends=0,final_history_appends=10,targets=1000,solves=10,
        model_level_numerical_validation='NOT_ESTABLISHED',GPU_continuation='NOT_TESTED',
        nonselected_weights='RUNTIME_POINTER_VERSION_GRAD_HOOK_BUFFER_GUARDS_AND_BASE_FINAL_HASH_ASSERT; no full base tensor reload by analysis',
        initial_marker=marker['status'],submission_agent_was_pending=True,all_raw_fullsha_new=True,
        shared_model_teacher_P_rehash=False,missing=['no W0 selected tensor loaded for new norm reconstruction','no full-model GPU continuation','no original FD/direct/selfKL numerical verification'])
    save(out/'state-summary.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);run(p.parse_args().output)
