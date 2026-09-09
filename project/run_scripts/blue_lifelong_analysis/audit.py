"""Bounded, sequential CPU rehash/reload audit. Does not import model/runtime code."""
import gc,hashlib,json,time
from collections import Counter
import torch
from .common import *

def tensor_sha(t):
    assert t.device.type=='cpu'
    a=t.detach().contiguous();h=hashlib.sha256(str((str(a.dtype),list(a.shape))).encode())
    v=a.view(torch.uint8).numpy().reshape(-1)
    for i in range(0,v.size,8<<20):h.update(memoryview(v[i:i+(8<<20)]))
    return h.hexdigest()

def content(sig):return dict(weights={k:v['sha256'] for k,v in sig['weights'].items()},cache=sig['cache_sha256'])

def main(out):
    torch.set_num_threads(4)
    s=sample();ids=[r['case_id'] for r in s['records']]
    inventory=[];chains=[];cps=[];targets=[];configs=[];sources=[];sample_rows=[];cache={}
    def verify(p,expected=None,group='source'):
        p=Path(p);key=str(p)
        if key not in cache:
            assert p.is_file()
            cache[key]=dict(path=key,bytes=p.stat().st_size,sha256=sha(p),symlink=p.is_symlink(),group=group)
        row=cache[key]
        if expected:assert (row['bytes'],row['sha256'])==(expected['bytes'],expected['sha256']),key
        return row
    # All shared source/model/config/stats locks are rehashed once per unique member, not once per arm.
    for a in [attempt(0),attempt(1)]:
        lock=read(a/'execution.lock.json');verify(a/'execution.lock.json')
        for m in lock['members']:sources.append(verify(m['path'],m))
        verify(lock['local_source_archive'],dict(bytes=Path(lock['local_source_archive']).stat().st_size,sha256=lock['local_source_sha256']))
    lock=read(attempt(0)/'execution.lock.json')
    data=read(lock['dataset']);byid={r['case_id']:r for r in data}
    for i,r in enumerate(s['records']):
        raw=byid[r['case_id']];assert digest(raw)==r['raw_record_sha256'] and digest(raw['requested_rewrite'])==r['request_sha256']
        assert digest(raw['requested_rewrite']['target_new'])==r['target_new_sha256']
        assert digest(raw['requested_rewrite']['target_true'])==r['target_true_sha256']
        sample_rows.append(dict(ordinal=i,case_hash=digest(r['case_id']),batch=i//100+1,request_sha256=r['request_sha256'],subject_relation_group=r['subject_relation_group'],target_new_sha256=r['target_new_sha256'],target_true_sha256=r['target_true_sha256']))
    prefix=read(s['prefix1000_file']['path']);assert digest(prefix['records'])==PREFIX_ROOT
    assert [r['case_id'] for r in prefix['records']]==ids[:1000]
    assert all(r['request_sha256']==q['request_sha256'] for r,q in zip(s['records'][:1000],prefix['records']))
    del data,byid
    pstack=torch.load(lock['projector'],map_location='cpu',weights_only=True,mmap=True)
    phashes={tuple(layers):tensor_sha(pstack[[l-4 for l in layers]].clone()) for layers in ([4,8],[4],[8])}
    assert list(pstack.shape)==[5,14336,14336]
    del pstack;gc.collect()
    for cell,arm in enumerate(ARMS):
        r=root(cell);t=read(r/'terminal.json');rt=read(r/'runtime.json');lk=read(attempt(cell)/'execution.lock.json')
        assert lk['checkpoint_batches']==SCHEDULE and lk['sample_root']==SAMPLE_ROOT
        assert rt['lock_sha256']==sha(attempt(cell)/'execution.lock.json')
        expected={m['path']:m for m in t['manifest_members']}
        actual={str(p.relative_to(r)) for p in r.rglob('*') if p.is_file()}
        assert actual==set(expected)|{'terminal.json'},(arm,'unmanifested members',actual-set(expected))
        for rel,m in expected.items():
            assert not (r/rel).is_symlink()
            inventory.append(dict(arm=arm,**verify(r/rel,m,'raw')))
        inventory.append(dict(arm=arm,**verify(r/'terminal.json',group='terminal')))
        hp=rt['hparams'];layers=hp['layers'];method=rt['spec']['method'];isalpha=method=='AlphaEdit'
        original=read(rt['spec']['original_config']['path']);cfg=read(rt['spec']['config'])
        assert [k for k in cfg if cfg[k]!=original.get(k)]==(['layers'] if len(layers)==1 else [])
        for k,v in cfg.items():assert hp[k]==v
        assert hp['blue'] and layers==([4,8] if cell<2 else [4] if cell<4 else [8])
        if isalpha:
            assert rt['binding']['projector_indices']==[l-4 for l in layers]
            assert rt['binding']['projector_sha256']==phashes[tuple(layers)]
        configs.append(dict(arm=arm,job=JOBS[cell],raw_root=str(r),runtime_sha256=sha(r/'runtime.json'),source_archive=lk['local_source_archive'],source_archive_sha256=lk['local_source_sha256'],source_head=rt['helper_head'],source_tree=lk['source_tree'],blue_head=rt['blue_head'],blue_tree=lk['blue_tree'],model_revision=rt['model_revision'],config_sha256=sha(rt['spec']['config']),sample_root=SAMPLE_ROOT,seed=lk['seed'],dtype=rt['dtype'],autocast=rt['autocast'],tf32_matmul=rt['tf32_matmul'],tf32_cudnn=rt['tf32_cudnn'],attention=rt['attention'],torch=rt['torch'],transformers=rt['transformers'],gpu=rt['gpu'],parameter_elements=rt['parameter_elements'],parameter_tensors=rt['parameter_tensors'],hparams=json.dumps(hp,sort_keys=True),binding=json.dumps(rt['binding'],sort_keys=True),tokenizer=json.dumps(rt['writer_tokenizer'],sort_keys=True),evaluator_tokenizer=json.dumps(rt['evaluator_tokenizer'],sort_keys=True)))
        prior=content(rt['W0']);zc=sc=hc=0;prior_cov=None;contexts=set();previous_cp=None;previous_batch=None
        for b in range(1,101):
            br=r/f'B{b:03d}';e=read(br/'entry.json');c=read(br/'commit.json');o=read(br/'native-observation.json')
            assert c['status']=='BATCH_COMMITTED' and c['requests']==100 and c['seen_requests']==b*100
            assert c['entry']==content(e['signature'])==prior
            assert e['request_ids']==ids[(b-1)*100:b*100]
            assert e['request_hashes']==[x['request_sha256'] for x in s['records'][(b-1)*100:b*100]]
            assert c['W_pointer_exact'] and not c['evaluator_mutation'] and not c['nonfinite']
            assert o['nonselected_pointer_version_exact'] and not o['cold_reset_inside_batch']
            assert (c['compute_z'],c['solve_calls'],c['history_append_passes'])==(100*len(layers),len(layers),int(isalpha))
            assert [(z['layer'],z['case_id']) for z in o['z']]==[(l,i) for l in layers for i in e['request_ids']]
            assert [x['layer'] for x in o['keys']]==layers*(2 if isalpha else 1)
            assert digest(o['z'])==o['z_hash_order']
            assert o['projector_asset_indices']==([l-4 for l in layers] if isalpha else [])
            if isalpha:assert (c['history_entries_in'],c['history_entries_out'])==((b-1)*100,b*100)
            else:
                assert c['history_entries_in'] is None and c['history_entries_out'] is None
                if prior_cov is not None:assert c['covariance_guard']==prior_cov
            prior_cov=c['covariance_guard']
            context=read(br/'contexts.json');assert digest(context)==c['context_hash'];contexts.add(c['context_hash'])
            tv=torch.load(br/'native-targets.pt',map_location='cpu',weights_only=True)
            assert tv['identities']==o['z'] and len(tv['values'])==c['compute_z']
            for z,v in zip(o['z'],tv['values']):
                assert tensor_sha(v)==z['sha256'] and torch.isfinite(v).all()
            targets.append(dict(arm=arm,batch=b,targets=len(tv['values']),file_sha256=expected[f'B{b:03d}/native-targets.pt']['sha256'],tensor_hash_order=o['z_hash_order'],status='CPU_HASH_FINITE_PASS'))
            del tv
            assert bool(c['checkpoint'])==(b in SCHEDULE)
            if b in SCHEDULE:
                path=br/'W-method-state.pt';cp=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
                md=cp['metadata'];w=cp['weights'];m=cp['cache_c']
                assert {k:tensor_sha(v) for k,v in w.items()}==c['endpoint']['weights']
                assert tensor_sha(m)==c['endpoint']['cache']
                assert list(m.shape)==([len(layers),14336,14336] if isalpha else [0])
                assert set(w)=={f'model.layers.{l}.mlp.down_proj.weight' for l in layers}
                assert all(list(v.shape)==[4096,14336] and v.dtype==torch.float32 and torch.isfinite(v).all() for v in w.values()) and torch.isfinite(m).all()
                assert md['state']==c['endpoint'] and md['seen_ids']==ids[:b*100] and md['sample_root']==SAMPLE_ROOT
                assert md['base_model_revision']==lk['revision'] and md['source']==lk['local_source_sha256'] and md['lock_sha256']==rt['lock_sha256']
                assert md['contexts']==context and md['cache_c_is_history']==isalpha
                assert all(k in md['rng'] for k in ['python','numpy','torch','cuda'])
                for k,v in w.items():
                    row=dict(arm=arm,batch=b,layer=int(k.split('.')[2]),key=k,shape=str(list(v.shape)),dtype=str(v.dtype),tensor_sha256=c['endpoint']['weights'][k],file_bytes=path.stat().st_size,file_sha256=c['checkpoint']['sha256'],cache_shape=str(list(m.shape)),cache_sha256=c['endpoint']['cache'],contexts_sha256=c['context_hash'],rng_present=True,restore_level='CPU_RELOAD_HASH_FINITE_ONLY; GPU_CONTINUATION_NOT_RUN',weight_norm=float(v.double().norm()))
                    if previous_cp is not None:row.update(previous_checkpoint=previous_batch,checkpoint_interval_net_norm=float((v.double()-previous_cp['weights'][k].double()).norm()))
                    cps.append(row)
                previous_cp=cp;previous_batch=b
            prior=c['endpoint'];zc+=c['compute_z'];sc+=c['solve_calls'];hc+=c['history_append_passes']
        assert len(contexts)==1
        assert (t['compute_z'],t['solve_calls'],t['history_append_passes'])==(zc,sc,hc)
        assert t['W_chain_links']==99 and t['cold_reset_count']==1 and t['W0_cache_bytes_pointer_restore'] and t['failure']==t['nonfinite']==0
        chains.append(dict(arm=arm,batches=100,requests=10000,checkpoints=12,W_links=99,history_links=99 if isalpha else 'NOT_APPLICABLE_STATIC_COV',compute_z=zc,solve_calls=sc,history_append_passes=hc,cold_reset=1,context_count=1,context_sha256=next(iter(contexts)),W0_restore='RUNTIME_ASSERTED_BYTES_POINTER; independent GPU replay0',nonselected_preservation='POINTER_VERSION_CHECK_ONLY; full_bytes_NOT_RECORDED',eval_nonmutation='RUNTIME_BEFORE_AFTER_POINTER_VERSION_SHA',failure=0,nonfinite=0,raw_members=len(expected),raw_bytes=sum(x['bytes'] for x in expected.values()),terminal_sha256=sha(r/'terminal.json')))
        del previous_cp;gc.collect()
        print('WHOLE_CHAIN_CPU_AUDIT_PASS',arm,'members',len(expected),'checkpoints',12,flush=True)
    csvwrite(out/'raw-member-inventory.csv',inventory)
    csvwrite(out/'source-member-inventory.csv',list({m['path']:m for m in sources}.values()))
    csvwrite(out/'checkpoint-tensors.csv',cps);csvwrite(out/'target-tensor-audit.csv',targets)
    csvwrite(out/'chain-integrity.csv',chains);csvwrite(out/'source-config-compatibility.csv',configs)
    csvwrite(out/'sample-hash-inventory.csv',sample_rows)
    save(out/'full-audit-receipt.json',dict(instruction_id=INSTRUCTION,status='CPU_ARTIFACT_AUDIT_COMPLETE',raw_member_root=digest(inventory),raw_members=len(inventory),raw_bytes=sum(m['bytes'] for m in inventory),chain_count=6,checkpoint_count=72,unique_request_count=10000,sample_root=SAMPLE_ROOT,prefix1000_root=PREFIX_ROOT,raw_model_gpu_forward=0,full_continuation_replay=0,limits=['nonselected byte hashes unavailable','historical runtime W0 restore cannot be independently replayed without GPU','no causal method correctness inference from hashes']))

if __name__=='__main__':main(cli().out)
