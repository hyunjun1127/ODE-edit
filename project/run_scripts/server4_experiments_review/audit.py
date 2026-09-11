"""New eight chains: streaming file SHA + sequential CPU selected tensor audit."""
import gc,json
import torch
from .common import *
from project.run_scripts.blue_lifelong_analysis.audit import tensor_sha,content

def main():
    torch.set_num_threads(2)
    s=sample();ids=[r['case_id'] for r in s['records']]
    inventories=[];chains=[];cps=[];sources=[];configs=[];targets=[];projectors=[];memo={}
    reused={r['path']:r for r in csvread(INHERITED/'source-member-inventory.csv')}
    def verify(p,expected=None):
        p=Path(p);key=str(p)
        if key not in memo:
            h=sha(p);memo[key]=dict(path=key,bytes=p.stat().st_size,sha256=h,verification='NEW_FILE_SHA')
        m=memo[key]
        if expected:assert m['sha256']==expected['sha256'] and m['bytes']==int(expected['bytes']),key
        return m
    for arm in NEWARMS:
        rp=root(arm);ap=attempt(arm);rt=read(rp/'runtime.json');lk=read(ap/'execution.lock.json');t=read(rp/'terminal.json')
        isalpha=family(arm)=='AlphaEdit';isbase=arm.startswith('BASE_');layers=rt['hparams']['layers']
        tm={x['path']:x for x in t['manifest_members']}
        assert {str(p.relative_to(rp)) for p in rp.rglob('*') if p.is_file()}==set(tm)|{'terminal.json'}
        for rel,m in tm.items():inventories.append(dict(arm=arm,**verify(rp/rel,m)))
        inventories.append(dict(arm=arm,**verify(rp/'terminal.json')))
        assert rt['lock_sha256']==sha(ap/'execution.lock.json') and lk['checkpoint_batches']==SCHEDULE and lk['sample_root']==SAMPLE_ROOT
        for m in lk['members']:
            p=Path(m['path'])
            if str(p) in reused and reused[str(p)]['sha256']==m['sha256'] and int(reused[str(p)]['bytes'])==m['bytes']:
                sources.append(dict(**m,verification='REUSED_PRIOR_SOURCE_ASSET_AUDIT',arm=arm))
            else:sources.append(dict(**verify(p,m),arm=arm))
        sources.append(dict(**verify(lk['local_source_archive']),arm=arm));assert sha(lk['local_source_archive'])==lk['local_source_sha256']
        cfg=read(rt['spec']['config']);original=read(rt['spec']['original_config']['path'])
        assert [k for k in cfg if cfg[k]!=original.get(k)]==([] if isbase else ['layers'])
        assert all(rt['hparams'][k]==v for k,v in cfg.items()) and bool(rt['hparams']['blue'])==(not isbase)
        assert layers==([4,5,6,7,8] if isbase else [int(arm[-6])])
        if isalpha:
            pstack=torch.load(lk['projector'],map_location='cpu',weights_only=True,mmap=True)
            indices=[lk['projector_source_layers'].index(l) for l in layers]
            selected=pstack[indices].clone();ph=tensor_sha(selected)
            assert indices==[l-4 for l in layers]==rt['binding']['projector_indices']
            assert list(pstack.shape)==[5,14336,14336] and ph==rt['binding']['projector_sha256']
            projectors.append(dict(arm=arm,physical_layers=str(layers),asset_indices=str(indices),selected_tensor_sha256=ph,source_file=lk['projector'],source_file_sha256=sha(lk['projector']),cpu_mapping='PASS'))
            del selected,pstack;gc.collect()
        configrow=dict(arm=arm,job=JOBS[ARMS.index(arm)],raw_root=str(rp),source_head=rt['helper_head'],source_tree=lk['source_tree'],blue_head=rt['blue_head'],blue_tree=lk['blue_tree'],source_archive=lk['local_source_archive'],source_archive_sha256=lk['local_source_sha256'],runtime_sha256=sha(rp/'runtime.json'),lock_sha256=rt['lock_sha256'],config_sha256=sha(rt['spec']['config']),model_revision=rt['model_revision'],sample_root=lk['sample_root'],seed=lk['seed'],dtype=rt['dtype'],native_scalar_policy=rt['native_scalar_policy'],gpu=rt['gpu'],attention=rt['attention'],torch=rt['torch'],transformers=rt['transformers'],autocast=rt['autocast'],tf32_matmul=rt['tf32_matmul'],tf32_cudnn=rt['tf32_cudnn'],hparams=json.dumps(rt['hparams'],sort_keys=True),binding=json.dumps(rt['binding'],sort_keys=True),tokenizer=json.dumps(rt['writer_tokenizer'],sort_keys=True),evaluator_tokenizer=json.dumps(rt['evaluator_tokenizer'],sort_keys=True),layers=str(layers),blue=not isbase)
        configs.append(configrow)
        prior=content(rt['W0']);zc=sc=hc=0;lastcov=None;contexts=set();prevcp=None;prevbatch=None
        for b in range(1,101):
            br=rp/f'B{b:03d}';e=read(br/'entry.json');c=read(br/'commit.json');o=read(br/'native-observation.json')
            assert c['entry']==prior==content(e['signature']) and e['request_ids']==ids[(b-1)*100:b*100]
            assert e['request_hashes']==[q['request_sha256'] for q in s['records'][(b-1)*100:b*100]]
            assert c['status']=='BATCH_COMMITTED' and c['requests']==100 and c['seen_requests']==100*b
            assert c['W_pointer_exact'] and not c['evaluator_mutation'] and not c['nonfinite']
            assert o['nonselected_pointer_version_exact'] and not o['cold_reset_inside_batch']
            expected_z=100 if isbase else len(layers)*100
            assert (c['compute_z'],c['solve_calls'],c['history_append_passes'])==(expected_z,len(layers),int(isalpha))
            zlayers=[layers[-1]] if isbase else layers
            assert [(z['layer'],z['case_id']) for z in o['z']]==[(l,i) for l in zlayers for i in e['request_ids']]
            assert [q['layer'] for q in o['keys']]==layers*(2 if isalpha else 1)
            assert o['projector_asset_indices']==([l-4 for l in layers] if isalpha else [])
            if isalpha:assert (c['history_entries_in'],c['history_entries_out'])==((b-1)*100,b*100)
            else:
                assert c['history_entries_in'] is None and c['history_entries_out'] is None
                if lastcov is not None:assert lastcov==c['covariance_guard']
            lastcov=c['covariance_guard']
            context=read(br/'contexts.json');assert digest(context)==c['context_hash'];contexts.add(c['context_hash'])
            tv=torch.load(br/'native-targets.pt',map_location='cpu',weights_only=True)
            assert tv['identities']==o['z'] and len(tv['values'])==expected_z and digest(o['z'])==o['z_hash_order']
            for z,v in zip(o['z'],tv['values']):assert tensor_sha(v)==z['sha256'] and torch.isfinite(v).all()
            targets.append(dict(arm=arm,batch=b,target_count=expected_z,target_order_sha256=o['z_hash_order'],status='CPU_TENSOR_HASH_FINITE_PASS'));del tv
            ev=read(br/'current.json');assert ev['weight_state']==c['endpoint']['weights'] and ev['cache_sha256']==c['endpoint']['cache'] and ev['before_after_exact'] and not ev['evaluator_controller_influence']
            assert bool(c['checkpoint'])==(b in SCHEDULE)
            if b in SCHEDULE:
                full=read(br/'seen-full.json');assert full['state']==c['endpoint'] and full['evaluation_type']=='CHECKPOINT_FINAL_W_ON_ALL_SEEN_REQUESTS'
                path=br/'W-method-state.pt';assert c['checkpoint']['sha256']==tm[str(path.relative_to(rp))]['sha256']
                cp=torch.load(path,map_location='cpu',weights_only=True,mmap=True);md=cp['metadata'];w=cp['weights'];cache=cp['cache_c']
                assert set(w)=={f'model.layers.{l}.mlp.down_proj.weight' for l in layers}
                assert {k:tensor_sha(v) for k,v in w.items()}==c['endpoint']['weights'] and tensor_sha(cache)==c['endpoint']['cache']
                assert list(cache.shape)==([len(layers),14336,14336] if isalpha else [0]) and torch.isfinite(cache).all()
                assert md['state']==c['endpoint'] and md['seen_ids']==ids[:b*100] and md['sample_root']==SAMPLE_ROOT
                assert md['base_model_revision']==lk['revision'] and md['contexts']==context and md['cache_c_is_history']==isalpha
                assert md['lock_sha256']==rt['lock_sha256'] and md['source']==lk['local_source_sha256']
                assert all(k in md['rng'] for k in ['python','numpy','torch','cuda'])
                for k,v in w.items():
                    assert list(v.shape)==[4096,14336] and v.dtype==torch.float32 and torch.isfinite(v).all()
                    row=dict(arm=arm,batch=b,layer=int(k.split('.')[2]),key=k,tensor_sha256=tensor_sha(v),shape=str(list(v.shape)),dtype=str(v.dtype),cache_shape=str(list(cache.shape)),cache_sha256=c['endpoint']['cache'],file_sha256=c['checkpoint']['sha256'],file_bytes=path.stat().st_size,weight_norm=float(v.double().norm()),restore_level='CPU_RELOAD_HASH_FINITE; GPU_CONTINUATION_NOT_RUN')
                    if prevcp is not None:row.update(previous_checkpoint=prevbatch,checkpoint_interval_net_norm=float((v.double()-prevcp['weights'][k].double()).norm()))
                    cps.append(row)
                prevcp=cp;prevbatch=b
                del full
            prior=c['endpoint'];zc+=c['compute_z'];sc+=c['solve_calls'];hc+=c['history_append_passes']
        assert (t['compute_z'],t['solve_calls'],t['history_append_passes'])==(zc,sc,hc)
        assert t['W_chain_links']==99 and t['cold_reset_count']==1 and t['W0_cache_bytes_pointer_restore'] and t['nonfinite']==t['failure']==0 and len(contexts)==1
        chains.append(dict(arm=arm,batches=100,requests=10000,checkpoints=12,W_links=99,history_links=99 if isalpha else 'NA_STATIC_COV',compute_z=zc,solve_calls=sc,history_append_passes=hc,cold_reset=1,context_sha256=next(iter(contexts)),terminal_sha256=sha(rp/'terminal.json'),raw_members=len(tm)+1,raw_bytes=sum(m['bytes'] for m in tm.values()),validation='NEW_RAW_SHA_CPU_TARGET_CP_STATE_CHAIN',failure=0,nonfinite=0,full_gpu_replay=0,nonselected_preservation='runtime pointer/version only; all bytes NOT_RECORDED'))
        del prevcp;gc.collect();print('NEW_CHAIN_VERIFIED',arm,'CP12',flush=True)
        # Local create-once per-arm progress, not a runtime journal mutation.
        save(LOCAL/(arm+'-audit.json'),chains[-1])
    for name,rows in [('new-raw-member-inventory',inventories),('new-source-member-inventory',sources),('new-chain-integrity',chains),('new-checkpoint-tensors',cps),('new-source-config-compatibility',configs),('new-target-tensor-audit',targets),('new-projector-binding',projectors)]:csvwrite(OUT/(name+'.csv'),rows)
    save(OUT/'new-eight-audit-receipt.json',dict(status='COMPLETE',chains=len(chains),checkpoints=96,raw_members=len(inventories),raw_bytes=sum(m['bytes'] for m in inventories),member_root=digest(inventories),gpu=0,scope='new eight only; old six audit reused'))

if __name__=='__main__':main()
