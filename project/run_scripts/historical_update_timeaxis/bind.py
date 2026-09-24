"""Shared CPU T0: actual full SHA once, mmap selected tensors, no model/GPU load."""
import gc
import importlib.util
import resource
import sys
import time
import types
from .common import *

def observation_modules():
    """Execute exact supplied evaluator bytes without importing a writer package."""
    name='project.run_scripts.alphaedit_strength_neutral_barrier'
    pkg=types.ModuleType(name);pkg.__path__=[];sys.modules[name]=pkg
    result={}
    for short in ('contracts','evaluator'):
        p=DESIGN/f'source-evidence/alphaedit_strength_neutral_barrier__{short}.py'
        fullname=name+'.'+short
        spec=importlib.util.spec_from_file_location(fullname,p)
        m=importlib.util.module_from_spec(spec);sys.modules[fullname]=m;spec.loader.exec_module(m);result[short]=m
    return result

def check_design():
    files={p.name:rows(p) for p in DESIGN.glob('*.csv')}
    assert len(files['fact-ledger.csv'])==10000 and len(files['state-bank.csv'])==173
    assert len(files['score-tasks.csv'])==383 and len(files['main-cells.csv'])==156 and len(files['pair-cells.csv'])==16
    facts=files['fact-ledger.csv'];states={r['state_id']:r for r in files['state-bank.csv']}
    for task in files['score-tasks.csv']:
        fs=[f for f in facts if f['cohort_id']==task['cohort_id'] and (task['case_selector']=='whole_cohort' or boolstr(f['pilot_selected']))]
        assert len(fs)==int(task['request_count']) and historical_digest([int(f['case_id']) for f in fs])==task['case_order_sha256']
    for c in files['main-cells.csv']:
        i,a,b,t=(int(c[k]) for k in ('cohort_index','update_start','anchor','eval_t'))
        assert (a,b)==TIMES[i:i+2]
        for key,mask in [('M_t',(1<<TIMES.index(t))-1),('B_t',((1<<TIMES.index(t))-1)&~(1<<i)),('M_b',(1<<TIMES.index(b))-1),('B_b',(1<<TIMES.index(a))-1)]:
            assert int(states[c[key+'_state']]['retained_interval_mask'],16)==mask
    assert sum(active(f,int(f['anchor_batch'])) for f in facts)==9966 and sum(active(f,100) for f in facts)==9784
    return facts

def main():
    started=time.monotonic();sys.path.insert(0,str(DEPS))
    import torch,transformers
    from transformers import AutoTokenizer
    from safetensors import safe_open
    from scripts.fixed_counterfact import load_prefix
    torch.set_num_threads(8)
    assert torch.__version__=='2.9.1+cu128' and transformers.__version__=='4.44.2'
    records=load_prefix(DATA,10000);facts=check_design()
    bindings=read(DESIGN/'asset-bindings.json')
    cp_candidates=read(ROOT/'receipts/design-received.json')['checkpoint_candidates']
    expected={(r['arm'],int(r['batch']),r['key']):r for r in rows(DESIGN/'checkpoint-tensor-hashes.csv')}
    checkpoints={f:{} for f in FAMILIES}
    out=ROOT/'inputs/t0-v1';out.mkdir(parents=True,exist_ok=True)
    for c in cp_candidates:
        p=Path(c['path']);rpath=out/'checkpoint-receipts'/f"{c['family']}-{c['batch']:03d}.json"
        if rpath.exists():
            rr=read(rpath);st=p.stat();assert (st.st_size,st.st_ino,st.st_mtime_ns)==(rr['bytes'],rr['inode'],rr['mtime_ns'])
        else:
            st=p.stat();h=sha(p);assert h==c['expected_sha256'] and st.st_size==c['expected_bytes']
            cp=torch.load(p,map_location='cpu',weights_only=True,mmap=True);md=cp['metadata'];weights={}
            assert set(cp['weights'])==set(KEYS) and md['batch']==c['batch']
            assert md['base_model_revision']==MODEL.name and md['sample_root']==bindings['ordered_root']
            assert md['seen_ids']==[r['case_id'] for r in records[:100*c['batch']]]
            for k in KEYS:
                w=cp['weights'][k];assert w.dtype==torch.float32 and list(w.shape)==[4096,14336] and torch.isfinite(w).all()
                h1=tensor_sha(w);assert h1==expected[c['family'],c['batch'],k]['tensor_sha256']==md['state']['weights'][k];weights[k]=h1
            rr=dict(path=str(p),bytes=st.st_size,sha256=h,inode=st.st_ino,mtime_ns=st.st_mtime_ns,weights=weights,verification='FRESH_FULLSHA_AND_SELECTED_TENSOR_HASH_FINITE',cache_shape=list(cp['cache_c'].shape),method=md['method'])
            save(rpath,rr);del w,cp;gc.collect()
        checkpoints[c['family']][str(c['batch'])]=rr
        print('T0_CP_VERIFIED',c['family'],c['batch'],flush=True)
    prior=read('/data/janghj/ODE-edit/local/native-delayed-write-e3/20260924-v1/attempt-v1/execution.lock.json')
    model=[]
    for m in prior['model_prior_fullSHA']:
        p=Path(m['path']);rpath=out/'model-receipts'/(p.name+'.json')
        if rpath.exists():
            rr=read(rpath);st=p.stat();assert (st.st_size,st.st_mtime_ns)==(rr['bytes'],rr['mtime_ns'])
        else:
            rr=record(p);assert rr['sha256']==m['sha256'] and rr['bytes']==m['bytes'];rr['mtime_ns']=p.stat().st_mtime_ns;save(rpath,rr)
        model.append(rr)
    index=read(MODEL/'model.safetensors.index.json');w0={}
    for k in KEYS:
        with safe_open(MODEL/index['weight_map'][k],framework='pt',device='cpu') as f:
            w=f.get_tensor(k).float();assert torch.isfinite(w).all();h=tensor_sha(w)
            assert h==bindings['w0_selected_weights'][k]['sha256'];w0[k]=h
        del w;gc.collect()
    obs=observation_modules();tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);tok.pad_token=tok.eos_token;tok.padding_side='right'
    contract=obs['contracts'];tokens=[]
    for r,f in zip(records,facts,strict=True):
        assert r['case_id']==int(f['case_id']) and historical_digest(r)==f['raw_record_sha256']
        req=r['requested_rewrite'];new=contract.target_token_ids(tok,req['target_new']['str']);true=contract.target_token_ids(tok,req['target_true']['str'])
        for kind,pidx,prompt in [('rewrite',0,req['prompt'].format(req['subject']))]+[('paraphrase',i,p) for i,p in enumerate(r['paraphrase_prompts'])]:
            ids=contract.prompt_token_ids(tok,prompt)
            assert max(len(ids)+len(new)-1,len(ids)+len(true)-1)<=read(MODEL/'config.json')['max_position_embeddings'],'OVERLONG_SEQUENCE'
            tokens.append(dict(case_id=r['case_id'],panel=kind,prompt_index=pidx,prompt_ids=ids,target_ids=new,competitor_ids=true,
                prompt_token_hash=digest(ids),target_token_hash=digest(new),competitor_token_hash=digest(true),target_version=req['target_new']['id'],
                pair_identity=historical_digest([r['case_id'],pidx,prompt,req['target_new']['str'],req['target_true']['str']])))
    tokenfile=out/'token-manifest.json';save(tokenfile,tokens)
    result=dict(status='PASS',stage='T0',instruction_id=INSTRUCTION,checkpoints=checkpoints,model_shards=model,
        model_index=record(MODEL/'model.safetensors.index.json'),model_config=record(MODEL/'config.json'),w0_selected=w0,
        model=str(MODEL),dataset=record(DATA/'counterfact.json'),ordered_root=bindings['ordered_root'],token_manifest=record(tokenfile),
        tokenizer=[record(MODEL/n) for n in ('tokenizer.json','tokenizer_config.json','special_tokens_map.json')],
        design_members=read(ROOT/'receipts/design-received.json')['members'],
        original_scores={family:{str(b):record(BASE/f'main-cell-{ci}'/f'B{b:03d}'/'seen-full.json') for b in TIMES[1:]} for ci,family in enumerate(FAMILIES,1)},
        max_host_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,seconds=time.monotonic()-started,
        model_loaded=False,GPU_calls=0,save_checkpoints=False,source_SHA=record(Path(__file__)))
    save(out/'runtime-binding.json',result)
    print('T0_CPU_PASS',record(out/'runtime-binding.json'),flush=True)

if __name__=='__main__':main()
