"""Read-only BLUE kernel binding, exact checkpoint restoration and asset seals."""
import importlib
import json
import os
from pathlib import Path
import random
import sys
import types
import torch
from .records import digest,tensor_sha
from .import_assets import ROOT,sha
from .objective import WEIGHT
from .panels import ENTRIES
from .microbatch import bounded_reader

MODEL='/mnt/raid5/janghj/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2'
PROJECTOR='/mnt/raid5/janghj/EasyEdit/examples/null_space_project_Meta-Llama-3-8B-Instruct.pt'
COVARIANCE='/mnt/raid5/janghj/EasyEdit/examples/data/stats/Meta-Llama-3-8B-Instruct/wikipedia_stats/model.layers.4.mlp.down_proj_float32_mom2_100000.npz'
DATA='/mnt/raid5/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1'

def namespace(name,path):
    if name not in sys.modules:
        mod=types.ModuleType(name);mod.__path__=[str(path)];mod.__package__=name;sys.modules[name]=mod

def kernel():
    src=ROOT/'imports/blue-source';sys.path.insert(0,str(src))
    # Avoid unrelated eager ROME writers; actual requested source files unchanged.
    for name in ['rome','util','AlphaEdit']:namespace(name,src/name)
    previous=os.getcwd()
    try:
        os.chdir(src)
        native=importlib.import_module('AlphaEdit.AlphaEdit_main')
    finally:os.chdir(previous)
    return native

def historical():
    namespace('blue_alphaedit_sequential_comparison',ROOT/'imports/historical/blue_alphaedit_sequential_comparison')
    return importlib.import_module('blue_alphaedit_sequential_comparison.integrity')

def load_entry(entry,records):
    bi,offset=ENTRIES[entry];base=ROOT/'imports/entries'
    cp=torch.load(base/f'B{bi:03d}/W-method-state.pt',map_location='cpu',weights_only=True)
    target=torch.load(base/f'B{bi+1:03d}/native-targets.pt',map_location='cpu',weights_only=True)
    native_entry=json.loads((base/f'B{bi+1:03d}/entry.json').read_text())
    md=cp['metadata'];oldsha=historical().tensor_sha
    assert md['batch']==bi and md['base_model_revision']==Path(MODEL).name
    assert md['seen_ids']==[r['case_id'] for r in records[:offset]]
    assert set(cp['weights'])=={WEIGHT} and cp['weights'][WEIGHT].shape==(4096,14336)
    assert oldsha(cp['weights'][WEIGHT])==md['state']['weights'][WEIGHT]==native_entry['signature']['weights'][WEIGHT]['sha256']
    assert oldsha(cp['cache_c'])==md['state']['cache']==native_entry['signature']['cache_sha256']
    current=records[offset:offset+100]
    assert native_entry['request_ids']==[r['case_id'] for r in current]
    assert native_entry['request_hashes']==[digest(r['requested_rewrite']) for r in current]
    assert digest(md['contexts'])==native_entry['context_hash']
    assert len(target['values'])==len(target['identities'])==100
    for r,z,ident in zip(current,target['values'],target['identities']):
        assert ident['case_id']==r['case_id'] and ident['layer']==4 and oldsha(z)==ident['sha256']
    return cp,target,current

def restore_rng(cp):
    import numpy as np
    r=cp['metadata']['rng'];random.setstate(r['python'])
    n=r['numpy'];np.random.set_state((n[0],np.asarray(n[1],dtype='uint32'),n[2],n[3],n[4]))
    torch.set_rng_state(torch.tensor(r['torch'],dtype=torch.uint8))
    # Different GPU architecture may have a different CUDA RNG byte schema;
    # no stochastic direct/evaluation path; preserve original receipt separately.
    torch.cuda.manual_seed_all(20260907)

def load_model(ledger):
    from transformers import AutoModelForCausalLM,AutoTokenizer
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    with ledger.time('model_load'):
        model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float32,local_files_only=True,
                                               low_cpu_mem_usage=True,attn_implementation='eager').cuda().eval()
        model.requires_grad_(False)
        tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
        tok.add_bos_token=False;tok.pad_token_id=tok.eos_token_id
        evaltok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);evaltok.pad_token_id=evaltok.eos_token_id
    assert tok.padding_side==evaltok.padding_side=='right'
    assert all(p.dtype==torch.float32 for p in model.parameters())
    return model,tok,evaltok

def native_write(model,tok,cp,targets,rows,p,ledger):
    native=kernel();hp=importlib.import_module('AlphaEdit.AlphaEdit_hparams').AlphaEditHyperParams.from_json(ROOT/'imports/config.json')
    assert hp.layers==[4] and hp.L2==1 and hp.blue and hp.kl_factor==.0625
    native.CONTEXT_TEMPLATES_CACHE=cp['metadata']['contexts']
    state=cp['cache_c'].clone();w=dict(model.named_parameters())[WEIGHT]
    with torch.no_grad():w.copy_(cp['weights'][WEIGHT].to(w.device))
    expected={r['case_id']:z for r,z in zip(rows,targets['values'])}
    original_z=native.compute_z;original_k=native.compute_ks;keys=[]
    def cached(model_,tok_,request,hp_,layer,contexts):
        assert layer==4 and contexts==cp['metadata']['contexts']
        ledger.add('native_z_cache_hit');return expected[request['case_id']].to('cuda')
    def capture(*a,**kw):
        with ledger.time('native_key_capture'):k=original_k(*a,**kw)
        ledger.add('native_key_capture')
        if not keys:keys.append(k.detach().T.clone())
        return k
    native.compute_z=cached;native.compute_ks=capture
    repr_tools=importlib.import_module('rome.repr_tools')
    original_reader=repr_tools.get_reprs_at_idxs
    repr_tools.get_reprs_at_idxs=bounded_reader(original_reader,ledger,2)
    original_solve=torch.linalg.solve
    def timed_solve(*args,**kwargs):
        with ledger.time('native_linear_solve'):
            value=original_solve(*args,**kwargs)
        ledger.add('native_linear_solve')
        return value
    torch.linalg.solve=timed_solve
    requests=[dict(r['requested_rewrite'],case_id=r['case_id']) for r in rows]
    try:
        with ledger.time('native_write'):
            native.apply_AlphaEdit_to_model(model,tok,requests,hp,cache_c=state,P=p[None])
    finally:
        native.compute_z=original_z;native.compute_ks=original_k
        repr_tools.get_reprs_at_idxs=original_reader
        torch.linalg.solve=original_solve
    assert ledger.counts['native_z_cache_hit']==100 and len(keys)==1
    assert torch.isfinite(w).all() and torch.isfinite(state).all()
    return w.detach().clone(),state,keys[0],hp,native
