"""Read-only original Llama/BLUE capture and evaluation binding.

Only restored L4 tensors are applied. No native optimizer, writer, history
append or checkpoint persistence is imported/called by the analysis runner.
"""
import importlib
import os
import random
import sys
import time
import types
from pathlib import Path
import numpy as np
import torch
from .common import *


def package(name,path):
    if name in sys.modules:
        assert list(sys.modules[name].__path__)==[str(path)], name
    else:
        m=types.ModuleType(name);m.__path__=[str(path)];m.__package__=name
        sys.modules[name]=m


def bind_sources():
    source=read(ATTEMPT/'inputs/source-ready.json')
    for member in source['members']:
        assert sha256(member['path'])==member['sha256'], member['path']
    blue=Path(source['blue']); helper=Path(source['helper'])
    sys.path.insert(0,str(blue))
    # Bypass original initializers which eagerly import unrelated writers.
    for name in ('util','rome','AlphaEdit'):
        package(name,blue/name)
    for name in ('blue_alphaedit_sequential_comparison','alphaedit_strength_neutral_barrier','ordered_response_barrier_ode'):
        package('project.run_scripts.'+name,helper/'project/run_scripts'/name)
    ks=importlib.import_module('AlphaEdit.compute_ks')
    z=importlib.import_module('AlphaEdit.compute_z')
    hp=importlib.import_module('AlphaEdit.AlphaEdit_hparams').AlphaEditHyperParams.from_json(source['config'])
    # Deliberate fail-closed sentinel: target optimization is forbidden here.
    def forbidden(*args,**kw): raise RuntimeError('NEW_Z_OPTIMIZATION_FORBIDDEN')
    z.compute_z=forbidden
    ev=importlib.import_module('project.run_scripts.blue_alphaedit_sequential_comparison.evaluation')
    assert hp.layers==[4] and hp.blue and hp.L2==1
    return ks,z,hp,ev


def stream_and_probe():
    m=source_map(); sample=read(mapped(SAMPLE,m));dataset=read(CONTRACT['paths']['dataset'])
    assert digest(sample['records'])==sample['ordered_root']==CONTRACT['identity']['sample_root']
    byid={int(r['case_id']):r for r in dataset};rows=[]
    for r in sample['records']:
        row=byid[r['case_id']];assert digest(row)==r['raw_record_sha256'];rows.append(row)
    assert len(rows)==10000 and len({r['case_id'] for r in rows})==10000
    ids={r['case_id'] for r in rows}
    import hashlib
    candidates=[r for r in dataset if int(r['case_id']) not in ids]
    candidates.sort(key=lambda r:(hashlib.sha256(f"20260920|{int(r['case_id'])}".encode()).hexdigest(),int(r['case_id'])))
    return rows,candidates[:512]


def checkpoint(batch):
    p=CPROOT/f'B{batch:03d}/W-method-state.pt'
    previous=[r for r in read(ATTEMPT/'results/geometry/checkpoint-input-verification.json') if r['path']==str(p)]
    assert len(previous)==1 and previous[0]['full_sha256'] and previous[0]['unchanged_during_hash']
    now=stat_identity(p)
    for field in ('bytes','mtime_ns','inode','device'):
        assert now[field]==previous[0][field], 'CHECKPOINT_STAT_DRIFT:'+str(p)
    obj=torch.load(p,map_location='cpu',weights_only=True,mmap=True)
    assert set(obj['weights'])=={WEIGHT}
    assert obj['weights'][WEIGHT].dtype==torch.float32 and obj['cache_c'].dtype==torch.float32
    return obj


def verify_model_assets():
    """Check complete original model/tokenizer inventory before model loading.

    Reuse a current stat-bound A00 full-hash receipt if provided; otherwise
    measure a fresh immutable read and include its cost (no hidden inference).
    """
    lock=read(mapped(S4RUN+'/execution.lock.json'));prefix=lock['snapshot'].rstrip('/')+'/'
    known_path=ATTEMPT/'inputs/model-asset-verification.json'
    known={m['path']:m for m in read(known_path)['members']} if known_path.exists() else {}
    records=[];start=time.perf_counter()
    for member in lock['members']:
        if not member['path'].startswith(prefix):continue
        relative=member['path'][len(prefix):];assert '..' not in Path(relative).parts
        path=Path(CONTRACT['paths']['model_snapshot'])/relative;now=stat_identity(path)
        assert now['bytes']==member['bytes'],'MODEL_ASSET_SIZE_MISMATCH'
        prior=known.get(str(path));reused=bool(prior and prior.get('full_sha256') and
            prior.get('sha256')==member['sha256'] and
            all(prior.get(k)==now[k] for k in ('bytes','mtime_ns','inode','device')))
        if not reused:assert sha256(path)==member['sha256'],'MODEL_ASSET_HASH_MISMATCH:'+relative
        after=stat_identity(path);assert after==now,'MODEL_ASSET_CHANGED_DURING_READ'
        records.append(dict(**now,sha256=member['sha256'],full_sha256=True,reused_stat_bound_receipt=reused))
    assert records and any('safetensors' in r['path'] for r in records)
    return dict(status='PASS',members=records,seconds=time.perf_counter()-start,
                source_execution_lock_sha256=sha256(mapped(S4RUN+'/execution.lock.json')))


def load():
    from transformers import AutoModelForCausalLM,AutoTokenizer
    import transformers,tokenizers
    assert torch.__version__=='2.9.1+cu128' and transformers.__version__=='4.44.2'
    torch.set_num_threads(8);random.seed(20260907);np.random.seed(20260907);torch.manual_seed(20260907)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    assets=verify_model_assets()
    t=time.perf_counter(); snapshot=CONTRACT['paths']['model_snapshot']
    model=AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,low_cpu_mem_usage=True,
        attn_implementation='eager').cuda().eval().requires_grad_(False)
    writer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True)
    writer.add_bos_token=False;writer.pad_token_id=writer.eos_token_id
    evaluator=AutoTokenizer.from_pretrained(snapshot,local_files_only=True);evaluator.pad_token_id=evaluator.eos_token_id
    assert writer.padding_side==evaluator.padding_side=='right'
    assert {p.dtype for p in model.parameters()}=={torch.float32}
    w=model.get_parameter(WEIGHT);w0=w.detach().cpu().clone()
    assert tensor_sha(w0)==CONTRACT['identity']['w0_fp32_tensor_sha256']
    receipt=dict(torch=torch.__version__,transformers=transformers.__version__,tokenizers=tokenizers.__version__,
        source=str(transformers.__file__),gpu=torch.cuda.get_device_name(),
        dtype='float32',attention=model.config._attn_implementation,autocast=torch.is_autocast_enabled(),
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,
        writer_tokenizer={'add_bos_token':writer.add_bos_token,'padding':writer.padding_side},
        evaluator_tokenizer={'add_bos_token':getattr(evaluator,'add_bos_token','NOT_EXPOSED'),'padding':evaluator.padding_side,
                             'kernel_padding':'manual_left','position_ids':'implicit','microbatch':16},
        w0_sha256=tensor_sha(w0),load_seconds=time.perf_counter()-t,
        slurm_job=os.environ.get('SLURM_JOB_ID'),save_checkpoints=False,model_assets=assets)
    return model,writer,evaluator,w0,receipt


def pointer_versions(model):
    return {n:(p.data_ptr(),p._version) for n,p in model.named_parameters() if n!=WEIGHT}


def set_weight(model,w):
    with torch.no_grad(): model.get_parameter(WEIGHT).copy_(w)


class PrefixComplete(Exception): pass


class PrefixProxy:
    """Keep original capture batching/tokens; stop before physical layer5.

    The L4 output hook completes before L5 pre-hook raises; the exception is
    swallowed only by this wrapper, leaving original repr_tools unchanged.
    """
    def __init__(self,model): self.model=model
    def __getattr__(self,name): return getattr(self.model,name)
    def __call__(self,*a,**kw):
        def stop(*unused): raise PrefixComplete()
        hook=self.model.model.layers[5].register_forward_pre_hook(stop)
        try:
            try: return self.model(*a,**kw)
            except PrefixComplete: return None
        finally: hook.remove()


def capture(model,tok,rows,contexts,bindings,early=False):
    ks,z,hp,_=bindings; proxy=PrefixProxy(model) if early else model
    requests=[dict(r['requested_rewrite'],case_id=int(r['case_id'])) for r in rows]
    t=time.perf_counter()
    packing=[];positions=[]
    def inputs_hook(module,args,kw):
        packing.append({key:value.detach().cpu().tolist() for key,value in kw.items()
                        if key in ('input_ids','attention_mask','position_ids','cache_position') and isinstance(value,torch.Tensor)})
    def position_hook(module,args,kw):
        positions.append({key:value.detach().cpu().tolist() for key,value in kw.items()
                          if key in ('position_ids','cache_position') and isinstance(value,torch.Tensor)})
    ih=model.register_forward_pre_hook(inputs_hook,with_kwargs=True)
    ph=model.model.layers[0].register_forward_pre_hook(position_hook,with_kwargs=True)
    try:
        with torch.no_grad():
            k=ks.compute_ks(proxy,tok,requests,hp,4,contexts).T.contiguous().cpu()
            bare,_=z.get_module_input_output_at_words(proxy,tok,4,
                context_templates=[r['prompt'] for r in requests],words=[r['subject'] for r in requests],
                module_template=hp.rewrite_module_tmp,fact_token_strategy=hp.fact_token)
            _,h=z.get_module_input_output_at_words(proxy,tok,4,
                context_templates=[r['prompt'] for r in requests],words=[r['subject'] for r in requests],
                module_template=hp.layer_module_tmp,fact_token_strategy=hp.fact_token)
    finally: ih.remove();ph.remove()
    torch.cuda.synchronize()
    repr_tools=importlib.import_module('rome.repr_tools')
    templates=[c.format(r['prompt']) for r in requests for group in contexts for c in group]
    words=[r['subject'] for r in requests for group in contexts for c in group]
    lookup=repr_tools.get_words_idxs_in_templates(tok,templates,words,hp.fact_token[len('subject_'):])
    return dict(K=k,bare_K=bare.T.contiguous().cpu(),h0=h.T.contiguous().cpu(),
                ids=[r['case_id'] for r in rows],contexts_sha256=digest(contexts),
                seconds=time.perf_counter()-t,early_stop=early,
                native_group_sizes=[len(x) for x in contexts],key_weights=[.5,.1,.1,.1,.1,.1],
                actual_packing=packing,actual_positions=positions,packing_sha256=digest(packing),
                subject_lookup=lookup,lookup_sha256=digest(lookup),
                tokenizer_padding=tok.padding_side,tokenizer_add_bos=tok.add_bos_token,
                capture_source_ready_sha256=sha256(ATTEMPT/'inputs/source-ready.json'))


def evaluate(model,tok,rows,bindings):
    # Original exact evaluator; only guard is lighter because no method state is
    # owned/mutable in this observation runner. Actual selected bytes are hashed.
    _,_,_,ev=bindings
    kernel=importlib.import_module('project.run_scripts.alphaedit_strength_neutral_barrier.evaluator')
    locality=importlib.import_module('project.run_scripts.ordered_response_barrier_ode.counterfact_locality_evaluator')
    before=pointer_versions(model);w=model.get_parameter(WEIGHT);ws=tensor_sha(w)
    pairs=kernel.counterfact_pairs(rows);pairs['locality_target_new']=locality.counterfact_locality_target_new_pairs(rows)
    t=time.perf_counter()
    raw={k:kernel.evaluate_pairs(model,tok,v,device=torch.device('cuda'),microbatch_size=16) for k,v in pairs.items()}
    assert before==pointer_versions(model) and ws==tensor_sha(w),'EVALUATOR_MUTATION'
    return dict(metrics=ev.reduce(raw),request_order=digest([r['case_id'] for r in rows]),
                seconds=time.perf_counter()-t,nonmutation=True,weight_sha256=ws,raw=raw)
