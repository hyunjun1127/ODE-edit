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
    obj=torch.load(p,map_location='cpu',weights_only=True,mmap=True)
    assert set(obj['weights'])=={WEIGHT}
    assert obj['weights'][WEIGHT].dtype==torch.float32 and obj['cache_c'].dtype==torch.float32
    return obj


def load():
    from transformers import AutoModelForCausalLM,AutoTokenizer
    import transformers,tokenizers
    assert torch.__version__=='2.9.1+cu128' and transformers.__version__=='4.44.2'
    torch.set_num_threads(8);random.seed(20260907);np.random.seed(20260907);torch.manual_seed(20260907)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
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
        slurm_job=os.environ.get('SLURM_JOB_ID'),save_checkpoints=False)
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
    with torch.no_grad():
        k=ks.compute_ks(proxy,tok,requests,hp,4,contexts).T.contiguous().cpu()
        bare,_=z.get_module_input_output_at_words(proxy,tok,4,
            context_templates=[r['prompt'] for r in requests],words=[r['subject'] for r in requests],
            module_template=hp.rewrite_module_tmp,fact_token_strategy=hp.fact_token)
        _,h=z.get_module_input_output_at_words(proxy,tok,4,
            context_templates=[r['prompt'] for r in requests],words=[r['subject'] for r in requests],
            module_template=hp.layer_module_tmp,fact_token_strategy=hp.fact_token)
    torch.cuda.synchronize()
    return dict(K=k,bare_K=bare.T.contiguous().cpu(),h0=h.T.contiguous().cpu(),
                ids=[r['case_id'] for r in rows],contexts_sha256=digest(contexts),
                seconds=time.perf_counter()-t,early_stop=early,
                native_group_sizes=[len(x) for x in contexts],key_weights=[.5,.1,.1,.1,.1,.1])


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
