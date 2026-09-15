"""Actual Llama SL-ZFlow execution helpers; no native target optimization."""
import hashlib
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from .llama_adapter import LlamaAffineOracle, WEIGHT, model_guard
from .native_binding import build_training_sequences,pack_sequences,capture_native_keys
from .flow_core import QuadraticGeometry, integrate
from .durable import tensor_sha256


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:
        json.dump(value,f,sort_keys=True,indent=2,ensure_ascii=True,allow_nan=False)
        f.flush();os.fsync(f.fileno())
    return dict(path=str(path),bytes=path.stat().st_size,sha256=file_sha(path))


def tensor_save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:
        torch.save(value,f);f.flush();os.fsync(f.fileno())
    return dict(path=str(path),bytes=path.stat().st_size,sha256=file_sha(path))


def capture_rng():
    n=np.random.get_state()
    return dict(python=random.getstate(),numpy=(n[0],n[1].tolist(),n[2],n[3],n[4]),
                torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())


def restore_rng(rng):
    random.setstate(rng['python']);n=rng['numpy']
    np.random.set_state((n[0],np.asarray(n[1],dtype=np.uint32),n[2],n[3],n[4]))
    torch.set_rng_state(rng['torch']);torch.cuda.set_rng_state_all(rng['cuda'])


def state_fingerprint(value):
    """Exact nested state identity, including RNG tensor bytes and container type."""
    def describe(item):
        if isinstance(item,torch.Tensor):
            return dict(kind='tensor',dtype=str(item.dtype),shape=list(item.shape),sha256=tensor_sha256(item))
        if isinstance(item,dict):
            return dict(kind='dict',items=[[str(k),describe(v)] for k,v in sorted(item.items())])
        if isinstance(item,(tuple,list)):
            return dict(kind=type(item).__name__,items=[describe(v) for v in item])
        return dict(kind=type(item).__name__,value=item)
    return digest(describe(value))


def load_model(lock):
    from transformers import AutoModelForCausalLM,AutoTokenizer
    import transformers
    if transformers.__version__!='4.44.2':raise RuntimeError('TRANSFORMERS_VERSION_MISMATCH')
    torch.set_num_threads(8);random.seed(20260907);np.random.seed(20260907)
    torch.manual_seed(20260907);torch.cuda.manual_seed_all(20260907)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    begin=time.perf_counter()
    model=AutoModelForCausalLM.from_pretrained(lock['model_path'],local_files_only=True,
                low_cpu_mem_usage=True,torch_dtype=torch.float32,attn_implementation='eager').cuda().eval()
    model.requires_grad_(False)
    tok=AutoTokenizer.from_pretrained(lock['model_path'],local_files_only=True)
    tok.add_bos_token=False;tok.pad_token_id=tok.eos_token_id
    evaltok=AutoTokenizer.from_pretrained(lock['model_path'],local_files_only=True)
    evaltok.pad_token_id=evaltok.eos_token_id
    if tok.padding_side!='right' or evaltok.padding_side!='right':raise RuntimeError('PADDING')
    if any(p.dtype!=torch.float32 for p in model.parameters()):raise RuntimeError('FULL_FP32')
    if model.config.num_hidden_layers!=32 or tuple(dict(model.named_parameters())[WEIGHT].shape)!=(4096,14336):
        raise RuntimeError('ACTUAL_LLAMA3_8B_SCHEMA')
    return model,tok,evaltok,dict(model_load_seconds=time.perf_counter()-begin,
        torch=str(torch.__version__),transformers=transformers.__version__,
        cuda=str(torch.version.cuda),device=torch.cuda.get_device_name(0),
        gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,
        model_dtype='float32',attention='eager',writer_add_bos=False,evaluator_add_bos=evaltok.add_bos_token,
        writer_padding='right',evaluator_packing='native manual left / microbatch16',
        base_selected_weight_sha256=tensor_sha256(dict(model.named_parameters())[WEIGHT]))


def prepare_geometry(keys,projector,history,*,direct_test=False):
    """One authoritative nonsymmetric LU factorization per MAIN batch."""
    begin=time.perf_counter();device=torch.device('cuda')
    k=keys.to(device);p=projector.to(device);m=history.to(device)
    n=p@(k@k.T+m)+torch.eye(k.shape[0],dtype=torch.float32,device=device)
    rhs=p@k
    lu,pivots,info=torch.linalg.lu_factor_ex(n,check_errors=False)
    if int(info)!=0:raise RuntimeError('NATIVE_LU_FACTOR_FAILURE')
    solved=torch.linalg.lu_solve(lu,pivots,rhs)
    residual=float((n@solved-rhs).double().norm()/rhs.double().norm().clamp_min(1e-30))
    b=solved.T.contiguous()
    raw=(b@m@b.T+b@b.T)/keys.shape[1]
    s=(raw+raw.T)*.5
    if not torch.isfinite(b).all() or not torch.isfinite(s).all():raise RuntimeError('NONFINITE_GEOMETRY')
    evidence=dict(factorizations=1,system='P@(K@K.T+M_entry)+I; nonsymmetric LU',
        m=keys.shape[1],q=keys.shape[1],E='identity',solve_relative_residual=residual,
        right_leakage=float((b-b@p).double().norm()/b.double().norm().clamp_min(1e-30)),
        history_pointer=history.data_ptr(),history_version=history._version,
        lambda_write=1,geometry_dtype='float32',native_z_calls=0)
    if direct_test:
        # Identical LU factorization, independent residual-inclusive RHS solve.
        gen=torch.Generator(device='cpu').manual_seed(20260907)
        x=torch.randn((4096,keys.shape[1]),generator=gen,dtype=torch.float32).to(device)*.001
        direct=torch.linalg.lu_solve(lu,pivots,p@k@x.T).T
        factored=x@b
        evidence['direct_rhs_relative_error']=float((direct-factored).double().norm()/direct.double().norm().clamp_min(1e-30))
        evidence['direct_rhs_max_abs']=float((direct-factored).abs().max())
    del lu,pivots,n,rhs,p,m,k,solved
    torch.cuda.empty_cache()
    geom=QuadraticGeometry(s,metric_ridge_relative=.001)
    evidence.update(seconds=time.perf_counter()-begin,S_min_eigenvalue=float(geom.s.min()),
        S_max_eigenvalue=float(geom.s.max()),metric_epsilon=geom.epsilon,
        clipped_negative_eigenvalues=geom.clipped_negative_eigenvalues)
    return b,s,geom,evidence


def prepare_batch(model,tok,records,contexts,projector,history,*,microbatch=2,direct_test=False):
    begin=time.perf_counter()
    requests=[dict(r['requested_rewrite'],case_id=r['case_id']) for r in records]
    binding=build_training_sequences(tok,requests,contexts)
    key_begin=time.perf_counter()
    keys,key_evidence=capture_native_keys(model,binding,pad_token_id=tok.pad_token_id,microbatch=microbatch)
    key_evidence['seconds']=time.perf_counter()-key_begin
    b,s,geom,geometry=prepare_geometry(keys,projector,history,direct_test=direct_test)
    batches=[pack_sequences(binding.sequences[i:i+microbatch],tok.pad_token_id)
             for i in range(0,len(binding.sequences),microbatch)]
    oracle=LlamaAffineOracle(model,b,batches,beta=.0625)
    return binding,keys,b,s,geom,oracle,dict(binding=binding.metadata,keys=key_evidence,
        geometry=geometry,prefix_teacher_seconds=oracle.preparation_seconds,
        total_preparation_seconds=time.perf_counter()-begin,physical_microbatch=microbatch)


def finite_json(value):
    if isinstance(value,float) and not np.isfinite(value):return None
    if isinstance(value,dict):return {k:finite_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [finite_json(v) for v in value]
    return value


def run_flow(oracle,geom,contract,*,technical_calls=None):
    options=dict(contract['integrator'])
    if technical_calls is not None:options['max_oracle_calls']=technical_calls
    begin=time.perf_counter()
    result=integrate(oracle,torch.zeros(oracle.shape,device=oracle.device),geom,
                     price=contract['objective']['price'],**options)
    if result.oracle_calls!=1+result.accepted_steps+result.rejected_steps:raise RuntimeError('ORACLE_COUNT')
    return result,dict(status=result.status,oracle_calls=result.oracle_calls,
        accepted=result.accepted_steps,rejected=result.rejected_steps,
        trace=finite_json(result.trace),terminal=finite_json(result.terminal_stats),
        oracle_events=oracle.events,work=oracle.work,flow_seconds=time.perf_counter()-begin)


def check_parity(evidence,tolerance,*,gradient=False):
    checks={k:evidence[k]<=tolerance[k] for k in
            ('max_logit_abs','logit_relative_l2','edit_abs_error','kl_abs_error')}
    if gradient:checks['gradient_relative_l2']=evidence['gradient_relative_l2']<=tolerance['gradient_relative_l2']
    return dict(**evidence,checks=checks,passed=all(checks.values()),tolerance=tolerance)


@torch.no_grad()
def probe_next_entry(model,tok,record,contexts):
    """Technical two-process continuation probe at next request, no writer/z."""
    binding=build_training_sequences(tok,[dict(record['requested_rewrite'],case_id=record['case_id'])],contexts)
    packed=pack_sequences(binding.sequences,tok.pad_token_id,device='cuda')
    h=model.model(input_ids=packed['input_ids'],attention_mask=packed['attention_mask'],
                  position_ids=packed['position_ids'],use_cache=False).last_hidden_state
    rows=torch.cat((packed['edit_rows'],packed['kl_rows']));cols=torch.cat((packed['edit_cols'],packed['kl_cols']))
    return model.lm_head(h[rows,cols]).float().cpu(),binding.metadata
