"""Task-owned single L4 runtime. Original native writer and assets are read-only.

M episodes reset W0/M0 independently. No S/R/L entrypoint exists here.
"""
import copy
import importlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from .common import digest, sha, tensor_sha, Timer, write, save_tensor
from .alltoken import FullWeightLlamaOracle, WEIGHT, model_guard
from .binding import pack, protected_sequences, score_rows, quality_ok
from . import geometry
from project.run_scripts.baseline_mechanism_first.fixtures import capture_rng, restore_rng
from project.run_scripts.low_cost_write_donor_pilot.fitting import NativeSingletonFitter, select_projector
from project.run_scripts.bg_tw_reference.ep_tw.model_adapter import TeacherStore, capture_native_fit


class Runtime:
    def __init__(self, lock, output):
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from scripts.fixed_counterfact import load_prefix
        self.lock, self.output, self.timing = lock, Path(output), {}
        self.records = load_prefix(lock['dataset_root'], 1000)
        if digest(self.records) != lock['records_digest'] or [r['case_id'] for r in self.records] != lock['sample_order']:
            raise ValueError('FIXED_PREFIX_IDENTITY')
        if torch.__version__ != lock['torch'] or transformers.__version__ != lock['transformers']:
            raise ValueError('PINNED_LIBRARY_IDENTITY')
        if os.environ.get('SLURMD_NODENAME') != 'server4' or not torch.cuda.is_available():
            raise ValueError('SLURM_SERVER4_GPU_REQUIRED')
        torch.set_num_threads(8); transformers.set_seed(lock['seed'])
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        sys.path.insert(0,lock['blue_root']); os.chdir(lock['blue_root'])
        self.module=importlib.import_module('AlphaEdit.AlphaEdit_main')
        HP=importlib.import_module('AlphaEdit.AlphaEdit_hparams').AlphaEditHyperParams
        self.hp=HP.from_json(lock['config4'])
        h=self.hp
        if not (h.blue and h.layers==[4] and h.L2==1 and h.v_weight_decay==.5 and h.clamp_norm_factor==.75
                and h.v_lr==.1 and h.v_num_grad_steps==25 and h.kl_factor==.0625):
            raise ValueError('NATIVE_HPARAMS')
        with Timer(self.timing,'model_load'):
            self.model=AutoModelForCausalLM.from_pretrained(lock['snapshot'],local_files_only=True,
                torch_dtype=torch.float32,low_cpu_mem_usage=True,attn_implementation='eager').cuda().eval()
        self.tok=AutoTokenizer.from_pretrained(lock['snapshot'],local_files_only=True)
        self.tok.add_bos_token=False;self.tok.pad_token_id=self.tok.eos_token_id
        self.etok=AutoTokenizer.from_pretrained(lock['snapshot'],local_files_only=True)
        self.etok.pad_token_id=self.etok.eos_token_id
        if self.tok.padding_side!='right' or self.etok.padding_side!='right':raise ValueError('TOKEN_PADDING')
        for p in self.model.parameters():p.requires_grad_(False)
        self.W=dict(self.model.named_parameters())[WEIGHT]
        if self.W.shape!=(4096,14336) or any(p.dtype!=torch.float32 for p in self.model.parameters()):
            raise ValueError('FULL_MODEL_FP32_L4_SHAPE')
        with Timer(self.timing,'P_load'):
            stack=torch.load(lock['projector'],weights_only=True,mmap=True,map_location='cpu')
            self.P,self.pmap=select_projector(stack,4);del stack
        self.M=torch.zeros_like(self.P); self.W0=self.W.detach().cpu().clone()
        cold=lock['cold_capsule']
        if sha(cold['path'])!=cold['sha256']:raise ValueError('COLD_CAPSULE_IDENTITY')
        common=json.loads(Path(cold['path']).read_text())
        self.context=copy.deepcopy(common['contexts']);self.rng=common['rng']
        if [[self.tok(x)['input_ids'] for x in g] for g in self.context]!=common['context_tokens']:
            raise ValueError('CONTEXT_TOKEN_IDENTITY')
        if tensor_sha(self.W0)!=common['W0']['4'] or self.pmap!=common['projector_mapping']['4']:
            raise ValueError('COLD_W0_OR_P4_IDENTITY')
        self.module.CONTEXT_TEMPLATES_CACHE=copy.deepcopy(self.context);self.module.COV_CACHE={}
        restore_rng(self.rng)
        self.fitter=NativeSingletonFitter(self.module,expected_source_sha256=lock['editor_sha256'],contexts=self.context)
        tm=lock['teacher_manifest']
        self.teacher=TeacherStore(lock['reference_root'],tm['path'],expected_manifest_sha=tm['sha256'],verify_payload_hashes=False)
        from project.run_scripts.baseline_mechanism_first.evaluation import bind_evaluation_sources
        self.observer_binding=bind_evaluation_sources(lock['historical_evaluator_root'],helper_root=lock['helper_scripts_root'])
        self.base_guard=model_guard(self.model);self.oracles=[]
        self.identity=dict(W0=tensor_sha(self.W0),M0=tensor_sha(self.M),P4=self.pmap,contexts=digest(self.context),
            context_tokens=digest(common['context_tokens']),rng=digest(self.rng),teacher=tm,records_digest=lock['records_digest'],
            torch=torch.__version__,transformers=transformers.__version__,microbatch=1,physical_layer=4)
        write(self.output/'runtime-load.json',dict(identity=self.identity,timing=self.timing,model_gpu_name=torch.cuda.get_device_name(),
            model_bytes=sum(p.numel()*p.element_size() for p in self.model.parameters())))

    def guard(self):
        after=model_guard(self.model)
        strip=lambda g:(tuple(x for x in g[0] if x[1]!=WEIGHT),g[1])
        if strip(after)!=strip(self.base_guard):raise RuntimeError('NONSELECTED_STATE_MUTATION')
        if any(p.requires_grad or p.grad is not None for p in self.model.parameters()):raise RuntimeError('UNFROZEN_OR_GRAD_STATE')
        if self.module.CONTEXT_TEMPLATES_CACHE!=self.context or self.module.COV_CACHE:raise RuntimeError('NATIVE_CONTEXT_CACHE_MUTATION')
        if torch.count_nonzero(self.M):raise RuntimeError('COLD_M_HISTORY_MUTATED')

    def sync_oracles(self):
        expected=self.W.detach().cpu()
        for oracle in self.oracles:oracle.acknowledge_selected_write(expected)
        self.guard()

    def copy_weight(self,weight):
        if weight.dtype!=torch.float32 or weight.shape!=self.W.shape or not torch.isfinite(weight).all():
            raise ValueError('INVALID_WEIGHT_WRITE')
        with torch.no_grad():self.W.copy_(weight.to(self.W.device))
        if not torch.equal(self.W.detach().cpu(),weight.cpu()):raise RuntimeError('INEXACT_WEIGHT_COPY')
        self.sync_oracles()

    def reset(self):
        self.oracles=[];self.copy_weight(self.W0);self.M.zero_();restore_rng(self.rng);self.guard()
        return dict(W=tensor_sha(self.W),M=tensor_sha(self.M),rng=digest(capture_rng()),independent_cold=True)

    def requests(self,records):
        return [dict(copy.deepcopy(r['requested_rewrite']),case_id=r['case_id']) for r in records]

    def native(self,records,directory,reuse=False):
        directory=Path(directory);before=self.reset()
        if reuse:
            if [r['case_id'] for r in records]!=self.lock['sample_order'][:100]:raise ValueError('REUSE_EPISODE_NOT_B1')
            path=Path(self.lock['reused_native_b1'])
            sealed=self.lock['reused_native_binding']
            if path.stat().st_size!=sealed['native']['bytes'] or sha(path)!=sealed['native']['sha256']:
                raise ValueError('REUSED_NATIVE_FILE_IDENTITY')
            result=torch.load(path,weights_only=True,mmap=True,map_location='cpu')
            receipt=result['receipt']
            if tensor_sha(result['weight'])!=sealed['b1_endpoint_verified'] or tensor_sha(result['weight'])!=receipt['endpoint_weight_sha256']:
                raise ValueError('REUSED_NATIVE_ENDPOINT_BYTES')
            if [r['case_id'] for r in result['target_observations']]!=[r['case_id'] for r in records]:raise ValueError('REUSED_NATIVE_TARGET_ORDER')
            if receipt['entry_weight_sha256']!=before['W'] or receipt['history_sha256']!=before['M']:
                raise ValueError('REUSE_W0_M0_MISMATCH')
            if receipt['projector_sha256']!=tensor_sha(self.P):raise ValueError('REUSE_P_MISMATCH')
            self.copy_weight(result['weight']);source=dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))
        else:
            with Timer(self.timing,'native_fit'):
                result=capture_native_fit(self.fitter,self.model,self.tok,self.hp,self.M,self.P,self.requests(records),layer=4)
            self.guard();source=save_tensor(directory/'native-capsule.pt',result)
        if result['receipt']['compute_z']!=len(records) or result['receipt']['history_append']!=0:
            raise ValueError('NATIVE_COUNTS')
        write(directory/'native-binding.json',dict(source=source,mode='REUSE' if reuse else 'RUN_MISSING',
            entry=before,endpoint=tensor_sha(result['weight']),case_ids=[r['case_id'] for r in records],receipt=result['receipt'],
            native_fit_new_calls=0 if reuse else 1,native_target_new_calls=0 if reuse else len(records)))
        restore_rng(self.rng)
        return result

    def reference_oracle(self,split='S64'):
        indices=list(self.teacher.indices(split));packs=[]
        for i in indices:
            packs.append(pack(self.teacher.ids[i]))
        def loader(i,cache):
            ids,lp,_=self.teacher.document(indices[i],'cpu')
            if not torch.equal(ids,cache.packed['input_ids']):raise ValueError('TEACHER_TOKEN_MISMATCH')
            return lp[0]
        oracle=FullWeightLlamaOracle(self.model,packs,teacher_loader=loader)
        self.oracles.append(oracle);return oracle

    def protected_oracle(self,records):
        packs,rows,unique,meta=protected_sequences(self.tok,self.etok,self.requests(records),self.context)
        oracle=FullWeightLlamaOracle(self.model,packs);self.oracles.append(oracle)
        # FP32 captured keys; no output-space or writer@K compression.
        values=[];prefixes={};actual_alias=[]
        for alias in meta['key_aliases']:
            ci,pos=alias['cache'],alias['position'];value=oracle.caches[ci].keys[0,pos]
            candidates=prefixes.setdefault(alias['prefix_sha'],[])
            found=next((i for i in candidates if torch.equal(values[i],value)),None)
            if found is None:found=len(values);values.append(value);candidates.append(found)
            actual_alias.append(dict(**alias,actual_key_column=found))
        K=torch.stack(values,1).contiguous()
        meta['actual_key_aliases']=actual_alias;meta['actual_distinct_key_columns']=len(values)
        meta['dedup_rule']='same exact token prefix AND identical captured FP32 key bytes only'
        meta['K_shape']=list(K.shape);meta['K_sha256']=tensor_sha(K)
        return oracle,rows,K,meta

    def factor_A(self,native):
        """Additional diagnostic map solve, never a replacement for saved W_N."""
        K=native['captures']['compute_ks'][0].T.to(self.W.device)
        P=self.P[0].to(self.W.device)
        with Timer(self.timing,'additional_A_map_solve'),torch.no_grad():
            lhs=P@(K@K.T+self.M[0].to(self.W.device))+self.hp.L2*torch.eye(K.shape[0],device=self.W.device)
            A=torch.linalg.solve(lhs,P@K).T.cpu()
        if not torch.isfinite(A).all():raise FloatingPointError('NONFINITE_A_MAP')
        return A

    def covariance(self,oracle,weight,gradient=False):
        # W0 cumulative activation drift, NOT KL and NOT displacement from WN.
        delta=weight.double()-self.W0.double();total=0.;g=torch.zeros_like(delta) if gradient else None;rows=[]
        with Timer(self.timing,'covariance_objective'):
            for i in range(len(oracle.caches)):
                K=oracle.capture_keys(i).double();response=delta@K
                loss=float(.5*response.square().sum()/K.shape[1]);rows.append(dict(index=i,loss=loss,valid_tokens=K.shape[1]));total+=loss
                if gradient:g.add_((response@K.T)/K.shape[1])
            if gradient:g.div_(len(rows))
        return total/len(rows),g,rows

    def byte_hash_nonselected(self):
        # T and M episode boundaries: no persistent full-model copy.
        return {n:tensor_sha(p) for n,p in self.model.named_parameters() if n!=WEIGHT}


def invariant(oracle,rows,anchor,weight,WN,ideal,K,allowed):
    actual=weight.double()-WN.double()
    result=geometry.invariant_diagnostics(ideal,actual,WN,K,allowed)
    if not result['ideal_pass']:raise RuntimeError('FP64_NULLSPACE_PROPOSAL_FAILURE')
    current=score_rows(oracle,weight,rows)
    result['max_NLL_difference']=max(abs(current[k]['nll']-anchor[k]['nll']) for k in anchor)
    strict=lambda x:{k for k,r in x.items() if r['branch']=='new' and r['strict']}
    pair=lambda x:{k for k,r in x.items() if r['kind']=='canonical' and r['branch']=='new'
                   and k.removesuffix('new')+'old' in x and r['nll']<x[k.removesuffix('new')+'old']['nll']}
    result['strict_symmetric_difference']=sorted(strict(current)^strict(anchor))
    result['pair_symmetric_difference']=sorted(pair(current)^pair(anchor))
    stats=[oracle.compare_logits(i,WN,weight,left_route='cached',right_route='cached') for i in range(len(oracle.caches))]
    result['logit_max']=max(r['max_abs'] for r in stats)
    result['logit_rms']=(sum(r['squared_error'] for r in stats)/sum(r['logit_elements'] for r in stats))**.5
    result['logit_elements']=sum(r['logit_elements'] for r in stats)
    result['pass']=bool(result['actual_response_pass'] and result['actual_leakage_pass'] and result['max_NLL_difference']<=1e-4
        and not result['strict_symmetric_difference'] and not result['pair_symmetric_difference']
        and result['logit_max']<=1e-3 and result['logit_rms']<=1e-4)
    return result
