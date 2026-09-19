"""New task-owned loader and own-entry writer; old runtime is read-only."""
import copy
import importlib
import json
import os
from pathlib import Path
import sys
import torch
from .config import check_lock
from project.run_scripts.en_execution_reuse.model import verify_large_asset_stats
from project.run_scripts.en_execution_reuse.matched_runner import load_teacher
from project.run_scripts.en_execution_reuse.current_oracle import MeasuredCurrentOracle
from project.run_scripts.single_layer_edit_preserving_correction.runtime import Runtime as Methods
from project.run_scripts.single_layer_edit_preserving_correction.alltoken import WEIGHT, model_guard, tensor_sha256
from project.run_scripts.single_layer_edit_preserving_correction.common import digest, sha, tensor_sha, write, save_tensor, Timer
from project.run_scripts.single_layer_edit_preserving_correction.binding import protected_sequences
from project.run_scripts.low_cost_write_donor_pilot.fitting import NativeSingletonFitter, select_projector
from project.run_scripts.baseline_mechanism_first.fixtures import restore_rng, capture_rng


class Runtime(Methods):
    def __init__(self, lock, output):
        check_lock(lock)
        import transformers, numpy, scipy
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from scripts.fixed_counterfact import load_prefix
        self.lock,self.output,self.timing=lock,Path(output),{}
        verify_large_asset_stats(lock)
        self.records=load_prefix(lock['dataset_root'],1000)
        if digest(self.records)!=lock['records_digest'] or [r['case_id'] for r in self.records]!=lock['sample_order']:
            raise ValueError('PREFIX1000_BEFORE_MODEL')
        for name,actual in [('torch',str(torch.__version__)),('transformers',transformers.__version__),
                            ('numpy',numpy.__version__),('scipy',scipy.__version__)]:
            if actual!=lock[name]:raise ValueError('PINNED_IMPORT_'+name)
        if str(Path(transformers.__file__).resolve())!=lock['transformers_import']:
            raise ValueError('TRANSFORMERS_IMPORT_PATH')
        if os.environ.get('SLURMD_NODENAME')!='server4' or not torch.cuda.is_available():
            raise ValueError('SERVER4_SLURM_GPU_REQUIRED')
        torch.set_num_threads(8);transformers.set_seed(20260916)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        sys.path.insert(0,lock['blue_root']);os.chdir(lock['blue_root'])
        self.module=importlib.import_module('AlphaEdit.AlphaEdit_main')
        HP=importlib.import_module('AlphaEdit.AlphaEdit_hparams').AlphaEditHyperParams
        self.hp=HP.from_json(lock['config4']);h=self.hp
        if not (h.blue and h.layers==[4] and h.L2==1 and h.v_weight_decay==.5 and h.clamp_norm_factor==.75
                and h.v_lr==.1 and h.v_num_grad_steps==25 and h.kl_factor==.0625 and h.v_loss_layer==31):
            raise ValueError('NATIVE_HPARAMS')
        with Timer(self.timing,'model_load'):
            self.model=AutoModelForCausalLM.from_pretrained(lock['snapshot'],local_files_only=True,
                torch_dtype=torch.float32,low_cpu_mem_usage=True,attn_implementation='eager').cuda().eval()
        self.tok=AutoTokenizer.from_pretrained(lock['snapshot'],local_files_only=True)
        self.tok.add_bos_token=False;self.tok.pad_token_id=self.tok.eos_token_id
        self.etok=AutoTokenizer.from_pretrained(lock['snapshot'],local_files_only=True)
        self.etok.pad_token_id=self.etok.eos_token_id
        if self.tok.padding_side!='right' or self.etok.padding_side!='right':raise ValueError('RIGHT_PADDING')
        for p in self.model.parameters():p.requires_grad_(False)
        self.W=dict(self.model.named_parameters())[WEIGHT]
        if self.W.shape!=(4096,14336) or any(p.dtype!=torch.float32 for p in self.model.parameters()):
            raise ValueError('FULL_MODEL_FP32')
        stack=torch.load(lock['projector'],weights_only=True,mmap=True,map_location='cpu')
        self.P,self.pmap=select_projector(stack,4);del stack
        self.M=torch.zeros_like(self.P);self.W0=self.W.detach().cpu().clone()
        cold=lock['cold_capsule']
        if sha(cold['path'])!=cold['sha256']:raise ValueError('COLD_CAPSULE_SHA')
        common=json.loads(Path(cold['path']).read_text())
        self.context,self.rng=copy.deepcopy(common['contexts']),common['rng']
        if [[self.tok(x)['input_ids'] for x in g] for g in self.context]!=common['context_tokens']:
            raise ValueError('CONTEXT_ACTUAL_TOKEN_IDS')
        if tensor_sha(self.W0)!=common['W0']['4'] or self.pmap!=common['projector_mapping']['4']:
            raise ValueError('W0_PROJECTOR_BINDING')
        self.module.CONTEXT_TEMPLATES_CACHE=copy.deepcopy(self.context);self.module.COV_CACHE={}
        restore_rng(self.rng)
        self.fitter=NativeSingletonFitter(self.module,expected_source_sha256=lock['editor_sha256'],contexts=self.context)
        from project.run_scripts.baseline_mechanism_first.evaluation import bind_evaluation_sources
        self.observer_binding=bind_evaluation_sources(lock['historical_evaluator_root'],helper_root=lock['helper_scripts_root'])
        self.base_guard,self.oracles=model_guard(self.model),[]
        self.identity=dict(W0=tensor_sha(self.W0),W0_header_bytes=tensor_sha256(self.W0),M0=tensor_sha(self.M),
            P4=self.pmap,contexts=digest(self.context),context_tokens=digest(common['context_tokens']),rng=digest(self.rng),
            records_digest=lock['records_digest'],physical_layer=4,torch=str(torch.__version__),
            transformers=transformers.__version__,attention='eager',matmul_tf32=False,cudnn_tf32=False,
            reference_microbatch=1,canonical_microbatch=16,data_id='EN-R512-G256-v1')
        write(self.output/'runtime-load.json',dict(identity=self.identity,timing=self.timing,
            GPU=torch.cuda.get_device_name(),model_bytes=sum(p.numel()*p.element_size() for p in self.model.parameters())))

    def guard(self):
        after=model_guard(self.model)
        strip=lambda g:(tuple(x for x in g[0] if x[1]!=WEIGHT),g[1])
        if strip(after)!=strip(self.base_guard):raise RuntimeError('NONSELECTED_STATE_MUTATION')
        if any(p.requires_grad or p.grad is not None for p in self.model.parameters()):raise RuntimeError('GRAD_STATE_MUTATION')
        if self.module.CONTEXT_TEMPLATES_CACHE!=self.context or self.module.COV_CACHE:
            raise RuntimeError('NATIVE_CONTEXT_MUTATION')
        if self.M.dtype!=torch.float32 or not torch.isfinite(self.M).all():raise RuntimeError('NATIVE_MEMORY_INVALID')

    def native(self, records, directory, *, fitter=None):
        """Fresh own-entry fitting; never reset a later branch to W0."""
        fitter=fitter or self.fitter;directory=Path(directory)
        entry=dict(W=tensor_sha(self.W),M=tensor_sha(self.M),rng=digest(capture_rng()))
        with Timer(self.timing,'native_fit'):
            result=fitter.fit(self.model,self.tok,self.hp,self.M,self.P,self.requests(records),layer=4,capture=True)
        self.sync_oracles();self.guard()
        if result['receipt']['history_append']!=0 or result['receipt']['compute_z']!=len(records):
            raise ValueError('NATIVE_CALLS_OR_INNER_HISTORY')
        source=save_tensor(directory/'native-capsule.pt',result)
        write(directory/'native-binding.json',dict(source=source,entry=entry,receipt=result['receipt'],
            case_ids=[r['case_id'] for r in records],native_fit_new_calls=1,native_target_new_calls=len(records)))
        return result

    def protected_oracle(self,records):
        packs,rows,_,meta=protected_sequences(self.tok,self.etok,self.requests(records),self.context)
        oracle=MeasuredCurrentOracle(self.model,packs);self.oracles.append(oracle)
        values=[];prefixes={};aliases=[]
        for alias in meta['key_aliases']:
            value=oracle.caches[alias['cache']].keys[0,alias['position']]
            candidates=prefixes.setdefault(alias['prefix_sha'],[])
            found=next((i for i in candidates if torch.equal(values[i],value)),None)
            if found is None:found=len(values);values.append(value);candidates.append(found)
            aliases.append(dict(**alias,actual_key_column=found))
        K=torch.stack(values,1).contiguous()
        meta.update(actual_key_aliases=aliases,actual_distinct_key_columns=len(values),K_shape=list(K.shape),
            K_sha256=tensor_sha(K),dedup_rule='same prefix AND identical FP32 key bytes')
        return oracle,rows,K,meta

    def reference(self):
        from project.run_scripts.en_execution_reuse.generated_oracle import GeneratedReferenceOracle
        store,ready=load_teacher(self,self.lock);self.generated_store=store
        oracle=GeneratedReferenceOracle(self.model,store)
        self.oracles.append(oracle)
        write(self.output/'generated-input-binding.json',dict(store=store.receipt,preparation=self.lock['generated_ready'],
            new_generated_documents=0,prior_payload_verification='REUSED_READY_SEAL',
            T0_current_execution_validation='REQUIRED',old_skip_inherited=False))
        return oracle
