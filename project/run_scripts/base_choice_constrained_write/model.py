"""Fresh same-host native adapter; shared EN/native source is read-only.

Reuses only full-weight cache/current invariant and the unmodified native fit.
No EN optimizer, old endpoint, old technical waiver or old teacher is used.
"""
import copy
import importlib
import json
import os
from pathlib import Path
import sys
import torch
from project.run_scripts.single_layer_edit_preserving_correction.runtime import Runtime as CacheRuntime
from project.run_scripts.single_layer_edit_preserving_correction.common import digest,tensor_sha,Timer,write
from project.run_scripts.single_layer_edit_preserving_correction.alltoken import WEIGHT,model_guard
from project.run_scripts.low_cost_write_donor_pilot.fitting import NativeSingletonFitter,select_projector
from project.run_scripts.baseline_mechanism_first.fixtures import restore_rng
from .config import require_scope
from .provenance import sha

class Runtime(CacheRuntime):
    def __init__(self,lock,output):
        require_scope(lock)
        import transformers
        from transformers import AutoTokenizer,AutoModelForCausalLM
        from scripts.fixed_counterfact import load_prefix
        self.lock,self.output,self.timing=lock,Path(output),{}
        self.records=load_prefix(lock['dataset_root'],100)
        if digest(self.records)!=lock['records_digest'] or [r['case_id'] for r in self.records]!=lock['sample_order']:
            raise ValueError('FIXED_B1_IDENTITY')
        if str(torch.__version__)!=lock['torch'] or transformers.__version__!=lock['transformers']:
            raise ValueError('LOCKED_LIBRARY_MISMATCH')
        if os.environ.get('SLURMD_NODENAME')!='server4' or not torch.cuda.is_available():
            raise ValueError('SLURM_SERVER4_GPU_REQUIRED')
        torch.set_num_threads(8);transformers.set_seed(lock['seed'])
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        sys.path.insert(0,lock['blue_root']);os.chdir(lock['blue_root'])
        self.module=importlib.import_module('AlphaEdit.AlphaEdit_main')
        HP=importlib.import_module('AlphaEdit.AlphaEdit_hparams').AlphaEditHyperParams
        self.hp=HP.from_json(lock['config4']);h=self.hp
        if not(h.blue and h.layers==[4] and h.L2==1 and h.v_weight_decay==.5 and h.clamp_norm_factor==.75 and
               h.v_lr==.1 and h.v_num_grad_steps==25 and h.kl_factor==.0625):raise ValueError('NATIVE_HPARAMS')
        with Timer(self.timing,'model_load'):
            self.model=AutoModelForCausalLM.from_pretrained(lock['snapshot'],local_files_only=True,
                torch_dtype=torch.float32,low_cpu_mem_usage=True,attn_implementation='eager').cuda().eval()
        self.tok=AutoTokenizer.from_pretrained(lock['snapshot'],local_files_only=True)
        self.tok.add_bos_token=False;self.tok.pad_token_id=self.tok.eos_token_id
        self.etok=AutoTokenizer.from_pretrained(lock['snapshot'],local_files_only=True)
        self.etok.pad_token_id=self.etok.eos_token_id
        if self.tok.padding_side!='right' or self.etok.padding_side!='right':raise ValueError('PADDING')
        for p in self.model.parameters():p.requires_grad_(False)
        self.W=dict(self.model.named_parameters())[WEIGHT]
        if self.W.shape!=(4096,14336) or any(p.dtype!=torch.float32 for p in self.model.parameters()):raise ValueError('FP32_L4')
        with Timer(self.timing,'P_load'):
            stack=torch.load(lock['projector'],weights_only=True,mmap=True,map_location='cpu')
            self.P,self.pmap=select_projector(stack,4);del stack
        self.M=torch.zeros_like(self.P);self.W0=self.W.detach().cpu().clone()
        cold=lock['cold_capsule']
        if sha(cold['path'])!=cold['sha256']:raise ValueError('COLD_CAPSULE_SHA')
        common=json.loads(Path(cold['path']).read_text());self.context=copy.deepcopy(common['contexts']);self.rng=common['rng']
        if [[self.tok(x)['input_ids'] for x in g] for g in self.context]!=common['context_tokens']:raise ValueError('CONTEXT_TOKENS')
        if tensor_sha(self.W0)!=common['W0']['4'] or self.pmap!=common['projector_mapping']['4']:raise ValueError('W0_P_IDENTITY')
        self.module.CONTEXT_TEMPLATES_CACHE=copy.deepcopy(self.context);self.module.COV_CACHE={};restore_rng(self.rng)
        self.fitter=NativeSingletonFitter(self.module,expected_source_sha256=lock['editor_sha256'],contexts=self.context)
        from project.run_scripts.baseline_mechanism_first.evaluation import bind_evaluation_sources
        self.observer_binding=bind_evaluation_sources(lock['historical_evaluator_root'],helper_root=lock['helper_scripts_root'])
        self.base_guard=model_guard(self.model);self.oracles=[];self.native_calls=0
        self.identity=dict(W0=tensor_sha(self.W0),M0=tensor_sha(self.M),P4=self.pmap,contexts=digest(self.context),
            context_tokens=digest(common['context_tokens']),rng=digest(self.rng),records_digest=lock['records_digest'],
            torch=str(torch.__version__),transformers=str(transformers.__version__),physical_layer=4,
            choice_microbatch=1,canonical_microbatch=16,base_capsule='NEW_W0_RAW_ARGMAX_NOT_OLD_KL_TEACHER')
        write(self.output/'runtime-load.json',dict(identity=self.identity,timing=self.timing,
            model_gpu_name=torch.cuda.get_device_name(),model_bytes=sum(p.numel()*p.element_size() for p in self.model.parameters())))

    def native(self,records,directory,reuse=False):
        if reuse or self.native_calls or len(records)!=100:raise ValueError('EXACTLY_ONE_FRESH_NATIVE_B100')
        self.native_calls+=1
        result=super().native(records,directory,reuse=False)
        self.sync_oracles()
        return result
