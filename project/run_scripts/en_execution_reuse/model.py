"""New generated-reference B1 binding. Shared native and old EN source unchanged."""
import copy
import importlib
import json
import os
from pathlib import Path
import sys
import torch
from .config import Scope, validate_runtime_policy
from .preparation import sha, create_json
from project.run_scripts.single_layer_edit_preserving_correction.runtime import Runtime as LegacyRuntime
from project.run_scripts.single_layer_edit_preserving_correction.common import digest, tensor_sha, Timer
from project.run_scripts.single_layer_edit_preserving_correction.alltoken import WEIGHT, model_guard, tensor_sha256
from project.run_scripts.low_cost_write_donor_pilot.fitting import NativeSingletonFitter, select_projector
from project.run_scripts.baseline_mechanism_first.fixtures import restore_rng
from project.run_scripts.single_layer_edit_preserving_correction.binding import protected_sequences
from .current_oracle import MeasuredCurrentOracle


def require_lock(lock, *, historical_preparation=False):
    if lock.get('lock_identity') != digest({k:v for k,v in lock.items() if k!='lock_identity'}):
        raise ValueError('EXECUTION_LOCK_DIGEST')
    value = dict(lock['scope'])
    value['arms'] = tuple(value['arms'])
    Scope(**value).require_batch(0)
    if historical_preparation and lock['stage']!='GENERATED_REFERENCE_PREPARATION':
        raise ValueError('HISTORICAL_POLICY_ONLY_FOR_READ_ONLY_PREPARATION')
    validate_runtime_policy(lock['runtime_policy'],historical_preparation=historical_preparation)
    if lock['stage'] not in ('GENERATED_REFERENCE_PREPARATION', 'MATCHED_B1'):
        raise ValueError('UNAUTHORIZED_STAGE')
    expected_resources=dict(GPU=1,CPU=8,mem_MiB=60416,wall_hours=24,node='server4',export='NONE',requeue=0,GPUhour_hardcap=None)
    if lock['resources'] != expected_resources or lock['sequential_authorized'] is not False or lock['auto_continue'] is not False:
        raise ValueError('LOCK_RESOURCE_OR_CONTINUATION')
    root=Path('/data/janghj/ODE-edit/local/en-execution-reuse/20260919-v1')
    phase='PREP' if lock['stage']=='GENERATED_REFERENCE_PREPARATION' else 'B1'
    output=Path(lock['output']).resolve()
    if not output.is_relative_to(root/phase) or output.name!='output':
        raise ValueError('OUTPUT_SCOPE_CONFINEMENT')


def verify_large_asset_stats(lock):
    """Rebind prior verified immutable bytes at admission AND load boundaries."""
    model_paths = set()
    for item in lock['prior_large_asset_binding']:
        prior = item['prior']
        p = Path(prior['path'])
        st = p.stat()
        if (st.st_size != prior['bytes'] or
            [st.st_dev, st.st_ino, st.st_mtime_ns] != item['current_stat']):
            raise ValueError('IMMUTABLE_LARGE_ASSET_STAT_CHANGED:'+str(p))
        if p.suffix == '.safetensors':
            model_paths.add(p.name)
    index = Path(lock['snapshot'])/'model.safetensors.index.json'
    expected = set(json.loads(index.read_text())['weight_map'].values())
    if expected != model_paths:
        raise ValueError('MODEL_SHARD_INVENTORY_INCOMPLETE')


class Runtime(LegacyRuntime):
    """Reuses native/current methods, not the old S64 TeacherStore constructor."""
    def __init__(self, lock, output):
        require_lock(lock)
        import transformers
        import numpy, scipy
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from scripts.fixed_counterfact import load_prefix
        self.lock, self.output, self.timing = lock, Path(output), {}
        verify_large_asset_stats(lock)
        self.records = load_prefix(lock['dataset_root'], 100)
        if digest(self.records) != lock['records_digest'] or [r['case_id'] for r in self.records] != lock['sample_order']:
            raise ValueError('EXACT_FIRST100_BEFORE_MODEL')
        if str(torch.__version__) != lock['torch'] or transformers.__version__ != lock['transformers']:
            raise ValueError('PINNED_IMPORT')
        if numpy.__version__!=lock['numpy'] or scipy.__version__!=lock['scipy']:
            raise ValueError('PINNED_GEOMETRY_BACKEND_VERSION')
        if str(Path(transformers.__file__).resolve()) != lock['transformers_import']:
            raise ValueError('TRANSFORMERS_IMPORT_PATH')
        if os.environ.get('SLURMD_NODENAME') != 'server4' or not torch.cuda.is_available():
            raise ValueError('SERVER4_SLURM_GPU_ONLY')
        torch.set_num_threads(8)
        transformers.set_seed(20260916)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        sys.path.insert(0, lock['blue_root'])
        os.chdir(lock['blue_root'])
        self.module = importlib.import_module('AlphaEdit.AlphaEdit_main')
        HP = importlib.import_module('AlphaEdit.AlphaEdit_hparams').AlphaEditHyperParams
        self.hp = HP.from_json(lock['config4'])
        h = self.hp
        if not (h.blue and h.layers == [4] and h.L2 == 1 and h.v_weight_decay == .5 and
                h.clamp_norm_factor == .75 and h.v_lr == .1 and h.v_num_grad_steps == 25 and h.kl_factor == .0625):
            raise ValueError('NATIVE_HPARAMS_DRIFT')
        with Timer(self.timing, 'model_load'):
            self.model = AutoModelForCausalLM.from_pretrained(lock['snapshot'], local_files_only=True,
                torch_dtype=torch.float32, low_cpu_mem_usage=True, attn_implementation='eager').cuda().eval()
        self.tok = AutoTokenizer.from_pretrained(lock['snapshot'], local_files_only=True)
        self.tok.add_bos_token = False
        self.tok.pad_token_id = self.tok.eos_token_id
        self.etok = AutoTokenizer.from_pretrained(lock['snapshot'], local_files_only=True)
        self.etok.pad_token_id = self.etok.eos_token_id
        if self.tok.padding_side != 'right' or self.etok.padding_side != 'right':
            raise ValueError('RIGHT_PADDING_REQUIRED')
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.W = dict(self.model.named_parameters())[WEIGHT]
        if self.W.shape != (4096,14336) or any(p.dtype != torch.float32 for p in self.model.parameters()):
            raise ValueError('FULL_FP32_L4_SCHEMA')
        with Timer(self.timing, 'projector_load'):
            stack = torch.load(lock['projector'], weights_only=True, mmap=True, map_location='cpu')
            self.P, self.pmap = select_projector(stack, 4)
            del stack
        self.M = torch.zeros_like(self.P)
        self.W0 = self.W.detach().cpu().clone()
        cold = lock['cold_capsule']
        if sha(cold['path']) != cold['sha256']:
            raise ValueError('COLD_CAPSULE_SHA')
        common = json.loads(Path(cold['path']).read_text())
        self.context, self.rng = copy.deepcopy(common['contexts']), common['rng']
        if [[self.tok(x)['input_ids'] for x in group] for group in self.context] != common['context_tokens']:
            raise ValueError('ACTUAL_CONTEXT_TOKEN_MISMATCH')
        if tensor_sha(self.W0) != common['W0']['4'] or self.pmap != common['projector_mapping']['4']:
            raise ValueError('W0_P_MAPPING')
        self.module.CONTEXT_TEMPLATES_CACHE = copy.deepcopy(self.context)
        self.module.COV_CACHE = {}
        restore_rng(self.rng)
        self.fitter = NativeSingletonFitter(self.module, expected_source_sha256=lock['editor_sha256'], contexts=self.context)
        from project.run_scripts.baseline_mechanism_first.evaluation import bind_evaluation_sources
        self.observer_binding = bind_evaluation_sources(lock['historical_evaluator_root'], helper_root=lock['helper_scripts_root'])
        self.base_guard, self.oracles = model_guard(self.model), []
        self.identity = dict(W0=tensor_sha(self.W0), W0_header_bytes=tensor_sha256(self.W0), M0=tensor_sha(self.M),
            P4=self.pmap, contexts=digest(self.context), context_tokens=digest(common['context_tokens']),
            rng=digest(self.rng), records_digest=lock['records_digest'], physical_layer=4,
            torch=str(torch.__version__), transformers=transformers.__version__,
            attention='eager', matmul_tf32=False, cudnn_tf32=False, reference_microbatch=1,
            canonical_microbatch=16, data_id='EN-R512-G256-v1')
        create_json(self.output/'runtime-load.json', dict(identity=self.identity, timing=self.timing,
            GPU=torch.cuda.get_device_name(), model_bytes=sum(p.numel()*p.element_size() for p in self.model.parameters())))

    def native(self, records, directory, reuse=True):
        if self.lock['stage'] != 'MATCHED_B1':
            raise ValueError('PREPARATION_CANNOT_FIT_OR_EDIT')
        if not reuse:
            raise ValueError('RETAINED_MATCHED_NATIVE_REUSE_LOCK_REQUIRES_REUSE')
        result = super().native(records, directory, reuse=True)
        self.sync_oracles()
        return result

    def protected_oracle(self,records):
        packs,rows,unique,meta=protected_sequences(self.tok,self.etok,self.requests(records),self.context)
        oracle=MeasuredCurrentOracle(self.model,packs)
        self.oracles.append(oracle)
        values=[];prefixes={};actual_alias=[]
        for alias in meta['key_aliases']:
            ci,pos=alias['cache'],alias['position'];value=oracle.caches[ci].keys[0,pos]
            candidates=prefixes.setdefault(alias['prefix_sha'],[])
            found=next((i for i in candidates if torch.equal(values[i],value)),None)
            if found is None:
                found=len(values);values.append(value);candidates.append(found)
            actual_alias.append(dict(**alias,actual_key_column=found))
        K=torch.stack(values,1).contiguous()
        meta.update(actual_key_aliases=actual_alias,actual_distinct_key_columns=len(values),
            dedup_rule='same exact token prefix AND identical captured FP32 key bytes only',
            K_shape=list(K.shape),K_sha256=tensor_sha(K))
        return oracle,rows,K,meta
