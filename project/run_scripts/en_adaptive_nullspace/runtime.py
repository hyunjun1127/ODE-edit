"""SH3 dedicated offline FP32 Llama loader with an explicit L4 write boundary."""
from copy import deepcopy
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
import torch
from scripts.fixed_counterfact import load_prefix
from project.run_scripts.baseline_mechanism_first.fixtures import restore_rng, capture_rng
from project.run_scripts.low_cost_write_donor_pilot.fitting import select_projector
from project.run_scripts.single_layer_edit_preserving_correction.alltoken import WEIGHT, model_guard
from project.run_scripts.single_layer_edit_preserving_correction.common import tensor_sha, digest
from .native import NativeRunner, NATIVE_SHA256, requests_from_records

READY_MANIFEST = Path('/data/janghj/ODE-edit/local/runtime/server3-experiment-ready-v1/manifest.json')
REVISION = '8afb486c1db24fe5011ec46dfbe5b5dccdb575c2'
COLD_SHA256 = '2d5d5c45bbbdf36ed859451242d7d84874d08cda4008d585945e565d9b3b1cb7'
DATA_SHA256 = '3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1'
ORDER_ROOT = '5b01356928ad2e2e46effad5a2c5621c87746b610992b7b92bf3ace16b6f5729'


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _nonselected(model):
    tensors, modules = model_guard(model)
    return tuple(x for x in tensors if x[1] != WEIGHT), modules


def verify_ready_inputs(manifest):
    """Reuse the completed large-file seals only when stable identity matches.

    Small native/config/evaluator closure is hashed once at load admission;
    no per-candidate whole-model or source hashing is performed.
    """
    if manifest['dataset']['dataset_sha256'] != DATA_SHA256 or manifest['dataset']['ordered_root'] != ORDER_ROOT:
        raise ValueError('FIXED10K_MANIFEST_IDENTITY')
    llama = manifest['models']['llama3-8b-inst']
    if llama['revision'] != REVISION or Path(llama['snapshot']).name != REVISION:
        raise ValueError('LLAMA_REVISION_IDENTITY')
    if not llama.get('experiment_ready') or manifest['native_source_sha256'] != NATIVE_SHA256:
        raise ValueError('COMPLETED_READY_NATIVE_REQUIRED')
    if _sha(manifest['context_capsule']) != COLD_SHA256:
        raise ValueError('EXACT_COLD_CAPSULE_REQUIRED')
    inventory = json.loads(Path(manifest['asset_inventory']).read_text())
    large = []
    for item in inventory:
        if item['model'] != 'llama3-8b-inst':
            continue
        p = Path(item['consumed_path'])
        st = p.stat()
        actual = [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns]
        if actual != item['stable_stat'] or st.st_size != item['size']:
            raise ValueError('READY_ASSET_STABLE_IDENTITY_CHANGED:' + str(p))
        if item['observed_sha256'] != item['sha256']:
            raise ValueError('READY_ASSET_UNVERIFIED_SHA')
        large.append(dict(path=str(p), size=st.st_size, sha256=item['sha256'],
                          stable_stat=actual, verification='prior_SHA_plus_unchanged_stable_identity'))
    index = json.loads((Path(llama['snapshot']) / 'model.safetensors.index.json').read_text())
    expected = {(Path(llama['snapshot']) / name).resolve() for name in set(index['weight_map'].values())}
    observed = {Path(i['consumed_path']).resolve() for i in inventory
                if i['model'] == 'llama3-8b-inst' and i['kind'] == 'weight_shard'}
    if expected != observed:
        raise ValueError('COMPLETE_MODEL_SHARD_CLOSURE_REQUIRED')
    source = []
    for item in json.loads(Path(manifest['source_allowlist']).read_text()):
        p = Path(item['destination'])
        if p.stat().st_size != item['size'] or _sha(p) != item['sha256']:
            raise ValueError('READY_SOURCE_CLOSURE_CHANGED:' + str(p))
        source.append(dict(path=str(p), bytes=item['size'], sha256=item['sha256']))
    return dict(large_assets=large, source_members=source,
                source_manifest_sha256=digest(source), large_asset_count=len(large))


class Runtime:
    def __init__(self, output):
        started = time.monotonic()
        self.output = Path(output).absolute()
        self.output.mkdir(parents=True, exist_ok=True)
        self.ready_manifest = json.loads(READY_MANIFEST.read_text())
        manifest = self.ready_manifest
        if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURMD_NODENAME') != 'ubuntu':
            raise ValueError('UBUNTU_SLURM_ALLOCATION_REQUIRED')
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError('EXACTLY_ONE_VISIBLE_ALLOCATED_GPU_REQUIRED')
        if os.environ.get('SLURM_GPUS_ON_NODE', '1') != '1':
            raise ValueError('PROJECT_GPU_CAP_ONE')
        if str(Path(sys.executable).absolute()) != manifest['python']:
            raise ValueError('DEDICATED_READY_PYTHON_REQUIRED')
        sys.dont_write_bytecode = True
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_DATASETS_OFFLINE'] = '1'
        os.environ['SAVE_CHECKPOINTS'] = 'false'
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        import transformers
        import numpy
        import scipy
        if (sys.version_info[:3], str(torch.__version__), transformers.__version__,
                numpy.__version__, scipy.__version__) != ((3, 12, 3), '2.9.1+cu128', '4.44.2', '2.2.6', '1.15.3'):
            raise ValueError('PINNED_READY_PACKAGE_VERSIONS')
        binding = verify_ready_inputs(manifest)
        self.records = load_prefix(manifest['dataset_root'], 300)
        self.requests = requests_from_records(self.records)
        native_root = Path(manifest['native_root']).resolve()
        sys.path.insert(0, str(native_root))
        os.chdir(native_root)
        self.module = importlib.import_module('AlphaEdit.AlphaEdit_main')
        if not Path(self.module.__file__).resolve().is_relative_to(native_root):
            raise ValueError('NATIVE_IMPORT_ESCAPED_READY_CLOSURE')
        HP = importlib.import_module('AlphaEdit.AlphaEdit_hparams').AlphaEditHyperParams
        self.hp = HP.from_json(manifest['native_config'])
        expected_hp = dict(blue=True, layers=[4], L2=1, v_weight_decay=.5,
            clamp_norm_factor=.75, v_lr=.1, v_num_grad_steps=25, kl_factor=.0625,
            v_loss_layer=31, fact_token='subject_last', rewrite_module_tmp='model.layers.{}.mlp.down_proj')
        if any(getattr(self.hp, k) != v for k, v in expected_hp.items()):
            raise ValueError('EXACT_NATIVE_CONFIG_REQUIRED')
        llama = manifest['models']['llama3-8b-inst']
        load_started = time.monotonic()
        self.model = transformers.AutoModelForCausalLM.from_pretrained(llama['snapshot'],
            local_files_only=True, torch_dtype=torch.float32, low_cpu_mem_usage=True,
            attn_implementation='eager').to('cuda').eval()
        self.load_seconds = time.monotonic() - load_started
        self.tok = transformers.AutoTokenizer.from_pretrained(llama['snapshot'], local_files_only=True)
        self.tok.add_bos_token = False
        self.tok.pad_token_id = self.tok.eos_token_id
        self.etok = transformers.AutoTokenizer.from_pretrained(llama['snapshot'], local_files_only=True)
        self.etok.pad_token_id = self.etok.eos_token_id
        if self.tok.padding_side != 'right' or self.etok.padding_side != 'right':
            raise ValueError('PINNED_RIGHT_PADDING_REQUIRED')
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.W = dict(self.model.named_parameters())[WEIGHT]
        if self.W.shape != (4096, 14336) or any(p.dtype != torch.float32 for p in self.model.parameters()):
            raise ValueError('FULL_FP32_LLAMA_L4_REQUIRED')
        self.W0 = self.W.detach().cpu().clone()
        stack = torch.load(llama['assets']['projector']['path'], weights_only=True,
                           mmap=True, map_location='cpu')
        if tuple(stack.shape) != (5, 14336, 14336) or stack.dtype != torch.float32:
            raise ValueError('RAW_NATIVE_PROJECTOR_STACK_SCHEMA')
        self.P, self.pmap = select_projector(stack, 4)
        del stack
        cold = json.loads(Path(manifest['context_capsule']).read_text())
        self.context, self.rng = deepcopy(cold['contexts']), deepcopy(cold['rng'])
        if [[self.tok(text)['input_ids'] for text in group] for group in self.context] != cold['context_tokens']:
            raise ValueError('EXACT_CONTEXT_TOKEN_IDS_REQUIRED')
        if tensor_sha(self.W0) != cold['W0']['4'] or self.pmap != cold['projector_mapping']['4']:
            raise ValueError('COLD_W0_OR_PHYSICAL_PROJECTOR_IDENTITY')
        self.module.CONTEXT_TEMPLATES_CACHE = deepcopy(self.context)
        self.module.COV_CACHE = {}
        restore_rng(self.rng)
        if digest(capture_rng()) != digest(self.rng):
            raise ValueError('EXACT_COLD_RNG_RESTORE')
        self.native_runner = NativeRunner(self.model, self.tok, self.hp, self.module,
                                         self.context, self.P, batch_size=16)
        self.oracles, self.objective = [], None
        self.base_guard = _nonselected(self.model)
        self.identity = dict(W0=tensor_sha(self.W0), raw_P4=self.pmap,
            context=digest(self.context), context_tokens=digest(cold['context_tokens']),
            rng=digest(self.rng), records=digest(self.records),
            case_ids=[r['case_id'] for r in self.records], batches=[[r['case_id'] for r in self.records[k:k+100]] for k in (0,100,200)],
            dataset_sha256=DATA_SHA256, ordered_root=ORDER_ROOT, model_revision=REVISION,
            dtype='float32', attention='eager', matmul_tf32=False, cudnn_tf32=False,
            torch=str(torch.__version__), transformers=transformers.__version__,
            numpy=numpy.__version__, scipy=scipy.__version__, python=sys.version.split()[0],
            z_request_chunk=16, visible_GPUs=torch.cuda.device_count(),
            slurm_job_id=os.environ['SLURM_JOB_ID'], node=os.environ['SLURMD_NODENAME'],
            GPU=torch.cuda.get_device_name(), source_manifest_sha256=binding['source_manifest_sha256'],
            readiness_manifest_sha256=_sha(READY_MANIFEST), save_checkpoints=False,
            exact_crash_resume='NOT_AVAILABLE')
        self.guard()
        receipt = dict(identity=self.identity, model_load_seconds=self.load_seconds,
            total_load_seconds=time.monotonic()-started, asset_binding=binding,
            model_bytes=sum(p.numel()*p.element_size() for p in self.model.parameters()))
        with (self.output / 'runtime-load.json').open('x') as stream:
            json.dump(receipt, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write('\n')

    def guard(self):
        if _nonselected(self.model) != self.base_guard:
            raise RuntimeError('NONSELECTED_PARAMETER_BUFFER_HOOK_MODE_MUTATION')
        if any(p.requires_grad or p.grad is not None for p in self.model.parameters()):
            raise RuntimeError('FROZEN_MODEL_WITHOUT_PARAMETER_GRAD_REQUIRED')
        if self.module.CONTEXT_TEMPLATES_CACHE != self.context or self.module.COV_CACHE:
            raise RuntimeError('NATIVE_CONTEXT_OR_COV_CACHE_MUTATION')
        if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
            raise RuntimeError('TF32_POLICY_MUTATION')

    def sync_oracles(self):
        """Explicit acknowledgment after a caller-owned native selected write."""
        self.guard()
        expected = self.W.detach()
        if self.objective is not None:
            self.objective.rebind(expected)
        seen = set()
        for oracle in self.oracles:
            if id(oracle) not in seen:
                oracle.acknowledge_selected_write(expected)
                seen.add(id(oracle))

    def install(self, weight):
        self.guard()
        if weight.dtype != torch.float32 or weight.shape != self.W.shape or not torch.isfinite(weight).all():
            raise ValueError('FINITE_ABSOLUTE_FP32_L4_ENDPOINT_REQUIRED')
        with torch.no_grad():
            self.W.copy_(weight.detach().to(self.W.device))
        self.sync_oracles()
