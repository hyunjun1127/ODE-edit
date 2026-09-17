"""Cold7 native L4 closure, but no operational P8/M8 or target-z8 path."""
import copy
import importlib
import os
import sys
import torch
from .common import digest, Timer
from project.run_scripts.baseline_mechanism_first.fixtures import capture_rng, restore_rng
from project.run_scripts.low_cost_write_donor_pilot.fitting import NativeSingletonFitter, select_projector, tensor_sha
from project.run_scripts.bg_tw_reference.ep_tw.model_adapter import TeacherStore, capture_native_fit

class Runtime:
    def __init__(self, lock, common):
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from scripts.fixed_counterfact import load_prefix
        from project.run_scripts.baseline_mechanism_first.evaluation import bind_evaluation_sources
        from .response import RepairResponse
        self.lock, self.timing = lock, {}
        self.records = load_prefix(lock['dataset_root'], 1000)
        assert digest(self.records) == lock['records_digest']
        assert torch.__version__ == lock['torch'] and transformers.__version__ == lock['transformers']
        assert os.environ.get('SLURMD_NODENAME') == 'server4'
        torch.set_num_threads(8); transformers.set_seed(lock['seed'])
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        sys.path.insert(0, lock['blue_root']); os.chdir(lock['blue_root'])
        self.module = importlib.import_module('AlphaEdit.AlphaEdit_main')
        HP = importlib.import_module('AlphaEdit.AlphaEdit_hparams').AlphaEditHyperParams
        self.hp = HP.from_json(lock['config4'])
        assert self.hp.blue and self.hp.layers == [4] and self.hp.L2 == 1
        assert (self.hp.v_weight_decay,self.hp.clamp_norm_factor,self.hp.v_lr,self.hp.v_num_grad_steps)==(.5,.75,.1,25)
        with Timer(self.timing, 'model_load'):
            self.model = AutoModelForCausalLM.from_pretrained(lock['snapshot'], local_files_only=True,
                torch_dtype=torch.float32, low_cpu_mem_usage=True, attn_implementation='eager').cuda().eval()
        self.tok = AutoTokenizer.from_pretrained(lock['snapshot'], local_files_only=True)
        self.tok.add_bos_token = False; self.tok.pad_token_id = self.tok.eos_token_id
        self.etok = AutoTokenizer.from_pretrained(lock['snapshot'], local_files_only=True)
        self.etok.pad_token_id = self.etok.eos_token_id
        assert self.tok.padding_side == self.etok.padding_side == 'right'
        self.params = dict(self.model.named_parameters())
        assert all(p.dtype == torch.float32 for p in self.params.values())
        for p in self.params.values(): p.requires_grad_(False)
        self.W = {l:self.params[f'model.layers.{l}.mlp.down_proj.weight'] for l in (4,8)}
        with Timer(self.timing,'P4_load'):
            stack = torch.load(lock['projector'],map_location='cpu',weights_only=True,mmap=True)
            self.P4,self.pmap = select_projector(stack,4); del stack
        self.M4 = torch.zeros_like(self.P4)
        assert self.pmap == common['projector_mapping']['4']
        self.context = copy.deepcopy(common['contexts'])
        self.module.CONTEXT_TEMPLATES_CACHE = copy.deepcopy(self.context); self.module.COV_CACHE = {}
        self.context_tokens = [[self.tok(x)['input_ids'] for x in group] for group in self.context]
        assert self.context_tokens == common['context_tokens']
        self.fitter = NativeSingletonFitter(self.module, expected_source_sha256=lock['editor_sha256'], contexts=self.context)
        teacher_ref = common['teacher_manifest']
        self.teacher = TeacherStore(lock['reference_root'],teacher_ref['path'],
            expected_manifest_sha=teacher_ref['sha256'],verify_payload_hashes=False)
        self.response = RepairResponse(self.model,self.etok,self.teacher)
        self.binding = bind_evaluation_sources(lock['historical_evaluator_root'],helper_root=lock['helper_scripts_root'])
        self.nonselected = {k:(p,p.data_ptr(),p._version) for k,p in self.params.items()
            if k not in ('model.layers.4.mlp.down_proj.weight','model.layers.8.mlp.down_proj.weight')}
        self.hooks = {k:(tuple(m._forward_hooks),tuple(m._forward_pre_hooks),tuple(m._backward_hooks)) for k,m in self.model.named_modules()}
        self.buffers = {k:(b,b.data_ptr(),b._version) for k,b in self.model.named_buffers()}
        assert {str(l):tensor_sha(w) for l,w in self.W.items()} == common['W0']
        self.common = common; restore_rng(common['rng']); self.guard()

    def guard(self):
        assert not self.model.training and not self.module.COV_CACHE
        assert all(not p.requires_grad and p.grad is None for p in self.params.values())
        assert self.module.CONTEXT_TEMPLATES_CACHE == self.context
        for k,(p,ptr,v) in self.nonselected.items():
            assert self.params[k] is p and p.data_ptr()==ptr and p._version==v, ('NONSELECTED_MUTATION',k)
        for k,(b,ptr,v) in self.buffers.items():
            assert b.data_ptr()==ptr and b._version==v, ('BUFFER_MUTATION',k)
        assert all(self.hooks[k]==(tuple(m._forward_hooks),tuple(m._forward_pre_hooks),tuple(m._backward_hooks))
            for k,m in self.model.named_modules())

    def state(self):
        return dict(W={str(l):tensor_sha(w) for l,w in self.W.items()}, M4=tensor_sha(self.M4),
            P4=tensor_sha(self.P4), contexts=digest(self.context),rng=digest(capture_rng()))

    def snapshot(self):
        return dict(W={l:w.detach().cpu().clone() for l,w in self.W.items()},M4=self.M4.clone(),
            contexts=copy.deepcopy(self.context),rng=capture_rng())

    def restore(self, snap):
        with torch.no_grad():
            for l,w in self.W.items(): w.copy_(snap['W'][l].to(w.device))
            self.M4.copy_(snap['M4'])
        self.module.CONTEXT_TEMPLATES_CACHE=copy.deepcopy(snap['contexts'])
        restore_rng(snap['rng']); self.guard()

    def apply8(self, weight, rng):
        assert weight.dtype == torch.float32 and weight.shape == self.W[8].shape
        assert torch.isfinite(weight).all(), 'NONFINITE_MATERIALIZED_WEIGHT'
        with torch.no_grad(): self.W[8].copy_(weight.to(self.W[8].device))
        restore_rng(rng); self.guard()

    def apply_weights(self, weights, rng):
        with torch.no_grad():
            for layer in (4,8):
                w=weights[layer]
                assert w.shape==self.W[layer].shape and w.dtype==torch.float32 and torch.isfinite(w).all()
                self.W[layer].copy_(w.to(self.W[layer].device))
        restore_rng(rng);self.guard()

    def observe(self, fn):
        before=self.state(); result=fn(); assert self.state()==before,'OBSERVER_STATE_MUTATION'
        self.guard(); return result

    def requests(self, records):
        return [dict(copy.deepcopy(r['requested_rewrite']),case_id=r['case_id']) for r in records]

    def fit(self, records):
        before=self.state()
        with Timer(self.timing,'native_target_solve_inclusive'):
            result=capture_native_fit(self.fitter,self.model,self.tok,self.hp,self.M4,self.P4,self.requests(records),layer=4)
        assert self.state()['W']['8']==before['W']['8'] and self.state()['M4']==before['M4']
        self.guard(); result['receipt']['input_state']=before
        return result

    def finalize(self, records):
        before=self.state()
        history=self.fitter.finalize(self.model,self.tok,self.requests(records),[(4,self.hp,self.M4,self.P4)])
        assert len(history)==1 and history[0]['history_append']==1
        assert self.state()['W']==before['W']; self.guard(); return history
