"""Actual all-token v2 backend, immutable structurally shared RAM snapshots.

No repair controller, target bank, full-model copy or disk W/M checkpoint.
Only the five permitted weight tensors can change; candidate histories cannot.
"""
import copy
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time
import weakref

import torch
from .common import LAYERS, COLD, digest, serial, save, tensor_save, Timer
from .controller import Scores
from .native import GeneralizedNativeFitter, select_projector, tensor_sha
from .metrics import OnlineMetrics, TeacherStore
from project.run_scripts.baseline_mechanism_first.fixtures import capture_rng, restore_rng


def fp32_gate(entry, native, gate):
    if not math.isfinite(gate) or not 0 <= gate <= 1:
        raise ValueError('GATE_OUT_OF_BOX')
    if entry.dtype != torch.float32 or native.dtype != torch.float32 or entry.shape != native.shape:
        raise ValueError('GATE_SCHEMA')
    if gate == 0: return entry
    if gate == 1: return native
    value = entry + gate * (native-entry)
    if not torch.isfinite(value).all(): raise ValueError('NONFINITE_GATE')
    return value


class Runtime:
    def __init__(self, lock, common=None):
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from scripts.fixed_counterfact import load_prefix
        from project.run_scripts.baseline_mechanism_first.evaluation import bind_evaluation_sources
        self.lock, self.timing = lock, {}
        self.records = load_prefix(lock['dataset_root'], 1000)
        assert digest(self.records) == lock['records_digest']
        assert torch.__version__ == lock['torch'] and transformers.__version__ == lock['transformers']
        assert os.environ.get('SLURMD_NODENAME') == 'server4'
        torch.set_num_threads(8); transformers.set_seed(20260916)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        sys.path.insert(0, lock['blue_root']); os.chdir(lock['blue_root'])
        self.module = importlib.import_module('AlphaEdit.AlphaEdit_main')
        HP = importlib.import_module('AlphaEdit.AlphaEdit_hparams').AlphaEditHyperParams
        base = HP.from_json(lock['config4']); self.hp = {}
        for layer in LAYERS:
            hp = copy.deepcopy(base); hp.layers = [layer]; self.hp[layer] = hp
        assert base.blue and base.layers == [4] and base.L2 == 1
        assert (base.v_weight_decay,base.clamp_norm_factor,base.v_lr,base.v_num_grad_steps,base.kl_factor)==(.5,.75,.1,25,.0625)
        with Timer(self.timing, 'model_load'):
            self.model = AutoModelForCausalLM.from_pretrained(lock['snapshot'],local_files_only=True,
                torch_dtype=torch.float32,low_cpu_mem_usage=True,attn_implementation='eager').cuda().eval()
        self.tok = AutoTokenizer.from_pretrained(lock['snapshot'],local_files_only=True)
        self.tok.add_bos_token=False;self.tok.pad_token_id=self.tok.eos_token_id
        self.etok=AutoTokenizer.from_pretrained(lock['snapshot'],local_files_only=True)
        self.etok.pad_token_id=self.etok.eos_token_id
        assert self.tok.padding_side==self.etok.padding_side=='right'
        self.params=dict(self.model.named_parameters())
        assert all(p.dtype==torch.float32 for p in self.params.values())
        for p in self.params.values():p.requires_grad_(False)
        self.W={l:self.params[f'model.layers.{l}.mlp.down_proj.weight'] for l in LAYERS}
        with Timer(self.timing,'projector_load'):
            stack=torch.load(lock['projector'],map_location='cpu',weights_only=True,mmap=True)
            selected={l:select_projector(stack,l) for l in LAYERS}
            self.P={l:selected[l][0] for l in LAYERS};self.pmap={str(l):selected[l][1] for l in LAYERS}
            del stack,selected
        self.M={l:torch.zeros_like(self.P[l]) for l in LAYERS}
        self.module.CONTEXT_TEMPLATES_CACHE=None;self.module.COV_CACHE={}
        if common is None:
            common=json.loads(Path(lock['cold_capsule']['path']).read_text())
        self.context=copy.deepcopy(common['contexts']);self.common_rng=common['rng']
        restore_rng(self.common_rng);self.module.CONTEXT_TEMPLATES_CACHE=copy.deepcopy(self.context)
        self.context_tokens=[[self.tok(x)['input_ids'] for x in group] for group in self.context]
        assert self.context_tokens==common['context_tokens']
        self.fitter=GeneralizedNativeFitter(self.module,expected_source_sha256=lock['editor_sha256'],contexts=self.context)
        teacher=common['teacher_manifest']
        self.teacher=TeacherStore(lock['reference_root'],teacher['path'],expected_manifest_sha=teacher['sha256'],verify_payload_hashes=False)
        self.metrics=OnlineMetrics(self.model,self.tok,self.etok,self.module,self.hp[4],self.context,self.teacher)
        self.observer=self.metrics._canonical
        self.binding=bind_evaluation_sources(lock['historical_evaluator_root'],helper_root=lock['helper_scripts_root'])
        selected_names={f'model.layers.{l}.mlp.down_proj.weight' for l in LAYERS}
        self.nonselected={k:(p,p.data_ptr(),p._version) for k,p in self.params.items() if k not in selected_names}
        self.hooks={k:(tuple(m._forward_hooks),tuple(m._forward_pre_hooks),tuple(m._backward_hooks)) for k,m in self.model.named_modules()}
        self.buffers={k:(b,b.data_ptr(),b._version) for k,b in self.model.named_buffers()}
        self._pool=weakref.WeakValueDictionary()
        self.cpuW={};self.whash={};self.mhash={l:tensor_sha(m) for l,m in self.M.items()}
        self.phash={l:tensor_sha(p) for l,p in self.P.items()}
        for l,w in self.W.items():
            self.cpuW[l],self.whash[l]=self.intern(w.detach().cpu().clone())
        self.w0=dict(self.cpuW)
        self._refresh_versions()
        for key,value in common['W0'].items():assert self.whash[int(key)]==value,('W0_MISMATCH',key)
        for key,value in common['projector_mapping'].items():assert self.pmap[key]==value,('P_MAPPING',key)
        self.guard()

    def intern(self,value):
        assert value.device.type=='cpu' and value.dtype==torch.float32
        key=tensor_sha(value)
        prior=self._pool.get(key)
        if prior is not None:
            assert prior.shape==value.shape and torch.equal(prior,value),'HASH_COLLISION'
            return prior,key
        self._pool[key]=value
        return value,key

    def _refresh_versions(self):
        self._wver={l:w._version for l,w in self.W.items()}
        self._wptr={l:w.data_ptr() for l,w in self.W.items()}
        self._mver={l:m._version for l,m in self.M.items()}
        self._pver={l:p._version for l,p in self.P.items()}
        self._cpuver={l:w._version for l,w in self.cpuW.items()}

    def guard(self, *, selected=True):
        assert not self.model.training and not self.module.COV_CACHE
        assert all(not p.requires_grad and p.grad is None for p in self.params.values())
        assert self.module.CONTEXT_TEMPLATES_CACHE==self.context
        for k,(p,ptr,v) in self.nonselected.items():
            assert self.params[k] is p and p.data_ptr()==ptr and p._version==v,('NONSELECTED_MUTATION',k)
        for k,(b,ptr,v) in self.buffers.items():assert b.data_ptr()==ptr and b._version==v,('BUFFER_MUTATION',k)
        assert all(self.hooks[k]==(tuple(m._forward_hooks),tuple(m._forward_pre_hooks),tuple(m._backward_hooks)) for k,m in self.model.named_modules())
        assert all(self.P[l]._version==self._pver[l] for l in LAYERS),'P_MUTATION'
        assert all(self.M[l]._version==self._mver[l] for l in LAYERS),'UNTRACKED_M_MUTATION'
        assert all(self.cpuW[l]._version==self._cpuver[l] for l in LAYERS),'CPU_SNAPSHOT_MUTATION'
        if selected:assert all(self.W[l]._version==self._wver[l] and self.W[l].data_ptr()==self._wptr[l] for l in LAYERS),'UNTRACKED_SELECTED_MUTATION'

    def state(self):
        self.guard()
        return dict(W={str(l):self.whash[l] for l in LAYERS},M={str(l):self.mhash[l] for l in LAYERS},
            P={str(l):self.phash[l] for l in LAYERS},contexts=digest(self.context),rng=digest(capture_rng()))

    def snapshot(self):
        state=self.state()
        return dict(W=dict(self.cpuW),M=dict(self.M),contexts=copy.deepcopy(self.context),rng=capture_rng(),
            whash=dict(self.whash),mhash=dict(self.mhash),state=state,
            wversions={l:w._version for l,w in self.cpuW.items()},mversions={l:m._version for l,m in self.M.items()},
            wschema={l:self.tensor_schema(w) for l,w in self.cpuW.items()},mschema={l:self.tensor_schema(m) for l,m in self.M.items()})

    @staticmethod
    def tensor_schema(t):return (id(t),t.data_ptr(),tuple(t.shape),str(t.dtype),str(t.device))

    @staticmethod
    def validate_snapshot(s):
        assert all(s['W'][l]._version==s['wversions'][l] for l in LAYERS),'IMMUTABLE_SNAPSHOT_W_CHANGED'
        assert all(s['M'][l]._version==s['mversions'][l] for l in LAYERS),'IMMUTABLE_SNAPSHOT_M_CHANGED'
        assert all(Runtime.tensor_schema(s['W'][l])==s['wschema'][l] for l in LAYERS),'IMMUTABLE_SNAPSHOT_W_REPLACED'
        assert all(Runtime.tensor_schema(s['M'][l])==s['mschema'][l] for l in LAYERS),'IMMUTABLE_SNAPSHOT_M_REPLACED'

    def restore(self,s):
        self.validate_snapshot(s)
        with torch.no_grad():
            for l,w in self.W.items():
                if self.whash.get(l)!=s['whash'][l] or w._version!=self._wver[l]:w.copy_(s['W'][l].to(w.device))
        self.cpuW=dict(s['W']);self.whash=dict(s['whash']);self.M=dict(s['M']);self.mhash=dict(s['mhash'])
        self.module.CONTEXT_TEMPLATES_CACHE=copy.deepcopy(s['contexts']);restore_rng(s['rng'])
        self._refresh_versions();self.guard()

    def adopt_native_weights(self,weights,rng):
        self.guard(selected=False)
        with torch.no_grad():
            for l,value in weights.items():
                value,key=self.intern(value)
                self.W[l].copy_(value.to(self.W[l].device));self.cpuW[l]=value;self.whash[l]=key
        restore_rng(rng);self._refresh_versions();self.guard()

    def observe(self,fn):
        before=self.state();result=fn();assert self.state()==before,'OBSERVER_STATE_MUTATION';return result

    def requests(self,records):
        return [dict(copy.deepcopy(r['requested_rewrite']),case_id=r['case_id']) for r in records]

    def fit(self,records,layer,instrument=True):
        before=self.snapshot()
        with Timer(self.timing,'native_fit_inclusive'):
            result=self.fitter.fit_capture(self.model,self.tok,self.hp[layer],self.M[layer],self.P[layer],
                self.requests(records),layer=layer,instrument=instrument)
        self.guard(selected=False)
        assert all(self.W[l]._version==self._wver[l] for l in LAYERS if l!=layer),'WRONG_LAYER_WRITE'
        self.adopt_native_weights({layer:result['weight']},before['rng'])
        assert self.mhash==before['mhash'],'INNER_HISTORY_CHANGED'
        result['receipt']['input_state']=before['state'];result['receipt']['output_state']=self.state()
        return result


class Backend:
    def __init__(self,rt,current,past,out):
        self.rt,self.current,self.past,self.out=rt,list(current),list(past),Path(out)
        self.out.mkdir(parents=True,exist_ok=False)
        self.fit_receipts=[];self.score_receipts=[];self.history_receipt=None
        self.namespace=digest(dict(source=rt.lock['source_head'],model=rt.lock['model_revision'],
            input=digest(self.current),past=digest(self.past),contexts=rt.context,context_tokens=rt.context_tokens,
            request_tokens=[rt.tok(r['requested_rewrite']['prompt'].format(r['requested_rewrite']['subject']))['input_ids'] for r in self.current],
            target_tokens=[rt.tok(r['requested_rewrite']['target_new']['str'])['input_ids'] for r in self.current],
            hparams={str(l):vars(rt.hp[l]) for l in LAYERS},state=rt.state(),teacher=rt.teacher.receipt))
        save(self.out/'namespace.json',dict(namespace=self.namespace,state=rt.state(),current=digest(self.current),past=digest(self.past)))

    def snapshot(self):return self.rt.snapshot()
    def restore(self,s):self.rt.restore(s)
    @staticmethod
    def snapshot_token(s):return digest({k:v for k,v in s['state'].items() if k!='M'})
    def validate_snapshot(self,s,expected_state_token,expected_history_token):
        self.rt.validate_snapshot(s)
        assert self.snapshot_token(s)==expected_state_token
        assert digest(s['state']['M'])==expected_history_token
    def state_token(self):return digest({k:v for k,v in self.rt.state().items() if k!='M'})
    def history_token(self):return digest(self.rt.state()['M'])

    def fit_native(self,layer):
        n=len(self.fit_receipts);folder=self.out/'fits'/f'{n:03d}-L{layer}';folder.mkdir(parents=True)
        save(folder/'entry.json',dict(state=self.rt.state(),layer=layer,current=digest(self.current)))
        try:result=self.rt.fit(self.current,layer)
        except BaseException as exc:
            evidence=None
            if hasattr(exc,'native_partial'):evidence=tensor_save(folder/'failure-partial.pt',exc.native_partial)
            save(folder/'failure.json',dict(error=repr(exc),partial=evidence,successful_fit=False))
            raise
        # Selected weight is shared only in RAM; disk retains target/key/native
        # components needed for diagnostics, not a claimed continuation state.
        payload={k:v for k,v in result.items() if k not in ('weight','receipt')}
        ref=tensor_save(folder/'native-evidence.pt',payload)
        receipt=save(folder/'receipt.json',result['receipt']|dict(evidence=ref,W_saved=False,
            history_append=0,weight_sha256=self.rt.whash[layer],exact_crash_resume=False))
        self.fit_receipts.append(receipt)
        return result['receipt']['adam_updates']

    def apply_gate(self,layer,before,native,gate):
        self.rt.validate_snapshot(before);self.rt.validate_snapshot(native)
        assert before['mhash']==native['mhash'],'GATE_HISTORY'
        assert all(before['whash'][l]==native['whash'][l] for l in LAYERS if l!=layer),'NATIVE_OTHER_LAYER'
        self.restore(before)
        value=fp32_gate(before['W'][layer],native['W'][layer],gate)
        self.rt.adopt_native_weights({layer:value},before['rng'])

    def score(self):
        before=self.rt.state();result=self.rt.observe(lambda:self.rt.metrics.score(self.current,self.past))
        path=self.out/'scores'/f'{len(self.score_receipts):03d}.json'
        self.score_receipts.append(save(path,dict(state=before,metrics=serial(result),controller_inputs_only=True)))
        return Scores(**result['controller'])

    def changed_layers(self,before,after):
        return tuple(l for l in LAYERS if before['whash'][l]!=after['whash'][l])
    def action_norm(self,before,after):
        return math.sqrt(math.fsum(float((after['W'][l].double()-before['W'][l].double()).square().sum())
            for l in self.changed_layers(before,after)))

    def finalize_history(self,layers):
        assert tuple(layers)==LAYERS and self.history_receipt is None,'HISTORY_ONCE_ALL5'
        before=self.rt.snapshot()
        # Clone before modifying so all candidate and entry histories stay immutable.
        self.rt.M={l:m.clone() for l,m in self.rt.M.items()};self.rt._refresh_versions()
        with Timer(self.rt.timing,'history_five_layers'):
            rows=self.rt.fitter.finalize(self.rt.model,self.rt.tok,self.rt.requests(self.current),
                [(l,self.rt.hp[l],self.rt.M[l],self.rt.P[l]) for l in LAYERS])
        self.rt.mhash={l:tensor_sha(m) for l,m in self.rt.M.items()};restore_rng(before['rng'])
        self.rt._refresh_versions();self.rt.guard()
        assert self.rt.whash==before['whash'] and len(rows)==5 and all(r['history_append']==1 for r in rows)
        self.history_receipt=save(self.out/'history.json',dict(rows=rows,entry=before['state'],selected=self.rt.state(),
            appends=5,candidate_appends=0,all_current_requests=len(self.current),gate_weighting=False))
