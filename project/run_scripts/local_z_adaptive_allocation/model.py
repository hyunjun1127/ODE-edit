"""Read-only native imports, actual all-token weights, isolated cold state."""
import ast
import copy
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time

import torch
from .common import digest, Timer
from .policy import materialized
from project.run_scripts.baseline_mechanism_first.fixtures import capture_rng, restore_rng
from project.run_scripts.low_cost_write_donor_pilot.fitting import NativeSingletonFitter, select_projector, tensor_sha
from project.run_scripts.bg_tw_reference.ep_tw.model_adapter import TeacherStore, EpisodeAdapter, capture_native_fit

def normalized(requests):
    out = copy.deepcopy(requests)
    for r in out:
        if not r['target_new']['str'].startswith(' '): r['target_new']['str'] = ' '+r['target_new']['str']
    return out

def terminal_writer(module):
    """Extract exact native K/readout/repeat/direct-solve/add statements.

    Unlike a singleton target injection, this function explicitly fixes readout
    to physical L8 independently of the physical writer. Residual has no native
    remaining-layer divisor: actual FP32 endpoint scaling follows outside.
    """
    tree = ast.parse(inspect.getsource(module))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'apply_AlphaEdit_to_model')
    branch = next(n for n in fn.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'hparams.blue')
    loop = branch.body[0]
    start = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'layer_ks')
    stop = next(i for i in range(start, len(loop.body)) if isinstance(loop.body[i], ast.With))
    body = copy.deepcopy(loop.body[start:stop+1])
    # Remove native diagnostic print only; retain every numerical expression.
    body = [n for n in body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == 'print')]
    prefix = ast.parse("i=0\nz_layer=8\nweights={hparams.rewrite_module_tmp.format(layer)+'.weight': weight}").body
    suffix = ast.parse("return {'K':layer_ks.detach().cpu(), 'H8':cur_zs.detach().cpu(), 'R':targets.detach().cpu()}").body
    adapter = ast.parse('def terminal(model,tok,requests,hparams,cache_c,P,layer,context_templates,zs,weight):\n pass').body[0]
    adapter.body = prefix + body + suffix
    code = ast.fix_missing_locations(ast.Module(body=[adapter], type_ignores=[]))
    ns = dict(vars(module)); exec(compile(code, module.__file__+':local-z-terminal-adapter', 'exec'), ns)
    return ns['terminal'], dict(native_statement_ast=digest(ast.dump(ast.Module(body=body, type_ignores=[]))),
        change='new explicit terminal readout8; native BLUE unscaled direct solve statements; FP32 gates outside',
        writer_readout_mapping={'4':8,'8':8}, inverse_calls=0)

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
        h4 = HP.from_json(lock['config4']); h8 = copy.deepcopy(h4); h8.layers = [8]
        assert h4.blue and h4.layers == [4] and h4.L2 == 1
        assert (h4.v_weight_decay,h4.clamp_norm_factor,h4.v_lr,h4.v_num_grad_steps)==(.5,.75,.1,25)
        self.hp = {4:h4,8:h8}
        with Timer(self.timing, 'model_load'):
            self.model = AutoModelForCausalLM.from_pretrained(lock['snapshot'], local_files_only=True,
                torch_dtype=torch.float32, low_cpu_mem_usage=True, attn_implementation='eager').cuda().eval()
        self.tok = AutoTokenizer.from_pretrained(lock['snapshot'],local_files_only=True)
        self.tok.add_bos_token=False; self.tok.pad_token_id=self.tok.eos_token_id
        self.etok = AutoTokenizer.from_pretrained(lock['snapshot'],local_files_only=True)
        self.etok.pad_token_id=self.etok.eos_token_id
        assert self.tok.padding_side == self.etok.padding_side == 'right'
        self.params=dict(self.model.named_parameters())
        assert all(p.dtype==torch.float32 for p in self.params.values())
        for p in self.params.values(): p.requires_grad_(False)
        self.W={l:self.params[f'model.layers.{l}.mlp.down_proj.weight'] for l in (4,8)}
        with Timer(self.timing,'projector_load'):
            stack=torch.load(lock['projector'],map_location='cpu',weights_only=True,mmap=True)
            selected={l:select_projector(stack,l) for l in (4,8)}
            self.P={l:selected[l][0] for l in (4,8)}
            self.pmap={str(l):selected[l][1] for l in (4,8)}; del stack
        self.M={l:torch.zeros_like(self.P[l]) for l in (4,8)}
        self.module.CONTEXT_TEMPLATES_CACHE=None; self.module.COV_CACHE={}
        if common is None:
            with Timer(self.timing,'native_context_generation'):
                self.context=copy.deepcopy(self.module.get_context_templates(self.model,self.tok))
            self.common_rng=capture_rng()
        else:
            self.context=copy.deepcopy(common['contexts']); self.common_rng=common['rng']
            restore_rng(self.common_rng)
        self.module.CONTEXT_TEMPLATES_CACHE=copy.deepcopy(self.context)
        self.context_tokens=[[self.tok(x)['input_ids'] for x in group] for group in self.context]
        self.fitter=NativeSingletonFitter(self.module,expected_source_sha256=lock['editor_sha256'],contexts=self.context)
        self.terminal,self.terminal_evidence=terminal_writer(self.module)
        teacher_ref=lock['teacher_manifest'] if common is None else common['teacher_manifest']
        self.teacher=TeacherStore(lock['reference_root'],teacher_ref['path'],
            expected_manifest_sha=teacher_ref['sha256'],verify_payload_hashes=False)
        self.observer=EpisodeAdapter(self.model,self.etok,'model.layers.4.mlp.down_proj.weight',self.teacher)
        self.binding=bind_evaluation_sources(lock['historical_evaluator_root'],helper_root=lock['helper_scripts_root'])
        self.nonselected={k:(p,p.data_ptr(),p._version) for k,p in self.params.items() if k not in
            ('model.layers.4.mlp.down_proj.weight','model.layers.8.mlp.down_proj.weight')}
        self.hooks={k:(tuple(m._forward_hooks),tuple(m._forward_pre_hooks),tuple(m._backward_hooks)) for k,m in self.model.named_modules()}
        self.buffers={k:(b,b.data_ptr(),b._version) for k,b in self.model.named_buffers()}
        self.w0={l:w.detach().cpu().clone() for l,w in self.W.items()}
        if common is not None:
            assert self.pmap==common['projector_mapping']
            assert {str(l):tensor_sha(w) for l,w in self.w0.items()}==common['W0']
            assert self.context_tokens==common['context_tokens']
        self.guard()

    def guard(self):
        assert not self.model.training and not self.module.COV_CACHE
        assert all(not p.requires_grad and p.grad is None for p in self.params.values())
        assert self.module.CONTEXT_TEMPLATES_CACHE==self.context
        for k,(p,ptr,v) in self.nonselected.items():
            assert self.params[k] is p and p.data_ptr()==ptr and p._version==v, ('NONSELECTED_MUTATION',k)
        for k,(b,ptr,v) in self.buffers.items():
            assert b.data_ptr()==ptr and b._version==v,('BUFFER_MUTATION',k)
        assert all(self.hooks[k]==(tuple(m._forward_hooks),tuple(m._forward_pre_hooks),tuple(m._backward_hooks)) for k,m in self.model.named_modules())

    def state(self):
        return dict(W={str(l):tensor_sha(w) for l,w in self.W.items()},M={str(l):tensor_sha(m) for l,m in self.M.items()},
            P={str(l):tensor_sha(p) for l,p in self.P.items()},contexts=digest(self.context),rng=digest(capture_rng()))

    def snapshot(self):
        return dict(W={l:w.detach().cpu().clone() for l,w in self.W.items()},M={l:m.clone() for l,m in self.M.items()},
                    contexts=copy.deepcopy(self.context),rng=capture_rng())

    def restore(self,s):
        with torch.no_grad():
            for l,w in self.W.items(): w.copy_(s['W'][l].to(w.device)); self.M[l].copy_(s['M'][l])
        self.module.CONTEXT_TEMPLATES_CACHE=copy.deepcopy(s['contexts']);restore_rng(s['rng']);self.guard()

    def apply(self, weights, rng):
        with torch.no_grad():
            for l in (4,8): self.W[l].copy_(weights[l].to(self.W[l].device))
        restore_rng(rng);self.guard()

    def observe(self, fn):
        before=self.state();result=fn();assert self.state()==before,'OBSERVER_STATE_MUTATION';self.guard();return result

    def requests(self, records):
        return [dict(copy.deepcopy(r['requested_rewrite']),case_id=r['case_id']) for r in records]

    def fit(self,records,layer):
        before=self.state()
        with Timer(self.timing,'local_native_fit'):
            result=capture_native_fit(self.fitter,self.model,self.tok,self.hp[layer],self.M[layer],self.P[layer],self.requests(records),layer=layer)
        result['receipt']['input_state']=before
        result['receipt']['clamp_iteration_hits']='NOT_RECORDED_NATIVE_COUNTER_ABSENT; final delta and radius retained'
        return result

    def targets8(self, records):
        """Original native compute_z exactly once/request, before any subwrite."""
        requests=normalized(self.requests(records)); captures=[]
        code=self.module.compute_z.__code__
        if sys.getprofile() is not None: raise RuntimeError('PROFILE_OWNER')
        def profile(frame,event,result):
            if event=='return' and frame.f_code is code and isinstance(result,torch.Tensor):
                d=frame.f_locals; step=d['opt'].state.get(d['delta'],{}).get('step',0)
                captures.append(dict(case_id=d['request']['case_id'],adam_updates=int(step),loss_evaluations=d['it']+1,
                    anchor=d['target_init'].detach().cpu().clone(),delta=d['delta'].detach().cpu().clone(),
                    radius=float(self.hp[8].clamp_norm_factor*d['target_init'].norm()),loss_final=float(d['loss'].detach())))
        start=time.monotonic();sys.setprofile(profile)
        try: zs=torch.stack([self.module.compute_z(self.model,self.tok,r,self.hp[8],8,self.context) for r in requests],1)
        finally:sys.setprofile(None)
        assert len(captures)==len(records) and torch.isfinite(zs).all()
        return zs.detach().cpu(),dict(compute_z=len(records),seconds=time.monotonic()-start,
            adam_updates=sum(x['adam_updates'] for x in captures),loss_evaluations=sum(x['loss_evaluations'] for x in captures)),captures

    def terminal_fit(self,records,layer,z):
        before=self.state();begin=time.monotonic()
        with torch.no_grad():
            tensors=self.terminal(self.model,self.tok,normalized(self.requests(records)),self.hp[layer],self.M[layer],
                self.P[layer],layer,self.context,z.to(self.W[layer].device),self.W[layer])
        assert self.state()['M']==before['M'] and self.state()['P']==before['P']
        if not torch.isfinite(self.W[layer]).all():raise ValueError('NONFINITE_TERMINAL_FIT')
        return dict(weight=self.W[layer].detach().cpu().clone(),captures=tensors,receipt=dict(layer=layer,readout_layer=8,
            solve=1,compute_z=0,history_append=0,seconds=time.monotonic()-begin,source=self.terminal_evidence,input_state=before))

    @torch.no_grad()
    def training_E(self,records):
        """Exact native rewrite prompts/target mask and last-layer NLL expression.

        No edit_output hook, KL penalty or regularizer. E is the native rewrite
        NLL at the actual all-token endpoint, contexts then requests equally.
        """
        hp=self.hp[4];tok=self.tok;rows=[];start=time.monotonic();ncontexts=0;tokens=0
        lm_w=self.module.nethook.get_parameter(self.model,f'{hp.lm_head_module}.weight').T
        ln_f=self.module.nethook.get_module(self.model,hp.ln_f_module)
        try:lm_b=self.module.nethook.get_parameter(self.model,f'{hp.lm_head_module}.bias')
        except LookupError:lm_b=torch.zeros(self.model.config.vocab_size,device='cuda')
        for r in normalized(self.requests(records)):
            target=tok(r['target_new']['str'],return_tensors='pt').to('cuda')['input_ids'][0]
            if target[0] in (tok.bos_token_id,tok.unk_token_id):target=target[1:]
            rewrite=[c.format(r['prompt'])+tok.decode(target[:-1]) for group in self.context for c in group]
            inputs=tok([p.format(r['subject']) for p in rewrite+['{} is a']],return_tensors='pt',padding=True).to('cuda')
            targets=torch.full((len(rewrite),inputs['input_ids'].shape[1]),-100,device='cuda',dtype=torch.long)
            for i in range(len(rewrite)):
                end=int(inputs['attention_mask'][i].sum());targets[i,end-len(target):end]=target
            with self.module.nethook.TraceDict(self.model,[hp.layer_module_tmp.format(hp.v_loss_layer)],retain_input=False,retain_output=True) as trace:
                self.model(**inputs)
            output=trace[hp.layer_module_tmp.format(hp.v_loss_layer)].output[0]
            if output.shape[1]!=targets.shape[1]:output=output.transpose(0,1)
            full=output[:len(rewrite)]
            logp=torch.log_softmax(ln_f(full)@lm_w.to(full.device)+lm_b.to(full.device),dim=2)
            values=logp.gather(2,torch.where(targets!=-100,targets,0).unsqueeze(2)).squeeze(2)
            losses=-(values*(targets!=-100).float()).sum(1)/target.size(0)
            if not torch.isfinite(losses).all():raise ValueError('NONFINITE_E')
            rows.append(dict(case_id=r['case_id'],context_nll=losses.cpu().tolist(),nll=float(losses.mean())))
            ncontexts+=len(rewrite);tokens+=int(inputs['attention_mask'].sum())
        return dict(E=math.fsum(r['nll'] for r in rows)/len(rows),rows=rows,requests=len(rows),contexts=ncontexts,
            model_forwards=len(rows),input_tokens=tokens,seconds=time.monotonic()-start,
            reduction='NATIVE_TOKEN_MEAN_CONTEXT_MEAN_REQUEST_MEAN',gradients=0)

    def score(self,records,past):
        e=self.observe(lambda:self.training_E(records));cur=self.observe(lambda:self.observer.current(records))
        old=self.observe(lambda:self.observer.current(past)) if past else None
        d=self.observe(lambda:self.observer.generic('S64'))
        return dict(E=e['E'],H=None if old is None else old['E'],D=d['D'],S_cur=cur['strict_ids'],
            S_past=[] if old is None else old['strict_ids'],details=dict(training=e,current=cur,past=old,generic=d))
