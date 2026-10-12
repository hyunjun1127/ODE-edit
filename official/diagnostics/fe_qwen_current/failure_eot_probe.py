"""FE + OURS basic position0 boundary preprocessing, not author-bitwise FE.

CLI BINDINGS NEW_OUTPUT: old b255 bindings + control_result exact prefix proof.
Cold W0, seven paired fits/replays and fourteen batched plain-prefix forwards.
No writer/history/evaluation/checkpoint/W&B; no sink list, M1 or key replacement.
InputAdapter creates scoped in-memory function copies; frozen modules unchanged.
"""
import argparse
import __future__
import ast
import copy
import hashlib
import importlib
import inspect
import sys
from pathlib import Path
import torch
from omegaconf import OmegaConf
from official.ours.pos0 import repair_prompt,PREFIX
from official.diagnostics.fe_qwen_current import failure_probe as base
from official.experiments.checkpoint import rng_snapshot,rng_restore
from official.experiments.prepare import write_new
from official.runners.server1.common import read,verify,member,require
from official.diagnostics.fe_qwen_current.storage import tensor_sha,state_digest

CASES=[21007,14707,15452,6263,11829,17428,6976]


def ast_id(node):return hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()


class InputAdapter:
    def __init__(self,tok,lookup,model_type='qwen2'):
        require(model_type in ('qwen2','gptj'),'BASIC_EOT_FAMILY')
        self.tok=tok;self.native_lookup=lookup;self.model_type=model_type;self.proofs={}
    def prompt(self,prompt,subject):
        return repair_prompt(self.tok,prompt,subject,self.model_type,
            lambda t,p,s:self.native_lookup(p,s,t,'subject_last',verbose=False))
    def lookup(self,p,s,t,strategy,**kw):
        require(t is self.tok and strategy=='subject_last','LOOKUP_CONTRACT')
        return self.native_lookup(self.prompt(p,s),s,t,strategy,**kw)
    def extraction(self,function):
        def call(model,tok,layer,**kw):
            require(tok is self.tok and kw['fact_token_strategy']=='subject_last','EXTRACTION_CONTRACT')
            updated=dict(kw);updated['context_templates']=[self.prompt(p,s) for p,s in zip(kw['context_templates'],kw['words'])]
            require(len(updated['context_templates'])==len(kw['context_templates'])==len(kw['words']),'EXTRACTION_ROWS')
            return function(model,tok,layer,**updated)
        return call
    def _compile(self,function,node,globals_extra,original):
        ns=dict(function.__globals__,**globals_extra)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),inspect.getsourcefile(function) or '<native-input-overlay>','exec',flags=__future__.annotations.compiler_flag),ns)
        result=ns[function.__name__];self.proofs[function.__name__]=dict(original_ast=ast_id(original),overlay_ast=ast_id(node),non_input_ast_exact=True)
        return result
    def fit(self,function):
        original=ast.parse(inspect.getsource(function)).body[0];node=copy.deepcopy(original);count=0
        class Insert(ast.NodeTransformer):
            def visit_Assign(inner,n):
                nonlocal count
                if len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='all_prompts':
                    count+=1
                    extra=ast.parse("all_prompts = [_eot_input(p, request['subject']) for p in all_prompts]\nrewriting_prompts, kl_prompts = all_prompts[:len(rewriting_prompts)], all_prompts[len(rewriting_prompts):]").body
                    for x in extra:x._input_overlay=True
                    return [n]+extra
                return n
        node=Insert().visit(node);require(count==2,'PINNED_NATIVE_FIT_INPUT_SITES')
        stripped=copy.deepcopy(node)
        class Strip(ast.NodeTransformer):
            def visit_Assign(inner,n):return None if getattr(n,'_input_overlay',False) else n
        require(ast_id(Strip().visit(stripped))==ast_id(original),'FIT_NON_INPUT_AST_CHANGED')
        return self._compile(function,node,dict(_eot_input=self.prompt),original)
    def supplier(self,function):
        original=ast.parse(inspect.getsource(function)).body[0];node=copy.deepcopy(original);count=0
        class Replace(ast.NodeTransformer):
            def visit_Subscript(inner,n):
                nonlocal count
                if isinstance(n.value,ast.Name) and n.value.id=='request' and isinstance(n.slice,ast.Constant) and n.slice.value=='prompt':
                    count+=1;c=ast.Call(func=ast.Name(id='_eot_input',ctx=ast.Load()),args=[n,ast.parse("request['subject']",mode='eval').body],keywords=[]);c._input_overlay=True;return c
                return inner.generic_visit(n)
        node=Replace().visit(node);require(count==2,'PINNED_REPLAY_INPUT_SITES')
        class Strip(ast.NodeTransformer):
            def visit_Call(inner,n):return n.args[0] if getattr(n,'_input_overlay',False) else inner.generic_visit(n)
        require(ast_id(Strip().visit(copy.deepcopy(node)))==ast_id(original),'REPLAY_NON_INPUT_AST_CHANGED')
        return self._compile(function,node,dict(_eot_input=self.prompt),original)
    def keys(self,function):
        node=ast.parse(inspect.getsource(function)).body[0]
        return self._compile(function,node,dict(get_module_input_output_at_words=self.extraction(function.__globals__['get_module_input_output_at_words'])),node)
    def writer(self,function,key_function):
        # Exposed for a separately authorized future prefix; NOT invoked by CLI.
        node=ast.parse(inspect.getsource(function)).body[0]
        return self._compile(function,node,dict(compute_ks=self.keys(key_function),
            get_module_input_output_at_words=self.extraction(function.__globals__['get_module_input_output_at_words'])),node)


def observe_fit(function,receipts):
    def call(*args,**kw):
        require(sys.getprofile() is None,'PROFILE_BUSY')
        def profile(frame,event,value):
            if frame.f_code is function.__code__ and event=='return':
                v=frame.f_locals;tokens=v['input_tok'];n=len(v['rewriting_prompts'])
                rows=[]
                for i,pos in enumerate(v['lookup_idxs']):
                    size=int(tokens['attention_mask'][i].sum());labels=v['rewriting_targets'][i,:size].tolist() if i<n else None
                    rows.append(dict(kind='rewrite' if i<n else 'KL',lookup=int(pos),
                        ids=tokens['input_ids'][i,:size].tolist(),mask=tokens['attention_mask'][i,:size].tolist(),labels=labels))
                receipts.append(dict(init=base.scalar_vector(v['target_init']),delta=base.scalar_vector(v['delta']),
                    returned=base.scalar_vector(value),probabilities=v['nll_loss_each'].detach().float().neg().exp().cpu().tolist(),
                    nll_each=v['nll_loss_each'].detach().float().cpu().tolist(),iterations=int(v['it'])+1,input_rows=rows))
        sys.setprofile(profile)
        try:return function(*args,**kw)
        finally:sys.setprofile(None)
    return call


def plain_geometry(model,tok,r,cfg,contexts,lookup,TraceDict,adapter=None,force_bare_control=False):
    prompts=[ctx.format(r['prompt']) for group in contexts for ctx in group]+['{} is a']
    if adapter:prompts=[adapter.prompt(p,r['subject']) for p in prompts]
    if force_bare_control:
        require(r['case_id']==17428 and lookup(prompts[0],r['subject'],tok,cfg.llms.fact_token,verbose=False)==1,'THEWARS_NONZERO_CONTROL')
        old=prompts[0];prompts[0]=PREFIX+old
        require(tok.encode(prompts[0].format(r['subject']))==[151643]+tok.encode(old.format(r['subject'])),'CONTROL_PREFIX_TOKEN_EXACT')
    positions=[lookup(p,r['subject'],tok,cfg.llms.fact_token,verbose=False) for p in prompts]
    inputs=tok([p.format(r['subject']) for p in prompts],return_tensors='pt',padding=True).to(next(model.parameters()).device)
    modules=dict(model.named_modules());rows={};handles=[]
    # Only lookup vectors copied; no complete activations retained.
    for l in cfg.llms.layers:
        def hook(m,args,out,layer=l):
            keys=torch.stack([args[0][i,p].detach().clone() for i,p in enumerate(positions)])
            kbar=torch.stack([keys[:1].mean(0),keys[1:6].mean(0)]).mean(0)
            rows[str(layer)]=dict(keys=[base.scalar_vector(x) for x in keys],Kbar=base.scalar_vector(kbar),
                bare_vs_bar_cosine=float(torch.nn.functional.cosine_similarity(keys[0].float(),kbar.float(),dim=0)))
        handles.append(modules[cfg.llms.rewrite_module_tmp.format(l)].register_forward_hook(hook))
        def block(m,args,out,layer=l):rows[str(layer)]['block']=[base.scalar_vector(base.hidden(out)[i,p]) for i,p in enumerate(positions)]
        handles.append(modules[cfg.llms.layer_module_tmp.format(l)].register_forward_hook(block))
    try:
        with torch.no_grad():model(**inputs)
    finally:
        for h in handles:h.remove()
    return dict(lookup_positions=positions,layers=rows,model_forwards=1,
        policy='FORCED_NONZERO_BARE_PREFIX_CONTROL_ONLY_NOT_FIT_OR_WRITER_POLICY' if force_bare_control else 'BASIC_POS0_ONLY' if adapter else 'ORIGINAL')


def run(bindings,output):
    b=read(bindings);v=base.validate_inputs(b);c=v['config'];lock=v['lock'];out=Path(output)
    proof=read(verify(b['control_result']));require(proof['status']=='PREFIX_STATE_EXACT','CONTROL_EXACT')
    cb=read(verify(proof['bindings']));require(cb['checkpoint']==b['checkpoint'] and cb['config']==b['config'],'CONTROL_BINDING')
    require(not out.exists(),'NEW_OUTPUT_ONLY')
    import transformers
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,'ROOT_GPU1')
    require(torch.__version__==lock['runtime']['torch'] and transformers.__version__==lock['runtime']['transformers'],'RUNTIME')
    ref=torch.load(verify(b['checkpoint']),map_location='cpu',mmap=True,weights_only=False)
    require(ref['cursor']['native_config']==v['runtime']['resolved'] and ref['identity']==v['pointer']['identity'],'REFERENCE_CONFIG')
    cfg=OmegaConf.create(ref['cursor']['native_config']);require(not cfg.lti and not cfg.negetive_prompt_test and cfg.llms.fact_token=='subject_last','NATIVE_SUPPORTED_INPUTS')
    expected_contexts=copy.deepcopy(ref['contexts']);del ref # no H allocation/copy/restore
    data=read(verify(c['native_data']));stream=read(verify(c['stream']));require([r['case_id'] for r in data]==[r['case_id'] for r in stream],'ORDER')
    requests=[]
    for cid in CASES:
        matches=[r for r in data if r['case_id']==cid];require(len(matches)==1,'SEVEN_EXACT_CASES');r=copy.deepcopy(matches[0]);r['target_new']=' '+r['target_new'];r['sample_idx']=0;requests.append(r)
    snap=read(verify(c['model_assets']))['model']['snapshot']
    model=transformers.AutoModelForCausalLM.from_pretrained(snap,torch_dtype=torch.bfloat16,trust_remote_code=True,local_files_only=True).to('cuda:0')
    require(model.config._attn_implementation==v['runtime']['attention_backend'],'ORIGINAL_BACKEND')
    tok=transformers.AutoTokenizer.from_pretrained(snap,trust_remote_code=True,local_files_only=True)
    if tok.pad_token is None:tok.pad_token=tok.eos_token
    sys.path.insert(0,lock['author_path']);native=importlib.import_module('algs.memit.memit_main');z=importlib.import_module('algs.memit.compute_z');s=importlib.import_module('algs.memit.native_sequential_z')
    for mod in (native,z,s):require(Path(mod.__file__).resolve().is_relative_to(Path(lock['author_path']).resolve()),'EXACT_NATIVE_IMPORT')
    require(native.CONTEXT_TEMPLATES_CACHE is None,'FRESH_CONTEXTS');native.set_random_seed(cfg.seed)
    contexts=native.get_context_templates(model,tok);require(contexts==expected_contexts,'ORIGINAL_GENERATED_CONTEXTS')
    saved_rng=rng_snapshot();versions={n:(id(p),p._version) for n,p in model.named_parameters()};grad={n:p.requires_grad for n,p in model.named_parameters()}
    hooks={n:(tuple(m._forward_hooks),tuple(m._forward_pre_hooks)) for n,m in model.named_modules()}
    selected={cfg.llms.rewrite_module_tmp.format(l)+'.weight' for l in cfg.llms.layers};hashes={n:tensor_sha(p) for n,p in model.named_parameters() if n in selected}
    adapter=InputAdapter(tok,z.find_fact_lookup_idx);fitted=adapter.fit(z.compute_z);supplier=adapter.supplier(s.current_batch_targets)
    # Build all routes for audit; writer and key extraction are never executed here.
    adapter.writer(native.batch_edit,native.compute_ks);rows=[]
    try:
        for r in requests:
            pair=dict(case_id=r['case_id'],variants={})
            for enabled in (False,True):
                rng_restore(saved_rng);fits=[];fit=fitted if enabled else z.compute_z;sup=supplier if enabled else s.current_batch_targets
                lookup=adapter.lookup if enabled else z.find_fact_lookup_idx
                stage={'replay':False};replay={};wrapped=observe_fit(fit,fits)
                def observed_fit(*a,**kw):
                    value=wrapped(*a,**kw);stage['replay']=True;return value
                pos=lookup(r['prompt'],r['subject'],tok,cfg.llms.fact_token,verbose=False)
                def replay_key(m,args,value):
                    if stage['replay']:replay.update(lookup=int(pos),key=base.scalar_vector(args[0][0,pos]),mlp=base.scalar_vector(base.hidden(value)[0,pos]))
                handle=dict(model.named_modules())[cfg.llms.rewrite_module_tmp.format(5)].register_forward_hook(replay_key)
                try:targets=sup(model,tok,[r],cfg,contexts,compute_z=observed_fit,find_fact_lookup_idx=lookup,TraceDict=z.nethook.TraceDict)
                finally:handle.remove()
                require(bool(replay),'REPLAY_KEY_CAPTURE')
                geom=plain_geometry(model,tok,r,cfg,contexts,z.find_fact_lookup_idx,z.nethook.TraceDict,adapter if enabled else None,
                    force_bare_control=enabled and r['case_id']==17428)
                require(state_digest(rng_snapshot())==state_digest(saved_rng),'FIT_OR_PLAIN_RNG_CHANGED')
                pair['variants']['eot' if enabled else 'original']=dict(fit=fits[0],targets={str(l):base.scalar_vector(t[:,0]) for l,t in targets.items()},replay_L5=replay,plain=geom)
            a=pair['variants']['original']['fit']['input_rows'];d=pair['variants']['eot']['fit']['input_rows']
            for old,new in zip(a,d):
                changed=old['lookup']==0;require(new['ids']==([151643]+old['ids'] if changed else old['ids']) and all(new['mask']),'EXACT_PREFIX_MASK')
                require(new['lookup']==(1 if changed else old['lookup']),'LOOKUP_SHIFT')
                if old['labels'] is not None:require(new['labels']==([-100]+old['labels'] if changed else old['labels']),'TARGET_LABELS_SHIFT_ONLY')
            pair['changed_fit_rows']=dict(rewrite=sum(x['lookup']==0 for x in a if x['kind']=='rewrite'),KL=sum(x['lookup']==0 for x in a if x['kind']=='KL'))
            rows.append(pair)
    finally:
        for n,p in model.named_parameters():p.requires_grad_(grad[n])
        rng_restore(saved_rng)
    require(versions=={n:(id(p),p._version) for n,p in model.named_parameters()},'WEIGHTS_MUTATED')
    require(hooks=={n:(tuple(m._forward_hooks),tuple(m._forward_pre_hooks)) for n,m in model.named_modules()},'HOOK_LEAK')
    require(hashes=={n:tensor_sha(p) for n,p in model.named_parameters() if n in selected},'SELECTED_HASH_CHANGED')
    verify(b['checkpoint']);out.mkdir(parents=True)
    result=dict(status='PAIRED_EOT_NO_WRITE_COMPLETE',rows=rows,source=member(__file__),bindings=member(bindings),overlay_proofs=adapter.proofs,
        fits=14,replays=14,plain_prefix_forwards=14,context_generation='ORIGINAL_NATIVE_W0_EQUAL',writer=0,history=0,checkpoint_save=0,evaluation=0,
        variant='FE_PLUS_OURS_BASIC_POSITION0_EOT_INPUT_PREPROCESSING',not_author_bitwise=True,recovery_claim=False,
        runtime=dict(torch=torch.__version__,transformers=transformers.__version__,backend=model.config._attn_implementation,gpu=torch.cuda.get_device_name(),
            tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,
            pad_token_id=tok.pad_token_id,padding_side=tok.padding_side))
    write_new(out/'eot-probe.json',result);return member(out/'eot-probe.json')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('bindings');p.add_argument('output');a=p.parse_args();print(run(a.bindings,a.output))
