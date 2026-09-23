"""Pinned original non-BLUE model binding and read-only prefix instrumentation.

No optimized-z adapter and no native source file is changed. Model state is
restored from the approved input checkpoints; new resume checkpoints forbidden.
"""
import importlib
import json
import os
from pathlib import Path
import random
import sys
import time
import copy
from contextlib import contextmanager

from .common import digest,file_sha,save,tensor_sha,tensor_file,rng_get,rng_set,rng_preserved,Timer

LAYERS=(4,5,6,7,8)

class PrefixComplete(Exception):pass

class Runtime:
    def __init__(self,root,repo):
        import numpy as np,torch,transformers
        self.root=Path(root);self.repo=Path(repo);self.design=self.root/'inputs/design'
        self.contract=json.loads((self.design/'contract.json').read_text())
        self.old=json.loads((self.design/'evidence/audits/global/2026-09-22-alphaedit-native-criticality-audit/target-native-execution.lock.json').read_text())
        binding_path=self.root/'receipts/native-input-binding-r2.json'
        if not binding_path.exists():binding_path=self.root/'receipts/native-input-binding-r1.json'
        self.binding=json.loads(binding_path.read_text())
        # The two old non-executable policy documents are retained as provenance
        # misses if unavailable; every executable/model/input member must match.
        assert all(x['path'].endswith(('PROTOCOL.md','fixed-counterfact-10k-policy.md')) for x in self.binding['unresolved'])
        for x in self.binding['members']:
            p=Path(x['resolved_path']);assert p.stat().st_size==x['bytes'],('ASSET_SIZE_CHANGED',str(p))
            if p.suffix=='.py':assert file_sha(p)==x['sha256'],('SOURCE_DRIFT',str(p))
        loader=self.old['fixed_dataset_binding']['loader']
        loader=next(x['resolved_path'] for x in self.binding['members'] if x['path']==loader)
        spec=importlib.util.spec_from_file_location('alpha_causal_fixed_data',loader)
        policy=importlib.util.module_from_spec(spec);spec.loader.exec_module(policy)
        self.rows=policy.load_prefix(self.old['fixed_dataset_binding']['root'],10000)
        self.byid={int(r['case_id']):r for r in self.rows}
        assert len(self.rows)==len(self.byid)==10000
        assert file_sha(self.old['dataset'])=='3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1'
        assert torch.__version__=='2.9.1+cu128' and transformers.__version__=='4.44.2'
        torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
        random.seed(self.old['seed']);np.random.seed(self.old['seed']);torch.manual_seed(self.old['seed'])
        sys.path.insert(0,self.old['blue_root']);os.chdir(self.old['blue_root'])
        self.native=importlib.import_module('AlphaEdit.AlphaEdit_main')
        self.repr=importlib.import_module('rome.repr_tools')
        from transformers import AutoModelForCausalLM,AutoTokenizer
        self.timer=Timer();t=time.monotonic()
        self.model=AutoModelForCausalLM.from_pretrained(self.old['snapshot'],local_files_only=True,low_cpu_mem_usage=True,attn_implementation='eager').cuda().eval()
        self.model.requires_grad_(False)
        self.tok=AutoTokenizer.from_pretrained(self.old['snapshot'],local_files_only=True)
        self.tok.add_bos_token=False;self.tok.pad_token_id=self.tok.eos_token_id
        self.evalt=AutoTokenizer.from_pretrained(self.old['snapshot'],local_files_only=True);self.evalt.pad_token_id=self.evalt.eos_token_id
        assert self.tok.padding_side==self.evalt.padding_side=='right'
        assert {p.dtype for p in self.model.parameters()}=={torch.float32}
        self.weights={l:self.model.model.layers[l].mlp.down_proj.weight for l in LAYERS}
        self.base_weights={l:w.detach().cpu().clone() for l,w in self.weights.items()}
        expected=json.loads((self.design/'source/runtime.json').read_text())['W0']['weights']
        for l,w in self.base_weights.items():assert tensor_sha(w)==expected[f'model.layers.{l}.mlp.down_proj.weight']['sha256']
        self.P=torch.load(self.old['projector'],weights_only=True,map_location='cpu',mmap=True)
        assert list(self.P.shape)==[5,14336,14336] and self.P.dtype==torch.float32
        self.hp=importlib.import_module('AlphaEdit.AlphaEdit_hparams').AlphaEditHyperParams(**self.contract['baseline']['hparams'])
        assert self.hp.blue is False and self.hp.L2==10 and self.hp.layers==list(LAYERS)
        self.cp_manifest=json.loads((self.design/'checkpoint-transfer-manifest.json').read_text())
        receipt=json.loads((self.root/'receipts/checkpoint-transfer-r1.json').read_text())
        assert receipt['status']=='PASS' and len(receipt['members'])==12
        self.cp_root=self.root/'inputs/checkpoints/BASE_ALPHAEDIT'
        first=self._load_cp(1)
        self.contexts=copy.deepcopy(first['metadata']['contexts'])
        assert list(map(len,self.contexts))==[1,5] and self.contexts[0]==['{}']
        self.native.CONTEXT_TEMPLATES_CACHE=copy.deepcopy(self.contexts)
        self.base_rng=rng_get();self.M=None;self.cp=None;self.batch=0
        self.model_load_seconds=time.monotonic()-t
        self.nonselected={n:(v.data_ptr(),v._version) for n,v in self.model.named_parameters() if n not in {f'model.layers.{l}.mlp.down_proj.weight' for l in LAYERS}}

    def _load_cp(self,batch):
        import torch
        p=self.cp_root/f'B{batch:03d}/W-method-state.pt'
        x=torch.load(p,map_location='cpu',weights_only=True,mmap=True)
        assert set(x)=={'weights','cache_c','metadata'}
        assert x['metadata']['batch']==batch and len(x['metadata']['seen_ids'])==batch*100
        assert list(x['cache_c'].shape)==[5,14336,14336]
        assert x['metadata']['method']=='AlphaEdit'
        return x

    def set_state(self,batch):
        import numpy as np,torch
        cp=self._load_cp(batch) if batch else None
        with torch.no_grad():
            for l,w in self.weights.items():
                v=cp['weights'][f'model.layers.{l}.mlp.down_proj.weight'] if cp else self.base_weights[l]
                assert v.dtype==torch.float32 and v.shape==w.shape and torch.isfinite(v).all()
                w.copy_(v.to(w.device))
        self.M=cp['cache_c'].clone() if cp else torch.zeros_like(self.P)
        assert torch.isfinite(self.M).all()
        if cp:
            m=cp['metadata'];assert m['contexts']==self.contexts
            r=m['rng'];n=r['numpy']
            pr=r['python']; python_state=(pr[0],tuple(pr[1]),pr[2])
            rng_set((python_state, (n[0],np.asarray(n[1],dtype=np.uint32),n[2],n[3],n[4]),torch.tensor(r['torch'],dtype=torch.uint8),[torch.tensor(x,dtype=torch.uint8) for x in r['cuda']]))
        else:rng_set(self.base_rng)
        self.native.CONTEXT_TEMPLATES_CACHE=copy.deepcopy(self.contexts)
        self.batch=batch;self.cp=cp
        self.assert_nonselected()

    def assert_nonselected(self):
        assert all((v.data_ptr(),v._version)==self.nonselected[n] for n,v in self.model.named_parameters() if n in self.nonselected),'NONSELECTED_MUTATION'

    def signature(self):
        return dict(weights={str(l):tensor_sha(w) for l,w in self.weights.items()},history=tensor_sha(self.M),context=digest(self.contexts),cursor=self.batch)

    def snapshot(self):
        return dict(w={l:v.detach().cpu().clone() for l,v in self.weights.items()},m=self.M.clone(),rng=rng_get(),cursor=self.batch,context=copy.deepcopy(self.contexts),cp=self.cp)

    def restore(self,s):
        import torch
        with torch.no_grad():
            for l,v in self.weights.items():v.copy_(s['w'][l].to(v.device))
            self.M.copy_(s['m'])
        rng_set(s['rng']);self.batch=s['cursor'];self.cp=s.get('cp');self.contexts=copy.deepcopy(s['context']);self.native.CONTEXT_TEMPLATES_CACHE=copy.deepcopy(self.contexts)
        self.assert_nonselected()

    def capture(self,records,contexts=True,features=False,full=False):
        """Collect exact subject positions, fixed native128 batching; all layers once."""
        import torch
        requests=[dict(r['requested_rewrite'],case_id=r['case_id']) for r in records]
        templates=[c.format(r['prompt']) for r in requests for group in (self.contexts if contexts else [['{}']]) for c in group]
        words=[r['subject'] for r in requests for group in (self.contexts if contexts else [['{}']]) for c in group]
        idxs=self.repr.get_words_idxs_in_templates(self.tok,templates,words,'last')
        texts=[p.format(w) for p,w in zip(templates,words)]
        keys={l:[] for l in LAYERS};points={};token_rows=[];t=time.monotonic()
        with rng_preserved(),torch.no_grad():
            for start in range(0,len(texts),128):
                pack=self.tok(texts[start:start+128],padding=True,return_tensors='pt').to('cuda')
                n=pack['input_ids'].shape[0];localidx=[i[0] for i in idxs[start:start+n]]
                token_rows.extend([dict(ids=ids[:int(mask.sum())].tolist(),lookup=int(idx)) for ids,mask,idx in zip(pack['input_ids'].cpu(),pack['attention_mask'].cpu(),localidx)])
                ii=torch.arange(n,device='cuda');jj=torch.tensor(localidx,device='cuda');hooks=[]
                def extract(x):
                    if isinstance(x,tuple):x=x[0]
                    assert x.shape[0]==n
                    return x[ii,jj].detach().cpu().clone()
                for l in LAYERS:
                    layer=self.model.model.layers[l]
                    def pre(module,args,l=l):keys[l].append(extract(args[0]))
                    hooks.append(layer.mlp.down_proj.register_forward_pre_hook(pre))
                    if features:
                        def reg(mod,name,before=False):
                            key=f'L{l}/{name}'
                            if before:hooks.append(mod.register_forward_pre_hook(lambda m,a,key=key:points.setdefault(key,[]).append(extract(a[0])) and None))
                            else:hooks.append(mod.register_forward_hook(lambda m,a,o,key=key:points.setdefault(key,[]).append(extract(o)) and None))
                        reg(layer,'residual_before_attention',True);reg(layer.self_attn,'attention_output')
                        reg(layer.post_attention_layernorm,'pre_mlp_residual',True);reg(layer.post_attention_layernorm,'rmsnorm_output')
                        reg(layer.mlp.gate_proj,'gate_preactivation');reg(layer.mlp.up_proj,'up_projection')
                if not full:
                    def stop(m,a,o):raise PrefixComplete()
                    hooks.append(self.model.model.layers[8].mlp.down_proj.register_forward_hook(stop))
                try:
                    try:self.model(**pack)
                    except PrefixComplete:
                        assert not full
                finally:
                    for h in reversed(hooks):h.remove()
        c=6 if contexts else 1
        result={l:torch.cat(x).reshape(len(records),c,-1) for l,x in keys.items()}
        means={}
        with torch.no_grad():
            for l,k in result.items():
                kg=k.cuda();vals=[]
                for v in kg:
                    vals.append(torch.stack([v[:1].mean(0),v[1:].mean(0)],0).mean(0) if contexts else v[0])
                means[l]=torch.stack(vals).cpu();del kg,vals
        feature={k:torch.cat(x).reshape(len(records),c,-1) for k,x in points.items()}
        if features:
            for l in LAYERS:feature[f'L{l}/swiglu_product']=result[l]
        return dict(keys=result,means=means,features=feature,seconds=time.monotonic()-t,token_receipt=dict(sequences=len(texts),sha256=digest(token_rows),writer_add_bos=False,padding='right',batch_size=128))

    def current_requests(self,entry):
        return [dict(r['requested_rewrite'],case_id=int(r['case_id'])) for r in self.rows[entry*100:(entry+1)*100]]

    def local_sidecar(self,batch,name):
        p=Path('/data/janghj/ODE-edit/local/fixed10k-native-baselines/attempt-v1/output/main-cell-1')/f'B{batch:03d}'/name
        bindings=json.loads((self.design/'evidence/audits/global/2026-09-22-alphaedit-native-criticality-audit/target-native-bindings.json').read_text())
        match=next((x for x in bindings['files'] if x['path']==str(p)),None)
        if p.is_file() and match:
            assert file_sha(p)==match['sha256'];return json.loads(p.read_text())
        if match and 'raw' in match:
            import hashlib
            assert hashlib.sha256(match['raw'].encode()).hexdigest()==match['sha256']
            return json.loads(match['raw'])
        raise RuntimeError('SIDECAR_BINDING_UNAVAILABLE:'+str(p))
