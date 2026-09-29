"""Task-local five-layer adapter; official writer/dependencies are read-only."""
import ast
import copy
import importlib.util
import inspect
import os
import random
import sys
import time
from pathlib import Path
from .common import *


class Trace:
    def __init__(self,native):
        self.native=native;self.rows=[];self.losses=[];self.history_calls=0;self.target_calls=0
        self.key_calls=0;self.key_rows=[];self.post_key=None;self.native_receipt={}
        lines,start=inspect.getsourcelines(native.compute_z)
        nodes=ast.walk(ast.parse(''.join(lines)))
        self.loss_line=next(start+n.lineno-1 for n in nodes if isinstance(n,ast.If) and isinstance(n.test,ast.Compare)
            and isinstance(n.test.left,ast.Name) and n.test.left.id=='loss')

    def __call__(self,frame,event,arg):
        code=frame.f_code; d=frame.f_locals
        if code is self.native.compute_z.__code__:
            if event=='call':self.target_calls+=1;self.losses=[]
            if event=='line' and frame.f_lineno==self.loss_line:
                self.losses.append(dict(iteration=int(d['it']),nll=float(d['nll_loss'].detach()),kl=float(d['kl_loss'].detach()),
                    decay=float(d['weight_decay'].detach()),loss=float(d['loss'].detach())))
            if event=='return' and arg is not None:
                step=d['opt'].state.get(d['delta'],{}).get('step',0)
                self.rows.append(dict(case_id=int(d['request']['case_id']),layer=int(d['layer']),target_norm=float(arg.detach().norm()),
                    anchor_norm=float(d['target_init'].detach().norm()),value_delta_norm=float(d['delta'].detach().norm()),
                    loss_evaluations=int(d['it'])+1,adam_updates=int(step),lookup_indices=list(d['lookup_idxs']),
                    input_ids=d['input_tok']['input_ids'].detach().cpu().tolist(),attention_mask=d['input_tok']['attention_mask'].detach().cpu().tolist(),
                    target_ids=d['target_ids'].detach().cpu().tolist(),losses=self.losses))
            return self
        if code is self.native.compute_ks.__code__:
            if event=='call':self.key_calls+=1
            if event=='return' and arg is not None:self.key_rows.append((d['layer'],arg.detach().cpu().clone().T))
            return self
        if code is self.native.apply_AlphaEdit_to_model.__code__:
            if event=='return' and arg is not None:
                self.native_receipt=dict(residual_norm=float(d['resid'].detach().norm()),divisors=[5,4,3,2,1],history_append_calls=5)
            return self
        return None

    def __enter__(self):
        require(sys.gettrace() is None,'EXISTING_TRACE');sys.settrace(self);return self

    def __exit__(self,*args):sys.settrace(None)


class Runtime:
    def __init__(self,config,checkpoint,repo):
        import numpy as np
        import torch
        self.torch=torch;self.config=config;self.layer=4;self.checkpoint=checkpoint;self.repo=Path(repo)
        for member in config['model_members']:
            s=Path(member['path']).stat()
            require(s.st_size==member['bytes'] and s.st_mtime_ns==member['mtime_ns'],'MODEL_MEMBER_STAT_CHANGED')
        for member in config['native_dependencies']:
            require(sha(member['path'])==member['sha256'],'NATIVE_DEPENDENCY_CHANGED:'+member['path'])
        sys.path.insert(0,str(DEPS));sys.path.insert(0,str(NATIVE));os.chdir(NATIVE)
        from transformers import AutoTokenizer,AutoModelForCausalLM
        import transformers
        require(transformers.__version__=='4.44.2' and torch.__version__=='2.9.1+cu128','RUNTIME')
        torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
        source=Path(config['official_native']['path']);require(sha(source)==config['official_native']['sha256'],'NATIVE_SOURCE')
        # Relative imports resolve exclusively to the already sealed original dependency tree.
        name='AlphaEdit.joint_official_native';spec=importlib.util.spec_from_file_location(name,source)
        self.native=importlib.util.module_from_spec(spec);sys.modules[name]=self.native;spec.loader.exec_module(self.native)
        from AlphaEdit.AlphaEdit_hparams import AlphaEditHyperParams
        self.hp=AlphaEditHyperParams(**dict(config['hparams'],layers=list(LAYERS),blue=False,L2=10))
        require(self.hp.layers==list(LAYERS) and not self.hp.blue and self.hp.L2==10,'FIVE_LAYER_CONTRACT')
        start=time.monotonic()
        self.model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,low_cpu_mem_usage=True,
            attn_implementation='eager',torch_dtype=torch.float32).cuda().eval();self.model.requires_grad_(False)
        self.model.config.use_cache=False
        require({p.dtype for p in self.model.parameters()}=={torch.float32},'MODEL_PRECISION')
        self.tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);self.tok.add_bos_token=False;self.tok.pad_token_id=self.tok.eos_token_id
        require(self.tok.padding_side=='right','PADDING')
        self.weights={l:self.model.model.layers[l].mlp.down_proj.weight for l in LAYERS}
        self.w0={l:w.detach().cpu().clone() for l,w in self.weights.items()}
        r=config['checkpoints'][checkpoint];p=Path(r['path']);st=p.stat()
        require((st.st_size,st.st_ino,st.st_mtime_ns)==(r['bytes'],r['inode'],r['mtime_ns']),'CP_CHANGED')
        self.cp=torch.load(p,map_location='cpu',weights_only=True,mmap=True)
        self.M=self.cp['cache_c'].clone();self.P=torch.load(config['projector']['path'],map_location='cpu',weights_only=True,mmap=True)
        pr=config['projector'];ps=Path(pr['path']).stat()
        require((ps.st_size,ps.st_ino,ps.st_mtime_ns)==(pr['bytes'],pr['inode'],pr['mtime_ns']),'PROJECTOR_FILE_CHANGED')
        self.contexts=copy.deepcopy(self.cp['metadata']['contexts']);self.native.CONTEXT_TEMPLATES_CACHE=copy.deepcopy(self.contexts)
        self.parent_hash={l:r['weights'][f'model.layers.{l}.mlp.down_proj.weight'] for l in LAYERS}
        self.history_parent=[tensor_sha(self.M[i]) for i in range(5)]
        self.fixed={n:(p.data_ptr(),p._version) for n,p in self.model.named_parameters() if n not in {f'model.layers.{l}.mlp.down_proj.weight' for l in LAYERS}}
        self.model_load_seconds=time.monotonic()-start;self.fits=0;self.loss_evals=0;self.adam=0;self.native_seconds=0
        self.set_rng();self.rng_entry=self.rng_get()

    def versions(self):return {n:(p.data_ptr(),p._version) for n,p in self.model.named_parameters()}

    def rng_get(self):
        import numpy as np
        return random.getstate(),np.random.get_state(),self.torch.get_rng_state(),self.torch.cuda.get_rng_state_all()

    def rng_set(self,x):
        import numpy as np
        random.setstate(x[0]);np.random.set_state(x[1]);self.torch.set_rng_state(x[2]);self.torch.cuda.set_rng_state_all(x[3])

    def set_rng(self):
        import numpy as np
        t=self.torch;r=self.cp['metadata']['rng'];n=r['numpy'];p=r['python']
        random.setstate((p[0],tuple(p[1]),p[2]));np.random.set_state((n[0],np.asarray(n[1],dtype=np.uint32),n[2],n[3],n[4]))
        t.set_rng_state(t.tensor(r['torch'],dtype=t.uint8))
        # One logical GPU: restore parent logical-device0 RNG, other original devices not allocated here.
        require(bool(r['cuda']),'PARENT_CUDA_RNG');t.cuda.set_rng_state(t.tensor(r['cuda'][0],dtype=t.uint8))

    def set_weights(self,which):
        with self.torch.no_grad():
            for l,w in self.weights.items():
                src=self.w0[l] if which=='W0' else self.cp['weights'][f'model.layers.{l}.mlp.down_proj.weight']
                w.copy_(src)


    def snapshot(self):
        return {l:w.detach().cpu().clone() for l,w in self.weights.items()}

    def restore(self,weights):
        with self.torch.no_grad():
            for l,w in self.weights.items():w.copy_(weights[l])

    def hashes(self):return {str(l):tensor_sha(w) for l,w in self.weights.items()}

    def check_fixed(self):
        require(all((p.data_ptr(),p._version)==self.fixed[n] for n,p in self.model.named_parameters() if n in self.fixed),'NONEDITABLE_MUTATION')

    def keys(self,requests,l):
        with self.torch.no_grad():
            return self.native.compute_ks(self.model,self.tok,requests,self.hp,l,self.contexts).T.detach()

    def append(self,requests):
        rows=[]
        for l in LAYERS:
            k=self.keys(requests,l).cpu();before=tensor_sha(self.M[l-4])
            self.M[l-4].add_(k@k.T)
            require(bool(self.torch.isfinite(self.M[l-4]).all()),'HISTORY_NONFINITE')
            rows.append(dict(layer=l,requests=len(requests),key_sha=tensor_sha(k),before=before,after=tensor_sha(self.M[l-4]),append_count=1))
        return rows

    def fit_batch(self,requests):
        t=self.torch;require(len(requests)==BATCH_SIZE,'NATIVE_USER_BS1');m_before=self.M.clone();before=self.hashes()
        trace=Trace(self.native);start=time.monotonic()
        self.fits+=len(requests);require(self.fits<=STEPS,'FIT_BUDGET')
        with trace:
            model,history=self.native.apply_AlphaEdit_to_model(self.model,self.tok,requests,self.hp,
                 cache_template=None,cache_c=self.M,P=self.P)
        require(model is self.model and history.data_ptr()==self.M.data_ptr(),'NATIVE_RETURN_OWNER')
        require(trace.target_calls==BATCH_SIZE and trace.key_calls==10 and len(trace.rows)==BATCH_SIZE,'NATIVE_COUNTS')
        hist=[]
        for l,k in trace.key_rows[5:]:
            require(t.equal(m_before[l-4]+k@k.T,self.M[l-4]),'HISTORY_EXACTLY_ONCE')
            hist.append(dict(layer=l,requests=BATCH_SIZE,key_sha=tensor_sha(k),before=tensor_sha(m_before[l-4]),after=tensor_sha(self.M[l-4]),append_count=1))
        self.check_fixed()
        require(all(bool(t.isfinite(w).all()) for w in self.weights.values()),'NATIVE_NONFINITE')
        self.loss_evals+=sum(r['loss_evaluations'] for r in trace.rows);self.adam+=sum(r['adam_updates'] for r in trace.rows)
        require(self.loss_evals<=25*self.fits and self.adam<=24*self.fits,'NATIVE_LOSS_ADAM_BUDGET')
        return dict(targets=trace.rows,history=hist,weight_before=before,weight_after=self.hashes(),
            native_calls=1,fit_requests=BATCH_SIZE,divisors=[5,4,3,2,1],seconds=time.monotonic()-start)
