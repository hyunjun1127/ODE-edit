"""Whole-five-weight FP64 subtraction, immutable copy restoration, exact evaluator."""
import contextlib
import gc
import random
import sys
import time
from .common import *
from .bind import observation_modules

class Backend:
    def __init__(self,binding,family):
        sys.path.insert(0,str(DEPS))
        import torch,transformers,numpy as np
        from transformers import AutoModelForCausalLM,AutoTokenizer
        assert torch.__version__=='2.9.1+cu128' and transformers.__version__=='4.44.2'
        self.torch=torch;self.family=family;self.binding=binding;self.start=time.monotonic();self.forward_seconds=0.;self.calls=0;self.sequences=0;self.padded=0;self.nonpadding=0;self.upload_bytes=0;self.materialize_seconds=0;self.restore_seconds=0
        torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
        random.seed(20260924);np.random.seed(20260924);torch.manual_seed(20260924)
        self.obs=observation_modules()
        self.tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);self.tok.pad_token=self.tok.eos_token;self.tok.padding_side='right'
        self.model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,low_cpu_mem_usage=True,attn_implementation='eager',torch_dtype=torch.float32).cuda().eval()
        for p in self.model.parameters():p.requires_grad_(False)
        assert {p.dtype for p in self.model.parameters()}=={torch.float32}
        self.weights={k:self.model.get_parameter(k) for k in KEYS}
        self.w0={k:v.detach().cpu().clone() for k,v in self.weights.items()}
        assert {k:tensor_sha(v) for k,v in self.w0.items()}==binding['w0_selected']
        self.fixed={k:(p.data_ptr(),p._version) for k,p in self.model.named_parameters() if k not in KEYS}
        self.rng=torch.get_rng_state().clone();self.cuda_rng=torch.cuda.get_rng_state().clone()
        self.load_seconds=time.monotonic()-self.start

    def endpoint(self,t):
        if t==0:return self.w0
        r=self.binding['checkpoints'][self.family][str(t)];p=Path(r['path']);s=p.stat()
        assert (s.st_size,s.st_ino,s.st_mtime_ns)==(r['bytes'],r['inode'],r['mtime_ns']),'CHECKPOINT_STAT_DRIFT'
        x=self.torch.load(p,map_location='cpu',weights_only=True,mmap=True)
        return x['weights'] # only selected storages; history mappings are released

    def copy(self,ws):
        with self.torch.no_grad():
            for k in KEYS:self.weights[k].copy_(ws[k]);self.upload_bytes+=ws[k].numel()*4

    def hashes(self):return {k:tensor_sha(self.weights[k]) for k in KEYS}

    def unchanged(self):
        for k,p in self.model.named_parameters():
            if k in self.fixed:assert (p.data_ptr(),p._version)==self.fixed[k],('NONSELECTED_MUTATION',k)
        assert self.torch.equal(self.rng,self.torch.get_rng_state()) and self.torch.equal(self.cuda_rng,self.torch.cuda.get_rng_state()),'RNG_MUTATION'

    @contextlib.contextmanager
    def state(self,recipe,force_removal=None):
        """No edited state/delta is ever persisted. layer temporaries are bounded."""
        # T1 diagonal checks supply an ACTUAL recipe plus an explicit removal.
        # Resolve that override before reading counterfactual-only fields.
        if force_removal is not None:
            t,remove=force_removal
        elif recipe['kind']=='ACTUAL':
            t=int(recipe['actual_checkpoint']);remove=[]
        else:
            t=int(recipe['construction_endpoint'])
            remove=[int(i) for i in recipe['removed_cohort_indices'].split(';') if i!='']
        entry=self.endpoint(t);entry_hash={k:tensor_sha(v) for k,v in entry.items()};self.copy(entry)
        begin=time.monotonic();detail={}
        try:
            if remove:
                with self.torch.no_grad():
                    for k in KEYS:
                        w=entry[k].double()
                        for i in remove:
                            a=self.endpoint(TIMES[i]);b=self.endpoint(TIMES[i+1]);u=b[k].double()-a[k].double();w.sub_(u)
                            del a,b,u
                        actual=w.float();assert self.torch.isfinite(actual).all();self.weights[k].copy_(actual);self.upload_bytes+=actual.numel()*4
                        del w,actual;gc.collect()
            h=self.hashes();detail.update(weight_hashes=h,state_weight_hash=digest(h),entry_hash=entry_hash,endpoint=t,removed=remove,restore='NOT_YET')
            self.materialize_seconds+=time.monotonic()-begin
            yield detail
        finally:
            begin=time.monotonic();self.copy(entry);assert self.hashes()==entry_hash,'RESTORE_BYTES';self.unchanged()
            detail['restore']='EXACT_BYTES';self.restore_seconds+=time.monotonic()-begin;del entry;gc.collect()

    def evaluate(self,pairs,mb=16,evidence_sink=None):
        torch=self.torch;begin=time.monotonic()
        with torch.autocast(device_type='cuda',enabled=False):
            result=self.obs['evaluator'].evaluate_pairs(self.model,self.tok,pairs,device=torch.device('cuda'),microbatch_size=mb)
        torch.cuda.synchronize();self.forward_seconds+=time.monotonic()-begin;self.calls+=(len(pairs)+mb-1)//mb;self.sequences+=len(pairs)
        enc=self.obs['evaluator']._encode_pair
        lengths=[sum(map(len,enc(self.tok,p)))-1 for p in pairs]
        self.nonpadding+=sum(lengths);self.padded+=sum(max(lengths[i:i+mb])*len(lengths[i:i+mb]) for i in range(0,len(lengths),mb))
        if evidence_sink is not None:evidence_sink(result)
        assert all(__import__('math').isfinite(r['nll']) for r in result)
        return result

    def cost(self):
        import resource
        return dict(seconds=time.monotonic()-self.start,model_load_seconds=self.load_seconds,forward_seconds=self.forward_seconds,
            forward_calls=self.calls,target_sequences=self.sequences,padded_tokens=self.padded,nonpadding_tokens=self.nonpadding,
            selected_H2D_bytes=self.upload_bytes,materialize_seconds=self.materialize_seconds,restore_seconds=self.restore_seconds,
            peak_GPU_bytes=self.torch.cuda.max_memory_allocated(),max_host_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            native_fit=0,history_append=0,checkpoint_saved=False,timers_nested=True)
