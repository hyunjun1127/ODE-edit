"""All injected native sites, including KL; history remains rewrite only."""
import torch
from project.run_scripts.jlz_shared_budget.entry import Adapter as FrozenAdapter
from project.run_scripts.jlz_realization.profile import move
from project.run_scripts.jlz_realization.geometry import mean_keys
from .common import require,tensor_sha,digest

class Adapter(FrozenAdapter):
    """Observe existing planner forward, without changing its graph or calls."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.last_virtual={};self.capture_virtual=False
        self.calls=dict(full_adapter_forward=0,native_cached_forward=0,native_full_forward=0,
                        layer_function_first=0,layer_function_recompute=0)
    def full(self,*args,**kwargs):
        self.calls['full_adapter_forward']+=1
        return super().full(*args,**kwargs)
    def recompute(self,fn,*args):
        invoked=0
        def measured(*values):
            nonlocal invoked
            self.calls['layer_function_recompute' if invoked else 'layer_function_first']+=1
            invoked+=1
            return fn(*values)
        return super().recompute(measured,*args)
    def native(self,group,D,capture=False):
        self.calls['native_cached_forward' if self.native_route=='cached' else 'native_full_forward']+=1
        result=super().native(group,D,capture)
        if self.capture_virtual and capture:
            require(len({r['request'] for r in group['rows']})==1,'FROZEN_G1_TERMINAL_CAPTURE')
            for j,row in enumerate(group['rows']):
                self.last_virtual[row['global_row']]={l:h[j].detach().cpu().clone() for l,h in result[2].items()}
        return result

@torch.no_grad()
def capture_native_sites(a,entry,layers):
    keys={l:[] for l in layers};hidden={l:[] for l in layers};rows=[];valid=padded=0
    for group in entry['groups']:
        ks={};hs={};handles=[]
        for l in layers:
            handles.append(a.blocks[l].mlp.down_proj.register_forward_pre_hook(lambda m,args,l=l:ks.update({l:args[0]})))
            handles.append(a.blocks[l].register_forward_hook(lambda m,args,out,l=l:hs.update({l:out})))
        try:
            tokens=move(group['tokens'],a.device);a.full(tokens)
            valid+=int(tokens['attention_mask'].sum());padded+=tokens['input_ids'].numel()
            for j,row in enumerate(group['rows']):
                rows.append(row)
                for l in layers:
                    keys[l].append(ks[l][j,row['lookup']].detach().cpu().clone())
                    hidden[l].append(hs[l][j,row['lookup']].detach().cpu().clone())
        finally:
            for h in handles:h.remove()
    require([r['global_row'] for r in rows]==list(range(len(entry['pack']['row_kind']))),'NATIVE_ROW_ORDER')
    raw={l:torch.stack(v).T for l,v in keys.items()}
    rew=[j for j,r in enumerate(rows) if r['kind']=='rewrite'];rwrows=[rows[j] for j in rew]
    return dict(keys=raw,hidden={l:torch.stack(v).T for l,v in hidden.items()},rows=rows,
        mean={l:mean_keys(raw[l][:,rew],rwrows,entry['pack']) for l in layers},
        calls=len(entry['groups']),valid_tokens=valid,padded_tokens=padded)

def virtual_terminal(a,entry,plan):
    order=list(range(len(entry['pack']['row_kind'])))
    require(set(a.last_virtual)==set(order),'ALL_NATIVE_TERMINAL_CAPTURE')
    result={l:torch.stack([a.last_virtual[i][l] for i in order]).T for l in a.sites}
    for l in a.sites:
        require(torch.equal(result[l][:,entry['pack']['canonical_rows']],plan['z'][l]),'SAME_EVALUATED_TERMINAL_Z')
    return result

def cache_identity(entry):
    def encode(v):
        if isinstance(v,torch.Tensor):return dict(shape=list(v.shape),dtype=str(v.dtype),sha256=tensor_sha(v))
        if isinstance(v,dict):return {str(k):encode(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)):return [encode(x) for x in v]
        return v
    return digest(encode(dict(groups=entry['groups'],teachers=entry['teachers'],anchors=entry['anchors'],pack_identity=entry['pack']['identity'])))
