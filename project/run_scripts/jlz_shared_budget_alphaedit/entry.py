"""Whole native requests per graph; anchors captured once in native FP32."""
import time
import torch
from project.run_scripts.jlz_realization.inputs import make_rows,batches
from project.run_scripts.jlz_realization.profile import LlamaAdapter,move
from project.run_scripts.jlz_writer_coupled.entry import cpu
from .common import require,tensor_sha

class Adapter(LlamaAdapter):
    native_route='cached'
    def native(self,group,D,capture=False):
        if self.native_route=='cached':return super().native(group,D,capture)
        rows=group['rows'];ix=torch.arange(len(rows),device=self.device)
        pos=torch.tensor([r['lookup'] for r in rows],device=self.device)
        owners=torch.tensor([r['request'] for r in rows],device=self.device)
        handles=[];found={}
        for l in self.sites:
            def hook(m,args,out,l=l):
                y=out.clone();y[ix,pos]=y[ix,pos]+D[l][:,owners].T
                if capture:found[l]=y[ix,pos]
                return y
            handles.append(self.blocks[l].register_forward_hook(hook))
        try:
            nh,fh=self.full(move(group['tokens'],self.device));return nh,fh,found
        finally:
            for h in handles:h.remove()

@torch.no_grad()
def prepare_entry(a,bench,pack,history,stats,requests_per_group=1):
    start=time.monotonic();rows=make_rows(pack);groups=[];anchors={l:[] for l in a.sites};hidden={l:[] for l in a.sites};teachers={}
    size=(pack['n_rw']+1)*requests_per_group
    for rowgroup,tokens in batches(rows,size,bench.tokenizer.pad_token_id,a.device):
        first={};found={};handles=[]
        def before(m,args,kw):
            first['x']=args[0].detach();first['kwargs']=cpu({k:v for k,v in kw.items() if k!='past_key_values'})
        handles.append(a.blocks[a.first].register_forward_pre_hook(before,with_kwargs=True))
        handles.append(a.blocks[a.first].self_attn.register_forward_hook(lambda m,args,out:first.update(attn=out[0].detach())))
        handles.append(a.blocks[a.first].mlp.down_proj.register_forward_pre_hook(lambda m,args:first.update(key=args[0].detach())))
        for l in a.sites:
            handles.append(a.blocks[l].register_forward_hook(lambda m,args,out,l=l:found.update({l:out})))
        try:
            _,final=a.full(tokens)
            for j,r in enumerate(rowgroup):
                if r['global_row'] in pack['canonical_rows']:
                    for l in a.sites:
                        h=found[l][j,r['lookup']].detach().clone()
                        hidden[l].append(h.cpu());anchors[l].append(h.norm().cpu())
                if r['kind']=='kl':teachers[r['request']]=a.head(final[j,r['lookup']]).log_softmax(-1).cpu()
            groups.append(dict(rows=rowgroup,tokens=cpu(tokens),cache=dict(key=cpu(first['key']),
                residual=cpu(first['x']+first['attn']),kwargs=first['kwargs'])))
        finally:
            for h in handles:h.remove()
    anchors={l:torch.stack(v).to(a.device) for l,v in anchors.items()}
    require(all(v.shape==(pack['n_requests'],) and bool((v>0).all()) and bool(torch.isfinite(v).all()) for v in anchors.values()),'ANCHORS')
    require(a.profile['anchor_layer'] in anchors,'UNSUPPORTED_EXTERNAL_ANCHOR')
    return dict(pack=pack,groups=groups,anchors=anchors,entry_hidden={l:torch.stack(v).T for l,v in hidden.items()},
        teachers=teachers,stats=stats,seconds=time.monotonic()-start,
        teacher_hash={r:tensor_sha(t) for r,t in teachers.items()},
        entry_weights={l:w.detach().clone() for l,w in a.weights.items()})
