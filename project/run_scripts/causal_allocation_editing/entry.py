"""Task-local native full-input entry capture; no future-KL crop shortcut."""
import time
import torch
from project.run_scripts.jlz_realization.inputs import batches
from project.run_scripts.jlz_writer_coupled.entry import cpu
from . import require,tensor_sha

def make_native_rows(pack):
    rows=[]
    for i,kind in enumerate(pack['row_kind']):
        n=int(pack['tokens']['attention_mask'][i].sum())
        require(n>0 and 0<=pack['lookup'][i]<n,'NATIVE_FULL_ROW_LOOKUP')
        tokens={k:v[i,:n].clone() for k,v in pack['tokens'].items()}
        rows.append(dict(tokens=tokens,lookup=pack['lookup'][i],target=pack['targets'][i,:n].clone(),
            kind=kind,request=pack['row_request'][i],global_row=i))
    return rows

@torch.no_grad()
def prepare_entry(a,bench,pack,history,stats,requests_per_group=1):
    """Unchanged native owner grouping/anchor FP32/teacher; full tokens retained."""
    start=time.monotonic();rows=make_native_rows(pack);groups=[]
    anchors={l:[] for l in a.sites};hidden={l:[] for l in a.sites};teachers={}
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
            for handle in handles:handle.remove()
    anchors={l:torch.stack(v).to(a.device) for l,v in anchors.items()}
    require(all(v.shape==(pack['n_requests'],) and bool((v>0).all()) and bool(torch.isfinite(v).all())
                for v in anchors.values()),'ANCHORS')
    require(a.profile['anchor_layer'] in anchors,'UNSUPPORTED_EXTERNAL_ANCHOR')
    return dict(pack=pack,groups=groups,anchors=anchors,entry_hidden={l:torch.stack(v).T for l,v in hidden.items()},
        teachers=teachers,stats=stats,seconds=time.monotonic()-start,
        teacher_hash={r:tensor_sha(t) for r,t in teachers.items()},
        entry_weights={l:w.detach().clone() for l,w in a.weights.items()},
        input_policy='ORIGINAL_NATIVE_FULL_REWRITE_AND_KL_TOKENS')
