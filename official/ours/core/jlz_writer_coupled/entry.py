"""One native entry pass: full-context keys, anchors, teacher, prefix cache."""
import time
import torch
from official.ours.common import require,digest
from .inputs import make_rows,batches
from .geometry import solve_layer_from_npz

def cpu(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
    if isinstance(value,tuple):return tuple(cpu(v) for v in value)
    return value

@torch.no_grad()
def prepare_entry(a,bench,pack,history,stats,microbatch=2):
    start=time.monotonic();rows=make_rows(pack);groups=[]
    raw={l:[] for l in a.sites};anchors={l:[] for l in a.sites};teachers={};kl_inputs={}
    for group,tokens in batches(rows,microbatch,bench.tokenizer.pad_token_id,a.device):
        seen={};handles=[];entry_block={}
        def pre_block(module,args,kwargs):
            entry_block['input']=args[0].detach()
            entry_block['kwargs']=cpu({k:v for k,v in kwargs.items() if k!='past_key_values'})
        handles.append(a.blocks[a.first].register_forward_pre_hook(pre_block,with_kwargs=True))
        handles.append(a.blocks[a.first].self_attn.register_forward_hook(lambda m,args,out:entry_block.update(attn=out[0].detach())))
        for l in a.sites:
            def key_hook(m,args,l=l):
                x=args[0];seen[l]=x.detach()
                if l==a.first:entry_block['key']=cpu(x)
            handles.append(a.blocks[l].mlp.down_proj.register_forward_pre_hook(key_hook))
            def anchor_hook(m,args,out,l=l):
                for j,r in enumerate(group):
                    if r['kind']=='rewrite' and r['global_row'] in pack['canonical_rows']:
                        anchors[l].append(out[j,r['lookup']].float().norm().cpu())
            handles.append(a.blocks[l].register_forward_hook(anchor_hook))
        try:
            _,final=a.full(tokens)
            for j,r in enumerate(group):
                if r['kind']=='kl':
                    teachers[r['request']]=a.head(final[j,r['lookup']]).log_softmax(-1).cpu()
                    kl_inputs[r['request']]=dict(input_ids=r['tokens']['input_ids'].tolist(),
                        attention_mask=r['tokens']['attention_mask'].tolist(),positions=list(range(len(r['tokens']['input_ids']))),readout=r['lookup'])
                else:
                    for l in a.sites:raw[l].append(seen[l][j,r['lookup']].cpu().clone())
            cache=dict(key=entry_block['key'],residual=cpu(entry_block['input']+entry_block['attn']),kwargs=entry_block['kwargs'])
            groups.append(dict(rows=group,tokens=cpu(tokens),cache=cache))
        finally:
            for h in handles:h.remove()
    B=pack['n_requests'];anchor={l:torch.stack(v).to(a.device) for l,v in anchors.items()}
    require(all(x.shape==(B,) and bool((x>0).all()) and bool(torch.isfinite(x).all()) for x in anchor.values()),'ENTRY_ANCHOR')
    P={};geometry={};keys={l:torch.stack(v).T for l,v in raw.items()}
    for l in a.sites:
        t=time.monotonic()
        P[l],geometry[l]=solve_layer_from_npz(stats[str(l)],history[l],keys[l].to(a.device),
            torch.tensor(pack['key_context_weights'],dtype=torch.float64,device=a.device),
            torch.tensor(pack['key_request'],device=a.device),B,lambda_c=a.profile['lambda_C'],residual_tolerance=1e-8)
        geometry[l]['seconds']=time.monotonic()-t
    return dict(pack=pack,groups=groups,P=P,anchors=anchor,teachers=teachers,kl_inputs=kl_inputs,
        entry_weights={l:w.detach().clone() for l,w in a.weights.items()},geometry=geometry,
        seconds=time.monotonic()-start,full_forward_rows=len(rows),entry_keys=keys)

@torch.no_grad()
def prepare_reference(a,bench,residents,microbatch=2):
    if not residents:return []
    pack=bench.prepare([r['record'] for r in residents]);rows=make_rows(pack)
    for r in rows:
        e=residents[r['request']]
        if r['kind']=='kl':
            frozen=e['kl_input']
            r['tokens']={k:torch.tensor(frozen[k],dtype=torch.long) for k in ('input_ids','attention_mask')}
            r['lookup']=frozen['readout'];r['target']=torch.full_like(r['tokens']['input_ids'],-100)
            require(frozen['positions']==list(range(len(r['target']))),'FROZEN_KL_POSITION_POLICY')
        else:
            r['context_index']=r['global_row']%(pack['n_rw']+1)
        r['n_rw']=pack['n_rw']
    return [dict(rows=g,tokens=cpu(t),cache=a.prefix(t)) for g,t in batches(rows,microbatch,bench.tokenizer.pad_token_id,a.device)]
