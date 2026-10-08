"""Fresh native complete-owner entry capture; capture L8 even for L4-only."""
import time
import torch
from official.ours.core.jlz_realization.inputs import batches
from official.ours.core.jlz_writer_coupled.entry import cpu
from official.ours.core.jlz_realized_subject.geometry import mean_keys
from .geometry import prior
from official.ours.common import require, tensor_sha
from official.ours.core.jlz_native_writer_aware.routes import annotate


def native_rows(pack):
    rows=[]
    for i,kind in enumerate(pack['row_kind']):
        n=int(pack['tokens']['attention_mask'][i].sum())
        require(n>0 and 0<=pack['lookup'][i]<n,'NATIVE_FULL_ROW_LOOKUP')
        rows.append(dict(tokens={k:v[i,:n].clone() for k,v in pack['tokens'].items()},
            target=pack['targets'][i,:n].clone(),lookup=pack['lookup'][i],kind=kind,
            request=pack['row_request'][i],global_row=i))
    return rows


@torch.no_grad()
def prepare_entry(a,bench,pack,history,stats,requests_per_group=1):
    require(requests_per_group==1,'COMPLETE_SINGLE_OWNER_GRAPH')
    start=time.monotonic();rows=native_rows(pack);groups=[]
    anchor_layer=a.profile['anchor_layer'];capture_sites=sorted(set(a.sites)|{anchor_layer})
    anchors={l:[] for l in capture_sites};hidden={l:[] for l in capture_sites}
    raw={l:[] for l in a.sites};teachers={};kl_inputs={}
    for rowgroup,tokens in batches(rows,pack['n_rw']+1,bench.tokenizer.pad_token_id,a.device):
        require(len({r['request'] for r in rowgroup})==1,'COMPLETE_OWNER_GROUP')
        first={};found={};keys={};handles=[]
        def before(m,args,kw):
            first['x']=args[0].detach();first['kwargs']=cpu({k:v for k,v in kw.items() if k!='past_key_values'})
        handles.append(a.blocks[a.first].register_forward_pre_hook(before,with_kwargs=True))
        handles.append(a.attention(a.first).register_forward_hook(lambda m,args,out:first.update(attn=out[0].detach())))
        for l in a.sites:
            handles.append(a.projection(l).register_forward_pre_hook(lambda m,args,l=l:keys.update({l:args[0].detach()})))
        for l in capture_sites:
            handles.append(a.blocks[l].register_forward_hook(lambda m,args,out,l=l:found.update({l:a.unwrap(out)})))
        try:
            _,final=a.full(tokens)
            for j,r in enumerate(rowgroup):
                if r['global_row'] in pack['canonical_rows']:
                    for l in capture_sites:
                        h=found[l][j,r['lookup']].detach().clone()
                        require(h.dtype==torch.float32,'NATIVE_ANCHOR_FP32')
                        hidden[l].append(h.cpu());anchors[l].append(h.norm().cpu())
                if r['kind']=='kl':
                    teachers[r['request']]=a.head(final[j,r['lookup']]).log_softmax(-1).cpu()
                    kl_inputs[r['request']]=dict(input_ids=r['tokens']['input_ids'].tolist(),
                        attention_mask=r['tokens']['attention_mask'].tolist(),readout=r['lookup'])
                else:
                    for l in a.sites:raw[l].append(keys[l][j,r['lookup']].cpu().clone())
            selected=[]
            for j,r in enumerate(rowgroup):
                pos=(torch.nonzero(r['target']!=-100).flatten().to(a.device) if r['kind']=='rewrite'
                     else torch.tensor([r['lookup']],device=a.device))
                selected.append(final[j,pos].detach().cpu())
            groups.append(dict(rows=rowgroup,tokens=cpu(tokens),native_c0_pending=True,
                native_c0_selected=selected,cache=dict(key=cpu(keys[a.first]),
                residual=cpu(a.entry_residual(first['x'],first['attn'])),kwargs=first['kwargs'])))
        finally:
            for handle in handles:handle.remove()
    B=pack['n_requests'];anchors={l:torch.stack(v).to(a.device) for l,v in anchors.items()}
    require(all(v.shape==(B,) and bool(torch.isfinite(v).all()) and bool((v>0).all())
        for v in anchors.values()),'NATIVE_POSITIVE_FINITE_ANCHORS')
    factors={};metadata={}
    for l in a.sites:factors[l],metadata[l]=prior(a,stats[str(l)],history[l],l)
    rwrows=[r for g in groups for r in g['rows'] if r['kind']=='rewrite']
    entry=dict(pack=pack,groups=groups,anchors=anchors,anchor_star=anchors[anchor_layer],
        entry_hidden={l:torch.stack(v).T for l,v in hidden.items()},teachers=teachers,kl_inputs=kl_inputs,
        mean_keys={l:mean_keys(torch.stack(v).T,rwrows,pack) for l,v in raw.items()},
        entry_keys={l:torch.stack(v).T for l,v in raw.items()},factors=factors,geometry=metadata,
        first_geometry={},entry_weights={l:w.detach().clone() for l,w in a.weights.items()},
        history_entry={l:history[l] for l in a.sites},teacher_hash={r:tensor_sha(t) for r,t in teachers.items()},
        seconds=time.monotonic()-start,capture_sites=capture_sites,
        input_policy='ORIGINAL_NATIVE_FULL_REWRITE_AND_KL_COMPLETE_OWNER')
    return annotate(entry)
