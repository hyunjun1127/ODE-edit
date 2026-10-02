"""One entry capture. No upper writer solve or state from older methods."""
import time
import torch
from project.run_scripts.jlz_writer_coupled.entry import cpu, prepare_reference
from .inputs import make_rows, batches
from .common import require
from .dynamic_solve import prior

@torch.no_grad()
def prepare_entry(a, bench, pack, history, stats, microbatch=2):
    start = time.monotonic(); rows = make_rows(pack); groups = []
    raw = {l: [] for l in a.sites}; anchors = {l: [] for l in a.sites}
    teachers = {}; kl_inputs = {}
    for group, tokens in batches(rows, microbatch, bench.tokenizer.pad_token_id, a.device):
        seen = {}; handles = []; first = {}
        def before(m, args, kw):
            first['input'] = args[0].detach()
            first['kwargs'] = cpu({k:v for k,v in kw.items() if k != 'past_key_values'})
        handles.append(a.blocks[a.first].register_forward_pre_hook(before, with_kwargs=True))
        handles.append(a.blocks[a.first].self_attn.register_forward_hook(lambda m,args,out:first.update(attn=out[0].detach())))
        for l in a.sites:
            def key(m, args, l=l): seen[l] = args[0].detach()
            def anchor(m, args, out, l=l):
                for j,r in enumerate(group):
                    if r['global_row'] in pack['canonical_rows']:
                        anchors[l].append(out[j,r['lookup']].float().norm().cpu())
            handles.append(a.blocks[l].mlp.down_proj.register_forward_pre_hook(key))
            handles.append(a.blocks[l].register_forward_hook(anchor))
        try:
            _, final = a.full(tokens)
            for j,r in enumerate(group):
                if r['kind'] == 'kl':
                    teachers[r['request']] = a.head(final[j,r['lookup']]).log_softmax(-1).cpu()
                    kl_inputs[r['request']] = dict(input_ids=r['tokens']['input_ids'].tolist(),
                        attention_mask=r['tokens']['attention_mask'].tolist(),
                        positions=list(range(len(r['tokens']['input_ids']))), readout=r['lookup'])
                else:
                    for l in a.sites: raw[l].append(seen[l][j,r['lookup']].cpu().clone())
            groups.append(dict(rows=group,tokens=cpu(tokens),cache=dict(key=cpu(seen[a.first]),
                residual=cpu(first['input']+first['attn']),kwargs=first['kwargs'])))
        finally:
            for h in handles: h.remove()
    B = pack['n_requests']; anchor = {l:torch.stack(v).to(a.device) for l,v in anchors.items()}
    require(all(x.shape == (B,) and bool((x > 0).all()) and bool(torch.isfinite(x).all()) for x in anchor.values()), 'ENTRY_ANCHOR')
    factors = {}; metadata = {}
    for l in a.sites:
        factors[l], metadata[l] = prior(stats[str(l)],history[l],a.device,a.profile['lambda_C'])
    return dict(pack=pack,groups=groups,anchors=anchor,teachers=teachers,kl_inputs=kl_inputs,
                entry_weights={l:w.detach().clone() for l,w in a.weights.items()},
                entry_keys={l:torch.stack(v).T for l,v in raw.items()}, factors=factors,
                first_geometry=None, geometry=metadata, seconds=time.monotonic()-start)
