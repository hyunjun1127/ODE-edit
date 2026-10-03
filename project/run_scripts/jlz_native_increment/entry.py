"""Native entry capture with scalar-only provenance; all tensors are RAM-only."""
import torch
from project.run_scripts.jlz_realization.entry import prepare_entry as capture_entry
from project.run_scripts.jlz_realization.profile import move
from .common import tensor_sha,require

@torch.no_grad()
def prepare_entry(a,bench,pack,history,stats,microbatch=2):
    entry=capture_entry(a,bench,pack,history,stats,microbatch)
    hidden={l:[] for l in a.sites}
    for group in entry['groups']:
        found={};handles=[]
        for l in a.sites:
            handles.append(a.blocks[l].register_forward_hook(lambda m,args,out,l=l:found.update({l:out})))
        try:
            a.full(move(group['tokens'],a.device))
            for j,r in enumerate(group['rows']):
                if r['kind']=='rewrite':
                    for l in a.sites:hidden[l].append(found[l][j,r['lookup']].cpu().clone())
        finally:
            for h in handles:h.remove()
    entry['entry_hidden']={l:torch.stack(v).T for l,v in hidden.items()}
    entry['entry_diagnostic_forward_groups']=len(entry['groups'])
    entry['teacher_hash']={r:tensor_sha(v) for r,v in entry['teachers'].items()}
    return entry
