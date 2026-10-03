"""Request-MEAN values and request-SUM gradients; no physical/past term."""
import time
import torch
from project.run_scripts.jlz_realization.subject import row_logprobs
from project.run_scripts.jlz_realization.profile import LlamaAdapter as CachedAdapter, move
from .common import require

class Adapter(CachedAdapter):
    native_route = 'cached'

    def native(self, group, D, capture=False):
        if self.native_route == 'cached':
            return super().native(group, D, capture)
        # Same full model reference. Hook lifetime includes backward because
        # this route does not use checkpoint/recompute.
        rows = group['rows']; ix = torch.arange(len(rows), device=self.device)
        positions = torch.tensor([r['lookup'] for r in rows], device=self.device)
        owners = torch.tensor([r['request'] for r in rows], device=self.device)
        handles, subjects = [], {}
        for layer in self.sites:
            def inject(module, args, out, layer=layer):
                result = out.clone()
                result[ix, positions] = result[ix, positions] + D[layer][:, owners].T
                if capture: subjects[layer] = result[ix, positions]
                return result
            handles.append(self.blocks[layer].register_forward_hook(inject))
        try:
            nll, final = self.full(move(group['tokens'], self.device))
            return nll, final, subjects
        finally:
            for handle in handles: handle.remove()

def native(adapter, entry, D, backward=False, components=False):
    started = time.monotonic(); B = entry['pack']['n_requests']; contexts = entry['pack']['n_rw']
    totals = dict(nll=0., kl=0., norm=0.)
    per_request={name:[0.]*B for name in totals}
    gradients = {name:{l:torch.zeros_like(d) for l,d in D.items()} for name in totals} if components else None
    calls, prediction_tokens, valid_tokens, padded_tokens = 0,0,0,0
    with torch.set_grad_enabled(backward):
        for group in entry['groups']:
            nh, fh, _ = adapter.native(group,D)
            probs = row_logprobs(adapter,group['rows'],nh,fh)
            terms = {name:nh.reshape(-1)[0]*0 for name in ('nll','kl')}
            for row, lp in zip(group['rows'],probs):
                prediction_tokens += len(lp)
                if row['kind'] == 'rewrite':
                    target = row['target'][row['target'] != -100].to(adapter.device)
                    value=-lp.gather(1,target[:,None]).mean()/contexts
                    terms['nll'] = terms['nll'] + value
                    per_request['nll'][row['request']]+=float(value.detach())
                else:
                    teacher = entry['teachers'][row['request']].to(adapter.device)
                    value=.0625*(lp.exp()*(lp-teacher)).sum()
                    terms['kl'] = terms['kl'] + value
                    per_request['kl'][row['request']]+=float(value.detach())
            for name, value in terms.items():
                require(bool(torch.isfinite(value)), 'NONFINITE_NATIVE_'+name)
                totals[name] += float(value.detach())/B
            if backward:
                if components:
                    for name,value in terms.items():
                        g = torch.autograd.grad(value,tuple(D.values()),retain_graph=name=='nll',allow_unused=True)
                        for (l,d),v in zip(D.items(),g):
                            if v is not None: gradients[name][l].add_(v)
                        calls += 1
                else:
                    sum(terms.values()).backward(); calls += 1
            valid_tokens += int(group['tokens']['attention_mask'].sum())
            padded_tokens += group['tokens']['input_ids'].numel()
        norm = .5*sum((torch.linalg.vector_norm(d,dim=0)/entry['anchors'][l].square()).sum() for l,d in D.items())
        require(bool(torch.isfinite(norm)), 'NONFINITE_NATIVE_NORM')
        totals['norm'] = float(norm.detach())/B
        per_request['norm']=(.5*sum(d.detach().norm(dim=0)/entry['anchors'][l].square() for l,d in D.items())).cpu().tolist()
        if backward:
            if components:
                gs = torch.autograd.grad(norm,tuple(D.values()))
                for l,v in zip(D,gs): gradients['norm'][l].copy_(v)
                for l,d in D.items(): d.grad = sum(gradients[n][l] for n in gradients)
            else: norm.backward()
            calls += 1
    return dict(mean=sum(totals.values()),components=totals,per_request=per_request,gradients=gradients,
                seconds=time.monotonic()-started,forward_groups=len(entry['groups']),backward_calls=calls,
                prediction_tokens=prediction_tokens,valid_tokens=valid_tokens,padded_tokens=padded_tokens)
