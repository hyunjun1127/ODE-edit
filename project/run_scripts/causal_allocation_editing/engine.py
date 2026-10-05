"""One actual all-token causal primal; whole-B streamed adjoint replay.

Only immutable prefix/entry metric are reusable between candidates. Boundaries,
K, P and FP32 payloads are candidate-local RAM; none is a disk checkpoint.
"""
import time
import torch
import torch.nn.functional as F
from project.run_scripts.jlz_native_writer_aware.physical import (
    Adapter as BaseAdapter, linear, materialize, move,
)
from project.run_scripts.jlz_native_writer_aware.routes import annotate
from project.run_scripts.jlz_realized_subject.subject import row_logprobs
from .geometry import mean_keys, ridge, compact_cost, cost_adjoint, solve_vjp


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def positions(group, device):
    rows = group['rows']
    return (torch.arange(len(rows), device=device),
            torch.tensor([r['lookup'] for r in rows], device=device),
            torch.tensor([r['global_row'] for r in rows], device=device))


class Adapter(BaseAdapter):
    def stage_full(self, layer, next_layer, key, residual, R, P, W, kw, route='direct'):
        def step(k, r, d, p, w):
            x = r + linear(k, d, p, w, route)
            nll = x if self.nll_layer == layer else x.new_empty(0)
            for j in range(layer+1, next_layer):
                x = self.unwrap(self.blocks[j](x, **kw))
                if j == self.nll_layer:
                    nll = x
            nk, nr = self.pre_projection(next_layer, x, kw)
            return nk, nr, nll
        return self.recompute(step, key, residual, R, P, W)

    def suffix(self, layer, key, residual, R, P, W, kw, route='direct'):
        def step(k, r, d, p, w):
            x = r + linear(k, d, p, w, route)
            nll = x if self.nll_layer == layer else x.new_empty(0)
            for j in range(layer+1, len(self.blocks)):
                x = self.unwrap(self.blocks[j](x, **kw))
                if j == self.nll_layer:
                    nll = x
            return nll, x
        return self.recompute(step, key, residual, R, P, W)


def selected_logits(a,rows,nh,fh):
    selected=[];widths=[]
    for index,row in enumerate(rows):
        pos=(torch.nonzero(row['target']!=-100).flatten().to(a.device) if row['kind']=='rewrite'
             else torch.tensor([row['lookup']],device=a.device))
        widths.append(len(pos));selected.extend((nh if row['kind']=='rewrite' else fh)[index,pos].unbind(0))
    return a.head(torch.stack(selected)),widths


def native_terms(a, entry, group, nh, fh,capture=None):
    """Owner-native token/context reduction; SUM across original requests."""
    n = entry['pack']['n_rw']
    terms = nh.reshape(-1)[0] * 0
    values = []
    logits,widths=selected_logits(a,group['rows'],nh,fh)
    if capture is not None:capture['selected_logits']=logits.detach().cpu()
    for row, lp in zip(group['rows'], logits.log_softmax(-1).split(widths)):
        if row['kind'] == 'rewrite':
            labels = row['target'][row['target'] != -100].to(a.device)
            value = -lp.gather(1, labels[:, None]).mean()
            terms = terms + value/n
        else:
            teacher = entry['teachers'][row['request']].to(a.device)
            value = (lp.exp() * (lp-teacher)).sum()
            terms = terms + float(a.profile.get('lambda_KL', .0625))*value
        values.append((row['kind'], row['request'], row['reduction_index'], float(value.detach())))
    require(bool(torch.isfinite(terms)), 'NONFINITE_NATIVE_TASK')
    return terms, values


class CausalObjective:
    def __init__(self, adapter, entry, events=None, route='direct'):
        self.a, self.entry, self.events, self.route = adapter, annotate(entry), events, route
        self.calls = dict(logical_candidates=0, primal_stage_groups=0,
                          suffix_groups=0, head_groups=0, reverse_stage_groups=0,
                          reverse_suffix_groups=0, head_backward_groups=0,
                          primal_solves=0, transpose_solves=0, gradient_channels=0)
        self.serial = 0
        require(adapter.nll_layer >= adapter.first, 'IMMUTABLE_EARLY_NLL_ADAPTER_UNSUPPORTED')

    def zeros(self):
        return {l:torch.zeros(self.a.dims[l][0], self.entry['pack']['n_requests'],
                              device=self.a.device, dtype=torch.float32) for l in self.a.sites}

    def _validate(self, u):
        require(tuple(u) == tuple(self.a.sites), 'U_LAYER_ORDER')
        B = self.entry['pack']['n_requests']
        for l, value in u.items():
            require(value.shape == (self.a.dims[l][0], B) and value.dtype == torch.float32
                    and bool(torch.isfinite(value).all()), 'U_SHAPE_DTYPE_FINITE')

    @torch.no_grad()
    def _primal(self, u, candidate, reference_R=None):
        a, entry = self.a, self.entry
        self._validate(u)
        groups = entry['groups']; rows = [r for g in groups for r in g['rows']]
        require([r['global_row'] for r in rows] == list(range(len(rows))), 'ROW_ORDER')
        rw = [i for i,r in enumerate(rows) if r['kind'] == 'rewrite']
        rwrows = [rows[i] for i in rw]
        R = {l:(u[l]*entry['anchors'][l][None,:]).detach().clone() for l in a.sites}
        # The one native endpoint is returned in R coordinates.  Do not alter
        # that endpoint through the lossy FP32 divide/multiply roundtrip.
        # This override is NEVER used by the main solver or a writer commit.
        if reference_R is not None:
            require(set(reference_R)==set(R), 'REFERENCE_R_LAYER_IDENTITY')
            for l in a.sites:
                value=reference_R[l]
                require(value.shape==R[l].shape and value.dtype==torch.float32
                        and bool(torch.isfinite(value).all()), 'REFERENCE_R_NATIVE_SCHEMA')
                R[l]=value.detach().to(a.device).clone()
        boundary = {a.first:[dict(key=g['cache']['key'], residual=g['cache']['residual']) for g in groups]}
        result = dict(candidate=candidate, entry_id=id(entry), u={l:x.detach().clone() for l,x in u.items()},
                      R=R, boundary=boundary, weights={}, P={}, K={}, raw={}, geometry={},
                      nll_hidden={}, final_hidden=[], nll_site={}, Q_by_layer={}, rows=rows,
                      cache_versions={},native_reference_exact_R=reference_R is not None)
        for index,l in enumerate(a.sites):
            keys=[]
            for group,b in zip(groups,boundary[l]):
                ix,pos,_=positions(group,'cpu')
                keys.append(b['key'][ix,pos].to(a.device))
            raw=torch.cat(keys); K=mean_keys(raw[rw].T,rwrows,entry['pack']).double()
            geo=ridge(K,entry['factors'][l]); self.calls['primal_solves']+=1
            P=geo['P']; W=materialize(entry['entry_weights'][l],R[l],P)
            require(bool(torch.isfinite(W).all()), 'NONFINITE_WEIGHT')
            result['weights'][l]=W.detach().cpu().clone(); result['K'][l]=K
            result['P'][l]=P; result['raw'][l]=raw.detach().cpu();result['geometry'][l]=geo['metadata']
            result['Q_by_layer'][l]=float(compact_cost(R[l],P,entry['factors'][l]['A']))
            result['cache_versions'][l]=(id(entry['factors'][l]['A']),entry['factors'][l]['A']._version,K._version,P._version)
            if index+1 < len(a.sites):
                nxt=a.sites[index+1]; boundary[nxt]=[]
                for gi,(group,b) in enumerate(zip(groups,boundary[l])):
                    c=move(b,a.device);kw=move(group['cache']['kwargs'],a.device)
                    nk,nr,nh=a.stage_full(l,nxt,c['key'],c['residual'],R[l],P,W,kw,self.route)
                    boundary[nxt].append(dict(key=nk.detach().cpu(),residual=nr.detach().cpu()))
                    if nh.numel():result['nll_hidden'][gi]=nh.detach().cpu(); result['nll_site'][gi]=l
                    self.calls['primal_stage_groups']+=1
            else:
                for gi,(group,b) in enumerate(zip(groups,boundary[l])):
                    c=move(b,a.device);kw=move(group['cache']['kwargs'],a.device)
                    nh,fh=a.suffix(l,c['key'],c['residual'],R[l],P,W,kw,self.route)
                    if nh.numel():result['nll_hidden'][gi]=nh.detach().cpu();result['nll_site'][gi]=l
                    result['final_hidden'].append(fh.detach().cpu());self.calls['suffix_groups']+=1
            del W
        B=entry['pack']['n_requests'];n=entry['pack']['n_rw']
        nll=torch.zeros(B,n,dtype=torch.float64);kl=torch.zeros(B,dtype=torch.float64)
        total=0.; seen=[]
        for gi,g in enumerate(groups):
            nh=result['nll_hidden'][gi].to(a.device);fh=result['final_hidden'][gi].to(a.device)
            terms,values=native_terms(a,entry,g,nh,fh);total+=float(terms);self.calls['head_groups']+=1
            seen.extend(r['global_row'] for r in g['rows'])
            for kind,owner,column,value in values:
                if kind=='rewrite':nll[owner,column]=value
                else:kl[owner]=value
        require(seen==list(range(len(rows))), 'NATIVE_LOSS_COVERAGE')
        result.update(task_sum=total,nll=nll,kl=kl,nll_sum=float(nll.mean(1).sum()),
                      kl_sum=float(kl.sum()),Q=sum(result['Q_by_layer'].values()),
                      G=(sum(float((R[l].double().norm(dim=0)*(.5/entry['anchors'][l].double().square())).sum()) for l in a.sites)
                         if reference_R is not None else
                         sum(float((u[l].double().norm(dim=0)*(.5/entry['anchors'][l].double())).sum()) for l in a.sites)))
        require(all(bool(torch.isfinite(torch.tensor(result[k],dtype=torch.float64))) for k in ('task_sum','Q','G')), 'NONFINITE_OBJECTIVE')
        return result

    def evaluate(self, u, gradient=False, lambda_Q=0., channels=False, candidate_id=None,reference_R=None):
        started=time.monotonic();self.serial+=1;self.calls['logical_candidates']+=1
        payload=self._primal(u,self.serial if candidate_id is None else candidate_id,reference_R)
        result={k:payload[k] for k in ('task_sum','nll_sum','kl_sum','nll','kl','Q','G')}
        result.update(smooth_sum=result['task_sum']+float(lambda_Q)*result['Q'],payload=payload,
                      per_request=payload['nll'].mean(1)+.0625*payload['kl'],gradient=None)
        if gradient or channels:result.update(self.backward(payload,lambda_Q,channels))
        result['seconds']=time.monotonic()-started;result['calls']=dict(self.calls)
        if self.events is not None:
            from .telemetry import candidate_summary
            self.events('candidate',candidate_summary(self.a,self.entry,payload,
                full_gradient=gradient or channels,lambda_Q=lambda_Q))
        return result

    def terminal_telemetry(self,payload):
        from .telemetry import candidate_summary
        result=candidate_summary(self.a,self.entry,payload,terminal=True,full_gradient=True)
        result.update(terminal_gradient_reused=True,terminal_extra_gradient=0,
                      gradient_policy='last ACCEPTED gradient already computed; telemetry adds no backward')
        return result

    def _head_adjoints(self,payload):
        a,entry=self.a,self.entry;out=[]
        for gi,g in enumerate(entry['groups']):
            nh=payload['nll_hidden'][gi].to(a.device).detach().requires_grad_(True)
            fh=payload['final_hidden'][gi].to(a.device).detach().requires_grad_(True)
            loss,_=native_terms(a,entry,g,nh,fh)
            dn,df=torch.autograd.grad(loss,(nh,fh),allow_unused=True)
            out.append((torch.zeros_like(nh).cpu() if dn is None else dn.detach().cpu(),
                        torch.zeros_like(fh).cpu() if df is None else df.detach().cpu()))
            self.calls['head_backward_groups']+=1
        return out

    def _reverse(self,payload,task_scale,cost_scale,route=None,cached=True,stop_solve=False):
        a,entry=self.a,self.entry;route=self.route if route is None else route
        require(payload['entry_id']==id(entry),'PAYLOAD_ENTRY_IDENTITY')
        head=self._head_adjoints(payload) if task_scale else None
        incoming=None;result={};ledger=[];rows=payload['rows']
        rw=[i for i,r in enumerate(rows) if r['kind']=='rewrite'];rwrows=[rows[i] for i in rw]
        for index in reversed(range(len(a.sites))):
            l=a.sites[index];factor=entry['factors'][l]
            require(payload['cache_versions'][l]==(id(factor['A']),factor['A']._version,payload['K'][l]._version,payload['P'][l]._version),'PAYLOAD_GEOMETRY_MUTATION')
            r=payload['R'][l].detach().requires_grad_(True);p=payload['P'][l].detach().requires_grad_(True)
            gr=torch.zeros_like(r);gp=torch.zeros_like(p);boundary_grad=[]
            fixed=payload['weights'][l].to(a.device)
            for gi,(g,b) in enumerate(zip(entry['groups'],payload['boundary'][l])):
                key=b['key'].to(a.device).detach().requires_grad_(True)
                res=b['residual'].to(a.device).detach().requires_grad_(True)
                kw=move(g['cache']['kwargs'],a.device)
                W=materialize(entry['entry_weights'][l],r,p) if route=='dense' else fixed
                outputs=[];seeds=[]
                if index+1<len(a.sites):
                    nk,nr,nh=a.stage_full(l,a.sites[index+1],key,res,r,p,W,kw,route)
                    outputs.extend((nk,nr));seeds.extend(t.to(a.device) for t in incoming[gi])
                    self.calls['reverse_stage_groups']+=1
                elif task_scale:
                    nh,fh=a.suffix(l,key,res,r,p,W,kw,route)
                    outputs.append(fh);seeds.append(head[gi][1].to(a.device)*task_scale)
                    self.calls['reverse_suffix_groups']+=1
                else:nh=key.new_empty(0)
                if task_scale and payload['nll_site'][gi]==l:
                    outputs.append(nh);seeds.append(head[gi][0].to(a.device)*task_scale)
                if outputs:
                    grads=torch.autograd.grad(outputs,(key,res,r,p),seeds,allow_unused=True)
                else:grads=(None,None,None,None)
                gk,gx,dr,dp=grads
                boundary_grad.append([torch.zeros_like(b['key']) if gk is None else gk.detach().cpu(),
                                      torch.zeros_like(b['residual']) if gx is None else gx.detach().cpu()])
                if dr is not None:gr.add_(dr)
                if dp is not None:gp.add_(dp)
            if cost_scale:
                dr,dp,_=cost_adjoint(r,p,factor['A']);gr.add_(dr,alpha=float(cost_scale));gp.add_(dp,alpha=float(cost_scale))
            gK,solve=solve_vjp(payload['K'][l],payload['P'][l],gp,factor,cached)
            self.calls['transpose_solves']+=1
            if stop_solve:gK=torch.zeros_like(gK)
            raw=payload['raw'][l].to(a.device).T.detach().requires_grad_(True)
            mean=mean_keys(raw[:,rw],rwrows,entry['pack']).double()
            grow=torch.autograd.grad(mean,raw,gK)[0].T.detach().cpu()
            for g,dst in zip(entry['groups'],boundary_grad):
                ix,pos,rid=positions(g,'cpu');dst[0][ix,pos]+=grow[rid]
            result[l]=gr.detach()*entry['anchors'][l][None,:]
            incoming=boundary_grad
            ledger.append(dict(layer=l,solve=solve,P_adjoint_norm=float(gp.norm()),K_adjoint_norm=float(gK.norm()),
                               gradient_norm=float(result[l].norm()),Q_once=True,whole_B=r.shape[1]))
        ordered={l:result[l] for l in a.sites}
        require(all(bool(torch.isfinite(t).all()) for t in ordered.values()),'NONFINITE_FULL_GRADIENT')
        return ordered,ledger

    def backward(self,payload,lambda_Q=0.,channels=False,route=None,cached=True,stop_solve=False):
        start=time.monotonic()
        if channels:
            gt,lt=self._reverse(payload,1.,0.,route,cached,stop_solve)
            gq,lq=self._reverse(payload,0.,1.,route,cached,stop_solve)
            gradient={l:gt[l]+float(lambda_Q)*gq[l] for l in self.a.sites}
            self.calls['gradient_channels']+=2
            if self.events is not None:
                self.events('gradient',dict(candidate=payload['candidate'],channels=['task','Q'],
                    norm_in_smooth_gradient=False,logical_candidate_added=False,
                    task_ledger=lt,Q_ledger=lq,gradient_norm={str(l):float(g.norm()) for l,g in gradient.items()}))
            return dict(gradient=gradient,task_gradient=gt,Q_gradient=gq,
                        reverse_ledger=dict(task=lt,Q=lq),reverse_seconds=time.monotonic()-start)
        gradient,ledger=self._reverse(payload,1.,float(lambda_Q),route,cached,stop_solve)
        self.calls['gradient_channels']+=1
        if self.events is not None:
            self.events('gradient',dict(candidate=payload['candidate'],channels=['task+lambda_Q*Q'],
                lambda_Q=lambda_Q,norm_in_smooth_gradient=False,logical_candidate_added=False,
                ledger=ledger,gradient_norm={str(l):float(g.norm()) for l,g in gradient.items()}))
        return dict(gradient=gradient,reverse_ledger=ledger,reverse_seconds=time.monotonic()-start)

    def dense_evaluate(self,u,lambda_Q=0.,stop_solve=False,native_full=False):
        """Tiny qualification reference; all original group shapes preserved."""
        a,entry=self.a,self.entry;leaves={l:x.detach().clone().requires_grad_(True) for l,x in u.items()}
        R={l:x*entry['anchors'][l][None,:] for l,x in leaves.items()}
        states=[move(dict(key=g['cache']['key'],residual=g['cache']['residual']),a.device) for g in entry['groups']]
        rows=[r for g in entry['groups'] for r in g['rows']];rw=[i for i,r in enumerate(rows) if r['kind']=='rewrite'];rwrows=[rows[i] for i in rw]
        Q=next(iter(leaves.values())).double().sum()*0;nhidden={};final=[];keys={};weights={}
        for index,l in enumerate(a.sites):
            subject=[]
            for g,b in zip(entry['groups'],states):
                ix,pos,_=positions(g,a.device);subject.append(b['key'][ix,pos])
            K=mean_keys(torch.cat(subject)[rw].T,rwrows,entry['pack']).double()
            if stop_solve:K=K.detach()
            # Dense means the complete causal/autograd graph and materialized
            # dW, not an obligation to retain five huge primal LU factors.
            # Same-A Woodbury is independently full-system tested on tiny raw
            # nonsymmetric fixtures; actual qualification compares its complete
            # autograd graph with the cached transpose replay route.
            P=(torch.linalg.solve(entry['factors'][l]['A'].to(a.device)+K@K.T,K)
               if native_full else ridge(K,entry['factors'][l])['P'])
            Q=Q+compact_cost(R[l],P,entry['factors'][l]['A'])
            W=materialize(entry['entry_weights'][l],R[l],P);keys[l]=K;weights[l]=W
            nxtstates=[]
            for gi,(g,b) in enumerate(zip(entry['groups'],states)):
                kw=move(g['cache']['kwargs'],a.device)
                if index+1<len(a.sites):
                    nk,nr,nh=a.stage_full(l,a.sites[index+1],b['key'],b['residual'],R[l],P,W,kw,'dense')
                    nxtstates.append(dict(key=nk,residual=nr))
                else:nh,fh=a.suffix(l,b['key'],b['residual'],R[l],P,W,kw,'dense');final.append(fh)
                if nh.numel():nhidden[gi]=nh
            states=nxtstates
        task=next(iter(leaves.values())).sum()*0
        for gi,g in enumerate(entry['groups']):task=task+native_terms(a,entry,g,nhidden[gi],final[gi])[0]
        loss=task+float(lambda_Q)*Q
        grad=torch.autograd.grad(loss,tuple(leaves.values()))
        return dict(smooth_sum=float(loss.detach()),task_sum=float(task.detach()),Q=float(Q.detach()),
                    gradient=dict(zip(leaves,(g.detach() for g in grad))),K={l:x.detach() for l,x in keys.items()},
                    weights={l:x.detach().cpu() for l,x in weights.items()},
                    geometry_reference='native_full_system' if native_full else 'same_raw_A_qualified_dual_autograd')
