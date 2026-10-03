"""Scalar-only diagnostics. No writer coefficients, bases or weight payloads."""
import time
import torch
from .common import require
from .subject import COEF


def ratio(x,y): return None if float(y)==0 else float(x/y)


@torch.no_grad()
def candidate(a,entry,R,q,built):
    rows=built['rows'];n=entry['pack']['n_rw'];B=entry['pack']['n_requests'];out={}
    owners=torch.tensor([r['request'] for r in rows],device=a.device)
    for l in a.sites:
        v=built['v'][l].detach();vnorm=v.norm(dim=1);rho=vnorm/entry['anchors'][l][owners]
        groups={}
        for kind in ('rewrite','kl'):
            ix=[i for i,r in enumerate(rows) if r['kind']==kind];x=rho[ix]
            groups[kind]=dict(count=len(ix),exceeds_075=int((x>.75).sum()),mean=float(x.mean()),max=float(x.max()),
                quantiles=torch.quantile(x,torch.tensor([.05,.5,.95],device=x.device)).tolist())
        Y=R[l].double()@built['geometry'][l]['P'].T@built['geometry'][l]['K']
        out[str(l)]=dict(R_norm=R[l].detach().norm(dim=0).tolist(),q_norm=q[l].detach().norm(dim=0).tolist(),
            row_identity=[(r['request'],r['global_row']%(n+1),r['kind']) for r in rows],
            v_norm=vnorm.tolist(),v_over_anchor=rho.tolist(),radius=groups,
            mean_key_rho=(Y.norm(dim=0)/entry['anchors'][l]).tolist(),
            native_group_weighted_context_rho=sum(entry['pack']['key_context_weights'][r['request']*n+r['global_row']%(n+1)]*float(rho[i]) for i,r in enumerate(rows) if r['kind']=='rewrite')/B)
    means={l:sum(x['mean_key_rho'])/B for l,x in out.items()};den=sum(means.values())
    ctx={l:x['native_group_weighted_context_rho'] for l,x in out.items()};dc=sum(ctx.values())
    for l,x in out.items():x.update(mean_share=ratio(means[l],den),context_share=ratio(ctx[l],dc))
    return out


@torch.no_grad()
def gradient_measure(R,scale,result):
    total=result['gradient'];B=next(iter(R.values())).shape[1]
    def vector_stats(gs, reference, variable):
        norm=torch.sqrt(sum(g.double().square().sum() for g in gs.values()))
        refnorm=torch.sqrt(sum(g.double().square().sum() for g in reference.values()))
        dot=sum((gs[l].double()*reference[l].double()).sum() for l in gs)
        radial=sum((gs[l].double()*variable[l].double()).sum() for l in gs)
        vn=torch.sqrt(sum(x.double().square().sum() for x in variable.values()))
        return dict(norm=float(norm),by_layer={str(l):float(g.double().norm()) for l,g in gs.items()},
                    cosine_total=ratio(dot,norm*refnorm),radial=ratio(radial,vn))
    qvar={l:R[l]/scale[l] for l in R};qtotal={l:B*g*scale[l] for l,g in total.items()}
    resultout=dict(R_mean=vector_stats(total,total,R),q_optimizer_SUM=vector_stats(qtotal,qtotal,qvar),components={})
    if result['components']:
        for name,gs in result['components'].items():
            weighted={l:COEF[name]*g for l,g in gs.items()}
            qraw={l:B*g*scale[l] for l,g in gs.items()};qw={l:COEF[name]*g for l,g in qraw.items()}
            resultout['components'][name]=dict(coefficient=COEF[name],R_raw=vector_stats(gs,total,R),
                R_weighted=vector_stats(weighted,total,R),q_raw=vector_stats(qraw,qtotal,qvar),q_weighted=vector_stats(qw,qtotal,qvar))
        combined={l:sum(COEF[k]*g[l] for k,g in result['components'].items()) for l in R}
        error=torch.sqrt(sum((combined[l]-total[l]).double().square().sum() for l in R)/sum(x.numel() for x in total.values()))
        reference=torch.sqrt(sum(x.double().square().sum() for x in total.values())/sum(x.numel() for x in total.values()))
        resultout['weighted_sum_RMS']=float(error);resultout['weighted_sum_limit']=float(1e-6+1e-3*reference)
        require(error<=1e-6+1e-3*reference,'COMPONENT_GRADIENT_SUM')
    return resultout


@torch.no_grad()
def effective_energy(entry,built):
    start=time.monotonic();out={}
    for l,W in built['weights'].items():
        A=entry['factors'][l]['A'].to(W.device);value=0.
        for begin in range(0,len(W),256):
            delta=(W[begin:begin+256]-entry['entry_weights'][l][begin:begin+256]).double()
            value+=float(((delta@A)*delta).sum())
        out[str(l)]=value;del A
    return dict(Q_effective=out,seconds=time.monotonic()-start,definition='tr((W_eff32-W_entry32) A (W_eff32-W_entry32)^T)',full_weight_saved=False)
