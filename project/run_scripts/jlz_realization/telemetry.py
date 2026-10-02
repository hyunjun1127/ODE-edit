"""Vectorized realization diagnostics; only approved M/D/Y tensors persist."""
import os
import numpy as np
import torch
from .common import require,member,write,tensor_sha

@torch.no_grad()
def save_diagnostics(path,tensors):
    require(set(tensors)=={'M','planned_D','realized_Y'},'TELEMETRY_ALLOWLIST')
    path.parent.mkdir(parents=True,exist_ok=True)
    require(not path.exists(),'TELEMETRY_CREATE_ONCE')
    temp=path.with_suffix('.tmp')
    arrays={k:v.detach().cpu().numpy() for k,v in tensors.items()}
    with temp.open('xb') as f:
        np.savez(f,**arrays);f.flush();os.fsync(f.fileno())
    os.link(temp,path);temp.unlink()
    with np.load(path,allow_pickle=False) as reread:
        require(all(np.array_equal(reread[k],v) for k,v in arrays.items()),'TELEMETRY_RELOAD')
    return dict(**member(path),tensors={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in arrays.items()},
        owner='SH4',purpose='entry/terminal realization diagnostic; NOT resume',checkpoint=False)

@torch.no_grad()
def measure(D,built,entry,out,candidate):
    layers={};planned=[];realized=[];artifacts=[]
    for l,d32 in D.items():
        d=d32.double();geo=built['geometry'][l];k=geo['K'];p=geo['P'];M=p.T@k;Y=d@M
        n=d.norm(dim=0);yn=Y.norm(dim=0);valid=n>0
        gamma=torch.full_like(n,float('nan'));ratio=gamma.clone();orth=gamma.clone();fit=gamma.clone()
        gamma[valid]=(d*Y).sum(0)[valid]/n[valid].square();ratio[valid]=yn[valid]/n[valid]
        orth[valid]=(Y[:,valid]-gamma[valid]*d[:,valid]).norm(dim=0)/n[valid]
        fit[valid]=(Y[:,valid]-d[:,valid]).norm(dim=0)/n[valid]
        self_action=d*M.diagonal();cross=Y-self_action
        actual_weight=(built['weights'][l].double()-entry['entry_weights'][l].double())@k
        actual_forward=(torch.nn.functional.linear(k.T.float(),built['weights'][l])-
                        torch.nn.functional.linear(k.T.float(),entry['entry_weights'][l])).T.double()
        def values(x):return [None if not np.isfinite(v) else v for v in x.cpu().tolist()]
        layers[str(l)]=dict(rho=(n/entry['anchors'][l].double()).tolist(),gamma=values(gamma),norm_ratio=values(ratio),
            orthogonal_relative=values(orth),fit_relative=values(fit),zero_D=(~valid).tolist(),Y_norm=yn.tolist(),
            M_diagonal=M.diagonal().tolist(),M_offdiagonal_F=float((M-torch.diag(M.diagonal())).norm()),
            self_norm=self_action.norm(dim=0).tolist(),cross_norm=cross.norm(dim=0).tolist(),self_cross_dot=(self_action*cross).sum(0).tolist(),
            Y64_vs_materialized_weight_RMS=float((Y-actual_weight).square().mean().sqrt()),
            materialized_weight_vs_Flinear_RMS=float((actual_weight-actual_forward).square().mean().sqrt()),
            actual_write_norm=float((built['weights'][l]-entry['entry_weights'][l]).norm()),
            kappa_entry_change=float((k-entry['mean_keys'][l].to(k)).norm()))
        planned.append(n);realized.append(yn)
        if candidate in (1,25):
            artifacts.append(save_diagnostics(out/f'candidate-{candidate:02d}-L{l}.npz',dict(M=M,planned_D=d32,realized_Y=Y)))
    for name,parts in (('planned_layer_share',planned),('realized_layer_share',realized)):
        stack=torch.stack(parts);den=stack.sum(0);share=stack/den.clamp_min(1e-300)
        for i,l in enumerate(D):layers[str(l)][name]=[float(v) if bool(ok) else None for v,ok in zip(share[i].cpu(),(den>0).cpu())]
    return layers,artifacts

@torch.no_grad()
def context_decomposition(a,entry,builder,D,rows,hidden,keys,teachers):
    selected=[j for j,r in enumerate(rows) if r['kind']=='rewrite']
    if not selected:return {}
    owners=torch.tensor([rows[j]['request'] for j in selected],device=a.device)
    result={}
    for l in a.sites:
        k=keys[l][selected].T.double();d=D[l][:,owners].double()
        kbar=builder['geometry'][l]['K'][:,owners];p=builder['P'][l]
        # Avoid creating another full d_out x d_in update.
        own_ideal=D[l].double()@(p.T@k);mean=D[l].double()@(p.T@kbar)
        own_fp=(torch.nn.functional.linear(k.T.float(),builder['weights'][l])-
                torch.nn.functional.linear(k.T.float(),entry['entry_weights'][l])).T.double()
        virtual=torch.stack([teachers[rows[j]['global_row']]['hidden'][l] for j in selected]).to(a.device).T.double()
        h=hidden[l][selected].T.double();pre=h-own_fp;gap=virtual-pre
        parts=[gap-d,d-mean,mean-own_ideal];summed=sum(parts);target=gap-own_ideal
        require(torch.allclose(summed,target,atol=1e-9,rtol=1e-8),'DECOMPOSITION_IDENTITY')
        result[str(l)]=dict(rows=[rows[j]['global_row'] for j in selected],
            norms=torch.stack([x.norm(dim=0) for x in parts]).T.cpu().tolist(),
            cross_dots=torch.stack([(parts[0]*parts[1]).sum(0),(parts[0]*parts[2]).sum(0),(parts[1]*parts[2]).sum(0)]).T.cpu().tolist(),
            sum_error_max=float((summed-target).abs().max()),virtual_actual_gap=(virtual-h).norm(dim=0).tolist(),
            ideal_minus_actual_own_RMS=float((own_ideal-own_fp).square().mean().sqrt()))
    return result
