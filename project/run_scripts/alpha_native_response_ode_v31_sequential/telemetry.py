"""Small same-state observers; none of their results feed the physical write."""
import torch
from project.run_scripts.native_response_ode_v31.algebra import nnls_response, turning
from .contracts import LAMBDA


def restricted_l8(e, psi, layers):
    c=torch.zeros(len(layers),dtype=torch.float64)
    if 8 in layers:
        i=layers.index(8);g=psi[:,i]@e;h=psi[:,i]@psi[:,i]
        c[i]=g.clamp(min=0)/(h+LAMBDA)
    return c


def controller_matrices(e,psi,metric,c):
    g=psi.T@e;h=psi.T@psi
    return dict(g=g.tolist(),full_H=h.tolist(),G=metric.tolist(),
        predicted_target_contribution=(g*c).tolist(),
        response_residual_cosine=[float(g[i]/(e.norm()*psi[:,i].norm()))
            if float(e.norm()*psi[:,i].norm())>0 else None for i in range(len(c))])


def initial_history_gram(dictionary,active,q):
    # This task uses source cold M0=0, certified at chain initialization.
    # Evaluate the L2 physical Gram in ACTUAL coordinates, with actual qref fixed.
    # Disjoint editable weight blocks have zero cross-layer physical inner product.
    lam=float(dictionary.family.hparams.L2)
    return torch.diag(torch.tensor([lam*dictionary.raw(b,frobenius=True)/
        (float(q[i])*dictionary.qref) for i,b in enumerate(active)],dtype=torch.float64))


def same_state_shadows(e,psi,metric,gf,q,layers,c,history_metric=None):
    g,h=psi.T@e,psi.T@psi
    singles=[]
    for i,l in enumerate(layers):
        s=torch.zeros_like(c);s[i]=g[i].clamp(min=0)/(h[i,i]+LAMBDA)
        objective=float(.5*s@(h+LAMBDA*metric)@s-g@s)
        singles.append(dict(layer=l,c=s.tolist(),objective=objective,
            objective_improvement=-objective,**turning(c,s,metric),
            frobenius_angle=turning(c,s,gf),coordinate_difference=(c-s).tolist(),
            physical_coefficient_difference=((c-s)/q.sqrt()).tolist()))
    result=dict(single_layer=singles,best_single_layer=min(singles,key=lambda x:(x['objective'],x['layer']))['layer'] if singles else None,
        joint_objective_improvement=float(g@c-.5*c@(h+LAMBDA*metric)@c),
        extra_forward_count=0,extra_jvp_count=0,controller_influence_count=0)
    if history_metric is not None and len(c):
        s=nnls_response(e,psi,history_metric,LAMBDA)
        result['history_cost']=dict(policy='INITIAL_HISTORY_COST_FIXED_BASIS',
            G_initial=history_metric.tolist(),G_actual=metric.tolist(),
            c_actual=c.tolist(),c_initial=s.coefficients.tolist(),
            physical_actual=(c/q.sqrt()).tolist(),physical_initial=(s.coefficients/q.sqrt()).tolist(),
            actual_predicted_progress=float(g@c),shadow_predicted_progress=float(g@s.coefficients),
            **turning(c,s.coefficients,metric),frobenius_angle=turning(c,s.coefficients,gf),
            basis_response_normalization_qref_frozen=True,controller_influence_count=0)
    return result


def layer_actions(dictionary,active,q,c):
    rows=[]
    for i,b in enumerate(active):
        physical=float(c[i]/q[i].sqrt());frob=dictionary.raw(b,frobenius=True)
        raw=dictionary.raw(b);l2=float(dictionary.family.hparams.L2)*frob
        rows.append(dict(layer=b.layer,raw_coefficient=physical,coefficient=float(c[i]),
            q=float(q[i]),direction_raw_action=raw,direction_frobenius_squared=frob,
            raw_native_velocity_action=physical**2*raw,normalized_native_velocity_action=float(c[i]**2),
            history_velocity_action=physical**2*(raw-l2),L2_velocity_action=physical**2*l2,
            frobenius_velocity_squared=physical**2*frob,
            history_action_computation='raw_native_action_minus_L2_frobenius_action'))
    return rows


def actual_delta_rows(current,previous,entry,names):
    rows=[]
    for l,name in names.items():
        d=current[name].cpu().double()-previous[name].cpu().double()
        net=current[name].cpu().double()-entry[name].cpu().double()
        rows.append(dict(layer=l,actual_step_DeltaW_squared=float(d.square().sum()),
                         actual_net_DeltaW_squared=float(net.square().sum()),
                         actual_nonzero=int((d!=0).sum())))
    return rows
