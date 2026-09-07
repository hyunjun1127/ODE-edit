"""Same-state, config-bound shadows; never a coefficient authority for writes."""
import torch
from project.run_scripts.native_response_ode_v31.algebra import nnls_response, turning
from project.run_scripts.alpha_native_response_ode_v31_sequential.telemetry import (
    controller_matrices, initial_history_gram, layer_actions, actual_delta_rows)


def same_state_shadows(e, psi, metric, gf, q, layers, c, config, history_metric=None):
    lam=config.lambda_response
    g,h=psi.T@e,psi.T@psi
    singles=[]
    for i,layer in enumerate(layers):
        s=torch.zeros_like(c);s[i]=g[i].clamp(min=0)/(h[i,i]+lam*metric[i,i])
        objective=float(.5*s@(h+lam*metric)@s-g@s)
        singles.append(dict(layer=layer,c=s.tolist(),objective=objective,
            objective_improvement=-objective,**turning(c,s,metric),
            frobenius_angle=turning(c,s,gf),physical_coefficient_difference=((c-s)/q.sqrt()).tolist()))
    result=dict(single_layer=singles,lambda_response=lam,config=config.receipt(),
        best_single_layer=min(singles,key=lambda x:(x['objective'],x['layer']))['layer'] if singles else None,
        joint_objective_improvement=float(g@c-.5*c@(h+lam*metric)@c),
        extra_forward_count=0,extra_jvp_count=0,controller_influence_count=0)
    if history_metric is not None and len(c):
        s=nnls_response(e,psi,history_metric,lam)
        result['history_cost']=dict(policy='INITIAL_HISTORY_COST_FIXED_BASIS',lambda_response=lam,
            G_initial=history_metric.tolist(),G_actual=metric.tolist(),c_actual=c.tolist(),c_initial=s.coefficients.tolist(),
            physical_actual=(c/q.sqrt()).tolist(),physical_initial=(s.coefficients/q.sqrt()).tolist(),
            actual_predicted_progress=float(g@c),shadow_predicted_progress=float(g@s.coefficients),
            **turning(c,s.coefficients,metric),frobenius_angle=turning(c,s.coefficients,gf),controller_influence_count=0)
    return result
