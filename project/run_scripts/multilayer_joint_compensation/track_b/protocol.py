"""B finite correction path; shared fixture/evaluator are owned by SH1.

No target optimization, independent bank/history construction, output selector,
fallback, adaptive h or extra h on correction. All callback state is arm-local.
"""
from dataclasses import dataclass
from typing import Callable
import time
import torch
from ..functional import FunctionalPanel
from ..linear_solve import WeightTree, Operator, add, scale, zeros, finite, dot, norm
from ..elastic_qp import solve_constrained


ARMS={'B-OS':((8,),1,False), 'B-BF4':((8,),4,False),
      'B-Frozen-BF4':((8,),4,True), 'N4-L4-FUNCTIONAL-OS':((4,),1,False)}


@dataclass
class BProblem:
    """Fixed batch data and native geometry, no fixture creation in B runner.

    validate_state checks FP32 candidate bytes/shape and immutable live-model
    guard (including WN4 for B support8 arms). Inner forward is full-sequence
    functional application. Caller physically materializes the endpoint once
    AFTER run and verifies exact bytes/forward; that is not claimed here.
    """
    wn: WeightTree
    support: tuple[int,...]
    base: FunctionalPanel
    past: FunctionalPanel
    current: FunctionalPanel
    native_metric: Operator
    native_metric_inverse: Operator
    project: Operator
    native_risks: tuple[float,float]
    current_reference_nll: float
    validate_state: Callable[[WeightTree],dict]
    fixture_identity: dict


def _from_linear(row):
    return dict(value=row.value,mean_nll=row.mean_nll,context_rows=row.context_rows)


def _validate_state(problem,weights):
    if any(w.dtype!=torch.float32 for w in weights):raise TypeError('B_STATE_FP32_REQUIRED')
    receipt=problem.validate_state(weights)
    if not receipt.get('selected_fp32_shape_valid') or not receipt.get('nonselected_unchanged'):
        raise RuntimeError('B_ACTUAL_WEIGHT_OR_NONSELECTED_GUARD_FAILURE')
    if receipt.get('history_append_count',0)!=0 or receipt.get('compute_z_count',0)!=0:
        raise RuntimeError('B_INNER_STATE_CONTAMINATION')
    return receipt


def run(problem:BProblem,arm:str,*,on_node=None):
    """Return final WeightTree + node rows (raw local only). One batch only.

    Caller owns bounded ledger serialization, final endpoint evaluation and
    commit/history-once. This function never borrows another arm's endpoint.
    """
    if arm not in ARMS:raise ValueError('UNAUTHORIZED_B_ARM')
    support,steps,frozen=ARMS[arm]
    if problem.support!=support or len(problem.wn)!=len(support):raise ValueError('B_SUPPORT_MAPPING')
    if any(w.dtype!=torch.float32 for w in problem.wn):raise TypeError('B_FULL_FP32_REQUIRED')
    if (problem.base.role,problem.past.role,problem.current.role)!=('base','past','current'):
        raise ValueError('B_PANEL_ROLE_MISMATCH')
    if any(p.tau!=.1 for p in (problem.base,problem.past,problem.current)):raise ValueError('B_TAU_LOCK')
    if not all(torch.isfinite(torch.tensor(r,dtype=torch.float64)) for r in problem.native_risks):
        raise FloatingPointError('B_NONFINITE_NATIVE_RISK')
    h=1/steps;sigmas=tuple(max(r,1e-3) for r in problem.native_risks)
    weights=tuple(w.detach().clone() for w in problem.wn);rows=[];derivatives=None
    entry_validation=_validate_state(problem,weights)
    for node in range(steps):
        start=time.monotonic();s=node*h;counts0={name:dict(p.counts) for name,p in [('base',problem.base),('past',problem.past),('current',problem.current)]}
        if not frozen or derivatives is None:
            bl=problem.base.linearize(weights);pl=problem.past.linearize(weights)
            cl=problem.current.linearize(weights,need_nll_gradient=True)
            derivative_weights=weights  # Immutable tensors: next node allocates new blocks.
            derivatives=(derivative_weights,bl,pl,cl)
            observed=[_from_linear(x) for x in (bl,pl,cl)]
        else:
            derivative_weights,bl,pl,cl=derivatives
            observed=[p.observe(weights) for p in (problem.base,problem.past,problem.current)]
        g_base=problem.project(scale(bl.gradient,1/sigmas[0]))
        g_past=problem.project(scale(pl.gradient,1/sigmas[1]))
        a=problem.project(cl.nll_gradient)
        u=problem.project(add(add(scale(g_base,2),g_past),cl.gradient,1/h))
        def operator(direction):
            x=problem.project(direction)
            out=scale(problem.native_metric(x),.01/h)
            out=add(out,problem.base.ggn(derivative_weights,x),2/sigmas[0])
            out=add(out,problem.past.ggn(derivative_weights,x),1/sigmas[1])
            out=add(out,problem.current.ggn(derivative_weights,x),1/h)
            return finite(problem.project(out))
        def precondition(direction):
            return problem.project(scale(problem.native_metric_inverse(problem.project(direction)),h/.01))
        # Actual current deficit uses the same WN teacher at every B node.
        t_e=0. if steps==1 else -max(observed[2]['mean_nll']-problem.current_reference_nll,0.)
        raw_budget=((1-.5*(s+h))*problem.native_risks[0],problem.native_risks[1])
        risks=tuple(o['value']/sigma for o,sigma in zip(observed[:2],sigmas))
        budgets=tuple(r/sigma for r,sigma in zip(raw_budget,sigmas))
        result=solve_constrained(operator,u,a,t_e,precondition=precondition,
            gradients=(g_base,g_past) if steps>1 else None,
            risks=risks if steps>1 else None,budgets=budgets if steps>1 else None,h=h)
        delta=finite(result.correction)
        permitted=problem.project(delta)
        leakage=norm(add(delta,permitted,-1))/(norm(delta)+torch.finfo(torch.float64).tiny)
        next_weights=finite(tuple((w+d).detach() for w,d in zip(weights,delta)))
        # C is finite displacement, h appears in K/slack already: no extra h.
        validation_receipt=_validate_state(problem,next_weights)
        after=[p.observe(next_weights) for p in (problem.base,problem.past,problem.current)]
        actual_delta=tuple(n-w for n,w in zip(next_weights,weights))
        actual_violations=tuple(o['value']/sigma-budget for o,sigma,budget in zip(after[:2],sigmas,budgets))
        count_delta={name:{k:p.counts[k]-counts0[name][k] for k in p.counts}
            for name,p in [('base',problem.base),('past',problem.past),('current',problem.current)]}
        row=dict(arm=arm,node=node,s=s,h=h,source_fixture=problem.fixture_identity,
            support=list(support),frozen_derivative_node=0 if frozen else node,
            before=observed,after=after,native_risks=list(problem.native_risks),sigmas=list(sigmas),
            normalized_risks=risks,raw_budget=raw_budget,normalized_budget=budgets,
            actual_budget_violation=actual_violations if steps>1 else None,
            current_reference_nll=problem.current_reference_nll,t_e=t_e,
            predicted_current_change=dot(a,delta),actual_current_change=after[2]['mean_nll']-observed[2]['mean_nll'],
            predicted_risk_change=[dot(g_base,delta),dot(g_past,delta)],
            actual_risk_change=[after[j]['value']/sigmas[j]-risks[j] for j in range(2)],
            correction_norm=norm(delta),actual_delta_norm=norm(actual_delta),
            total_delta_from_wn=norm(add(next_weights,problem.wn,-1)),
            native_action=dot(delta,problem.native_metric(delta)),
            actual_native_action=dot(actual_delta,problem.native_metric(actual_delta)),
            permitted_range_relative_residual=leakage,
            fp32_addition_residual_norm=norm(add(actual_delta,delta,-1)),
            solver=result.diagnostics,state_validation=validation_receipt,entry_validation=entry_validation if node==0 else None,
            counts=count_delta,wall_seconds=time.monotonic()-start,
            compute_z=0,history_append=0,adaptive_gain=0,rollback=0,semantic_filter=0,
            endpoint_selected_by_performance=False,inner_physical_assignment=0,
            forward_boundary='FULL_SEQUENCE_FUNCTIONAL_FP32_SELECTED_WEIGHTS',
            terminal_materialization='CALLER_REQUIRED_AFTER_RUN')
        rows.append(row)
        if on_node is not None:on_node(row,next_weights,actual_delta)
        weights=next_weights
    return weights,rows
