"""Independent arithmetic at saved local coefficients; no new optimizer run."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .review_b1 import dump
from .review_completed_b1 import read, member


def audit(mu,j,receipt):
    radius=receipt['radius'];scales=np.asarray(receipt['row_scales'],dtype=np.float64)
    if len(scales)!=len(mu):raise ValueError('B1_EMPTY_HISTORY_CURRENT_LINEAR_ROWS_ONLY')
    matrix=radius*j/scales[:,None];lower=(np.minimum(mu,0.)-mu)/scales
    rows=[]
    for phase in ('phase1','phase2'):
        p=receipt[phase];a=np.asarray(p['coefficients'],dtype=np.float64);x=a/radius
        margin=mu+j@a;negative=np.minimum(margin,0.);risk=float(np.mean(negative**2))
        rg=2*(radius*j).T@negative/len(mu)
        gradient=rg if phase=='phase1' else 2*x
        slack=matrix@x-lower;ball_slack=1-float(x@x)
        residual=gradient.copy();complementarity=[];nonzero=[]
        dual=p['dual_audit'];risk_scale=max(float(receipt['initial_reference_risk']),1e-12,
            float(receipt['phase1']['certification_gap_tolerance']))
        for name,multiplier in zip(dual['labels'],dual['multipliers'],strict=True):
            if multiplier<0 or not np.isfinite(multiplier):raise ValueError('INVALID_SAVED_DUAL')
            if name.startswith('linear:'):
                index=int(name.split(':')[1]);grad=-matrix[index];value=-slack[index]
            elif name=='ball':grad=2*x;value=-ball_slack
            elif name=='risk_bound':grad=rg/risk_scale;value=(risk-p['risk_limit'])/risk_scale
            else:raise ValueError('UNKNOWN_DUAL_ROW')
            residual+=multiplier*grad;complementarity.append(abs(multiplier*value))
            if multiplier:nonzero.append(dict(label=name,multiplier=multiplier))
        if not np.isclose(risk,p['risk'],rtol=1e-12,atol=1e-14):raise ValueError('RISK_ARITHMETIC')
        if not np.allclose(residual,dual['stationarity_vector'],rtol=1e-10,atol=1e-12):
            raise ValueError('STATIONARITY_ARITHMETIC')
        rows.append(dict(phase=phase,coefficients=a.tolist(),risk=risk,min_scaled_slack=float(slack.min()),
            ball_slack=ball_slack,stationarity_l2=float(np.linalg.norm(residual)),
            complementarity_max=max(complementarity,default=0.),nonzero_duals=nonzero))
    return rows


def run(output,destination):
    out=Path(output)
    if not (out/'terminal.json').exists():raise ValueError('TERMINAL_ONLY')
    mu=np.asarray([r['mu'] for r in read(out/'B1/reference-native.json')['rows']],dtype=np.float64)
    results={}
    for arm in ('DEC_LINE','DEC_MODES_CUM'):
        root=out/'B1/arms'/arm;s=read(root/'selection-ledger.json');receipt=s['details'].get('solver')
        if not receipt or s['details']['solver_status']!='LOCAL_TWO_PHASE_SOLVED':
            results[arm]=dict(status='NO_CERTIFIED_TWO_PHASE_SOLUTION',source_status=s['stop_reason']);continue
        path=root/'jacobians.pt';j=torch.load(path,map_location='cpu',weights_only=True,mmap=True)['J'].numpy()
        results[arm]=dict(status='SAVED_COEFFICIENT_ARITHMETIC_MATCH',phases=audit(mu,j,receipt),input=member(path))
    dump(Path(destination),dict(results=results,new_solver_runs=0,new_model_calls=0,
        scientific_claim='LOCAL_LINEARIZED_PROBLEM_ONLY; NO_GLOBAL_NONLINEAR_OPTIMUM'))
    return results


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--destination',required=True)
    a=p.parse_args();print(json.dumps(run(a.output,a.destination)))
