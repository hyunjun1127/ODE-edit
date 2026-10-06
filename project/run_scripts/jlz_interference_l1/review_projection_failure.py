"""Regression of one preserved actual PRICE failure, not a synthetic suite."""
import argparse,hashlib,json,math
from pathlib import Path
from . import require,member,write
from .projection import _lengths

def review(error_path,original_projection,out):
    error_path,original_projection,out=map(Path,(error_path,original_projection,out))
    require(not out.exists(),'CREATE_ONCE_FAILURE_REGRESSION')
    evidence=json.loads(error_path.read_text())
    require(evidence['error']=='PROJECTION_SORTED_BREAKPOINT_ROOT_UNAVAILABLE','EXACT_REAL_FAILURE')
    v=evidence['operands'];ns,ws,ds,beta=(v[k] for k in ('proposal_norm','weights','caps','beta'))
    require(all(math.isfinite(x) for x in ns+ws+ds+[beta]),'FINITE_ORIGINAL_OPERANDS')
    scope={};exec(compile(original_projection.read_text(),str(original_projection),'exec'),scope)
    old_error=None
    try:scope['_lengths'](ns,ws,ds,beta)
    except RuntimeError as e:old_error=str(e)
    require(old_error==evidence['error'],'REPRODUCE_ORIGINAL_EXACT_FAILURE')
    ts,tau,on,convention=_lengths(ns,ws,ds,beta)
    formula=[min(d,max(n-tau*w,0.)) for n,w,d in zip(ns,ws,ds)]
    spend=sum(w*t for w,t in zip(ws,ts));delta=spend-beta
    limits=dict(primal=1e-10*max(1,beta),complementarity=1e-10*max(1,tau*beta))
    require(ts==formula and tau>=0 and on,'UNCHANGED_LENGTH_FORMULA')
    require(abs(delta)<=limits['primal'] and abs(tau*delta)<=limits['complementarity'],'SEALED_FP64_KKT')
    require(all(0<=t<=d for t,d in zip(ts,ds)),'LOCAL_LENGTH_CAP')
    result=dict(status='PASS_REAL_FAILURE_SCALAR_REGRESSION',failure=member(error_path),
        original_source=member(original_projection),repaired_source=member(Path(__file__).with_name('projection.py')),
        original_failure_reproduced=old_error,original_operands=v,projected_norm=ts,tau=tau,convention=convention,
        spend=spend,spend_minus_beta=delta,complementarity=abs(tau*delta),fixed_limits=limits,
        method_changed=False,tolerances_changed=False,new_toy_fixtures=0,model_load=0,GPU_calls=0,
        actual_full_FP32_payload_check='NOT_REPLAYED_NO_CHECKPOINT; runtime same gates retained')
    write(out,result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--error',required=True);p.add_argument('--original-projection',required=True)
    p.add_argument('--out',required=True);a=p.parse_args();print(json.dumps(review(a.error,a.original_projection,a.out)))
