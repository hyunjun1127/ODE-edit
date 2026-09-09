"""Independent NLL-pair reducer, not importing the execution evaluator/reducer."""
from collections import defaultdict
import math
import numpy as np
from .common import MULT,digest,stats

def preferred(new,true,tag):
    assert tag in MULT and math.isfinite(new) and math.isfinite(true)
    return true<new if tag=='NS' else new<true

def reduce_rows(rows,tag):
    n=len(rows);assert n
    groups=defaultdict(list)
    bits=[]
    for r in rows:
        b=preferred(r['new_nll'],r['true_nll'],tag)
        assert b==r['success']
        assert abs(r['margin']-(r['true_nll']-r['new_nll']))<1e-12
        groups[r['case_id']].append(r);bits.append(b)
        for side in ('new','true'):
            assert 0<=r[side+'_token_correct']<=r[side+'_token_count']
            assert r[side+'_strict']==(r[side+'_token_correct']==r[side+'_token_count'])
    out=dict(metric=tag,numerator=sum(bits),denominator=n,rate=sum(bits)/n,
             request_denominator=len(groups),pair_strict_num=sum(all(preferred(r['new_nll'],r['true_nll'],tag) for r in g) for g in groups.values()),
             ties=sum(r['new_nll']==r['true_nll'] for r in rows),
             bit_order_sha256=digest([(r['identity'],b) for r,b in zip(rows,bits)]))
    for side in ('new','true'):
        vals=[r[side+'_nll'] for r in rows]
        for k,v in stats(vals).items():out[side+'_nll_prompt_'+k]=v
        for k,v in stats([np.mean([r[side+'_nll'] for r in g]) for g in groups.values()]).items():out[side+'_nll_request_'+k]=v
        out[side+'_strict_num']=sum(r[side+'_strict'] for r in rows)
        out[side+'_strict_den']=n
        out[side+'_token_correct']=sum(r[side+'_token_correct'] for r in rows)
        out[side+'_token_den']=sum(r[side+'_token_count'] for r in rows)
    # Common margin is true-new for all categories; positive favors new, NS prefers negative.
    for k,v in stats([r['margin'] for r in rows]).items():out['margin_prompt_'+k]=v
    for k,v in stats([np.mean([r['margin'] for r in g]) for g in groups.values()]).items():out['margin_request_'+k]=v
    return out

def reduce_eval(raw,ids,arm,batch,scope):
    assert raw['requests']==len(ids)
    result=[]
    for tag in MULT:
        m=raw['metrics'][tag];rows=m['rows']
        assert [(r['case_id'],r['prompt_index']) for r in rows]==[(i,p) for i in ids for p in range(MULT[tag])]
        assert len({r['identity'] for r in rows})==len(rows)
        r=reduce_rows(rows,tag)
        for k in ['numerator','denominator','rate','bit_order_sha256']:assert r[k]==m[k],(arm,batch,tag,k)
        result.append(dict(arm=arm,batch=batch,scope=scope,**r))
    return result
