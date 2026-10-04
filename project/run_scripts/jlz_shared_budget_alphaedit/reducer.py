"""CPU scalar reducer independent of observer aggregation; no torch/model calls."""
import math
from .common import require

def reduce_rows(rows):
    seen=set();group={}
    for r in rows:
        require(r['identity'] not in seen,'DUPLICATE_RAW');seen.add(r['identity'])
        family=r['kind'];require(family in ('R','P','N'),'FAMILY')
        for target in ('true','new'):
            require(math.isfinite(r[target+'_nll']),'FINITE_NLL')
            n,k=r[target+'_token_count'],r[target+'_token_correct']
            require(type(n) is int and type(k) is int and n>0 and 0<=k<=n,'TOKEN_COUNTS')
            require(type(r[target+'_strict']) is bool and r[target+'_strict']==(k==n),'STRICT')
        label='true' if family=='N' else 'new';n,k=r[label+'_token_count'],r[label+'_token_correct']
        v=group.setdefault(family,dict(denominator=0,numerator=0,desired_token_count=0,desired_token_correct=0,
            strict_numerator=0,new_strict_numerator=0,true_nll=[],new_nll=[],macro=[]))
        v['denominator']+=1;v['numerator']+=int(r[label+'_nll']<r[('new' if label=='true' else 'true')+'_nll'])
        v['desired_token_count']+=n;v['desired_token_correct']+=k;v['strict_numerator']+=int(k==n)
        v['new_strict_numerator']+=int(r['new_strict']);v['true_nll'].append(r['true_nll']);v['new_nll'].append(r['new_nll']);v['macro'].append(k/n)
    for v in group.values():
        d=v['denominator'];v['strict_denominator']=d;v['rate']=v['numerator']/d
        v['token_micro']=v['desired_token_correct']/v['desired_token_count']
        v['prompt_macro']=math.fsum(v.pop('macro'))/d
        for label in ('true','new'):v[label+'_nll_mean']=math.fsum(v.pop(label+'_nll'))/d
    return group
