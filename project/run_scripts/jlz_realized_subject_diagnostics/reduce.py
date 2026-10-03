"""Independent CPU raw arithmetic; no production evaluator/reducer import."""
import math
import statistics


def require(ok,message):
    if not ok:raise ValueError(message)


def distribution(values):
    require(values and all(math.isfinite(x) for x in values),'FINITE_DISTRIBUTION')
    ordered=sorted(values)
    def q(p):
        x=(len(ordered)-1)*p;i=int(x);j=min(i+1,len(ordered)-1)
        return ordered[i]+(x-i)*(ordered[j]-ordered[i])
    return dict(mean=statistics.mean(values),median=q(.5),p90=q(.9),p95=q(.95),p99=q(.99),min=min(values),max=max(values))


def validate(rows,expected=None):
    require(len({r['identity'] for r in rows})==len(rows),'DUPLICATE_ROW')
    for r in rows:
        require(r['kind'] in ('R','P','N'),'KIND')
        for label in ('new','true'):
            n,c=r[label+'_token_count'],r[label+'_token_correct']
            require(type(n) is int and type(c) is int and n>0 and 0<=c<=n,'TOKEN_COUNTS')
            require(type(r[label+'_strict']) is bool and r[label+'_strict']==(n==c),'STRICT')
            require(math.isfinite(r[label+'_nll']),'NONFINITE_NLL')
        require(abs(r['margin_true_minus_new']-(r['true_nll']-r['new_nll']))<1e-12,'MARGIN_SIGN')
    if expected is not None:
        require([r['identity'] for r in rows]==[r['identity'] for r in expected],'IDENTITY_ORDER')
        for r,e in zip(rows,expected):
            require(all(r[k]==e[k] for k in ('case_id','kind','prompt_index','new_token_identity','true_token_identity','new_token_count','true_token_count')),'TOKEN_TARGET_IDENTITY')


def success(r):return r['true_nll']<r['new_nll'] if r['kind']=='N' else r['new_nll']<r['true_nll']


def metrics(rows):
    validate(rows);result={}
    for kind in ('R','P','N'):
        group=[r for r in rows if r['kind']==kind]
        if not group:continue
        desired='true' if kind=='N' else 'new';num=sum(success(r) for r in group);n=len(group)
        tc=sum(r[desired+'_token_correct'] for r in group);tn=sum(r[desired+'_token_count'] for r in group)
        margin=[r['new_nll']-r['true_nll'] if kind=='N' else r['true_nll']-r['new_nll'] for r in group]
        result[kind]=dict(numerator=num,denominator=n,percent=100*num/n,TF_token_correct=tc,TF_token_count=tn,
            TF_token_micro=tc/tn,TF_prompt_macro=statistics.mean(r[desired+'_token_correct']/r[desired+'_token_count'] for r in group),
            TF_strict_numerator=sum(r[desired+'_strict'] for r in group),TF_strict_denominator=n,
            ties=sum(r['new_nll']==r['true_nll'] for r in group),
            true_NLL=distribution([r['true_nll'] for r in group]),new_NLL=distribution([r['new_nll'] for r in group]),
            desired_margin=distribution(margin))
    return result


def paired(before,after):
    validate(after,before);result={}
    for kind in ('R','P','N'):
        pairs=[(b,a) for b,a in zip(before,after) if b['kind']==kind]
        if not pairs:continue
        d='true' if kind=='N' else 'new'
        result[kind]=dict(denominator=len(pairs),lost=sum(success(b) and not success(a) for b,a in pairs),
            gained=sum(not success(b) and success(a) for b,a in pairs),both_success=sum(success(b) and success(a) for b,a in pairs),
            both_failure=sum(not success(b) and not success(a) for b,a in pairs),
            strict_lost=sum(b[d+'_strict'] and not a[d+'_strict'] for b,a in pairs),
            strict_gained=sum(not b[d+'_strict'] and a[d+'_strict'] for b,a in pairs))
    return result


def paired_rows(before,after):
    validate(after,before);output=[]
    for b,a in zip(before,after):
        d='true' if b['kind']=='N' else 'new';sign=-1 if b['kind']=='N' else 1
        output.append(dict(identity=b['identity'],case_id=b['case_id'],kind=b['kind'],prompt_index=b['prompt_index'],
            before_success=success(b),after_success=success(a),lost=success(b) and not success(a),gained=not success(b) and success(a),
            strict_lost=b[d+'_strict'] and not a[d+'_strict'],strict_gained=not b[d+'_strict'] and a[d+'_strict'],
            new_nll_delta=a['new_nll']-b['new_nll'],true_nll_delta=a['true_nll']-b['true_nll'],
            desired_margin_delta=sign*(a['margin_true_minus_new']-b['margin_true_minus_new'])))
    return output


def interaction(masked):
    names=('NONE','SUBJECT_ONLY','NONSUBJECT_ONLY','ALL');base=masked['NONE']
    for name in names:validate(masked[name],base)
    rows=[]
    for i,r in enumerate(base):
        m={name:masked[name][i]['new_nll']-masked[name][i]['true_nll'] for name in names}
        rows.append(dict(identity=r['identity'],case_id=r['case_id'],prompt_index=r['prompt_index'],
            **m,interaction=m['ALL']-m['SUBJECT_ONLY']-m['NONSUBJECT_ONLY']+m['NONE']))
    return rows,distribution([r['interaction'] for r in rows]) if rows else None
