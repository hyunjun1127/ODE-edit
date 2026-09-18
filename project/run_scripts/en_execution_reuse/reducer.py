"""Independent CPU raw-NLL reducer. No evaluator/torch/model imports."""
import hashlib
import json
import math
import statistics


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def quantile(values,p):
    values=sorted(values)
    if not values:return None
    x=(len(values)-1)*p;i=int(x);j=min(i+1,len(values)-1)
    return values[i]+(values[j]-values[i])*(x-i)


def distribution(values):
    return dict(mean=statistics.fmean(values),median=statistics.median(values),
        p05=quantile(values,.05),p95=quantile(values,.95),p99=quantile(values,.99),min=min(values),max=max(values))


def reduce_raw(value,records):
    if value['selection_seal']['status']!='SELECTION_SEALED':raise ValueError('NO_ENDPOINT_SEAL')
    ids=[r['case_id'] for r in records]
    if len(ids)!=100 or len(set(ids))!=100 or value['requests']!=100:raise ValueError('B100_CARDINALITY')
    if value['request_order']!=digest(ids):raise ValueError('B100_ORDER')
    raw=value['raw'];result={};all_rows=[]
    expected_keys={f'{prefix}_target_{branch}' for prefix in ('rewrite','rephrase','locality') for branch in ('new','true')}
    if set(raw)!=expected_keys:raise ValueError('RAW_SIX_PANELS')
    for tag,prefix,multiple in (('RS','rewrite',1),('PS','rephrase',2),('NS','locality',10)):
        expected=[]
        for record in records:
            r=record['requested_rewrite']
            prompts=[r['prompt'].format(r['subject'])] if tag=='RS' else record['paraphrase_prompts' if tag=='PS' else 'neighborhood_prompts']
            if len(prompts)!=multiple:raise ValueError('PROMPT_MULTIPLICITY')
            expected.extend((record['case_id'],i,p,r['target_new']['str'],r['target_true']['str']) for i,p in enumerate(prompts))
        new=raw[prefix+'_target_new'];true=raw[prefix+'_target_true']
        if len(new)!=100*multiple or len(true)!=len(new):raise ValueError('RAW_DENOMINATOR')
        rows=[]
        for a,b,e in zip(new,true,expected,strict=True):
            for row,branch,target in ((a,'new',e[3]),(b,'true',e[4])):
                if (row['case_id'],row['prompt_index'],row['prompt'],row['target'])!=(e[0],e[1],e[2],target):
                    raise ValueError('PROMPT_TARGET_CASE_ORDER_IDENTITY')
                if row['kind']!=prefix+'_target_'+branch:raise ValueError('RAW_KIND')
                if not math.isfinite(row['nll']):raise ValueError('NONFINITE_NLL')
                if not row['target_token_ids'] or len(row['token_correct'])!=len(row['target_token_ids']):
                    raise ValueError('TARGET_TOKEN_CARDINALITY')
                expected_correct=[x==y for x,y in zip(row['token_predictions'],row['target_token_ids'],strict=True)]
                if row['token_correct']!=expected_correct or row['all_tokens_correct']!=all(expected_correct):
                    raise ValueError('STRICT_TOKEN_IDENTITY')
            margin=(a['nll']-b['nll']) if tag=='NS' else (b['nll']-a['nll'])
            identity=digest([*e,a['target_token_ids'],b['target_token_ids']])
            rows.append(dict(panel=tag,case_id=e[0],prompt_index=e[1],identity=identity,
                new_nll=a['nll'],true_nll=b['nll'],desired_margin=margin,success=margin>0,tie=margin==0,
                new_strict=a['all_tokens_correct'],true_strict=b['all_tokens_correct'],
                new_correct_tokens=sum(a['token_correct']),new_tokens=len(a['target_token_ids']),
                true_correct_tokens=sum(b['token_correct']),true_tokens=len(b['target_token_ids'])))
        success=sum(r['success'] for r in rows)
        result[tag]=dict(numerator=success,denominator=len(rows),percent=100*success/len(rows),ties=sum(r['tie'] for r in rows),
            new_strict=sum(r['new_strict'] for r in rows),true_strict=sum(r['true_strict'] for r in rows),
            distributions={f:distribution([r[f] for r in rows]) for f in ('new_nll','true_nll','desired_margin')},rows=rows)
        prior=value['metrics'][tag]
        if prior['denominator']!=len(rows) or sum(bool(r['success']) for r in prior['rows'])!=success:
            raise ValueError('PUBLISHED_REDUCER_DISAGREEMENT')
        all_rows.extend(rows)
    strict=[]
    for case in ids:
        r=next(x for x in result['RS']['rows'] if x['case_id']==case)
        p=[x for x in result['PS']['rows'] if x['case_id']==case]
        strict.append(dict(case_id=case,rewrite_strict=r['new_strict'],two_P_strict=all(x['new_strict'] for x in p),
            R_two_P_strict=r['new_strict'] and all(x['new_strict'] for x in p),
            R_two_P_NLL_joint=r['success'] and all(x['success'] for x in p)))
    return dict(metrics=result,strict={k:sum(x[k] for x in strict) for k in strict[0] if k!='case_id'},
        strict_rows=strict,rows=all_rows,endpoint=value['selection_seal']['endpoint_weight_sha256'],
        pair_identity_root=digest([r['identity'] for r in all_rows]))


def pair(before,after):
    if before['pair_identity_root']!=after['pair_identity_root']:raise ValueError('PAIRED_IDENTITY_ROOT')
    result={}
    for tag in ('RS','PS','NS'):
        a=before['metrics'][tag]['rows'];b=after['metrics'][tag]['rows'];rows=[]
        for x,y in zip(a,b,strict=True):
            if x['identity']!=y['identity']:raise ValueError('PAIRED_ROW_IDENTITY')
            rows.append(dict(case_id=x['case_id'],prompt_index=x['prompt_index'],identity=x['identity'],
                lost=x['success'] and not y['success'],gained=not x['success'] and y['success'],
                new_nll_delta=y['new_nll']-x['new_nll'],true_nll_delta=y['true_nll']-x['true_nll'],
                desired_margin_delta=y['desired_margin']-x['desired_margin']))
        result[tag]=dict(denominator=len(rows),lost=sum(r['lost'] for r in rows),gained=sum(r['gained'] for r in rows),
            delta_pp=100*(sum(r['gained'] for r in rows)-sum(r['lost'] for r in rows))/len(rows),
            distributions={f:distribution([r[f] for r in rows]) for f in ('new_nll_delta','true_nll_delta','desired_margin_delta')},rows=rows)
    return result
