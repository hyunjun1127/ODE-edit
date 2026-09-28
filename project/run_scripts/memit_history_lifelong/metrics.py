"""Same canonical forwards; add scalar reductions and paired identities only."""
import math
from project.run_scripts.blue_alphaedit_sequential_comparison.evaluation import evaluate as canonical_evaluate
from .io import digest

FULL_BATCHES=[1,5,10,20,30,40,50,60,70,80,90,100]

def summarize(rows,tag):
    if not rows:return {'denominator':0}
    desired='true' if tag=='NS' else 'new';n=len(rows)
    correct=sum(r[desired+'_token_correct'] for r in rows);count=sum(r[desired+'_token_count'] for r in rows)
    assert count>0 and all(r[desired+'_token_count']>0 for r in rows)
    return dict(numerator=sum(r['success'] for r in rows),denominator=n,rate=sum(r['success'] for r in rows)/n,
      desired=desired,tf_token_correct=correct,tf_token_count=count,tf_token_micro=correct/count,
      tf_prompt_macro=sum(r[desired+'_token_correct']/r[desired+'_token_count'] for r in rows)/n,
      tf_strict_numerator=sum(r[desired+'_strict'] for r in rows),tf_strict_denominator=n,
      tf_strict=sum(r[desired+'_strict'] for r in rows)/n,
      new_nll=sum(r['new_nll'] for r in rows)/n,true_nll=sum(r['true_nll'] for r in rows)/n,
      desired_nll=sum(r[desired+'_nll'] for r in rows)/n,margin_true_minus_new=sum(r['margin'] for r in rows)/n,
      bit_order_sha256=digest([(r['identity'],r['success']) for r in rows]))

def augment(result):
    for tag,m in result['metrics'].items():m.update(summarize(m['rows'],tag))
    return result

def evaluate(*a,**kw):return augment(canonical_evaluate(*a,**kw))

def merge(past,current,full=True):
    out={}
    for tag in (('RS','PS','NS') if full else ('RS',)):
        rows=(past['metrics'][tag]['rows'] if past else [])+current['metrics'][tag]['rows']
        assert len({r['identity'] for r in rows})==len(rows)
        out[tag]=dict(rows=rows,**summarize(rows,tag))
    return dict(requests=(past['requests'] if past else 0)+current['requests'],metrics=out,current_rows_reused=True)

def paired(before,after):
    b={r['identity']:r for r in before};a={r['identity']:r for r in after}
    assert b.keys()==a.keys(),'PAIRED_IDENTITY_MISMATCH'
    lost=[k for k in b if b[k]['success'] and not a[k]['success']]
    gained=[k for k in b if not b[k]['success'] and a[k]['success']]
    return dict(denominator=len(b),before_success=sum(r['success'] for r in b.values()),after_success=sum(r['success'] for r in a.values()),lost=len(lost),gained=len(gained),lost_ids=lost,gained_ids=gained)

def active_case_ids(records):
    # Evaluation-only strata: same fact+target repetitions retain occurrences;
    # a later different target supersedes previous target versions.
    latest={}
    for r in records:
        x=r['requested_rewrite'];latest[(x['subject'],x['relation_id'])]=x['target_new']['str']
    return {int(r['case_id']) for r in records if latest[(r['requested_rewrite']['subject'],r['requested_rewrite']['relation_id'])]==r['requested_rewrite']['target_new']['str']}

def strata(result,seen):
    active=active_case_ids(seen)
    return {tag:{name:summarize([r for r in m['rows'] if (r['case_id'] in active)==flag],tag) for name,flag in [('active',True),('superseded',False)]} for tag,m in result['metrics'].items()}
