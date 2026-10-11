"""Conditional final benchmark binding and independent CPU public-query audit.

No submission, model loading, selection on benchmark scores, or source mutation.
"""
import math
from pathlib import Path
from official.experiments.prepare import digest
from official.runners.server1.common import read, verify
from official.diagnostics.fe_qwen_current.contract import SPEC


def validated_selection(result_member):
    from official.diagnostics.fe_qwen_current.collect import collect
    path=verify(result_member)
    saved=read(path)
    actual=collect(Path(path).parent,write_result=False)
    if saved!=actual:raise ValueError('SELECTION_REAUDIT_MISMATCH')
    selection=actual['selection']
    if selection['status']!='SELECTED' or selection['selected_L2'] not in SPEC['promotable_L2']:
        raise ValueError('BENCHMARK_REQUIRES_SELECTED_TUNED_ARM')
    if {float(k) for k in actual['results']}!=set(SPEC['L2_arms']):
        raise ValueError('FOUR_ARM_COVERAGE_REQUIRED')
    return actual


def native_records(rows):
    """One unexpanded native edit request per original common stream record.

    Query expansion belongs only to the public evaluator, never the writer.
    """
    output=[]
    for row in rows:
        r=row['requested_rewrite']
        if r['prompt'].count('{}')!=1 or not r['subject'] or not r['target_new']['str']:
            raise ValueError('NATIVE_REQUEST_SCHEMA')
        output.append(dict(case_id=row['case_id'],prompt=r['prompt'],subject=r['subject'],
            target_new=r['target_new']['str'],target_true=r['target_true']['str'],
            paraphrase_prompts=row['paraphrase_prompts'],neighborhood_prompts=row['neighborhood_prompts']))
    return output


def reduce_zsre(raw,plan):
    """Recompute request-macro E/G/loc_ans from token predictions + exact queries."""
    cases=raw['cases'];groups=('rewrite','paraphrase','neighborhood')
    if (len(cases)!=plan['requests'] or raw['query_sha256']!=plan['query_sha256']
        or raw['stream_sha256']!=plan['stream_sha256']
        or digest(plan['queries'])!=plan['query_sha256']
        or raw['summary']['requests']!=len(cases)):
        raise ValueError('PUBLIC_QUERY_IDENTITY')
    expected={}
    for q in plan['queries']:
        expected.setdefault((q['case_index'],q['group']),[]).append(q)
    rates={g:[] for g in groups};counts={g:0 for g in groups}
    for index,(case,original) in enumerate(zip(cases,plan['cases'])):
        if any(case[k]!=original[k] for k in ('case_id','occurrence_index')):
            raise ValueError('PUBLIC_QUERY_CASE_ORDER')
        for g in groups:
            observations=case[g+'_observations'];queries=expected.get((index,g),[])
            if not queries or len(observations)!=len(queries):raise ValueError('PUBLIC_QUERY_COVERAGE')
            bits=[]
            for obs,q in zip(observations,queries):
                if any(obs[k]!=q[k] for k in ('prompt_index','token_index','target_token_id')):
                    raise ValueError('PUBLIC_QUERY_TOKEN_MAPPING')
                if type(obs['predicted_token_id']) is not int or obs['predicted_token_id']<0:
                    raise ValueError('PUBLIC_QUERY_PREDICTION')
                bit=obs['predicted_token_id']==obs['target_token_id']
                if type(obs['correct']) is not bool or obs['correct']!=bit:raise ValueError('PUBLIC_QUERY_CORRECTNESS')
                bits.append(bit)
            if case[g+'_prompts_correct']!=bits:raise ValueError('PUBLIC_QUERY_BITS')
            rates[g].append(sum(bits)/len(bits));counts[g]+=len(bits)
    metrics={label:100*math.fsum(rates[g])/len(cases) for g,label in zip(groups,('Efficacy','Generalization','Specificity'))}
    metrics['Specificity_loc_ans']=metrics['Specificity']
    vals=[metrics[k] for k in ('Efficacy','Generalization','Specificity')]
    metrics['Score']=0.0 if min(vals)==0 else 3/sum(1/x for x in vals)
    for k,v in metrics.items():
        if k=='Score' and k not in raw['summary']:continue
        if not math.isfinite(raw['summary'][k]) or abs(raw['summary'][k]-v)>1e-10:
            raise ValueError('PUBLIC_QUERY_REQUEST_MACRO')
    if counts!=raw['token_denominators'] or counts!=plan['token_denominators']:
        raise ValueError('PUBLIC_QUERY_DENOMINATORS')
    work=raw['work']
    if work['queries']!=sum(counts.values()) or work['physical_forward_calls']<=0 or not math.isfinite(work['elapsed_seconds']) or work['elapsed_seconds']<=0:
        raise ValueError('PUBLIC_QUERY_WORK')
    return dict(status='CPU_RAW_QUERY_REDUCED',requests=len(cases),metrics=metrics,
        token_denominators=counts,query_sha256=plan['query_sha256'],GPU_parity_claim=False)


def verify_query_subset(full,subset,rows):
    """Bind every current/milestone query to the CPU-parity-verified W0 plan."""
    wanted=[(r['case_id'],r['occurrence_index']) for r in rows]
    positions={(r['case_id'],r['occurrence_index']):i for i,r in enumerate(full['cases'])}
    old_to_new={positions[key]:i for i,key in enumerate(wanted)}
    expected=[dict(q,case_index=old_to_new[q['case_index']]) for q in full['queries'] if q['case_index'] in old_to_new]
    if (subset['queries']!=expected or subset['query_sha256']!=digest(expected)
        or subset['stream_sha256']!=digest(rows)):
        raise ValueError('FINAL_QUERY_SUBSET_PARITY')


def collect_final(root):
    from official.experiments.prepare import write_new
    from official.runners.server1.common import member
    from official.evaluation.factual import validate_retained_observation
    from official.evaluation.reduce import counterfact
    root=Path(root);lock=read(root/'lock.json');results=[]
    for cm in lock['configs']:
        c=read(verify(cm));out=Path(c['output'])
        proof=validated_selection(c['selected_result'])
        if c['stage']!='final_benchmark' or c['L2']!=proof['selection']['selected_L2']:
            raise ValueError('FINAL_SELECTED_SETTING')
        completed=read(out/'COMPLETE.json')
        def identity(receipt):
            i=receipt['identity']
            i=i.get('external_identity',i)
            if (i['config_sha256']!=c['config_sha256'] or i['source']!=lock['source_commit']
                or i['L2']!=c['L2'] or i['dataset']!=c['dataset']
                or i['selected_result']!=c['selected_result']):raise ValueError('FINAL_IDENTITY')
        identity(completed)
        if (completed['edits'],completed['batches'])!=(2000,20):raise ValueError('FINAL_W20_COVERAGE')
        verify(completed['checkpoint']);verify(completed['endpoint'])
        source_rows=read(verify(c['stream']));audits={};full_queries=None
        for b in range(21):
            commit=read(out/'commits'/f'B{b:02d}.json');identity(commit)
            if (commit['batch'],commit['edits'])!=(b,b*100):raise ValueError('FINAL_COMMIT_CURSOR')
            labels=['W0'] if b==0 else [f'B{b:02d}-current']
            if b in SPEC['milestones']:labels.append(f'B{b:02d}-all-seen')
            if len(commit['raw'])!=len(labels):raise ValueError('FINAL_ENDPOINT_COVERAGE')
            for label,raw_member in zip(labels,commit['raw']):
                raw=read(verify(raw_member));identity(raw)
                evaluation=read(out/'evaluations'/f'{label}.json');identity(evaluation)
                if evaluation['raw']!=raw_member:raise ValueError('FINAL_RAW_BINDING')
                plan=read(verify(evaluation['queries']))
                expected=source_rows if b==0 else (source_rows[:b*100] if label.endswith('all-seen') else source_rows[(b-1)*100:b*100])
                if [r['case_id'] for r in raw['cases']]!=[r['case_id'] for r in expected]:raise ValueError('FINAL_COHORT')
                if c['dataset']=='zsre':
                    if plan['stream_sha256']!=digest(expected):raise ValueError('FINAL_QUERY_STREAM')
                    if b==0:full_queries=plan
                    verify_query_subset(full_queries,plan,expected)
                    audit=reduce_zsre(raw,plan)
                    if b==0 and plan['query_sha256']!=read(verify(c['query_parity']))['query_sha256']:
                        raise ValueError('FINAL_QUERY_PARITY_BINDING')
                else:
                    validation=validate_retained_observation(raw,'cf',input_signature=plan)
                    metrics=counterfact(raw['cases'])
                    if any(metrics[k]!=raw['summary'][k] for k in ('Efficacy','Generalization','Specificity','Score','requests')):
                        raise ValueError('FINAL_CF_REDUCTION')
                    audit=dict(validation=validation,metrics=metrics)
                audits[label]=dict(audit,raw=raw_member)
        row=dict(dataset=c['dataset'],L2=c['L2'],config=cm,source=lock['source_commit'],
            checkpoint=completed['checkpoint'],W20=audits['B20-all-seen'],W0=audits['W0'],
            endpoints=audits,requests=2000,committed_batches=20,status='W20_CPU_RAW_VERIFIED')
        if c['dataset']=='zsre':row['Loc_delta_from_W0_pp']=audits['B20-all-seen']['metrics']['Specificity']-audits['W0']['metrics']['Specificity']
        results.append(row)
    if sorted(r['dataset'] for r in results)!=['cf','zsre'] or len({r['L2'] for r in results})!=1:
        raise ValueError('FINAL_PAIR_SAME_SETTING')
    receipt=dict(rows=results,both_W20_complete=True,recovery_claim=False,GPU_parity_claim=False)
    write_new(root/'final-result.json',receipt)
    return receipt


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();collect_final(a.root)
