"""Observation-only schema adaptation of original seen-full JSON.

The original file intentionally lacks request_order. Derive that metadata only
after verifying every ordered case/prompt/target identity, never from its size.
Native code and original files remain unchanged.
"""
from .contracts import ContractBoundary
from .case_population import source_digest
from .performance_compare import compare_rows

MULTIPLICITY={'RS':1,'PS':2,'NS':10}


def normalize(doc, records):
    if doc['requests']!=len(records):raise ContractBoundary('PERFORMANCE_REQUEST_COUNT')
    order=source_digest([r['case_id'] for r in records])
    if 'request_order' in doc and doc['request_order']!=order:
        raise ContractBoundary('PERFORMANCE_REQUEST_ORDER')
    metrics={}
    for tag,mult in MULTIPLICITY.items():
        expected=[]
        for r in records:
            w=r['requested_rewrite']
            prompts=([w['prompt'].format(w['subject'])] if tag=='RS' else
                     r['paraphrase_prompts' if tag=='PS' else 'neighborhood_prompts'])
            if len(prompts)!=mult:raise ContractBoundary('PERFORMANCE_PROMPT_CARDINALITY')
            expected.extend((r['case_id'],i,source_digest([r['case_id'],i,p,w['target_new']['str'],w['target_true']['str']])) for i,p in enumerate(prompts))
        m=doc['metrics'][tag];rows=m['rows']
        actual=[(r['case_id'],r['prompt_index'],r['identity']) for r in rows]
        if actual!=expected or len(set(actual))!=len(actual):raise ContractBoundary('PERFORMANCE_ORDERED_IDENTITY')
        check=compare_rows(rows,rows,tag)
        if m['denominator']!=len(expected) or m['numerator']!=check['original_numerator']:
            raise ContractBoundary('PERFORMANCE_REDUCER_MISMATCH')
        metrics[tag]=m
    return dict(doc,request_order=order,metrics=metrics)


def from_rows(groups, records):
    metrics={}
    for tag in MULTIPLICITY:
        rows=[r for group in groups for r in group['metrics'][tag]['rows']]
        n=sum(r['success'] for r in rows)
        metrics[tag]=dict(rows=rows,numerator=n,denominator=len(rows),rate=n/len(rows),
                          bit_order_sha256=source_digest([(r['identity'],r['success']) for r in rows]))
    return normalize(dict(requests=len(records),metrics=metrics),records)


def subset(doc,start,end,records):
    groups={tag:dict(rows=doc['metrics'][tag]['rows'][start*m:end*m]) for tag,m in MULTIPLICITY.items()}
    return from_rows([dict(metrics=groups)],records[start:end])
