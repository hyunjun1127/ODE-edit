"""One-shot CPU raw audit and selection; never creates a benchmark/GPU job."""
import argparse
from pathlib import Path
from official.experiments.prepare import digest, write_new, file_sha
from official.runners.server1.common import read, verify, member
from official.evaluation.factual import validate_retained_observation
from official.evaluation.reduce import counterfact
from official.diagnostics.fe_qwen_current.contract import SPEC, select
from official.diagnostics.fe_qwen_current.observer import NUMERIC_CODES

def collect(root, *, write_result=True):
    root=Path(root);lock=read(root/'lock.json');results={};proofs=[]
    contract=read(verify(lock['selection_lock']))
    if contract['spec']!=SPEC or contract['spec_sha256']!=digest(SPEC) or len(lock['configs'])!=4:
        raise ValueError('FOUR_ARM_IMMUTABLE_SELECTION_LOCK')
    for m in lock['configs']:
        c=read(verify(m));out=Path(c['output']);l2=c['L2']
        if l2 in results or c['source']!=lock['source_commit'] or c['config_sha256']!=digest({k:v for k,v in c.items() if k!='config_sha256'}):
            raise ValueError('SELECTION_CONFIG_IDENTITY')
        if c['selection_contract_sha256']!=digest(SPEC):raise ValueError('SPEC_CHANGED')
        def identity(value):
            actual=value['identity']
            actual=actual.get('external_identity',actual)
            if actual['config_sha256']!=c['config_sha256'] or actual['source']!=lock['source_commit'] or actual['L2']!=l2:
                raise ValueError('RESULT_IDENTITY')
        if not (out/'COMPLETE.json').exists():
            failure=read(out/'FAILURE.json');identity(failure)
            if failure['status']!='NUMERICAL_FAILURE' or failure['code'] not in NUMERIC_CODES:
                raise ValueError('UNRESOLVED_NONNUMERICAL_FAILURE')
            results[l2]=dict(status='NUMERICAL_FAILURE',verified_failure=True,
                failure_receipt_sha256=file_sha(out/'FAILURE.json'))
            proofs.append(dict(L2=l2,failure=member(out/'FAILURE.json')));continue
        completed=read(out/'COMPLETE.json');identity(completed)
        if completed['status']!='W20_COMPLETE' or (completed['edits'],completed['batches'])!=(2000,20):raise ValueError('PARTIAL_ENDPOINT')
        raw=read(verify(completed['endpoint']));verify(completed['checkpoint'])
        identity(raw)
        input_rows=read(verify(c['stream']))
        if [x['case_id'] for x in raw['cases']]!=[x['case_id'] for x in input_rows]:raise ValueError('W20_COHORT')
        queries=read(out/'queries/B20-all-seen.json')
        audit=validate_retained_observation(raw,'cf',input_signature=queries)
        reduced=counterfact(raw['cases'])
        for key in ('Efficacy','Generalization','Specificity','Score','requests'):
            if reduced[key]!=raw['summary'][key]:raise ValueError('RAW_REDUCTION_MISMATCH')
        for b in range(1,21):
            commit=read(out/'commits'/f'B{b:02d}.json');identity(commit)
            if (commit['batch'],commit['edits'])!=(b,b*100):raise ValueError('COMMIT_COVERAGE')
            for endpoint in commit['raw']:verify(endpoint)
        results[l2]=dict(reduced,committed_batches=20,status='W20_COMPLETE')
        proofs.append(dict(L2=l2,raw=completed['endpoint'],queries=member(out/'queries/B20-all-seen.json'),audit=audit))
    selection=select(results)
    receipt=dict(selection=selection,results={str(k):v for k,v in results.items()},proofs=proofs,
        selection_lock=member(root/'selection-lock.json'),benchmark_submitted=False,
        no_GPU_or_model_calls=True,source=lock['source_commit'])
    if write_result:write_new(root/'selection-result.json',receipt)
    return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();collect(a.root)
