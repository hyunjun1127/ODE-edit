"""Read-only independent receipt/row audit and compact publication (CPU only)."""
import argparse
import subprocess
import unittest
from .common import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',default='attempt-v1');p.add_argument('--preflight',action='store_true');a=p.parse_args()
    run=ROOT/a.attempt;lock=read(run/'execution.lock.json');r=read(run/'submission.json')
    dest=AUDIT/a.attempt;dest.mkdir(parents=True,exist_ok=True)
    if a.preflight:
        from .test_core import Core
        suite=unittest.defaultTestLoader.loadTestsFromTestCase(Core);result=unittest.TestResult();suite.run(result)
        assert result.wasSuccessful(),(result.failures,result.errors)
        checks=dict(CPU_tests=result.testsRun,CPU_tests_pass=result.wasSuccessful(),GPU_model_validation='NOT_OBSERVED_AT_SUBMISSION',
            whole_parameter_names=list(KEYS),physical_selected_weight_count=5,task_counts=dict(states=173,score_tasks=383,main_cells=156,pair_cells=16),
            readonly_CP_count=24,source_native_fit=0,source_new_CP_save=0,independent_agent_red=False,owner_review=True,
            source_freeze=read(run/'source-and-lock-receipt.json'),T0=record(ROOT/'inputs/t0-v1/runtime-binding.json'),submission=record(run/'submission.json'),release=record(run/'release.json'))
        save(dest/'cpu-and-submission.json',checks)
        status=dict(instruction_id=INSTRUCTION,status='SUBMITTED_NOT_COMPLETED',family_jobs=r['family_jobs'],collector=r['collector'],
            execution_source=lock['source_commit'],execution_tree=lock['source_tree'],execution_lock=record(run/'execution.lock.json'),
            monitoring_active=True,automatic_new_scientific_submission=False,save_checkpoints=False,initial_pause_inherited=False,
            cpu_T0='PASS',gpu_T1='NOT_OBSERVED_AT_SUBMISSION',report_pending=True)
        save(REPO/'tasks/status/historical-update-timeaxis-20260924-v1/server4.json',status)
        save(REPO/'runs/odeedit_historical_update_timeaxis_s4_20260924/submission-r1.json',status)
        return
    out=run/'output';failures=[];rows_count=0;states={};score_tasks=set()
    binding=read(lock['T0']['path']);tokens={(r['case_id'],r['panel'],r['prompt_index']):r for r in read(binding['token_manifest']['path'])}
    for family in FAMILIES:
        for receipt in sorted((out/family/'tasks').glob('*/PASS.json')):
            x=read(receipt);assert x['status']=='PASS' and x['identity']['source_sha256']==lock['source_sha256'];assert x['state']['restore']=='EXACT_BYTES'
            assert sha(x['scores']['path'])==x['scores']['sha256'];rs=read(x['scores']['path'])
            assert len(rs)==int(x['task']['prompt_count']);keys=set()
            for s in rs:
                key=(s['case_id'],s['panel'],s['prompt_index']);assert key not in keys;keys.add(key)
                assert s['margin']==s['competitor_nll']-s['target_nll'] and s['valid'] and s['missing_reason'] is None
                assert s['state_weight_hash']==x['state']['state_weight_hash'] and s['evaluator_signature']
                assert 0<=s['target_token_correct']<=s['target_token_count'] and 0<=s['competitor_token_correct']<=s['competitor_token_count']
                token=tokens[key]
                for prefix,field in [('target','target_ids'),('competitor','competitor_ids')]:
                    prediction=s[prefix+'_predictions'];labels=token[field];assert len(prediction)==len(labels)==s[prefix+'_token_count']
                    count=sum(a==b for a,b in zip(prediction,labels,strict=True));assert count==s[prefix+'_token_correct']
                    assert (count==len(labels))==s[prefix+'_strict_tf']
                assert all(s[k]==token[k] for k in ('prompt_token_hash','target_token_hash','competitor_token_hash','target_version'))
            rows_count+=len(rs);score_tasks.add(x['task']['task_id']);states[x['task']['state_id']]=x['state']['state_weight_hash']
        f=out/family/'FAILURE.json'
        if f.exists():failures.append(dict(family=family,receipt=record(f),stage=read(f)['stage'],error=read(f)['error']))
    result=dict(status='CPU_RECEIPT_AUDIT',unique_tasks=len(score_tasks),unique_states=len(states),stored_score_rows=rows_count,failures=failures,independent_agent=False,newGPU=0)
    save(dest/'terminal-receipt-audit.json',result);print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
