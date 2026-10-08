"""Explicit USER generation repair: cancel only the exact old three-job graph."""
import json
from pathlib import Path
import subprocess
from .submit import job
from .run import TASK


def main():
    root=Path('/data/janghj/ODE-edit/local/price-ridge-m1-m3-2k-20261008')
    old=root/'attempt-r1'; receipt=root/'generation-repair-cancellation.json'
    if receipt.exists(): raise RuntimeError('ALREADY_RECONCILED_NO_REPEAT')
    expected={'61209':'collector','61207':'GPTJ_M1','61208':'LLAMA_REPRO'}
    before={j:job(j) for j in expected}
    def check(j):
        row=job(j)
        if not (row['UserId']=='janghj(1025)' and row['ReqNodeList']=='server4'
                and row['JobName']==TASK+'-'+expected[j]
                and row['Command']==str(old/(expected[j]+'.sh'))
                and row['WorkDir']==str(old/'source')):
            raise RuntimeError('CANCEL_IDENTITY_MISMATCH_'+j)
        return row
    for j in expected: check(j)
    result=dict(authority='USER_REQUEST_GENERATION_REPAIR_AND_RERUN',before=before,actions=[])
    def persist(): receipt.write_text(json.dumps(result,indent=2)+'\n')
    persist()
    # Downstream first: afterany cancellation must not start the old collector.
    for j in expected:
        current=check(j)
        if current['JobState'] in ('PENDING','RUNNING','CONFIGURING','COMPLETING'):
            subprocess.run(['scancel',j],check=True)
            result['actions'].append(dict(job_id=j,action='scancel',before=current))
            persist()
    result['after']={j:job(j) for j in expected};persist()
    print(json.dumps({j:r['JobState'] for j,r in result['after'].items()}))


if __name__=='__main__':main()
