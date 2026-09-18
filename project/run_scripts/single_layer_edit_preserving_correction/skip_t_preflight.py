"""Reuse prior FULL_READ/CPU82; check only waiver import/routing and exact source delta."""
import ast
import datetime
import json
from pathlib import Path
import subprocess
import sys
from .common import ROOT,member,sha,write
from .control import PACKAGE


def main():
    repo=Path.cwd();out=ROOT/'receipts/skip-t-r1'
    prior_path=ROOT/'receipts/full-read-m0.json'
    prior=json.loads(prior_path.read_text());reused=[]
    for item in prior['files']:
        p=Path(item['path'])
        if sha(p)!=item['sha256']:raise ValueError('FULL_READ_IDENTITY_CHANGED:'+str(p))
        reused.append(dict(path=str(p),sha256=item['sha256'],read_level='EXACT_PRIOR_FULL_READ_REUSED'))
    names=['messages/head/2026-09-18-sh4-enfc-skip-t-all-m.md',
        'plans/global/2026-09-18-single-layer-edit-preserving-correction-sh4-dispatch/skip-t-all-m-override.json']
    new=[]
    for relative in names:
        p=repo/relative;data=p.read_bytes()
        target=out/'authoritative'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as f:f.write(data)
        new.append(dict(source=member(p),sealed=member(target),read_level='FULL_READ'))
    old=json.loads((ROOT/'M/attempt-v1/execution.lock.json').read_text())
    changes=[]
    for p in sorted((repo/PACKAGE).rglob('*.py')):
        ast.parse(p.read_bytes())
        previous=Path(old['source_root'])/p.relative_to(repo)
        if not previous.exists() or sha(previous)!=sha(p):changes.append(str(p.relative_to(repo)))
    run=subprocess.run([sys.executable,'-B','-m','unittest',
        'project.run_scripts.single_layer_edit_preserving_correction.tests.test_skip_t','-v'],
        text=True,capture_output=True)
    if run.returncode:raise ValueError(run.stdout+run.stderr)
    subprocess.run(['bash','-n',str(repo/PACKAGE/'run.sbatch')],check=True)
    from . import skip_t_control,runner
    result=dict(status='NARROW_CPU_ROUTING_PASS_NOT_NUMERICAL_VALIDATION',
        time=datetime.datetime.now(datetime.timezone.utc).isoformat(),new_T=0,new_model_load=0,
        full_numerical_validation='NOT_ESTABLISHED',T='SKIPPED_USER_DIRECTED',
        prior_fullread=member(prior_path),prior_CPU82=member(ROOT/'receipts/cpu-preflight-paired-stop-r1.json'),
        reused_read=reused,new_read=new,source_delta_from_M76bb903=changes,
        unit_output=run.stdout+run.stderr,unit_count=4,imports='runner/skip_t_control import only',
        old_terminal=member(ROOT/'receipts/paired-stop-r1/scheduler-terminal.json'),
        prior_cost_GPU_seconds=3231,source_members=[member(repo/PACKAGE/n) for n in
            ('skip_t_control.py','validation_route.py','runner.py','tests/test_skip_t.py')])
    print(json.dumps(write(out/'preflight-fullread.json',result)))


if __name__=='__main__':main()
