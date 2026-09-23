"""CPU publication of already captured control evidence; no scheduler/result query."""
import argparse
import json
from pathlib import Path
import re
from .common import file_sha,save,digest
from .control import dependency_matches

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--observation',type=Path,required=True)
    a=p.parse_args();c=a.root/'controls/attempt-r3'
    lock=json.loads((c/'execution.lock.json').read_text());s=json.loads((c/'submission.json').read_text());h=json.loads((c/'held-inspection.json').read_text());admission=json.loads((c/'admission.json').read_text())
    obs=json.loads(a.observation.read_text())
    assert s['status']=='REGISTERED_INSPECTED_RELEASED' and h['status']=='PASS'
    assert s['jobs']==h['jobs']=={k:v['id'] for k,v in obs['jobs'].items()}
    assert obs['jobs']['gate']['reason']=='Resources' and obs['jobs']['gate']['priority']>0
    assert obs['node_resources']['GPU_configured']==obs['node_resources']['GPU_allocated']==8
    assert all(x['state']=='PENDING' for x in obs['jobs'].values())
    events=[json.loads(x) for x in (c/'submission-events.jsonl').read_text().splitlines()]
    for phase,job in s['jobs'].items():
        assert dependency_matches(h['inspection'][phase],s['dependency_graph'][phase])
        assert sum(x.get('action')=='release' and x.get('job_id')==job for x in events)==1
        assert sum(x.get('inspection')=='PASS' and x.get('job_id')==job for x in events)==1
    evidence=[]
    for path in [c/'execution.lock.json',c/'submission.json',c/'held-inspection.json',c/'submission-events.jsonl',c/'admission.json',c/'storage-preflight.json',a.observation,
                 a.root/'repair-r2/cpu-validation-v3/receipt.json',a.root/'failure-diagnosis-r1/diagnosis.json',a.root/'failure-diagnosis-r1/bos-cpu-reproduction.json']:
        evidence.append(dict(path=str(path),bytes=path.stat().st_size,sha256=file_sha(path)))
    package=a.repo/'experiment-reports/servers/server4/alpha-key-causal-20260923-r1/autonomous-repair-r2'
    audit=a.repo/'audits/servers/server4/alpha-key-causal-20260923-r1/autonomous-repair-r2'
    handoff=dict(observation=obs,jobs=s['jobs'],dependency_graph=s['dependency_graph'],all_held_inspected_released=True,
        source_commit=lock['source_commit'],source_tree=lock['source_tree'],archive=lock['archive'],lock_sha256=file_sha(c/'execution.lock.json'),
        attempt=lock['attempt'],allowed_phases=lock['allowed_phases'],scientific_families=94,followups='FOLLOWUP_NOT_SUBMITTED',
        project_gpu_cap=2,task_gpu_cap=2,resources=lock['resources'],reducer_resources=lock['reducer_resources'],admission=admission,
        CPU_tests=143,actual_GPU_gate='NOT_OBSERVED',prior_failed_gpu_seconds=44,prior_CPU_freeze_gpu_seconds=0,
        new_allocated_gpu_seconds_at_last_observation=0,new_actual_results='NOT_OBSERVED; no later query',
        monitoring_active=False,automatic_resume=False,new_full_state_checkpoint=False,old_source_raw_inputs='KEEP',
        publication='OWN_BRANCH_ONLY; GH owns main integration',independent_red_this_repair=False,
        evidence=evidence,evidence_root=digest(evidence),NO_BROADCAST_NOT_REQUIRED=True)
    save(package/'handoff.json',handoff)
    save(audit/'handoff.json',handoff)
    # Static links/tables only. HTML rendering is recorded separately, not visual PASS.
    report=package/'submission-handoff-ko.md';text=report.read_text()
    for link in re.findall(r'\]\(([^)]+)\)',text):
        if link=='rooted-receipt.json':continue
        assert (package/link).exists(),('BROKEN_LINK',link)
    table=[x for x in text.splitlines() if x.startswith('|')]
    assert len({len(x.split('|')) for x in table})==1
    renderer='NOT_RUN_TOOL_UNAVAILABLE'
    try:
        import markdown
    except ImportError:pass
    else:
        with (package/'submission-handoff-ko.html').open('x') as f:f.write('<!doctype html><meta charset="utf-8">'+markdown.markdown(text,extensions=['tables']))
        renderer='GENERATED_HTML; visual review NOT_RUN'
    checks=dict(status='PASS_STATIC_ONLY',source_lock_receipt_bindings=True,held_full_args_and_release_count=True,
        table_columns=True,local_links=True,HTML=renderer,independent_agent_red=False,
        no_scheduler_calls_in_publisher=True,no_model_calls=True,no_scientific_output_reads=True)
    save(package/'publication-checks.json',checks)
    members=[dict(path=x.name,bytes=x.stat().st_size,sha256=file_sha(x)) for x in sorted(package.iterdir()) if x.is_file()]
    save(package/'rooted-receipt.json',dict(members=members,root=digest(members)))
    print(json.dumps(dict(report_sha256=file_sha(report),handoff_sha256=file_sha(package/'handoff.json'),rooted_receipt_sha256=file_sha(package/'rooted-receipt.json')),indent=2))

if __name__=='__main__':main()
