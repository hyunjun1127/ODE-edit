"""사용자 중단 후 기존 scalar/compact 산출물만 게시. 모델/Slurm 호출 없음."""
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from project.run_scripts.jlz_two_arm.collect import (
    read, sha, atomic_json, csv_table, extract_rows, validate_rows,
    reduce_rows, validate_ledger,
)
LOCAL = Path('/data/janghj/ODE-edit/local/state/user-stop-all-20261002-v1')
RUN = Path('/data/janghj/ODE-edit/local/jlz-twoarm/20261002-bs100x20-v1/attempt-r2-source-inventory')
AUDIT = ROOT / 'audits/servers/server4/2026-10-02-user-stop-all'
REPORT = ROOT / 'experiment-reports/servers/server4/jlz-twoarm-bs100x20-20261002-v1/user-stop-20261002-r1'


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    evidence = read(LOCAL / 'cancellation-evidence.json')
    accounts = list(csv.DictReader(evidence['accounting']['output'].splitlines(), delimiter='|'))
    for row in accounts:
        # 종료한 미할당 dependency는 start=None, elapsed=0, TRES 빈 값의 결합으로0.
        allocated = row['AllocTRES']
        gpu = 1 if 'gres/gpu=1' in allocated else 0
        row['allocated_gpu_seconds'] = gpu * int(row['ElapsedRaw'])
    if not (REPORT / 'parent-accounting.csv').exists():
        csv_table(REPORT / 'parent-accounting.csv', accounts)
    stop = dict(instruction_id=evidence['instruction_id'], nonce=evidence['nonce'],
                cancel_actor='GH', cancel_time_utc='2026-10-02T03:01:58Z',
                cancel_order=['56962', '56960,56961'], cancel_rc=[0, 0],
                actor_evidence=evidence['actor_evidence'], sh4_cancel_calls=0,
                host=evidence['host'], session=evidence['session'], owner=evidence['user'],
                queue_snapshot_utc=evidence['observed_utc'], active_owned_jobs=0,
                queue_stdout=evidence['queue_after']['output'],
                states={r['JobIDRaw']:r['State'] for r in accounts if r['JobIDRaw'] in ['56960','56961','56962']},
                monitoring_active=False, automatic_resume=False, new_gpu_calls=0,
                checkpoint_saved=False, raw_disposition='KEEP',
                source='2af2af3ba7a4e2e6b5d0ac5c8e53841fcb275b1b',
                execution_lock_sha256=sha(RUN/'execution.lock.json'),
                submission_sha256=sha(RUN/'submission.json'),
                status='STOPPED_USER', new_experiment_authorization_required=True,
                local_evidence=dict(path=str(LOCAL/'cancellation-evidence.json'),sha256=sha(LOCAL/'cancellation-evidence.json')))
    if (AUDIT/'cancellation-receipt.json').exists():
        assert read(AUDIT/'cancellation-receipt.json')==stop
    else:
        atomic_json(AUDIT / 'cancellation-receipt.json', stop)
    config = read(RUN/'config.json')
    data = read(config.get('stream_path', config.get('stream')))
    records = {r['case_id']:r for r in data}; ids=[r['case_id'] for r in data[:2000]]
    metrics=[]; commits=[]; chains={}; inventory=[]
    for stage, size in [('pilot',4),('main',100)]:
        for arm in ['A','B']:
            folder=RUN/f'{stage}-{arm}'; ledger=read(folder/'ledger.json')
            chain=validate_ledger(ledger,ids,stage=stage,schedule=config['schedules'][stage])
            chain.update(status='COMPLETED' if stage=='pilot' else 'PARTIAL_CANCELLED_USER',
                committed_requests=len(ledger)*size, next_entry_exists=(folder/f'B{len(ledger)+1:03d}-entry.json').exists(),
                terminal_recorded=(folder/'terminal.json').exists(), interrupted_oracle_calls='NOT_RECORDED',
                cancellation_rollback='NOT_VERIFIED',exact_resume='NOT_AVAILABLE')
            chains[f'{stage}-{arm}']=chain
            for item in ledger:
                b=item['batch']; receipt=folder/f'B{b:03d}-committed.json'
                assert read(receipt)==item and item['checkpoint_saved'] is False
                assert item['commit_gate']['weight_bitwise'] is True
                assert item['config_sha256']==sha(RUN/'config.json')
                commits.append(dict(stage=stage,arm=arm,batch=b,calls=item['solver']['calls'],
                    status=item['solver']['status'],history_appends=item['history_appends'],seconds=item.get('seconds'),
                    receipt=str(receipt),sha256=sha(receipt)))
                # 마지막 endpoint와 full-N milestone만 최소 산술 재집계. 새평가0.
                if b not in {len(ledger),10}: continue
                p=folder/f'W{b:02d}-observations.json'; obj=read(p); rows=extract_rows(obj)
                seen=ids[:b*size]; current=ids[(b-1)*size:b*size]
                validate_rows(rows,records,{'R':seen,'P':seen,'N':seen if b in (5,10,20) else current})
                assert obj['state']==item['post'] and obj['no_mutation'] and not obj['optimizer_feedback']
                if item['observation'].get('sha256'):
                    assert sha(p)==item['observation']['sha256']
                for kind,values in reduce_rows(rows).items():
                    metrics.append(dict(stage=stage,arm=arm,batch=b,panel='allseen' if kind!='N' or b in (5,10,20) else 'current',kind=kind,**values))
            for p in sorted(folder.iterdir()):
                if not p.is_file():continue
                # 작은 종료 scalar 파일만 해시; model/teacher/CP는 재해시하지 않음.
                inventory.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),disposition='LOCAL_KEEP_GIT0'))
    csv_table(REPORT/'stored-endpoint-metrics.csv',metrics)
    csv_table(REPORT/'commit-summary.csv',commits)
    atomic_json(REPORT/'raw-artifact-inventory.json',inventory)
    summary=dict(status='STOPPED_USER_PARTIAL',chains=chains,planned_main_commits=40,
        committed_main_batches=sum(x['commits'] for k,x in chains.items() if k.startswith('main')),
        interrupted_main_batches=['A/B012','B/B012'],final_W20='NOT_MEASURED',
        final_2k_baseline_comparison='NOT_MEASURED',main_parent_gpu_seconds=49748,
        current_attempt_parent_gpu_seconds=sum(r['allocated_gpu_seconds'] for r in accounts if int(r['JobIDRaw'])>=56957),
        prior_failed_gpu_seconds=1116,research_parent_gpu_seconds=sum(r['allocated_gpu_seconds'] for r in accounts),
        allocation_is_not_utilization=True,numerical_certification='NOT_ESTABLISHED',
        checkpoint_saved=False,exact_resume='NOT_AVAILABLE',new_model_calls=0,
        independent_reducer='기존 collect.py의 scalar reducer만 재사용; full20 collector 실행0',
        independent_agent_review='이번 게시에서 미실시; owner 최소검산',
        artifact_broadcast='NO_BROADCAST_NOT_REQUIRED')
    atomic_json(REPORT/'summary.json',summary)
    print(json.dumps({'summary':summary,'last_main_metrics':[r for r in metrics if r['stage']=='main' and r['batch']==11]},ensure_ascii=False))


if __name__=='__main__': main()
