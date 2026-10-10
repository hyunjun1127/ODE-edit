"""Compact factual handoff; no model/checkpoint payload publication."""
from pathlib import Path
from official.runners.server1.common import read,member
from official.experiments.prepare import write_new
ROOT=Path('/mnt/raid5/janghj/ODE-edit/local/fe-original-w0-2k-20261011')
AUDIT=Path(__file__).parent
sub=read(ROOT/'registration-r1/submission.json')
admission=read(ROOT/'registration-r1/admission.json')
rows=[]
for key,row in sub['jobs'].items():
    c=read(row['config']['path']);state=sub['initial_snapshot'][key]
    rows.append(dict(condition=key,job_id=row['job_id'],job_name=row['name'],state=state['JobState'],reason=state['Reason'],
        dependencies=row['dependencies'],config=row['config'],config_sha256=c['config_sha256'],stream=c['stream'],ordered_stream_sha256=c['stream_sha256'],
        hparams=c['llms'],query_proof=c['query_proof'],output=c['output'],latest_checkpoint=c['output']+'/checkpoint/latest.pt',
        z_cache=c['output']+'/z-cache',WB='NOT_OBSERVED_AT_REGISTRATION',generation='DEFERRED' if c['dataset']=='cf' else 'NOT_APPLICABLE'))
receipt=dict(instruction='USER-FE-ORIGINAL-W0-RESET-20261011-R1',session='01a04939-f93a-7b50-bca0-65438eab2062',
    source=sub['source'],official_tree=sub['official_tree'],execution_lock=sub['execution_lock'],
    runner=member(Path('official/runners/fe_original.py')),author_patch=member(Path('official/runners/fe_original_author.patch')),
    author_commit='478134dfb24b43f4e18b47e8500893ce3f9cc50f',author_clone=str(ROOT/'author'),author_lock=member(ROOT/'author-lock.json'),
    runtime=str(ROOT/'venv/bin/python'),resources=admission['resources'],cap=2,graph=admission['graph'],
    held_inspected=sub['held_inspected'],released=sub['released'],jobs=rows,
    CPU_tests=15,CPU_current_allseen_schema_requests=[100,500,2000],GPU_qualification='NOT_RUN_USER_DISABLED',
    GPU_scientific_completion='NOT_OBSERVED',cancelled_FE_jobs=['62530'],preserved_nonFE_jobs=['63125'],
    deleted_checkpoints=11,deleted_logical_bytes=54211593935,backup_created=False,raw_log_config_source_preserved=True,
    deletion_receipts=[member(AUDIT/n) for n in ('checkpoint-deleted.json','native-checkpoint-deleted.json','corrected-qwen-deleted.json')],
    broadcast='NO_BROADCAST_NOT_REQUIRED',README_owner='GH',automatic_retry=False)
write_new(AUDIT/'submission.json',receipt)
for name in ('checkpoint-delete-manifest.json','native-checkpoint-delete-manifest.json','corrected-qwen-delete-manifest.json'):
    # Keep originals; exact manifests and post-unlink receipts remain separate.
    assert (AUDIT/name).is_file()
write_new(AUDIT/'validation.json',dict(CPU_tests=15,passed=True,actual_GPU=False,
    command="CUDA_VISIBLE_DEVICES='' python -m unittest official.runners.test_fe_original -q",
    extra_CPU_payloads=['current/post100','all_seen/post500','all_seen/post2000'],
    source=sub['source'],checks=['BF16 raw bytes','author default attention','author exact patch/YAML','W0 z single precompute',
    'latest overwrite and restore','shared host lock and exception cleanup','stale/missing metadata recovery',
    'corrupt payload rejection','resume raw preservation','two-lane DAG','CF R/P/N and official counts','tracking config']))
print(dict(source=sub['source'],jobs={r['condition']:r['job_id'] for r in rows},deleted_bytes=54211593935))
