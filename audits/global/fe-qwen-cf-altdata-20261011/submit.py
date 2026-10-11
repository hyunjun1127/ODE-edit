"""Register one native training/evaluation lane without changing existing jobs."""
import json
import os
from pathlib import Path
import sys
import traceback

WORKTREE = Path('/mnt/raid5/janghj/ODE-edit/local/worktrees/fe-qwen-cf-altdata-20261011')
ROOT = Path(__file__).resolve().parent / 'native-r1'
sys.path.insert(0, str(WORKTREE))
from official.runners.server1 import submit as ctl
from official.runners.server1.common import member, read, verify
from official.experiments.prepare import digest, write_new

assert not (ROOT / 'submission.json').exists()
prepared = read(ROOT / 'preparation.json')
config = read(verify(prepared['config']))
assert digest({k:v for k,v in config.items() if k != 'identity_sha256'}) == config['identity_sha256']
assert config['model_dtype'] == 'bfloat16' and config['current_batch_targets_only']
assert config['history'] == 'native add_old_keys=True' and not config['covariance_repair']
assert config['entrypoint'] == 'python main.py'
selection = read(verify(prepared['selection']))
assert selection['source_slice_zero_based'] == [2000, 4000] and selection['case_overlap_with_reference'] == 0
lock = read(verify(prepared['execution_lock']))
for entry in config['source_members'] + lock['source_members'] + lock['configs'] + prepared['scripts']:
    verify(entry)
assert os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize > 110 * 2**30
before = ctl.inventory()
assert before['allocated_gpus'] <= 2 and before['admitted_DAG_width'] <= 2, 'NEW_LANE_REQUIRES_AVAILABLE_DIAGNOSTIC_CAP3'
write_new(ROOT / 'queue-before.json', before)
jobs = {}
submitted = []
released = False
try:
    for mode, script, cwd, mem in [('train', 'qwen25.sh', ROOT/'author', 96),
                                  ('evaluate', 'qwen25-eval.sh', ROOT/'evaluation-source', 64)]:
        name = 's1-qwen25-cf-fe-next2k-' + mode
        argv = ['sbatch', '--hold', '--parsable', '--job-name='+name, '--partition=gpu',
                '--nodelist=devbox', '--qos=lab_gpu_s1', '--gres=gpu:1', '--cpus-per-task=8',
                f'--mem={mem}G', '--time=12:00:00', '--export=NONE', '--no-requeue',
                '--chdir='+str(cwd), '--output='+str(ROOT/(mode+'-%j.out')),
                '--error='+str(ROOT/(mode+'-%j.err'))]
        parent = jobs['train']['job_id'] if mode == 'evaluate' else None
        if parent:
            argv += ['--dependency=afterany:'+parent]
        argv.append(str(ROOT/script))
        job = ctl.command(argv).split(';')[0]
        assert job.isdigit()
        submitted.append(job)
        detail = ctl.metadata(ctl.command(['scontrol', 'show', 'job', job, '-o']))
        assert detail['JobState'] == 'PENDING' and detail['Reason'] == 'JobHeldUser'
        assert detail['Command'] == str(ROOT/script) and detail['WorkDir'] == str(cwd)
        assert detail['UserId'].split('(')[0] == 'janghj' and detail['ReqNodeList'] == 'devbox'
        assert detail['QOS'] == 'lab_gpu_s1' and ctl.gpu_count(detail['ReqTRES']) == 1
        assert ctl.gpu_count(detail.get('AllocTRES', '')) == 0 and ctl.requested_cpu_matches(detail, 8)
        assert ctl.memory_MiB(detail['MinMemoryNode']) == mem * 1024
        assert ctl.dependencies(detail['Dependency']) == ([('afterany', parent)] if parent else [])
        assert ctl.command(['scontrol', 'write', 'batch_script', job, '-']).strip() == (ROOT/script).read_text().strip()
        jobs[mode] = dict(job_id=job, name=name, argv=argv, held_detail=detail, script=member(ROOT/script))
    held = ctl.inventory()
    assert held['allocated_gpus'] <= 3 and held['admitted_DAG_width'] <= 3
    write_new(ROOT/'held-inspection.json', dict(jobs=jobs, inventory=held))
    # Releasing the dependent job first preserves a single GPU lane in every ordering.
    for job in reversed(submitted):
        ctl.command(['scontrol', 'release', job])
    released = True
    after = ctl.inventory()
    assert after['allocated_gpus'] <= 3 and after['admitted_DAG_width'] <= 3
    for old in before['jobs']:
        current = next((row for row in after['jobs'] if row['key'] == old['key']), None)
        if current:
            for key in ('Command', 'WorkDir', 'JobName', 'ReqTRES', 'QOS', 'TimeLimit'):
                assert current['detail'][key] == old['detail'][key]
    receipt = dict(schema='fe-qwen-cf-altdata-submission-v1', jobs=jobs, after=after,
                   config=prepared['config'], selection=prepared['selection'],
                   author_source_commit=prepared['author_source_commit'], native_patch_source=prepared['native_patch_source'],
                   evaluator_source_commit=prepared['evaluator_source_commit'],
                   normal_cap=2, authorized_diagnostic_cap=3, additional_lanes=1,
                   authorization='User requested data-only Qwen FE retry; existing FE diagnosis server1 cap2 exception',
                   existing_job_mutations=0, scientific_change='CF fixed10k positions 2001-4000 only',
                   GPU_results='NOT_OBSERVED')
    write_new(ROOT/'submission.json', receipt)
    print(json.dumps({mode: {key: value for key, value in entry.items() if key in ('job_id', 'name')}
                      for mode, entry in jobs.items()}, indent=2), flush=True)
except BaseException:
    if not released:
        for job in submitted:
            ctl.command(['scancel', job])
    write_new(ROOT/'submission-failure.json', dict(submitted=submitted, released=released, error=traceback.format_exc()))
    raise
