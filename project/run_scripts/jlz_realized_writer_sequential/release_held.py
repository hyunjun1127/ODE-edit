"""One-shot narrow repair of held registration; NEVER submit/retry science."""
import argparse, getpass, json, os, subprocess
from pathlib import Path
from project.run_scripts.jlz_shared_budget.submit import resource_inventory
from .common import *
from .submit import STAGES, verify_frozen, inspect, command, verify_parallel_dependencies

def main():
    p = argparse.ArgumentParser(); p.add_argument('--attempt', type=Path, required=True); p.add_argument('--cpu-receipt', type=Path, required=True)
    args = p.parse_args(); attempt = args.attempt.resolve()
    require(attempt == LOCAL / 'attempt-r1', 'EXACT_EXISTING_HELD_ATTEMPT')
    require(not (attempt / 'submission.json').exists() and not list(attempt.glob('released-*.json')), 'NO_REPEAT_RELEASE')
    lock, c = verify_frozen(attempt)
    require(lock['source_commit'] == '2ff0ecc63dda2c830fbd68e6a2d76f3317a446f7', 'ORIGINAL_SOURCE_KEEP')
    cpu = json.loads(args.cpu_receipt.read_text()); require(cpu['passed'], 'REGISTRATION_REPAIR_CPU')
    for row in cpu['source']:
        verify(row)
    mapping = {}; ids = {}; checks = []
    for stage in STAGES:
        r = json.loads((attempt / ('submitted-' + stage + '.json')).read_text())
        require(r['nonce'] == NONCE and r['status'] == 'HELD', 'EXACT_REGISTRATION_NONCE')
        ids[stage] = r['job']; mapping[stage] = {k: r[k] for k in ('job', 'dependency', 'argv')}
        checks.append(inspect(r['job'], stage, r['dependency'], r['argv'], attempt, c['resources']))
    require(len(set(ids.values())) == 3, 'UNIQUE_EXACT_JOBS')
    barrier = verify_parallel_dependencies(ids, mapping)
    before = resource_inventory(tuple(ids.values()))
    caps = Path('/data/janghj/ODE-edit/servers/local/gpu-caps.tsv').read_text()
    local = int(next(s for s in caps.splitlines() if s.startswith('server4\t')).split('\t')[2])
    tracked = int(next(s for s in (ROOT / 'control/gpu-concurrency-policy.tsv').read_text().splitlines() if s.startswith('server4\t')).split('\t')[1])
    cap = min(3, local, tracked)
    require(cap >= 2 and (barrier or sum(j['gpus'] for j in before['jobs']) + 2 <= cap), 'PROJECT_CAP_KEEP_HELD')
    helper = subprocess.run(['bash', str(ROOT / 'scripts/check-slurm-resource-cap.sh'), 'server4', '2', '59392M'],
        env=dict(os.environ, AGENT_GPU_CAPS_FILE='/data/janghj/ODE-edit/servers/local/gpu-caps.tsv'), capture_output=True, text=True)
    require(helper.returncode == 0 or (helper.returncode == 4 and barrier), 'RESOURCE_HELPER_KEEP_HELD')
    verify_frozen(attempt)
    write(attempt / 'held-inspection.json', dict(checks=checks, before=before, prerelease=before, project_cap=cap, task_cap=2,
        aggregate_new_GPU=2, aggregate_new_CPU=16, aggregate_new_host_mib=118784,
        identical_resource_barrier=sorted(barrier), helper=dict(rc=helper.returncode, output=helper.stdout + helper.stderr),
        partition=command(['scontrol', 'show', 'partition', 'gpu']), node=command(['scontrol', 'show', 'node', 'server4']),
        all_held_before_release=True, no_other_job_mutation=True, arm_science_dependency=False,
        narrow_repair='None empty dependency TypeError before any release; re-inspect original jobs, no new submissions',
        repair_source=command(['git', 'rev-parse', 'HEAD'], cwd=ROOT), repair_CPU_receipt=member(args.cpu_receipt), original_execution_source=lock['source_commit']))
    for stage in reversed(STAGES):
        response = command(['scontrol', 'release', ids[stage]])
        write(attempt / ('released-' + stage + '.json'), dict(job=ids[stage], success=True, result=response, new_submission=False))
    write(attempt / 'submission.json', dict(nonce=NONCE, status='RELEASED', jobs=ids, mapping=mapping, source=lock['source_commit'],
        lock=member(attempt / 'execution.lock.json'), held=member(attempt / 'held-inspection.json'),
        repair_source=command(['git', 'rev-parse', 'HEAD'], cwd=ROOT), duplicate_submissions=0,
        main_initial='NOT_OBSERVED', actual_new_GPU='NOT_OBSERVED', monitoring_active=False, automatic_resume=False))
    print(json.dumps(dict(status='RELEASED', jobs=ids, original_source=lock['source_commit'], new_submissions=0)))

if __name__ == '__main__':
    main()
