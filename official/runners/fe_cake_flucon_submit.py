"""Held FE/CAKE Flu/Con registration: frozen source, inspect, release; no retry.

Same pattern as official.runners.server2.flucon_submit: sbatch --hold, inspect the
held jobs (owner, script, node, QOS, resources, dependency), then release. Each
--lane is one serial afterany chain on one GPU; existing own GPU jobs on the node
count toward the cap (gpu-caps.tsv, at most 2). A CPU collector follows all lanes.
"""
import argparse
import getpass
import re
import shlex
import shutil
from pathlib import Path

from official.experiments.prepare import digest, write_new
from official.runners.server1 import submit as control
from official.runners.server1.common import member, read, verify
from .fe_cake_flucon import AUTHORITY, LOCAL, NODES, require, tracking_values

PYTHON = '/mnt/raid5/janghj/EasyEdit/.venv/bin/python'
CAPS = Path('/mnt/raid5/janghj/ODE-edit/servers/local/gpu-caps.tsv')
RESOURCES = dict(
    server1=dict(node='devbox', qos='lab_gpu_s1', partition='gpu', cpus=8, memory_MiB=65536, gres='gpu:1'),
    server2=dict(node='server2', qos='lab_gpu_s2', partition='gpu', cpus=6, memory_MiB=59392, gres='gpu:a6000:1'))
COLLECTOR_MiB, GPU_HOURS, COLLECTOR_HOURS = 24576, 48, 4


def lane_rows(lanes):
    """Planned GPU jobs as graph rows: one serial chain per lane."""
    rows = []
    for lane in lanes:
        for index, key in enumerate(lane):
            rows.append(dict(key='planned-' + key, gpus=1, parents=['planned-' + lane[index - 1]] if index else []))
    return rows


def register(server, preparation, attempt, lanes):
    require(server in NODES, 'SERVER')
    res = RESOURCES[server]
    preparation = Path(preparation).absolute(); prep = read(preparation)
    require(prep['registration_authority'] == AUTHORITY and prep['server'] == server and prep['configs'], 'PREPARATION')
    configs = {c['key']: (m, c) for m in prep['configs'] for c in [read(verify(m))]}
    flat = [key for lane in lanes for key in lane]
    require(lanes and len(flat) == len(set(flat)) and set(flat) == set(configs), 'LANES_COVER_PREPARATION_ONCE')
    require(len({c['original']['checkpoint']['sha256'] for _, c in configs.values()}) == len(configs), 'DISTINCT_CHECKPOINTS')
    attempt = Path(attempt).absolute()
    require(attempt.is_relative_to(LOCAL) and not attempt.exists(), 'NEW_ATTEMPT_UNDER_TASK')
    repo = Path(__file__).resolve().parents[2]
    git = lambda *argv: control.command(['git', *argv], cwd=repo)
    source, tree = git('rev-parse', 'HEAD'), git('rev-parse', 'HEAD:official')
    git('merge-base', '--is-ancestor', source, 'origin/main')
    require(not git('status', '--porcelain', '--', 'official'), 'CLEAN_OFFICIAL_TREE')
    for _, c in configs.values():
        require(c['registration_authority'] == AUTHORITY and c['server'] == server
                and c['config_sha256'] == digest({k: v for k, v in c.items() if k != 'config_sha256'}), 'CONFIG')
        o = c['original']
        for m in (o['checkpoint'], o['pointer'], o['terminal'], o['config'], c['assets'], c['stream'],
                  c['reference'], c['evaluator']):
            verify(m)
        tracking_values(c, source)
    caps = [r.split('\t') for r in CAPS.read_text().splitlines() if r.startswith(server + '\t')]
    require(len(caps) == 1 and caps[0][1] == res['node'], 'CAP_ROW')
    cap = min(2, int(caps[0][2])); require(cap > 0 and res['memory_MiB'] <= int(caps[0][3]), 'CAP_MEMORY')
    existing = control.inventory(node=res['node'])
    planned = lane_rows(lanes)
    width = control.graph_width(existing['jobs'] + planned)
    require(width == existing['admitted_DAG_width'] + len(lanes) <= cap, 'GPU_CAP')
    free = shutil.disk_usage(LOCAL).free
    require(free >= (4 + len(flat)) * (1 << 30), 'STORAGE')
    attempt.mkdir(parents=True); (attempt / 'scripts').mkdir(); (attempt / 'logs').mkdir()
    write_new(attempt / 'admission.json', dict(instruction=AUTHORITY, server=server, resources=res,
        existing={k: v for k, v in existing.items() if k != 'jobs'},
        existing_jobs=[{k: v for k, v in row.items() if k not in ('detail', 'original')} for row in existing['jobs']],
        cap=cap, lanes=lanes, admitted_width_after=width, cap_member=member(CAPS), disk_free_bytes=free,
        existing_jobs_changed=False))
    frozen = control.freeze_source(dict(source=dict(main_commit=source, official_tree=tree)), attempt,
                                   repository=repo, runner=control.command)
    lock = dict(instruction=AUTHORITY, source_commit=source, official_tree=tree, source_directory=frozen['directory'],
        source_members=frozen['members'], configs=prep['configs'], preparation=member(preparation),
        qualification='NOT_RUN_USER_DISABLED')
    lock_path = attempt / 'execution-lock.json'; write_new(lock_path, lock)
    env = dict(PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
        OMP_NUM_THREADS=str(res['cpus']), MKL_NUM_THREADS=str(res['cpus']), OPENBLAS_NUM_THREADS=str(res['cpus']),
        OFFICIAL_CODE_COMMIT=source, OFFICIAL_TREE_SHA256=tree, WANDB_CONSOLE='off', WANDB_SAVE_CODE='false')
    jobs, held = {}, {}
    entries = [(key, lane[i - 1] if i else None) for lane in lanes for i, key in enumerate(lane)] + [('collector', None)]
    prefix = {'server1': 's1-flucon-', 'server2': 's2-flucon-'}[server]
    for key, parent in entries:
        gpu = key != 'collector'
        parents = [jobs[parent]['job_id']] if gpu and parent else ([] if gpu else [j['job_id'] for j in jobs.values()])
        argv = [PYTHON, '-B', '-u', '-m', 'official.runners.fe_cake_flucon']
        argv += (['run', '--config', configs[key][0]['path'], '--lock', str(lock_path)] if gpu else
                 ['collect', '--preparation', str(preparation), '--output', str(attempt / 'collector-results.json'),
                  '--lock', str(lock_path)])
        values = dict(env) if gpu else dict(env, CUDA_VISIBLE_DEVICES='')
        text = '#!/bin/bash\nset -euo pipefail\n' + ''.join('export ' + k + '=' + shlex.quote(v) + '\n' for k, v in values.items())
        text += 'cd ' + shlex.quote(frozen['directory']) + '\nexec ' + shlex.join(argv) + '\n'
        script = attempt / 'scripts' / f'{key}.sh'
        with script.open('x') as handle:
            handle.write(text)
        mem, hours, name = (res['memory_MiB'] if gpu else COLLECTOR_MiB), (GPU_HOURS if gpu else COLLECTOR_HOURS), prefix + key
        cmd = ['sbatch', '--parsable', '--hold', '--export=NONE', '--no-requeue', '--partition=' + res['partition'],
            '--qos=' + res['qos'], '--nodelist=' + res['node'], '--nodes=1', '--ntasks=1',
            f"--cpus-per-task={res['cpus']}", f'--mem={mem}M', '--time=' + ('2-00:00:00' if gpu else '04:00:00'),
            '--job-name=' + name, '--chdir=' + frozen['directory'],
            '--output=' + str(attempt / 'logs' / f'{key}-%j.out'), '--error=' + str(attempt / 'logs' / f'{key}-%j.err')]
        if gpu:
            cmd += ['--gres=' + res['gres']]
        if parents:
            cmd += ['--dependency=afterany:' + ':'.join(parents)]
        cmd += [str(script)]
        response = control.command(cmd); job_id = response.split(';')[0]
        require(re.fullmatch(r'\d+', job_id), 'SBATCH_RESPONSE')
        deps = [('afterany', p) for p in parents]
        jobs[key] = dict(job_id=job_id, name=name, config=configs[key][0] if gpu else None, script=member(script),
            argv=argv, dependencies=deps, gpus=int(gpu),
            output=configs[key][1]['output'] if gpu else str(attempt / 'collector-results.json'))
        write_new(attempt / f'submitted-{key}.json', dict(response=response, command=cmd, **jobs[key]))
        d = control.metadata(control.command(['scontrol', 'show', 'job', job_id, '--oneliner']))
        require(d['UserId'].split('(')[0] == getpass.getuser() and d['JobState'] == 'PENDING'
                and d['Reason'] == 'JobHeldUser' and d['Command'] == str(script) and d['WorkDir'] == frozen['directory']
                and d['JobName'] == name and d['ReqNodeList'] == res['node'] and d['QOS'] == res['qos']
                and d['Partition'] == res['partition'] and d['Requeue'] == '0', 'HELD_JOB_FIELDS')
        require(control.gpu_count(d['ReqTRES']) == int(gpu) and control.gpu_count(d.get('AllocTRES', '')) == 0
                and control.requested_cpu_matches(d, res['cpus']) and control.memory_MiB(d['MinMemoryNode']) == mem
                and control.seconds(d['TimeLimit']) == hours * 3600
                and control.dependencies(d.get('Dependency', '')) == sorted(deps), 'HELD_JOB_RESOURCES')
        require(control.command(['scontrol', 'write', 'batch_script', job_id, '-']).strip() == text.strip(), 'HELD_SCRIPT')
        held[key] = d
    fresh = control.inventory(node=res['node'])
    require(fresh['admitted_DAG_width'] <= cap, 'GPU_CAP_AFTER_HOLD')
    write_new(attempt / 'held-inspection.json', dict(jobs=held, inventory={k: v for k, v in fresh.items() if k != 'jobs'},
        inventory_jobs=[{k: v for k, v in r.items() if k not in ('detail', 'original')} for r in fresh['jobs']],
        lock=member(lock_path), GPU_qualification=False))
    for job in jobs.values():
        control.command(['scontrol', 'release', job['job_id']])
    snapshot = {k: control.metadata(control.command(['scontrol', 'show', 'job', j['job_id'], '--oneliner']))
                for k, j in jobs.items()}
    write_new(attempt / 'submission.json', dict(instruction=AUTHORITY, server=server, source=source, official_tree=tree,
        jobs=jobs, lanes=lanes, execution_lock=member(lock_path), released=True, held_inspected=True,
        initial_snapshot=snapshot, online_readback='NOT_OBSERVED_BEFORE_STARTUP', existing_jobs_changed=False))
    print({k: dict(job_id=j['job_id'], state=snapshot[k]['JobState'], dependencies=j['dependencies'])
           for k, j in jobs.items()}, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--server', required=True, choices=sorted(NODES)); p.add_argument('--preparation', required=True)
    p.add_argument('--attempt', required=True); p.add_argument('--lane', action='append', required=True)
    a = p.parse_args()
    register(a.server, a.preparation, a.attempt, [lane.split(',') for lane in a.lane])
