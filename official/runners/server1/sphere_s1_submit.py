"""Held registration of the re-authorized Qwen SPHERE jobs on server1 (devbox): one lane, no retry.

User 2026-10-11, "server1 cap 남으니 그냥 올려": resubmit on server1 only the cancelled 62873
(CF Flu/Con of the SPHERE W20 of 62079) and 63303 (zsRE B11-B20 resume from the 62538 B10).
Both join the existing own devbox lane after its last GPU job, tail -> Flu/Con -> resume, with
afterany edges, so the own devbox GPU DAG width does not grow (server1 cap 2, gpu-caps.tsv).
The resume job evaluates B15/B20 inline (as 63303), so it is one GPU job.

Same pattern as fe_cake_flucon_submit and server4/sphere_b10_submit: create-once attempt, a frozen
`git archive` of a main commit, sbatch --hold, exact held inspection (scontrol show job and
write batch_script), then release. The launchers are the cancelled jobs' launchers with only host
bindings replaced (checked). No existing job is changed and nothing is retried.
"""
import argparse
import getpass
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import traceback

from official.experiments.prepare import digest, file_sha, write_new
from official.runners.server1 import submit as control
from official.runners.server1 import sphere_flucon_s1 as flucon
from official.runners.server1 import sphere_resume_s1 as resume
from official.runners.server1.common import member, read, verify
from official.runners.server1.sphere_s1_common import AUTHORITY, LOCAL, received, require, transfer_rows

PYTHON = '/mnt/raid5/janghj/EasyEdit/.venv/bin/python'
CAPS = Path('/mnt/raid5/janghj/ODE-edit/servers/local/gpu-caps.tsv')
NODE, QOS, PARTITION, GRES = 'devbox', 'lab_gpu_s1', 'gpu', 'gpu:1'
CPUS, MEMORY_MiB, WALL = 6, 59392, '2-00:00:00'  # 62873 and 63303 both: 6 CPUs, 59392M, 48 h
NAMES = dict(flucon='s1-flucon-qwen25-sphere-62079', resume='s1-qwen25-zsre-sphere-resume-b10')
CANCELLED_NAMES = ('s2-flucon-qwen25-sphere-62079', 's4-qwen25-zsre-sphere-resume-b10')
FLUCON_ENV = ('PYTHONDONTWRITEBYTECODE', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'TOKENIZERS_PARALLELISM',
              'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'OFFICIAL_CODE_COMMIT',
              'OFFICIAL_TREE_SHA256', 'WANDB_CONSOLE', 'WANDB_SAVE_CODE')  # 62873's launcher order
RESUME_ENV = ('PYTHONDONTWRITEBYTECODE', 'PYTHONUNBUFFERED', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE',
              'TOKENIZERS_PARALLELISM', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'WANDB_CONSOLE', 'WANDB_SAVE_CODE')  # 63303's launcher order
S4_ROOT = resume.SERVER4_ROOT


def launcher(keys, values, workdir, argv):
    """The 62873/63303 launcher form: exports in order, cd to the frozen source, exec."""
    return ('#!/bin/bash\nset -euo pipefail\n' + ''.join('export ' + k + '=' + shlex.quote(values[k]) + '\n' for k in keys)
            + 'cd ' + shlex.quote(str(workdir)) + '\nexec ' + shlex.join(argv) + '\n')


def environment(source, tree):
    return dict(PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS=str(CPUS), MKL_NUM_THREADS=str(CPUS),
                OPENBLAS_NUM_THREADS=str(CPUS), OFFICIAL_CODE_COMMIT=source, OFFICIAL_TREE_SHA256=tree,
                WANDB_CONSOLE='off', WANDB_SAVE_CODE='false')


def replaced(text, pairs):
    for old, new in pairs:
        require(old in text, 'LAUNCHER_TEMPLATE_TOKEN:' + old)
        text = text.replace(old, new)
    return text


def lane_rows(jobs, tail):
    """Existing own devbox GPU jobs plus the planned chain tail -> flucon -> resume."""
    rows = [dict(key=r['key'], gpus=r['gpus'], parents=list(r['parents'])) for r in jobs]
    return rows + [dict(key='planned-flucon', gpus=1, parents=[str(tail)]),
                   dict(key='planned-resume', gpus=1, parents=['planned-flucon'])]


def admit(existing, tail, cap):
    jobs = existing['jobs']
    require(str(tail) in {r['key'] for r in jobs if r['gpus']}, 'LANE_TAIL_NOT_OWN_DEVBOX_GPU_JOB')
    require(str(tail) in control.frontier(jobs), 'LANE_TAIL_HAS_GPU_DESCENDANT')
    width = control.graph_width(lane_rows(jobs, tail))
    require(width <= cap and width <= max(1, existing['admitted_DAG_width']), 'SERVER1_GPU_CAP')
    return width


def held_fields(job_id, name, script, workdir, dependency):
    d = control.metadata(control.command(['scontrol', 'show', 'job', job_id, '--oneliner']))
    require(d['UserId'].split('(')[0] == getpass.getuser() and d['JobState'] == 'PENDING' and d['Reason'] == 'JobHeldUser'
            and d['Command'] == str(script) and d['WorkDir'] == str(workdir) and d['JobName'] == name
            and d['ReqNodeList'] == NODE and d['QOS'] == QOS and d['Partition'] == PARTITION and d['Requeue'] == '0',
            'HELD_JOB_FIELDS')
    require(control.gpu_count(d['ReqTRES']) == 1 and control.gpu_count(d.get('AllocTRES', '')) == 0
            and control.requested_cpu_matches(d, CPUS) and control.memory_MiB(d['MinMemoryNode']) == MEMORY_MiB
            and control.seconds(d['TimeLimit']) == control.seconds(WALL)
            and control.dependencies(d.get('Dependency', '')) == [('afterany', str(dependency))], 'HELD_JOB_RESOURCES')
    require(control.command(['scontrol', 'write', 'batch_script', job_id, '-']).strip() == Path(script).read_text().strip(),
            'HELD_BATCH_SCRIPT')
    return d


def check_flucon(preparation, source):
    prep = read(preparation)
    require(prep['registration_authority'] == AUTHORITY and prep['server'] == 'server1' and len(prep['configs']) == 1,
            'FLUCON_PREPARATION')
    c = read(verify(prep['configs'][0]))
    require(c['registration_authority'] == AUTHORITY and c['key'] == flucon.KEY
            and c['config_sha256'] == digest({k: v for k, v in c.items() if k != 'config_sha256'}), 'FLUCON_CONFIG')
    o = c['original']
    for m in (o['checkpoint'], o['pointer'], o['terminal'], o['config'], c['assets'], c['stream'], c['reference'], c['evaluator']):
        verify(m)
    for m in read(c['assets']['path'])['model']['members']:
        s = Path(m['path']).stat()
        require((s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
                == (m['device'], m['inode'], m['bytes'], m['mtime_ns'], m['ctime_ns']), 'MODEL_FILE_STAT')
    flucon.tracking_values(c, source)
    return prep, c


def check_resume(root):
    c = read(root/'transfer.json'); resume.validate(c)
    require(not (root/'submitted.json').exists() and not (root/'execution-lock.json').exists(), 'RESUME_ALREADY_REGISTERED')
    require(read(root/'preparation/READY.json')['config'] == member(root/'transfer.json'), 'RESUME_READY')
    derivation = read(root/'preparation/derivation-63303.json')
    require(derivation['transfer'] == member(root/'transfer.json') and derivation['scientific_fields_changed'] == [],
            'RESUME_DERIVATION')
    for m in c['evidence']:
        verify(m)
    require(file_sha(c['parent_checkpoint']['path']) == c['parent_checkpoint']['sha256'] == resume.PARENT_SHA, 'RESUME_PARENT_SHA')
    st = os.statvfs(root)
    require(st.f_bavail * st.f_frsize >= c['min_free_bytes'] + 2 * 2**30, 'STORAGE_ADMISSION')
    budget = read(root/'preparation/storage.json')
    for i in (0, 1):
        s = (root/'output/checkpoint'/f'slot-{i}.pt').stat()
        require(s.st_size == budget['slot_capacity'] and s.st_blocks * 512 >= s.st_size, 'CHECKPOINT_SLOT_NOT_RESERVED')
    return c, derivation


def register(preparation, attempt, tail, receipt=None):
    require(socket.gethostname() == NODE, 'DEVBOX_ONLY')
    attempt, root = Path(attempt).absolute(), resume.ROOT
    require(attempt.is_relative_to(LOCAL) and not attempt.exists(), 'NEW_ATTEMPT_UNDER_TASK')
    repo = Path(__file__).resolve().parents[3]
    git = lambda *argv: control.command(['git', *argv], cwd=repo)
    source, tree = git('rev-parse', 'HEAD'), git('rev-parse', 'HEAD:official')
    git('merge-base', '--is-ancestor', source, 'origin/main')
    require(not git('status', '--porcelain', '--', 'official'), 'CLEAN_OFFICIAL_TREE')
    prep, config = check_flucon(Path(preparation).absolute(), source)
    transfer, derivation = check_resume(root)

    # The cancelled launchers, copied with SHA-256 on both sides, are the templates.
    rows = transfer_rows(receipt or (LOCAL/'transfer/receipt.tsv'))
    old_flucon = flucon.OUTPUT/'inputs/flucon-62873-registration/registration-r1/scripts/qwen25-sphere-62079.sh'
    received(rows, old_flucon, flucon.FLUCON_BASE + '/registration-r1/scripts/qwen25-sphere-62079.sh')
    old_resume = root/'inputs/server4-63303/scripts/run.sh'
    received(rows, old_resume, S4_ROOT + '/scripts/run.sh')

    caps = [r.split('\t') for r in CAPS.read_text().splitlines() if r.startswith('server1\t')]
    require(len(caps) == 1 and caps[0][1] == NODE and MEMORY_MiB <= int(caps[0][3]), 'CAP_ROW')
    cap = min(2, int(caps[0][2]))
    queue = control.command(['squeue', '--noheader', '--user', getpass.getuser(), '--format=%i|%j']).splitlines()
    require(not any(line.split('|', 1)[1] in (*NAMES.values(), *CANCELLED_NAMES) for line in queue), 'DUPLICATE_ACTIVE_JOB')
    existing = control.inventory(node=NODE)
    width = admit(existing, tail, cap)
    physical = control.resource_preflight(dict(cap=cap, resources=dict(control.DEFAULT_RESOURCES, node=NODE, qos=QOS,
        partition=PARTITION, cpus=CPUS, memory_MiB=MEMORY_MiB, wall_seconds=control.seconds(WALL))))
    free = os.statvfs(LOCAL).f_bavail * os.statvfs(LOCAL).f_frsize
    require(free >= 64 * 2**30, 'STORAGE_RESERVE')

    attempt.mkdir(parents=True); (attempt/'scripts').mkdir(); (attempt/'logs').mkdir()
    write_new(attempt/'admission.json', dict(instruction=AUTHORITY, node=NODE, cap=cap, lane_tail=str(tail),
        planned=lane_rows(existing['jobs'], tail), admitted_width_after=width,
        existing={k: v for k, v in existing.items() if k != 'jobs'},
        existing_jobs=[{k: v for k, v in r.items() if k not in ('detail', 'original')} for r in existing['jobs']],
        resources=dict(cpus=CPUS, memory_MiB=MEMORY_MiB, wall=WALL, gres=GRES, qos=QOS, partition=PARTITION),
        physical=physical, cap_member=member(CAPS), disk_free_bytes=free, existing_jobs_changed=False))
    frozen = control.freeze_source(dict(source=dict(main_commit=source, official_tree=tree)), attempt,
                                   repository=repo, runner=control.command)
    workdir = Path(frozen['directory'])
    flucon_lock = attempt/'flucon-execution-lock.json'
    write_new(flucon_lock, dict(instruction=AUTHORITY, source_commit=source, official_tree=tree,
        source_directory=frozen['directory'], source_members=frozen['members'], configs=prep['configs'],
        preparation=member(Path(preparation).absolute()), qualification='NOT_RUN_USER_DISABLED'))
    write_new(root/'execution-lock.json', dict(authority=AUTHORITY, code_commit=source, official_tree_sha256=tree,
        source_directory=frozen['directory'], config=member(root/'transfer.json'), parent_checkpoint=transfer['parent_checkpoint'],
        members=[dict(relative=str(Path(m['path']).relative_to(workdir)), sha256=m['sha256']) for m in frozen['members']]))

    env = environment(source, tree)
    flucon_argv = [PYTHON, '-B', '-u', '-m', 'official.runners.server1.sphere_flucon_s1', 'run',
                   '--config', prep['configs'][0]['path'], '--lock', str(flucon_lock)]
    resume_argv = [PYTHON, '-B', '-u', '-m', 'official.runners.server1.sphere_resume_s1', 'run', '--root', str(root)]
    texts = dict(flucon=launcher(FLUCON_ENV, env, workdir, flucon_argv), resume=launcher(RESUME_ENV, env, workdir, resume_argv))
    old = read(flucon.OUTPUT/'inputs/flucon-62873-registration/registration-r1/execution-lock.json')
    require(texts['flucon'] == replaced(old_flucon.read_text(), [
        (old['source_commit'], source), (old['official_tree'], tree), (old['source_directory'], str(workdir)),
        ('official.runners.server2.flucon_eval', 'official.runners.server1.sphere_flucon_s1'),
        (flucon.FLUCON_BASE + '/' + flucon.FILES['eval_config'][2], prep['configs'][0]['path']),
        (flucon.FLUCON_BASE + '/registration-r1/execution-lock.json', str(flucon_lock))]), 'FLUCON_LAUNCHER_NOT_62873_FORM')
    require(texts['resume'] == replaced(old_resume.read_text(), [
        ('/data/janghj/EasyEdit/.venv/bin/python', PYTHON), ('official.runners.server4.sphere_b10_resume', 'official.runners.server1.sphere_resume_s1'),
        (S4_ROOT + '/source', str(workdir)), ('--root ' + S4_ROOT, '--root ' + str(root))]), 'RESUME_LAUNCHER_NOT_63303_FORM')
    scripts = {}
    for key, text in texts.items():
        scripts[key] = attempt/'scripts'/f'{key}.sh'
        with scripts[key].open('x') as handle:
            handle.write(text)

    submitted, records, released = [], {}, False
    try:
        parent = str(tail)
        for key in ('flucon', 'resume'):
            cmd = ['sbatch', '--parsable', '--hold', '--export=NONE', '--no-requeue']
            cmd += ['--kill-on-invalid-dep=yes'] if key == 'resume' else []  # as 63303; 62873 had none
            cmd += ['--partition=' + PARTITION, '--qos=' + QOS, '--nodelist=' + NODE, '--nodes=1', '--ntasks=1',
                    f'--cpus-per-task={CPUS}', f'--mem={MEMORY_MiB}M', '--time=' + WALL, '--job-name=' + NAMES[key],
                    '--chdir=' + str(workdir), '--output=' + str(attempt/'logs'/f'{key}-%j.out'),
                    '--error=' + str(attempt/'logs'/f'{key}-%j.err'), '--gres=' + GRES,
                    '--dependency=afterany:' + parent, str(scripts[key])]
            response = control.command(cmd); job = response.split(';')[0]
            require(job.isdigit(), 'SBATCH_RESPONSE')
            submitted.append(job)
            records[key] = dict(job_id=job, name=NAMES[key], dependencies=[('afterany', parent)], command=cmd,
                                script=member(scripts[key]), argv=flucon_argv if key == 'flucon' else resume_argv)
            write_new(attempt/f'submitted-{key}.json', dict(records[key], response=response))
            records[key]['held'] = held_fields(job, NAMES[key], scripts[key], workdir, parent)
            parent = job
        held = control.inventory(node=NODE)
        require(held['admitted_DAG_width'] <= cap, 'GPU_CAP_AFTER_HOLD')
        write_new(attempt/'held-inspection.json', dict(jobs=records, inventory={k: v for k, v in held.items() if k != 'jobs'},
            inventory_jobs=[{k: v for k, v in r.items() if k not in ('detail', 'original')} for r in held['jobs']],
            flucon_lock=member(flucon_lock), resume_lock=member(root/'execution-lock.json'), GPU_qualification=False))
        for job in submitted:
            control.command(['scontrol', 'release', job])
        released = True
        after = control.inventory(node=NODE)
        require(after['admitted_DAG_width'] <= cap, 'GPU_CAP_AFTER_RELEASE')
        for previous in existing['jobs']:
            now = next((r for r in after['jobs'] if r['key'] == previous['key']), None)
            if now:
                for field in ('Command', 'WorkDir', 'JobName', 'ReqTRES', 'QOS', 'TimeLimit'):
                    require(now['detail'].get(field) == previous['detail'].get(field), 'OTHER_JOB_CHANGED:' + previous['key'])
                require(now['parents'] == previous['parents'], 'OTHER_JOB_DEPENDENCY_CHANGED:' + previous['key'])
        snapshot = {k: control.metadata(control.command(['scontrol', 'show', 'job', r['job_id'], '--oneliner']))
                    for k, r in records.items()}
        result = dict(instruction=AUTHORITY, server='server1', node=NODE, source=source, official_tree=tree,
            jobs={k: {f: v for f, v in r.items() if f != 'held'} for k, r in records.items()}, chain=[str(tail), *submitted],
            width_after=after['admitted_DAG_width'], cap=cap, flucon_config=prep['configs'][0],
            resume_config=member(root/'transfer.json'), derivation_63303=member(root/'preparation/derivation-63303.json'),
            flucon_preparation=member(Path(preparation).absolute()), initial_snapshot=snapshot, held_inspected=True,
            released=True, automatic_retry=False, existing_jobs_changed=False, GPU_qualification='NOT_RUN_USER_DISABLED',
            resubmission_of=dict(flucon='62873', resume='63303'))
        write_new(attempt/'submission.json', result)
        print(json.dumps(dict(chain=result['chain'], width=result['width_after'],
                              states={k: v['JobState'] for k, v in snapshot.items()})), flush=True)
        return result
    except BaseException:
        if not released:
            for job in submitted:  # only the held jobs this pass created
                subprocess.run(['scancel', job], check=False)
        write_new(attempt/'submission-failure.json', dict(submitted=submitted, released=released, error=traceback.format_exc()))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--preparation', required=True); p.add_argument('--attempt', required=True)
    p.add_argument('--after', required=True, help='last GPU job of the own devbox lane (afterany)')
    p.add_argument('--receipt'); a = p.parse_args()
    register(a.preparation, a.attempt, a.after, a.receipt)
