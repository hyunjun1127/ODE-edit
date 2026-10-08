"""Exact USER-scoped replacement of old generation-heavy S2 jobs, not cleanup."""
import getpass
import json
from pathlib import Path

from .generation_common import ARMS, LOCAL, member, read, require, verify, write
from .generation_cache_submit import inventory
from .generation_submit import command, field

OLD = LOCAL / 'cache-repair-r2/attempt-r1'
OUT = LOCAL / 'final-generation-v1'
SOURCE = 'f979efa69ce76a00b3e35c295452d01e14d1f13c'
IDS = dict(BASE_MEMIT='61364', BASE_ALPHAEDIT='61365', CAKE='61366',
           ALPHAEDIT_BLUE='61367', PRUNE='61368', RECT='61369', collector='61370')
ACTIVE = ('PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED')


def transition():
    require(not (OUT / 'replacement-cancellation.json').exists(), 'ONE_DELIBERATE_TRANSITION')
    old = read(OLD / 'submission.json')
    lock = read(OLD / 'execution.lock.json')
    require(old['jobs'] == IDS and lock['source_commit'] == SOURCE
        and old['source_commit'] == SOURCE and lock['owner'] == getpass.getuser(),
        'EXACT_OLD_OWN_SOURCE_MAPPING')
    verify(old['lock'])
    for item in [lock['archive'], *lock['launchers']]:
        verify(item)
    before = inventory()
    write(OUT / 'owned-project-before.json', before)
    # New USER says all current S2 experiments; inventory found only this exact
    # six-baseline GPU DAG. Do not silently mutate an unclassified task.
    others = [row for row in before['project'] if row['job'] not in IDS.values()]
    require(not others, 'OTHER_LIVE_PROJECT_REQUIRES_SOURCE_SCHEDULE_CLASSIFICATION')
    events = []
    for role in ('collector', 'RECT', 'PRUNE', 'ALPHAEDIT_BLUE', 'CAKE', 'BASE_ALPHAEDIT', 'BASE_MEMIT'):
        job = IDS[role]
        detail = command(['scontrol', 'show', 'job', job, '--oneliner'])
        require(field(detail, 'JobId') == job
            and (field(detail, 'UserId') or '').startswith(getpass.getuser() + '(')
            and field(detail, 'ReqNodeList') == 'server2'
            and field(detail, 'Command') == str(OLD / (role + '.sh'))
            and field(detail, 'WorkDir') == str(OLD / 'source')
            and field(detail, 'JobName') == old['task_id'] + '-' + role,
            'EXACT_CANCEL_OWNER_NODE_COMMAND_SOURCE_ROLE')
        status = field(detail, 'JobState')
        event = dict(job=job, role=role, before_state=status, before=detail,
                     source_commit=SOURCE, cancelled=False)
        if status in ACTIVE:
            event['result'] = command(['scancel', job])
            event['cancelled'] = True
        events.append(event)
        write(OUT / ('cancellation-' + role + '.json'), event)
    after = command(['squeue', '-h', '-j', ','.join(IDS.values()), '-o', '%i|%T|%u|%N|%r'])
    require(not after.strip(), 'EXACT_OLD_TARGETS_STILL_ACTIVE')
    accounting = command(['sacct', '-n', '-X', '-P', '-j', ','.join(IDS.values()),
        '--format=JobIDRaw,JobName%80,State,ElapsedRaw,AllocTRES%100,NodeList,User'])
    result = dict(status='USER_SCHEDULE_REPLACEMENT_CANCELLED', jobs=IDS, events=events,
        after_exact_queue=after, accounting=accounting,
        inventory_scope=before['queries'], other_live_project_GPU_jobs=others,
        all_old_jobs_were_pending=all(e['before_state'] == 'PENDING' for e in events),
        current_running_project_GPU_jobs=sum(r['allocated_GPUs'] for r in before['project']),
        old_source=SOURCE, old_lock=member(OLD / 'execution.lock.json'),
        old_raw_and_source_preserved=True, no_deletes=True, all_unrelated_jobs_unchanged=True,
        no_checkpoint_rescue=True, model_loads=0, new_GPU_computation=0, automatic_retry=False)
    write(OUT / 'replacement-cancellation.json', result)
    print(json.dumps({key:result[key] for key in ('status','jobs','after_exact_queue',
        'all_old_jobs_were_pending','current_running_project_GPU_jobs','accounting')}))
    return result


if __name__ == '__main__':
    transition()
