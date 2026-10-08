"""Prospective resource-only scheduling; immutable submitted source is untouched.

The cap3 path is opt-in and pinned to one direct USER resource override. It is
not a general policy bypass, a new scientific permission, or a runtime change.
Callers must still verify fresh Slurm resources and the full admitted DAG.
"""
import hashlib
import json
from pathlib import Path

TASK = 'gpt2xl-baselines-native-generation-repair'
ARMS = ('BASE_MEMIT', 'BASE_ALPHAEDIT', 'ALPHAEDIT_BLUE')
SESSION = '01a04939-f93a-7b50-bca0-65438eab2062'
OVERRIDE = 'plans/updates/server1/gpu-cap3-20261008/user-override.json'
OVERRIDE_SHA = 'f1db5f8aa6dd578eaf45ac939e01a9df05b14d2b7ffff1da642a4e5728ce334c'
INSTRUCTION = 'USER-DIRECT-SERVER1-GPU-CAP3-20261008-R1'


def _require(condition, message):
    if not condition:
        raise RuntimeError('NATIVE_REPO_ADMISSION_' + message)


def _member(path, payload):
    return dict(path=str(path), bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())


def _rows(path, expected_columns):
    payload = path.read_bytes()
    rows = {}
    for line in payload.decode('utf-8').splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        fields = line.split('\t')
        _require(len(fields) == expected_columns, 'POLICY_COLUMNS')
        _require(fields[0] not in rows, 'DUPLICATE_SERVER_ROW')
        rows[fields[0]] = fields
    return rows, _member(path, payload)


def _positive_integer(value, name):
    _require(type(value) is int and value > 0, name)
    return value


def resolve_admission(root, *, server='server1', task_id=TASK, task_cap=2,
                      apply_user_override=False, current_three_arm_scope=False,
                      per_job_gpus=1, expected_node=None, expected_memory_mib=None):
    """Return a typed effective cap and compact authority receipt without writes.

    Normal callers remain min(canonical, enabled local, task). To opt into the
    current three-arm USER override, pass apply_user_override=True. If the old
    task_cap=2 represented the superseded combined task ceiling, also explicitly
    pass current_three_arm_scope=True; a distinct stricter task_cap (e.g.1), or
    omitted scope flag, remains strict. Per-job GPU1 is never increased.
    """
    root = Path(root)
    task_cap = _positive_integer(task_cap, 'POSITIVE_TASK_LIMIT')
    per_job_gpus = _positive_integer(per_job_gpus, 'POSITIVE_PER_JOB_LIMIT')
    _require(type(apply_user_override) is bool and type(current_three_arm_scope) is bool,
             'EXPLICIT_BOOLEAN_SCOPE')
    canonical_rows, canonical_member = _rows(root / 'control/gpu-concurrency-policy.tsv', 2)
    local_rows, local_member = _rows(root / 'servers/local/gpu-caps.tsv', 5)
    _require(server in canonical_rows and server in local_rows, 'SERVER_ROW_REQUIRED')
    local = local_rows[server]
    try:
        canonical_cap, local_cap, memory = int(canonical_rows[server][1]), int(local[2]), int(local[3])
    except ValueError as exc:
        raise RuntimeError('NATIVE_REPO_ADMISSION_INTEGER_POLICY') from exc
    _require(canonical_cap > 0 and local_cap > 0, 'ENABLED_POSITIVE_SERVER_CAP')
    _require(memory > 0 and local[1], 'NODE_MEMORY_PRESENT')
    if expected_node is not None:
        _require(local[1] == expected_node, 'LOCAL_NODE_UNCHANGED')
    if expected_memory_mib is not None:
        _require(memory == expected_memory_mib, 'LOCAL_MEMORY_UNCHANGED')

    project_cap = canonical_cap
    task_ceiling = task_cap
    override_member = None
    instruction = None
    historical_ceiling_superseded = False
    if apply_user_override:
        _require(server == 'server1' and task_id == TASK, 'EXACT_OVERRIDE_SERVER_TASK_SCOPE')
        _require(per_job_gpus == 1, 'OVERRIDE_PER_JOB_GPU1_UNCHANGED')
        _require(canonical_cap == 2, 'PINNED_HISTORICAL_CANONICAL_CAP2')
        _require(local[1] == 'devbox' and memory == 183296, 'OVERRIDE_LOCAL_NODE_MEMORY_UNCHANGED')
        _require(local_cap <= 3, 'LOCAL_NO_UNAUTHORIZED_CAP_INCREASE')
        path = root / OVERRIDE
        _require(path.is_file() and not path.is_symlink(), 'EXACT_OVERRIDE_FILE_REQUIRED')
        payload = path.read_bytes()
        override_member = _member(path, payload)
        _require(override_member['sha256'] == OVERRIDE_SHA, 'EXACT_OVERRIDE_BYTES')
        value = json.loads(payload)
        _require(value.get('schema') == 1 and value.get('instruction_id') == INSTRUCTION
                 and value.get('owner') == dict(server='server1', session=SESSION)
                 and value.get('server1_combined_project_GPU_cap') == 3
                 and value.get('server2_project_GPU_cap') == 2
                 and value.get('RUNNING_mutation') is False
                 and value.get('existing_scientific_source_config_archive_WB_identity') == 'IMMUTABLE'
                 and value.get('NoCP') is True and value.get('automatic_retry') is False
                 and value.get('recurring_monitor') is False, 'EXPLICIT_RESOURCE_ONLY_AUTHORITY')
        project_cap = 3
        instruction = INSTRUCTION
        # Only this exact three-arm current task's historical combined ceiling
        # is superseded; independent stricter task limits remain strict.
        if current_three_arm_scope and task_cap == 2:
            task_ceiling = 3
            historical_ceiling_superseded = True
    else:
        _require(not current_three_arm_scope, 'THREE_ARM_SCOPE_NEEDS_EXPLICIT_OVERRIDE')

    cap = min(project_cap, local_cap, task_ceiling)
    _require(per_job_gpus <= cap, 'PER_JOB_FITS_EFFECTIVE_CAP')
    return dict(server=server, task_id=task_id, effective_cap=cap,
                canonical_cap=canonical_cap, local_cap=local_cap, project_cap=project_cap,
                requested_task_cap=task_cap, effective_task_ceiling=task_ceiling,
                historical_combined_task_ceiling_superseded=historical_ceiling_superseded,
                per_job_gpus=per_job_gpus, node=local[1], memory_mib_per_gpu=memory,
                instruction_id=instruction, authority_member=override_member,
                canonical_policy=canonical_member, local_policy=local_member,
                scheduling_only=True, science_authorization_granted=False,
                existing_scientific_source_config_immutable=True,
                fresh_scheduler_and_admitted_DAG_check_required=True)


def order(cap):
    """Exact three-arm resource lanes; collector dependency stays caller-owned."""
    _require(type(cap) is int and cap in (1, 2, 3), 'DAG_CAP1_2_3_ONLY')
    if cap == 1:
        return {arm: [] if index == 0 else [ARMS[index - 1]]
                for index, arm in enumerate(ARMS)}
    if cap == 2:
        return dict(BASE_MEMIT=[], BASE_ALPHAEDIT=[], ALPHAEDIT_BLUE=['BASE_MEMIT'])
    return {arm: [] for arm in ARMS}
