"""Preparation-only DAG. No scheduler or network calls are made here."""
from official.runners.server4.qwen_plan import rows

USER_SUBMISSION_HOLD = False
QUALIFICATION_STATUS = 'NOT_RUN_USER_DISABLED'
AUTHORITY = 'USER-GH-SH4-QWEN-OFFICIAL-NO-GPU-QUAL-20261009-R1'


def require_execution_enabled():
    if USER_SUBMISSION_HOLD:
        raise RuntimeError('USER_PREPARATION_ONLY_NO_SUBMIT_OR_EXECUTION')


def plan():
    nodes = []
    predecessor = 'TUNING_FRONTIER_REVALIDATE_BEFORE_SUBMISSION'
    for row in rows():
        name = row['logical_main_row']
        nodes.append(dict(symbol=name, kind='GPU_MAIN', GPUs=1,
                          dependency_symbol=predecessor, job_id=None,
                          config_sha256=row['config']['config_sha256']))
        predecessor = name + '-archive'
        nodes.append(dict(symbol=predecessor, kind='CPU_ARCHIVE_GATE', GPUs=0,
                          dependency_symbol=name, job_id=None,
                          blocks_next_until='VERIFIED_DESTINATION_AND_SAFE_SOURCE_DISPOSITION'))
    return dict(status='AUTHORIZED_STORAGE_INTEGRATION_PENDING', submission_enabled=True,
                authority=AUTHORITY, qualification=QUALIFICATION_STATUS,
                project_GPU_cap=2, baseline_max_concurrent_GPUs=1,
                existing_tuning_keep=True, job_ids=[], main_rows=12, nodes=nodes,
                historical_frontier_hint='61674 (not a fresh scheduler assertion)',
                failure_policy='No automatic retry; no later run after failed storage gate',
                remaining_gates=['reviewed exact source/config freeze',
                                 'archive caller proof integration and CPU tests',
                                 'serial checkpoint storage peak budget',
                                 'fresh scheduler/storage admission'])
