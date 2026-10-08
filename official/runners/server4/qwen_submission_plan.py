"""Preparation-only DAG. No scheduler or network calls are made here."""
from official.runners.server4.qwen_plan import rows

USER_SUBMISSION_HOLD = True


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
    return dict(status='PREPARATION_ONLY_USER_HOLD', submission_enabled=False,
                project_GPU_cap=2, baseline_max_concurrent_GPUs=1,
                existing_tuning_keep=True, job_ids=[], main_rows=12, nodes=nodes,
                historical_frontier_hint='61674 (not a fresh scheduler assertion)',
                failure_policy='No automatic retry; no later run after failed storage gate',
                remaining_gates=['reviewed exact source/config freeze',
                                 'native/resume qualification at authorized execution',
                                 'archive caller proof integration and CPU tests',
                                 'qualification checkpoint retention/storage peak budget',
                                 'fresh scheduler/storage admission after USER releases hold'])
