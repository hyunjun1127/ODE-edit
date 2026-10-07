"""Pure scheduling/count plan, not a Slurm submission or a scientific READY claim."""
import re
from .generation_common import ARMS, expected_counts, require

W0_OWNER='BASE_MEMIT'
PREDECESSOR={'BASE_ALPHAEDIT':'BASE_MEMIT','CAKE':'BASE_MEMIT',
             'ALPHAEDIT_BLUE':'BASE_ALPHAEDIT','PRUNE':'CAKE','RECT':'ALPHAEDIT_BLUE'}

def dependencies(role, frontier, jobs, cap):
    require(type(cap) is int and 1<=cap<=2, 'TASK_PROJECT_CAP')
    require(role in (*ARMS,'collector'), 'EXACT_ROLE')
    require(len(set(frontier))==len(frontier) and all(re.fullmatch(r'[1-9][0-9]*',j) for j in frontier),
            'EXACT_FRONTIER_PARENT_IDS')
    require(set(jobs)<=set(ARMS) and len(set(jobs.values()))==len(jobs)
            and all(re.fullmatch(r'[1-9][0-9]*',j) for j in jobs.values()), 'EXACT_REGISTERED_IDS')
    if role=='collector':
        require(set(jobs)==set(ARMS), 'ALL_SIX_BEFORE_COLLECTOR')
        return [jobs[a] for a in ARMS]
    require(set(jobs)==set(ARMS[:ARMS.index(role)]), 'DETERMINISTIC_REGISTRATION_ORDER')
    if role==W0_OWNER:return list(frontier)
    if cap==1:return [jobs[ARMS[ARMS.index(role)-1]]]
    return [jobs[PREDECESSOR[role]]]

def counts():
    # Pre20; post16 current + four all-seen prefix endpoints. No extra current
    # generation at W5/10/15/20: current is reduced from the measured prefix.
    pre_cases=20*100
    post_cases=16*100+sum((500,1000,1500,2000))
    per_arm_cases=pre_cases+post_cases
    return dict(edit_applications=6*2000,unique_request_occurrences=2000,
        native_per_arm={arm:{key:value*20 for key,value in expected_counts(arm).items()} for arm in ARMS},
        generation_endpoint_calls_per_arm=40,
        generation_current_subset_reductions_per_arm=4,
        generation_edit_state_case_observations_per_arm=per_arm_cases,
        cold_W0_generation_case_observations=2000,
        planned_generation_case_observations=2000+len(ARMS)*per_arm_cases,
        prompt_count_if_fixed10_per_case=10*(2000+len(ARMS)*per_arm_cases),
        observation_counts_include_cache_reuse=True,
        B1_pre_cases_reusable_from_exact_cold_W0=100*len(ARMS),
        additional_unique_generation_case_upper_bound=2000+len(ARMS)*(per_arm_cases-100),
        additional_generation_per_metric=0,checkpoint_saves=0,
        cost_status='COUNT_PLAN_NOT_MEASURED_TIME_OR_MEMORY')

def ready(config):
    """Fail before any submission when immutable evaluator/reference is unbound."""
    generation=config['generation']
    require(generation.get('common_source_status')=='READY_BOUND', 'SH1_GENERATION_SOURCE_NOT_BOUND')
    require(generation.get('reference_status')=='READY_VERIFIED', 'GENERATION_REFERENCE_NOT_READY')
    require(generation.get('W0_owner')==W0_OWNER and generation.get('no_GPU_file_poll') is True,
            'SINGLE_COLD_GENERATION_OWNER')
    require(config['noCP'] and not config['z_disk_cache'], 'NOCP_NO_Z_DISK_CACHE')
    return True
