"""Approved narrow W0 reference-state exception, never a shared method bypass.

Job/auth/identifier routines are read-only imports from experiment_tracking.
Only this cold reference task permits comparison edits>0 with model state0.
"""
import math
from project.run_scripts.experiment_tracking import schema as shared

ENTITY, PROJECT, SDK_VERSION = shared.ENTITY, shared.PROJECT, shared.SDK_VERSION
identifier, endpoint, load_env = shared.identifier, shared.endpoint, shared.load_env
job_identity, run_name = shared.job_identity, shared.run_name
require = shared.require
REFERENCE_SCHEMA = 'w0-cohort-reference-v1'
ZERO_CONFIG = {'actual_model_edits','actual_applied_edits','pre_state_edits','post_state_edits'}
REFERENCE_CONFIG = {'reference_only','evaluation_model_state','edits_axis_semantics','w0_reference_schema'} | ZERO_CONFIG
OPTION_CONFIG = {'include_current_pre'}
CONFIG_KEYS = shared.CONFIG_KEYS | REFERENCE_CONFIG | OPTION_CONFIG
FIELDS = ('count','success_count','success_pct','token_acc_pct','prompt_acc_pct',
          'strict_acc_pct','true_nll','new_nll','margin_true_minus_new')
PREFIXES = ('W0_first2000','current/post','current/pre','all_seen/post')
GROUPS = tuple(prefix+'/'+kind for prefix in PREFIXES for kind in 'RPN') + ('w0/current/N','w0/all_seen/N')
PERFORMANCE = {group+'/'+field for group in GROUPS for field in FIELDS}
PERFORMANCE |= {prefix+'/success_harmonic_pct' for prefix in PREFIXES}
AXIS_FIELDS = {'edits','reference_cohort_edits','actual_model_edits','actual_applied_edits',
               'pre_state_edits','post_state_edits','reference_only','evaluation_model_state','edits_axis_semantics'}
PROGRESS = {'step','phase_id','status_code','time/elapsed_seconds','time/phase_seconds',
            'time/allocated_gpu_seconds','memory/gpu_allocated_bytes','memory/gpu_reserved_bytes',
            'memory/host_rss_bytes','resource/gpus','resource/cpus','logging/dropped_points'}
METRICS = PERFORMANCE | AXIS_FIELDS | PROGRESS | {'batch'}


def config(values):
    require(type(values) is dict and set(values) <= CONFIG_KEYS, 'W0_CONFIG_WHITELIST')
    require(REFERENCE_CONFIG <= values.keys(), 'W0_REFERENCE_CONFIG_REQUIRED')
    require(values['reference_only'] is True and values['evaluation_model_state'] == 'W0'
            and values['edits_axis_semantics'] == 'reference_cohort_progress'
            and values['w0_reference_schema'] == REFERENCE_SCHEMA, 'W0_REFERENCE_PROFILE')
    require(all(type(values[key]) is int and values[key]==0 for key in ZERO_CONFIG),'W0_CONFIG_ACTUAL_STATE_ZERO')
    require('include_current_pre' not in values or type(values['include_current_pre']) is bool,'W0_PRE_ALIAS_FLAG')
    base = shared.config({key:value for key,value in values.items() if key not in REFERENCE_CONFIG | OPTION_CONFIG})
    require(base['server']=='server1' and base['model']=='gpt2xl' and base['model_family']=='gpt2'
            and base['writer']=='none' and base['role']=='scientific'
            and base['metric_schema']=='price-first2k-scalar-v1', 'W0_NOT_SCIENTIFIC_WRITER_NONE')
    base.update({key:values[key] for key in REFERENCE_CONFIG})
    base.update({key:values[key] for key in OPTION_CONFIG & values.keys()})
    return base


def bind_job_identity(values, environ=None):
    cfg = config(values)
    reference = {key:cfg[key] for key in (REFERENCE_CONFIG | OPTION_CONFIG) & cfg.keys()}
    base = shared.bind_job_identity({key:value for key,value in cfg.items() if key not in REFERENCE_CONFIG | OPTION_CONFIG}, environ)
    require(base['execution_backend']=='slurm', 'W0_REAL_JOB_REQUIRED_NO_SMOKE')
    base.update(reference)
    return config(base)


def performance(values):
    return bool(set(values) & PERFORMANCE)


def metrics(values, cfg):
    cfg = config(cfg)
    require(type(values) is dict and 0 < len(values) <= len(METRICS)
            and set(values) <= METRICS, 'W0_METRIC_WHITELIST')
    for key,value in values.items():
        if key in ('evaluation_model_state','edits_axis_semantics'):
            require(value == cfg[key], 'W0_REFERENCE_CATEGORICAL_IDENTITY')
        elif key == 'reference_only':
            require(value is True, 'W0_REFERENCE_BOOLEAN')
        else:
            require(type(value) in (int,float) and math.isfinite(value), 'W0_BUILTIN_FINITE_SCALAR')
        if key in ('actual_model_edits','actual_applied_edits','pre_state_edits','post_state_edits'):
            require(type(value) is int and value == 0, 'W0_ACTUAL_MODEL_STATE_MUST_BE_ZERO')
        if key in ('edits','reference_cohort_edits','batch','step','phase_id','status_code'):
            require(type(value) is int and value >= 0, 'W0_INTEGER_AXIS')
    if performance(values):
        from .curves import validate_curve_payload
        validate_curve_payload(values, cfg)
    else:
        require(not set(values) & PERFORMANCE, 'W0_UNMEASURED_METRIC')
        # Progress does not advance the comparison cohort axis or invent metrics.
        if 'edits' in values:
            require(values['edits']==0, 'W0_PROGRESS_NOT_COHORT_CURVE')
        if 'reference_cohort_edits' in values:
            require(values['reference_cohort_edits']==0, 'W0_PROGRESS_NOT_COHORT_CURVE')
    return dict(values)


class AxisState:
    def __init__(self):
        self.last = None
        self.curves = []

    def check(self, values):
        if not performance(values):
            return
        edits = values['edits']
        require(self.last is None and edits==0 or self.last is not None and edits==self.last+100,
                'W0_EXACT_MONOTONIC_COHORT_AXIS')

    def accept(self, values):
        self.check(values)
        if performance(values):
            self.last = values['edits']
            self.curves.append(dict(values))


def define_axes(run):
    run.define_metric('edits')
    run.define_metric('reference_cohort_edits')
    run.define_metric('step')
    for prefix in (*PREFIXES,'w0/current/N','w0/all_seen/N'):
        run.define_metric(prefix+'/*', step_metric='edits', step_sync=False)
    for prefix in ('time/*','memory/*','resource/*','logging/*'):
        run.define_metric(prefix, step_metric='step', step_sync=False)
