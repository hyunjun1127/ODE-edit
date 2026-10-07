"""CPU scalar producer only; no generation, scorer, tensor copy or evaluation.

The shared SH1 observer must supply independently reduced, hash-bound sums.
These functions do not reinterpret missing observations as measured zero.
"""
import math

from project.run_scripts.jlz_interference_l1.cap_tracking import batch_values
from project.run_scripts.jlz_interference_l1.comparison_bridge import metric_row
from .common import METHODS, MILESTONES, PROFILE, TASK, require

PREFIXES = ('current/pre', 'current/post', 'all_seen/post', 'W0_first2000',
            'w0/current', 'w0/all_seen')
REASONS = ('missing_generation_prompts', 'missing_reference', 'zero_generated_vector',
           'zero_reference_vector', 'nonfinite_score', 'length_cap_no_continuation')
WRITERS = {'MEMIT': 'memit', 'PRUNE': 'prune', 'RECT': 'rect',
           'ALPHAEDIT': 'alphaedit', 'ALPHAEDIT_BLUE': 'alphaedit-blue', 'CAKE': 'cake'}
CONFIG_KEYS = ('generation_metric_schema', 'generation_profile', 'generation_eval_seed',
               'reference_assets_sha256', 'generation_source_sha', 'baseline')


def generation_values(prefix, aggregate, planned):
    """Map canonical sums/counts, keeping bits/cosine distinct from percentages."""
    require(prefix in PREFIXES, 'GENERATION_PREFIX')
    require(type(planned) is int and planned > 0, 'GENERATION_PLANNED')
    require(type(aggregate) is dict and aggregate['planned_count'] == planned,
            'GENERATION_COHORT_DENOMINATOR')
    result = {prefix + '/generation/planned_count': planned}
    for kind, key in (('fluency', 'ngram_entropy'), ('consistency', 'reference_score')):
        count, total = aggregate[kind + '_count'], aggregate[kind + '_sum']
        require(type(count) is int and 0 <= count <= planned, 'GENERATION_VALID_COUNT')
        require(type(total) in (int, float) and math.isfinite(total) and total >= 0,
                'GENERATION_FINITE_SUM')
        require(count > 0 or total == 0, 'MISSING_METRIC_HAS_SUM')
        if kind == 'consistency':
            # Preserve native FP64 endpoint ULPs under SH1's exact same raw
            # scalar policy. This is neither a clamp nor a score tolerance.
            require(total <= count*(1+4*math.ulp(1.)), 'COSINE_SUM_RANGE')
        result[prefix + '/generation/' + kind + '_count'] = count
        if count:
            result[prefix + '/' + kind + '/' + key] = total / count
    for field in ('generation_prompt_count', 'generated_token_count'):
        value = aggregate[field]
        require(type(value) is int and value >= 0, 'GENERATION_INTEGER_COUNT')
        result[prefix + '/generation/' + field] = value
    reasons = aggregate['reason_counts']
    require(set(reasons) <= set(REASONS), 'GENERATION_UNKNOWN_REASON')
    for reason, count in reasons.items():
        require(type(count) is int and count >= 0, 'GENERATION_REASON_COUNT')
        result[prefix + '/generation/missing_' + reason + '_count'] = count
    return result


def endpoint_values(pre, current, all_seen, batch, gen_pre, gen_current, gen_all_seen):
    """The current cohort remains100 at a measured cumulative milestone."""
    require(type(batch) is int and 1 <= batch <= 20, 'BATCH_RANGE')
    result = batch_values(pre, current, all_seen, batch)
    result.update(generation_values('current/pre', gen_pre, 100))
    result.update(generation_values('current/post', gen_current, 100))
    if batch in MILESTONES:
        require(gen_all_seen is not None, 'MILESTONE_GENERATION_REQUIRED')
        result.update(generation_values('all_seen/post', gen_all_seen, batch * 100))
    else:
        require(gen_all_seen is None, 'UNMEASURED_GENERATION_ENDPOINT')
    return result


def w0_values(rpn, generation):
    return dict(edits=0, batch=0, pre_state_edits=0, post_state_edits=0,
                **metric_row('W0_first2000', rpn, 2000),
                **generation_values('W0_first2000', generation, 2000))


def scientific_config(method, run_instance, source_sha, config_sha, generation):
    require(method in METHODS, 'BASELINE_METHOD')
    require(generation['profile'] == PROFILE, 'GENERATION_PROFILE')
    # Actual Slurm metadata is captured/validated by the shared parent helper.
    # No made-up job number is attached before submission or in CPU fixtures.
    return dict(server='server4', task_id=TASK, arm='LLAMA_BASE_' + method,
                attempt=run_instance, source_sha=source_sha, config_sha=config_sha,
                model='llama3', model_family='llama3', baseline=method,
                writer=WRITERS[method], role='scientific',
                metric_schema='price-first2k-scalar-v1',
                generation_metric_schema='counterfact-cake-generation-metrics-v1',
                generation_profile=PROFILE, generation_eval_seed=20261007,
                reference_assets_sha256=generation['assets_sha256'],
                generation_source_sha=generation['source_sha'])


def transport_support():
    """Inspect the actual helper, never claim SDK acceptance is remote delivery."""
    from project.run_scripts.experiment_tracking import schema
    keys = {p + '/' + section + '/' + field for p in PREFIXES
            for section, field in (('fluency', 'ngram_entropy'),
                                   ('consistency', 'reference_score'))}
    for p in PREFIXES:
        keys.update(p + '/generation/' + field for field in
                    ('planned_count', 'fluency_count', 'consistency_count',
                     'generation_prompt_count', 'generated_token_count'))
        keys.update(p + '/generation/missing_' + reason + '_count' for reason in REASONS)
    missing = dict(metrics=sorted(keys - schema.METRICS),
                   config=sorted(set(CONFIG_KEYS) - schema.CONFIG_KEYS))
    return dict(status='KEYS_PRESENT_REQUIRES_WORKER_AND_AXIS_CHECK'
                if not any(missing.values()) else 'GENERATION_TRANSPORT_NOT_AVAILABLE',
                missing=missing, online_validation='NOT_OBSERVED')
