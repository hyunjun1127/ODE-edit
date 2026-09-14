"""CPU-only paired request-cluster bootstrap of already observed NLL rows.

No file/model/evaluator access. This resamples requests in one fixed observed
order/population, NOT independent edit-order replications or causal assignment.
Every observed prompt in a sampled request travels together, including P2/N10.
"""
import hashlib
import json
import math

import numpy as np

from .review_metrics import key, outcome, require


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False)


def _sha(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _percentile_interval(values):
    # Match the independent reducer: linear interpolation at (n-1)*p.
    lo, hi = np.quantile(values, [.025, .975], method='linear')
    return dict(low=float(lo), high=float(hi), confidence=.95,
                includes_zero=bool(lo <= 0 <= hi))


def request_cluster_bootstrap(before, after, metric, *, seed=20260914,
                              resamples=1000, block_size=64):
    """Return paired preference delta-pp and desired-target NLL mean delta.

    ``before``/``after`` have exactly the same full identity keys. Their row
    order may differ; pairing explicitly uses (case_id,prompt_index,identity).
    Requests are resampled with replacement, and all their observed prompt
    pairs stay together. Point estimates and each replicate are prompt-weighted
    ratios. This also defines partial-request conditional groups without
    inventing missing prompts: each replicate uses its actual prompt count.

    Desired NLL is new_nll for RS/PS and true_nll for NS; positive delta is harm.
    Preference delta is after-before in percentage points; positive is gain.
    ``includes_zero`` is descriptive and must not become a performance gate.
    """
    require(metric in ('RS', 'PS', 'NS'), 'BOOTSTRAP_METRIC')
    require(type(seed) is int and seed >= 0, 'BOOTSTRAP_SEED')
    require(type(resamples) is int and resamples >= 2, 'BOOTSTRAP_RESAMPLES')
    require(type(block_size) is int and block_size > 0, 'BOOTSTRAP_BLOCK_SIZE')
    left, right = {}, {}
    for source, target in [(before, left), (after, right)]:
        for row in source:
            identity = key(row)
            require(identity not in target, 'BOOTSTRAP_DUPLICATE_IDENTITY')
            target[identity] = row
    require(set(left) == set(right), 'BOOTSTRAP_PAIRED_IDENTITY_SET')
    target_field = 'true_nll' if metric == 'NS' else 'new_nll'
    # Sort full keys canonically so shuffled serialization gives the same draws.
    identities = sorted(left, key=_canonical)
    clusters = {}
    for identity in identities:
        a, b = left[identity], right[identity]
        cluster = clusters.setdefault(_canonical(a['case_id']), [])
        delta = int(outcome(b, metric)) - int(outcome(a, metric))
        nll_delta = float(b[target_field]) - float(a[target_field])
        require(math.isfinite(nll_delta), 'BOOTSTRAP_NONFINITE_NLL_DELTA')
        cluster.append((delta, nll_delta))
    cluster_keys = sorted(clusters)
    n_requests, n_prompts = len(cluster_keys), len(identities)
    counts = np.asarray([len(clusters[k]) for k in cluster_keys], dtype=np.int64)
    successes = np.asarray([sum(x[0] for x in clusters[k]) for k in cluster_keys], dtype=np.int64)
    nll_sums = np.asarray([math.fsum(x[1] for x in clusters[k]) for k in cluster_keys], dtype=np.float64)
    method = dict(name='PAIRED_REQUEST_CLUSTER_PERCENTILE_BOOTSTRAP',
                  seed=seed, random_generator='numpy.random.PCG64',
                  numpy_version=np.__version__, requested_resamples=resamples,
                  interval='PERCENTILE_2.5_97.5_LINEAR_N_MINUS_1',
                  resampling_unit='case_id/request',
                  prompt_weighting='ratio_of_cluster_sums_to_resampled_prompt_count',
                  pairing='EXACT_CASE_PROMPT_TARGET_IDENTITY_SET_NOT_ROW_POSITION',
                  observed_prompts_per_request=sorted(set(counts.tolist())),
                  desired_target_nll_field=target_field,
                  delta_direction='after_minus_before',
                  preference_positive='GAIN', desired_nll_positive='HARM',
                  fixed_single_order=True,
                  independent_order_replications=False,
                  checkpoint_or_prompt_independence_claim=False,
                  causal_interpretation=False, CI_zero_is_gate=False)
    result = dict(metric=metric, request_denominator=n_requests,
                  prompt_denominator=n_prompts, paired_identity_sha256=_sha(identities),
                  method=method)
    if not n_requests:
        result.update(status='NOT_AVAILABLE_EMPTY_GROUP', actual_resamples=0,
                      preference_delta_pp=None, preference_delta_pp_ci95=None,
                      desired_target_nll_mean_delta=None,
                      desired_target_nll_mean_delta_ci95=None)
        return result

    result.update(status='OBSERVED_REQUEST_CLUSTER_CI_NOT_ORDER_GENERALIZATION',
                  preference_delta_pp=100.*int(successes.sum())/n_prompts,
                  desired_target_nll_mean_delta=math.fsum(nll_sums.tolist())/n_prompts)
    # At most 65536 draws per allocation except for one intrinsically larger
    # request population. Never allocate resamples × total_requests at once.
    actual_block = min(block_size, max(1, 65536//n_requests))
    preference, nll = np.empty(resamples), np.empty(resamples)
    generator = np.random.Generator(np.random.PCG64(seed))
    for start in range(0, resamples, actual_block):
        stop = min(start + actual_block, resamples)
        sampled = generator.integers(0, n_requests, size=(stop-start, n_requests))
        denominator = counts[sampled].sum(axis=1)
        preference[start:stop] = 100.*successes[sampled].sum(axis=1)/denominator
        nll[start:stop] = nll_sums[sampled].sum(axis=1)/denominator
    require(np.isfinite(preference).all() and np.isfinite(nll).all(),
            'BOOTSTRAP_NONFINITE_REPLICATE')
    method.update(actual_block_resamples=actual_block,
                  maximum_draw_matrix_entries=actual_block*n_requests,
                  replicate_distribution_sha256=_sha(dict(
                      preference_delta_pp=preference.tolist(),
                      desired_target_nll_mean_delta=nll.tolist())))
    result.update(actual_resamples=resamples,
                  preference_delta_pp_ci95=_percentile_interval(preference),
                  desired_target_nll_mean_delta_ci95=_percentile_interval(nll))
    return result
