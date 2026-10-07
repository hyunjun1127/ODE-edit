"""Independent stdlib reduction of one fresh, occurrence-ordered W0 observation.

Legacy case/token identities are preserved, not deduplicated. An occurrence is
identified by its ordinal and panel position, so repeated case IDs are legal.
No model/scorer call, scheduler, SDK, tensor, or shared validator is invoked.
"""
import math

from project.run_scripts.jlz_interference_l1.comparison_bridge import metric_row

KINDS = ('R', 'P', 'N')
MULT = {'R': 1, 'P': 2, 'N': 10}
REQUESTS = 2000
BATCH_SIZE = 100
PANELS = tuple((kind, index) for kind in KINDS for index in range(MULT[kind]))
FIELDS = ('count', 'success_count', 'success_pct', 'token_acc_pct',
          'prompt_acc_pct', 'strict_acc_pct', 'true_nll', 'new_nll',
          'margin_true_minus_new')
REFERENCE_SCHEMA = 'w0-cohort-reference-v1'
REFERENCE_METADATA = dict(reference_only=True, evaluation_model_state='W0',
                          edits_axis_semantics='reference_cohort_progress')
ZERO_STATE = dict(actual_model_edits=0, actual_applied_edits=0,
                  pre_state_edits=0, post_state_edits=0)
DEFAULT_CONFIG = dict(REFERENCE_METADATA, **ZERO_STATE, writer='none', role='scientific',
                      metric_schema='price-first2k-scalar-v1',
                      w0_reference_schema=REFERENCE_SCHEMA)
IDENTITY_FIELDS = ('identity', 'case_id', 'kind', 'prompt_index',
                   'new_token_identity', 'true_token_identity')
MILESTONES = (500, 1000, 1500, 2000)


def ensure(value, message):
    if not value:
        raise ValueError(message)


def attach_occurrence_ordinals(rows, expected=None):
    """Return new dictionaries; never rewrite original saved rows/identities.

    The canonical 13-row panel order is checked before assigning ordinal. If an
    ordinal already exists it must be correct; this function never repairs it.
    When expected is provided, return ``(annotated_rows, annotated_expected)``.
    """
    def annotate(values):
        ensure(type(values) is list and len(values) % 13 == 0, 'W0_OCCURRENCE_PANEL_LENGTH')
        result = []
        for index, row in enumerate(values):
            ordinal, panel = divmod(index, 13)
            kind, prompt_index = PANELS[panel]
            ensure(type(row['kind']) is str and type(row['prompt_index']) is int
                   and row['kind'] == kind and row['prompt_index'] == prompt_index,
                   'W0_OCCURRENCE_PANEL_ORDER')
            ensure(type(row['case_id']) is int and row['case_id'] == values[ordinal * 13]['case_id'],
                   'W0_OCCURRENCE_CASE_ORDER')
            if 'occurrence_ordinal' in row:
                ensure(type(row['occurrence_ordinal']) is int
                       and row['occurrence_ordinal'] == ordinal, 'W0_ORDINAL_CONFLICT')
            result.append(dict(row, occurrence_ordinal=ordinal))
        return result
    actual = annotate(rows)
    if expected is None:
        return actual
    return actual, annotate(expected)


def validate_rows(rows, expected=None, require_full=False):
    """Validate ordered occurrence identity without legacy-ID deduplication."""
    ensure(type(rows) is list and len(rows) % 13 == 0, 'W0_PANEL_CARDINALITY')
    if require_full:
        ensure(len(rows) == REQUESTS * 13, 'W0_FULL_26000_ROWS_REQUIRED')
    if expected is not None:
        ensure(type(expected) is list and len(rows) == len(expected), 'W0_EXPECTED_CARDINALITY')
    seen = set()
    first = rows[0]['occurrence_ordinal'] if rows else 0
    ensure(type(first) is int and 0 <= first <= REQUESTS, 'W0_ORDINAL_START')
    if require_full:
        ensure(first == 0, 'W0_FULL_ORDINAL_START_ZERO')
    for index, row in enumerate(rows):
        ordinal = first + index // 13
        kind, panel_index = PANELS[index % 13]
        ensure(type(row.get('occurrence_ordinal')) is int
               and row['occurrence_ordinal'] == ordinal and 0 <= ordinal < REQUESTS,
               'W0_EXACT_OCCURRENCE_ORDER')
        ensure(row['kind'] == kind and type(row['prompt_index']) is int
               and row['prompt_index'] == panel_index, 'W0_EXACT_PANEL_ORDER')
        ensure(type(row['case_id']) is int
               and row['case_id'] == rows[(index // 13) * 13]['case_id'],
               'W0_EXACT_CASE_IN_OCCURRENCE')
        key = (ordinal, kind, panel_index)
        ensure(key not in seen, 'W0_DUPLICATE_OCCURRENCE_PANEL')
        seen.add(key)
        ensure(row.get('endpoint') == 'W0', 'W0_NOT_EDITED_ENDPOINT')
        for field in ZERO_STATE:
            if field in row:
                ensure(type(row[field]) is int and row[field] == 0, 'W0_RAW_STATE_NOT_ZERO')
        if expected is not None:
            reference = expected[index]
            ensure(reference.get('occurrence_ordinal') == ordinal
                   and type(reference.get('occurrence_ordinal')) is int, 'W0_EXPECTED_ORDINAL')
            for field in IDENTITY_FIELDS:
                ensure(type(row[field]) is type(reference[field]) and row[field] == reference[field],
                       'W0_ORDERED_IDENTITY:' + field)
        for field in ('identity', 'new_token_identity', 'true_token_identity'):
            ensure(type(row[field]) is str and row[field], 'W0_TOKEN_IDENTITY_TYPE')
        for label in ('new', 'true'):
            nll = row[label + '_nll']
            count, correct = row[label + '_token_count'], row[label + '_token_correct']
            ensure(type(nll) in (int, float) and math.isfinite(nll) and nll >= 0,
                   'W0_NONFINITE_OR_NEGATIVE_NLL')
            ensure(type(count) is int and type(correct) is int and count > 0
                   and 0 <= correct <= count, 'W0_TOKEN_COUNTS')
            ensure(type(row[label + '_strict']) is bool
                   and row[label + '_strict'] == (correct == count), 'W0_STRICT_TOKENS')
        ensure(type(row['margin_true_minus_new']) in (int, float)
               and math.isfinite(row['margin_true_minus_new'])
               and math.isclose(row['margin_true_minus_new'], row['true_nll'] - row['new_nll'],
                                rel_tol=1e-12, abs_tol=1e-12), 'W0_MARGIN_SIGN')
        if 'margin_new_minus_true' in row:
            ensure(type(row['margin_new_minus_true']) in (int, float)
                   and math.isfinite(row['margin_new_minus_true'])
                   and math.isclose(row['margin_new_minus_true'], row['new_nll'] - row['true_nll'],
                                    rel_tol=1e-12, abs_tol=1e-12), 'W0_SECOND_MARGIN_SIGN')
    return True


def reduce_rows(rows):
    """Independent prompt and token arithmetic; missing families stay missing.

    Callers validate complete occurrence panels once before selecting family-only
    rows. The reducer never keys, thresholds or deduplicates by case ID.
    """
    result = {}
    for kind in KINDS:
        group = [row for row in rows if row['kind'] == kind]
        if not group:
            continue
        desired = 'true' if kind == 'N' else 'new'
        n = len(group)
        count = sum(row[desired + '_token_count'] for row in group)
        correct = sum(row[desired + '_token_correct'] for row in group)
        success = sum(row['true_nll'] < row['new_nll'] if kind == 'N'
                      else row['new_nll'] < row['true_nll'] for row in group)
        strict = sum(row[desired + '_strict'] for row in group)
        result[kind] = dict(denominator=n, numerator=success, rate=success / n,
            desired_token_count=count, desired_token_correct=correct,
            token_micro=correct / count,
            prompt_macro=math.fsum(row[desired + '_token_correct'] / row[desired + '_token_count']
                                   for row in group) / n,
            strict_numerator=strict, strict_denominator=n, strict_rate=strict / n,
            true_nll_mean=math.fsum(row['true_nll'] for row in group) / n,
            new_nll_mean=math.fsum(row['new_nll'] for row in group) / n,
            true_minus_new_mean=math.fsum(row['true_nll'] - row['new_nll'] for row in group) / n)
    return result


def metric_values(prefix, groups, requests):
    """Use the existing read-only PRICE mapping, with typed N-only references."""
    if set(groups) == set(KINDS):
        return metric_row(prefix, groups, requests)
    result = {}
    for kind, group in groups.items():
        ensure(kind in KINDS and group['denominator'] == requests * MULT[kind], 'W0_GROUP_DENOMINATOR')
        values = dict(count=group['denominator'], success_count=group['numerator'],
            success_pct=100 * group['rate'], token_acc_pct=100 * group['token_micro'],
            prompt_acc_pct=100 * group['prompt_macro'], strict_acc_pct=100 * group['strict_rate'],
            true_nll=group['true_nll_mean'], new_nll=group['new_nll_mean'],
            margin_true_minus_new=group['true_nll_mean'] - group['new_nll_mean'])
        result.update({prefix + '/' + kind + '/' + field: value for field, value in values.items()})
    return result  # No missing-as-zero or harmonic with unavailable R/P/N.


def _metadata(x):
    return dict(REFERENCE_METADATA, **ZERO_STATE, edits=x, reference_cohort_edits=x)


def build_curves(rows, expected, include_current_pre=False):
    """Build x0 + 20 current points; four exact prefixes share the same raw.

    Each point is a scalar payload, never an edited endpoint. No additional
    forward is called, and caller-owned rows/expected metadata are not mutated.
    """
    ensure(type(include_current_pre) is bool, 'W0_ALIAS_OPTION_TYPE')
    validate_rows(rows, expected, require_full=True)
    full = reduce_rows(rows)
    payloads = [dict(_metadata(0), **metric_values('W0_first2000', full, REQUESTS))]
    for batch in range(1, 21):
        x = BATCH_SIZE * batch
        # Exact occurrence slices; not sets of case IDs or numeric thresholds.
        current = reduce_rows(rows[(x - BATCH_SIZE) * 13:x * 13])
        payload = dict(_metadata(x), **metric_values('current/post', current, BATCH_SIZE))
        payload.update(metric_values('w0/current', {'N': current['N']}, BATCH_SIZE))
        if include_current_pre:
            payload.update(metric_values('current/pre', current, BATCH_SIZE))
        if x in MILESTONES:
            prefix = reduce_rows(rows[:x * 13])
            payload.update(metric_values('all_seen/post', prefix, x))
            payload.update(metric_values('w0/all_seen', {'N': prefix['N']}, x))
        payloads.append(payload)
    config = dict(DEFAULT_CONFIG, include_current_pre=include_current_pre)
    for payload in payloads:
        validate_curve_payload(payload, config)
    return payloads


def _check_group(payload, prefix, kinds, requests):
    expected = set()
    rates = []
    for kind in kinds:
        base = prefix + '/' + kind + '/'
        expected.update(base + field for field in FIELDS)
        ensure(all(base + field in payload for field in FIELDS), 'W0_MISSING_METRIC:' + base)
        values = {field: payload[base + field] for field in FIELDS}
        n, correct = values['count'], values['success_count']
        ensure(type(n) is int and n == requests * MULT[kind]
               and type(correct) is int and 0 <= correct <= n, 'W0_PAYLOAD_COUNTS')
        for field in FIELDS[2:]:
            value = values[field]
            ensure(type(value) in (int, float) and math.isfinite(value), 'W0_PAYLOAD_FINITE')
            if field.endswith('_pct'):
                ensure(0 <= value <= 100, 'W0_PAYLOAD_PERCENT')
            if field in ('true_nll', 'new_nll'):
                ensure(value >= 0, 'W0_PAYLOAD_NLL_NONNEGATIVE')
        ensure(math.isclose(values['success_pct'], 100 * correct / n,
                            rel_tol=1e-12, abs_tol=1e-10), 'W0_PAYLOAD_PREFERENCE_RATE')
        ensure(math.isclose(values['margin_true_minus_new'], values['true_nll'] - values['new_nll'],
                            rel_tol=1e-12, abs_tol=1e-10), 'W0_PAYLOAD_MARGIN')
        rates.append(values['success_pct'])
    if tuple(kinds) == KINDS:
        key = prefix + '/success_harmonic_pct'
        expected.add(key)
        harmonic = 0. if any(value == 0 for value in rates) else 3 / math.fsum(1 / value for value in rates)
        ensure(key in payload and type(payload[key]) in (int, float)
               and math.isfinite(payload[key])
               and math.isclose(payload[key], harmonic, rel_tol=1e-12, abs_tol=1e-10),
               'W0_PAYLOAD_HARMONIC')
    return expected


def validate_curve_payload(payload, config):
    """Task-only truthful state/axis exception; no general POST_STATE_AXIS bypass."""
    ensure(type(payload) is dict and type(config) is dict, 'W0_PAYLOAD_CONFIG_TYPE')
    for field, value in DEFAULT_CONFIG.items():
        actual = config.get(field)
        ensure(type(actual) is type(value) and actual == value, 'W0_CONFIG_IMMUTABLE:' + field)
    for field in ZERO_STATE:
        if field in config:
            ensure(type(config[field]) is int and config[field] == 0,
                   'W0_CONFIG_STATE_NOT_ZERO:' + field)
    for field, value in {**REFERENCE_METADATA, **ZERO_STATE}.items():
        actual = payload.get(field)
        ensure(type(actual) is type(value) and actual == value, 'W0_REFERENCE_STATE:' + field)
    x = payload.get('edits')
    ensure(type(x) is int and 0 <= x <= REQUESTS and x % BATCH_SIZE == 0,
           'W0_REFERENCE_COHORT_AXIS')
    ensure(type(payload.get('reference_cohort_edits')) is int
           and payload['reference_cohort_edits'] == x, 'W0_REFERENCE_AXIS_IDENTITY')
    allowed = set(_metadata(x))
    if x == 0:
        allowed.update(_check_group(payload, 'W0_first2000', KINDS, REQUESTS))
    else:
        allowed.update(_check_group(payload, 'current/post', KINDS, BATCH_SIZE))
        allowed.update(_check_group(payload, 'w0/current', ('N',), BATCH_SIZE))
        for field in FIELDS:
            ensure(payload['w0/current/N/' + field] == payload['current/post/N/' + field],
                   'W0_CURRENT_N_SAME_SUBSET')
        if config.get('include_current_pre', False):
            ensure(type(config['include_current_pre']) is bool, 'W0_PRE_ALIAS_CONFIG_TYPE')
            allowed.update(_check_group(payload, 'current/pre', KINDS, BATCH_SIZE))
            ensure(all(payload[key.replace('current/post', 'current/pre', 1)] == value
                       for key, value in payload.items() if key.startswith('current/post/')),
                   'W0_PRE_POST_ALIAS')
        if x in MILESTONES:
            allowed.update(_check_group(payload, 'all_seen/post', KINDS, x))
            allowed.update(_check_group(payload, 'w0/all_seen', ('N',), x))
            for field in FIELDS:
                ensure(payload['w0/all_seen/N/' + field] == payload['all_seen/post/N/' + field],
                       'W0_PREFIX_N_SAME_SUBSET')
    ensure(set(payload) == allowed, 'W0_UNKNOWN_OR_WRONG_SCOPE_METRIC')
    return dict(payload)


def curve_coverage(payloads, config, require_complete=True):
    """Validate accepted/readback point coverage; unavailable points stay explicit."""
    ensure(type(require_complete) is bool and type(payloads) is list, 'W0_COVERAGE_TYPE')
    checked = [validate_curve_payload(payload, config) for payload in payloads]
    xs = [payload['edits'] for payload in checked]
    ensure(xs == sorted(set(xs)), 'W0_COVERAGE_MONOTONIC_UNIQUE')
    required = list(range(0, REQUESTS + 1, BATCH_SIZE))
    missing = [x for x in required if x not in xs]
    if require_complete:
        ensure(not missing, 'W0_CURVE_COVERAGE_INCOMPLETE')
    return dict(schema=REFERENCE_SCHEMA, status='COMPLETE' if not missing else 'INCOMPLETE',
        full_points=[x for x in xs if x == 0], current_points=[x for x in xs if x > 0],
        all_seen_points=[x for x in xs if x in MILESTONES], missing_points=missing,
        reference_only=True, evaluation_model_state='W0', actual_model_edits=0,
        actual_applied_edits=0, new_forward_calls_by_reducer=0,
        current_coverage=len([x for x in xs if x > 0]),
        all_seen_coverage=len([x for x in xs if x in MILESTONES]))
