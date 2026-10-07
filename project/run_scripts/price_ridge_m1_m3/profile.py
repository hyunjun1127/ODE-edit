"""Explicit five-cell matrix; an unqualified lambda is never substituted."""
import copy
import math

CELLS = {
    'LLAMA_REPRO': ('llama3', False, False, False),
    'GPTJ_M1': ('gptj', True, False, True),
    'GPT2XL_M1_M2': ('gpt2xl', True, True, False),
    'GPT2XL_M1_M3': ('gpt2xl', True, False, True),
    'GPT2XL_M1_M2_M3': ('gpt2xl', True, True, True),
}
NATIVE = {'llama3': (list(range(4, 9)), 25, .1, 15000.),
          'gptj': (list(range(3, 9)), 25, .5, 15000.),
          'gpt2xl': (list(range(13, 18)), 20, .5, 20000.)}


def modified_profile(base, cell, realization):
    model, m1, m2, m3 = CELLS[cell]
    layers, evaluations, lr, native = NATIVE[model]
    if (base.get('writer', 'memit'), base['eligible_layers'], base['K_eval'], base['max_updates'],
        base['lr'], base['lambda_C'], base['arm'], base['beta_base'], base['cap_mode']) != (
            'memit', layers, evaluations, evaluations-1, lr, native, 'CAP075', .75, 'native'):
        raise ValueError('TASK_NATIVE_PROFILE_IDENTITY')
    if realization['model'] != model or realization['layer'] != layers[0]:
        raise ValueError('B1_REALIZATION_MODEL_LAYER_IDENTITY')
    median = realization['median_realization']
    if not math.isfinite(median):
        raise ValueError('B1_REALIZATION_NONFINITE')
    if model == 'llama3' and median < .5:
        raise ValueError('LLAMA_NATIVE_NOOP_PREMISE_FAILED')
    p = copy.deepcopy(base)
    if m1:
        p['price_m1_anchor_guard'] = True
    p['K_grace'] = (evaluations-1)//2 if m2 else 12
    if m3 and median < .5:
        cal = realization.get('calibration')
        if not cal or cal.get('status') != 'SAVED_B1_CALIBRATED':
            raise ValueError('NOT_RECORDED_SAVED_B1_CALIBRATION')
        value = cal['lambda_C']
        if not math.isfinite(value) or not 0 < value <= native or abs(cal['median']-.5) > .01:
            raise ValueError('M3_CALIBRATION_INVALID')
        p['lambda_C'] = value
    return p


def forbid_w0(config):
    """Caller must never fall back to W0 forwards, including generation."""
    if config.get('w0_policy') != 'REUSE_ONLY_NO_FORWARD':
        raise ValueError('NEW_W0_FORBIDDEN')
    if config.get('generate_contexts', False):
        raise ValueError('NEW_CONTEXT_FORWARD_FORBIDDEN')
