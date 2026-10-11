"""Read-only FE batch instrumentation and a no-future-target execution guard."""
from contextlib import contextmanager

import torch


@contextmanager
def observe_batch(native, requests, hparams, weights, *, writer=None):
    writer = native if writer is None else writer
    expected = [r['case_id'] for r in requests]
    versions = {k: v._version for k, v in weights.items()}
    owners = dict(compute_z=native, compute_ks=writer, get_module_input_output_at_words=writer)
    originals = {name: getattr(owner, name) for name, owner in owners.items()}
    captured = dict(case_ids=[], targets=[], keys={}, current_outputs={}, history_keys={})

    def target(model, tok, request, *args, **kwargs):
        index = len(captured['case_ids'])
        if index >= len(expected) or request['case_id'] != expected[index]:
            raise ValueError('FE_TARGET_OUTSIDE_CURRENT_BATCH')
        if versions != {k: v._version for k, v in weights.items()}:
            raise ValueError('FE_TARGET_ENTRY_MODEL_CHANGED')
        result = originals['compute_z'](model, tok, request, *args, **kwargs)
        if result is None or set(result) != set(hparams.layers):
            raise ValueError('FE_TARGET_INCOMPLETE')
        captured['case_ids'].append(request['case_id'])
        captured['targets'].append({str(k): v.detach().cpu().clone() for k, v in result.items()})
        return result

    def keys(model, tok, rows, hp, layer, *args, **kwargs):
        result = originals['compute_ks'](model, tok, rows, hp, layer, *args, **kwargs)
        key = str(layer)
        bucket = 'keys' if key not in captured['keys'] else 'history_keys'
        if key in captured[bucket]:
            raise ValueError('FE_DUPLICATE_KEY_CAPTURE')
        captured[bucket][key] = result.detach().cpu().clone()
        return result

    def output(model, tok, layer, *args, **kwargs):
        result = originals['get_module_input_output_at_words'](model, tok, layer, *args, **kwargs)
        if kwargs.get('track') == 'out':
            captured['current_outputs'][str(layer)] = result.detach().cpu().clone()
        elif writer is not native and isinstance(result, tuple):
            captured['current_outputs'][str(layer)] = result[1].detach().cpu().clone()
        return result

    native.compute_z, writer.compute_ks = target, keys
    writer.get_module_input_output_at_words = output
    try:
        yield captured
        if captured['case_ids'] != expected:
            raise ValueError('FE_CURRENT_BATCH_TARGETS_MISSING')
    finally:
        for name, function in originals.items():
            setattr(owners[name], name, function)


def summarize(captured):
    result = dict(case_ids=captured['case_ids'], z_requests=len(captured['targets']), layers={})
    for layer, keys in captured['keys'].items():
        targets = torch.stack([row[layer] for row in captured['targets']])
        current = captured['current_outputs'].get(layer)
        values = dict(target_norm=targets.double().norm(dim=1).tolist(),
                      key_norm=keys.double().norm(dim=1).tolist())
        if current is not None:
            values['residual_norm'] = (targets - current).double().norm(dim=1).tolist()
        if layer in captured['history_keys']:
            values['write_history_key_max_abs_difference'] = float((keys - captured['history_keys'][layer]).abs().max())
        result['layers'][layer] = values
    return result
