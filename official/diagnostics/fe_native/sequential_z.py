"""Batch-entry firstforward targets for the author's unmodified FE writer.

Installed into algs/memit in a pinned author checkout. No ODE/EasyEdit imports.
The optimization and forward replay are the author's operations; only their
placement moves from all-future W0 precomputation to the current batch.
"""
import torch


def current_batch_targets(model, tok, requests, cfg, contexts, *, compute_z,
                          find_fact_lookup_idx, TraceDict):
    if not requests or len(requests) > cfg.bs:
        raise ValueError("CURRENT_BATCH_ONLY")
    layers = sorted(cfg.llms.layers)
    first = layers[0]
    versions = {name: (id(p), p._version) for name, p in model.named_parameters()}
    # apply_memit_to_model already adds the author's leading target space.
    fitted = [compute_z(model, tok, request, cfg, first, contexts) for request in requests]
    targets = {first: torch.stack(fitted, dim=1)}
    for layer in layers[1:]:
        targets[layer] = []
    device = next(model.parameters()).device
    for index, request in enumerate(requests):
        inputs = tok(request['prompt'].format(request['subject']), return_tensors='pt').to(device)
        lookup = find_fact_lookup_idx(request['prompt'], request['subject'], tok,
                                     cfg.llms.fact_token, verbose=False)

        def inject(output, name):
            if name == cfg.llms.layer_module_tmp.format(first):
                hidden = output[0] if isinstance(output, (tuple, list)) else output
                if hidden.shape[0] == 1:
                    hidden[0, lookup, :] = targets[first][:, index]
                else:
                    hidden[lookup, 0, :] = targets[first][:, index]
            return output

        with torch.no_grad():
            with TraceDict(model, layers=[cfg.llms.layer_module_tmp.format(l) for l in layers],
                           edit_output=inject) as traced:
                model(**inputs)
            for layer in layers[1:]:
                output = traced[cfg.llms.layer_module_tmp.format(layer)].output
                hidden = output[0] if isinstance(output, (tuple, list)) else output
                value = hidden[0, lookup, :] if hidden.shape[0] == 1 else hidden[lookup, 0, :]
                targets[layer].append(value.detach().clone())
    for layer in layers[1:]:
        targets[layer] = torch.stack(targets[layer], dim=1)
    if versions != {name: (id(p), p._version) for name, p in model.named_parameters()}:
        raise RuntimeError('MODEL_CHANGED_DURING_BATCH_TARGETS')
    for layer, value in targets.items():
        if value.ndim != 2 or value.shape[1] != len(requests) or not torch.isfinite(value).all():
            raise ValueError(f'INVALID_BATCH_TARGETS_LAYER_{layer}')
    return targets
