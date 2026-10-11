"""Pinned FE author writer/history with our current-batch-entry target fitter.

Only ``batch_edit`` is called: the author's all-future W0 cache loader is never
used. Its RHS solve, pre-write FP64 key Gram and single-cast weight assignment
are retained. This is a diagnostic backend, not a baseline redefinition.
"""
from copy import deepcopy
import importlib.util
import hashlib
from pathlib import Path
import sys

import torch
from omegaconf import OmegaConf

from .memit_fe_history import native as target_native
from .fe_covariance import guarded_author_solve

AUTHOR_COMMIT = '478134dfb24b43f4e18b47e8500893ce3f9cc50f'
WRITER_SHA256 = '6934f007ef6066690933a38a5a8d5bfde1e17d5ebbb5879210e4f6d037d26f9d'
BACKEND = 'AUTHOR_WRITER_CURRENT_BATCH_Z'
NAMES = {'llama3': 'llama3-8b', 'gptj': 'gpt-j-6b', 'qwen25': 'qwen2.5-7b'}


def load_writer(root, model, hparams, covariances):
    """Caller verifies the immutable author source manifest before importing."""
    root = Path(root).resolve()
    if hashlib.sha256((root/'algs/memit/FE-memit_main.py').read_bytes()).hexdigest() != WRITER_SHA256:
        raise ValueError('FE_AUTHOR_WRITER_SOURCE_CHANGED')
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location(
        'algs.memit.fe_sequential_diagnostic', root/'algs/memit/FE-memit_main.py')
    author = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(author)
    # Refuse a conflicting already-imported top-level author package.
    for name in ('algs.memit.compute_ks', 'algs.memit.compute_z', 'util.nethook'):
        module = sys.modules[name]
        if not Path(module.__file__).resolve().is_relative_to(root):
            raise ValueError('FE_AUTHOR_IMPORT_OUTSIDE_PINNED_TREE')
    cfg = OmegaConf.load(root/'configs/config.yaml')
    cfg.llms = OmegaConf.load(root/'configs/llms'/f'{NAMES[model]}.yaml')
    cfg.algs = OmegaConf.load(root/'configs/algs/memit.yaml')
    for name in ('layers', 'fact_token', 'rewrite_module_tmp', 'layer_module_tmp',
                 'clamp_norm_factor', 'v_num_grad_steps', 'v_lr', 'v_weight_decay',
                 'v_loss_layer', 'kl_factor', 'mom2_dataset', 'mom2_n_samples', 'mom2_dtype'):
        if getattr(cfg.llms, name) != getattr(hparams, name):
            raise ValueError('FE_AUTHOR_HPARAM_MISMATCH: '+name)
    if list(cfg.llms.mom2_update_weight) != [hparams.mom2_update_weight]*len(hparams.layers):
        raise ValueError('FE_AUTHOR_COVARIANCE_COEFFICIENT_MISMATCH')
    if not cfg.algs.add_old_keys or cfg.algs.L2 != 0 or cfg.negetive_prompt_test:
        raise ValueError('FE_AUTHOR_HISTORY_CONFIG_MISMATCH')
    cfg.bs = 100
    cfg.num_z_samples = 100
    cfg.model_dtype = 'float32'
    author.covs = [covariances[hparams.rewrite_module_tmp.format(l)] for l in hparams.layers]
    return author, cfg


def apply(model, tok, requests, hparams, *, history, author, config):
    layers = [str(l) for l in hparams.layers]
    if not requests or set(history) != set(layers):
        raise ValueError('FE_AUTHOR_CURRENT_BATCH_HISTORY_MAPPING')
    weights = {f'{hparams.rewrite_module_tmp.format(l)}.weight':
               target_native.nethook.get_parameter(model, f'{hparams.rewrite_module_tmp.format(l)}.weight')
               for l in hparams.layers}
    before = {k: v.detach().cpu().clone() for k, v in weights.items()}
    metadata = {k: (id(p), p.data_ptr(), p._version) for k, p in model.named_parameters()}
    versions = {k: p._version for k, p in weights.items()}
    pending = torch.stack([history[l] for l in layers])
    if pending.dtype != torch.float32 or pending.device.type != 'cpu' or not torch.isfinite(pending).all():
        raise ValueError('FE_AUTHOR_HISTORY_SHAPE_DTYPE_FINITE')
    normalized = deepcopy(requests)
    contexts = target_native.get_context_templates(model, tok)
    author.CONTEXT_TEMPLATES_CACHE = deepcopy(contexts)
    targets = {l: [] for l in hparams.layers}
    try:
        for i, request in enumerate(normalized):
            if not request['target_new'].startswith(' '):
                request['target_new'] = ' '+request['target_new']
            if '{}' not in request['prompt']:
                if request['subject'] not in request['prompt']:
                    raise ValueError('FE_AUTHOR_SUBJECT_MISSING')
                request['prompt'] = request['prompt'].replace(request['subject'], '{}')
            # The author writer indexes only this batch's newly computed columns.
            request['sample_idx'] = i
            if versions != {k: p._version for k, p in weights.items()}:
                raise ValueError('FE_TARGET_ENTRY_MODEL_CHANGED')
            current = target_native.compute_z(model, tok, request, hparams, hparams.layers, contexts)
            if set(current) != set(hparams.layers):
                raise ValueError('FE_TARGET_INCOMPLETE')
            for layer in hparams.layers:
                targets[layer].append(current[layer].detach())
        if versions != {k: p._version for k, p in weights.items()}:
            raise ValueError('FE_TARGET_ENTRY_MODEL_CHANGED')
        zs = {l: torch.stack(values, dim=1) for l, values in targets.items()}
        with guarded_author_solve(author) as solves:
            author.batch_edit(config, model, tok, normalized, next(iter(weights.values())).device, pending, zs)
        if not torch.isfinite(pending).all() or not all(torch.isfinite(w).all() for w in weights.values()):
            raise ValueError('FE_AUTHOR_COMMIT_NONFINITE')
        for name, p in model.named_parameters():
            old = metadata[name]
            if id(p) != old[0] or p.data_ptr() != old[1] or (name not in weights and p._version != old[2]):
                raise ValueError('FE_AUTHOR_NONSELECTED_OR_IDENTITY_MUTATION')
        # Do not advance H if any layer or target fit failed.
        history.update({l: pending[i] for i, l in enumerate(layers)})
        return solves
    except BaseException:
        with torch.no_grad():
            for name, weight in weights.items():
                weight.copy_(before[name])
        raise
