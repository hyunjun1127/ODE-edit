"""Reuse JVP NLL kernel with a separate tokenizer; publish hashes/scalars only."""
import math
import sys
import types
from pathlib import Path
from .integrity import digest, signature


def bind_observation_only_package():
    # Reuse exact evaluator bytes without executing the historical package's
    # __init__, which eagerly imports an unrelated EasyEdit barrier writer.
    name = 'project.run_scripts.alphaedit_strength_neutral_barrier'
    path = Path(__file__).resolve().parents[1]/'alphaedit_strength_neutral_barrier'
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        package.__package__ = name
        sys.modules[name] = package
    assert sys.modules[name].__path__ == [str(path)]


def reduce(raw):
    out = {}
    for category, tag, reverse in [('rewrite', 'RS', False), ('rephrase', 'PS', False), ('locality', 'NS', True)]:
        if category + '_target_new' not in raw:
            continue
        rows = []
        for a, b in zip(raw[category + '_target_new'], raw[category + '_target_true'], strict=True):
            assert (a['case_id'], a['prompt_index'], a['prompt']) == (b['case_id'], b['prompt_index'], b['prompt'])
            assert math.isfinite(a['nll']) and math.isfinite(b['nll'])
            rows.append(dict(case_id=a['case_id'], prompt_index=a['prompt_index'],
                identity=digest([a['case_id'], a['prompt_index'], a['prompt'], a['target'], b['target']]),
                new_nll=a['nll'], true_nll=b['nll'], margin=b['nll']-a['nll'],
                success=(b['nll'] < a['nll'] if reverse else a['nll'] < b['nll']),
                new_strict=a['all_tokens_correct'], true_strict=b['all_tokens_correct'],
                new_token_correct=sum(a['token_correct']), new_token_count=len(a['token_correct']),
                true_token_correct=sum(b['token_correct']), true_token_count=len(b['token_correct'])))
        assert len({r['identity'] for r in rows}) == len(rows)
        n = sum(r['success'] for r in rows)
        out[tag] = dict(numerator=n, denominator=len(rows), rate=n/len(rows),
                        bit_order_sha256=digest([(r['identity'], r['success']) for r in rows]), rows=rows)
    return out


def evaluate(model, tok, records, weights, cache, full=True):
    import torch
    bind_observation_only_package()
    from project.run_scripts.alphaedit_strength_neutral_barrier.evaluator import counterfact_pairs, evaluate_pairs
    from project.run_scripts.ordered_response_barrier_ode.counterfact_locality_evaluator import counterfact_locality_target_new_pairs
    before = signature(weights, cache)
    pairs = counterfact_pairs(records)
    if full:
        pairs['locality_target_new'] = counterfact_locality_target_new_pairs(records)
    else:
        pairs = {k: pairs[k] for k in ('rewrite_target_new', 'rewrite_target_true')}
    raw = {k: evaluate_pairs(model, tok, v, device=torch.device('cuda'), microbatch_size=16) for k, v in pairs.items()}
    if before != signature(weights, cache):
        raise RuntimeError('EVALUATOR_STATE_MUTATION')
    return dict(requests=len(records), request_order=digest([r['case_id'] for r in records]),
                metrics=reduce(raw), before_after_exact=True, evaluator_controller_influence=0,
                weight_state={k: v['sha256'] for k, v in before['weights'].items()}, cache_sha256=before['cache_sha256'])
