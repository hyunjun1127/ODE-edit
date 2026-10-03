"""Observer-only native R/P/N scoring; never optimizer or memory feedback.

Evaluation semantics and the raw-row schema explicitly reuse the historical
observer, not its writer/optimizer. The reducer has no torch/model dependency.
"""
import math
import time
import unicodedata
from pathlib import Path

from .common import digest, require, state, write
from .writer import rng_snapshot, rng_equal


def active_flags(records):
    flags = {r['case_id']: True for r in records}
    versions = {}
    for record in records:
        rewrite = record['requested_rewrite']
        claim = (unicodedata.normalize('NFC', ' '.join(rewrite['subject'].split())),
                 rewrite['relation_id'])
        target = rewrite['target_new'].get('id', rewrite['target_new']['str'])
        for old, old_target in versions.get(claim, []):
            if old_target != target:
                flags[old] = False
        versions.setdefault(claim, []).append((record['case_id'], target))
    return flags


def reduce_rows(rows):
    """Independent strict preference and teacher-forced token/prompt reduction."""
    require(len({r['identity'] for r in rows}) == len(rows), 'DUPLICATE_OBSERVER_ROW')
    result = {}
    for row in rows:
        require(row['kind'] in {'R', 'P', 'N'}, 'UNKNOWN_OBSERVER_KIND')
        for label in ('new', 'true'):
            require(isinstance(row[label + '_nll'], (int, float))
                    and math.isfinite(row[label + '_nll']), 'NONFINITE_OBSERVER')
            count, correct = row[label + '_token_count'], row[label + '_token_correct']
            require(type(count) is int and type(correct) is int
                    and count > 0 and 0 <= correct <= count, 'INVALID_OBSERVER_TOKEN_COUNTS')
            require(type(row[label + '_strict']) is bool
                    and row[label + '_strict'] == (count == correct), 'INVALID_OBSERVER_STRICT')
    for kind in sorted({r['kind'] for r in rows}):
        group = [r for r in rows if r['kind'] == kind]
        desired = 'true' if kind == 'N' else 'new'
        success = [r['true_nll'] < r['new_nll'] if kind == 'N'
                   else r['new_nll'] < r['true_nll'] for r in group]
        count = sum(r[desired + '_token_count'] for r in group)
        correct = sum(r[desired + '_token_correct'] for r in group)
        result[kind] = dict(
            denominator=len(group), numerator=sum(success), rate=sum(success) / len(group),
            true_nll_mean=sum(r['true_nll'] for r in group) / len(group),
            new_nll_mean=sum(r['new_nll'] for r in group) / len(group),
            desired_token_count=count, desired_token_correct=correct, token_micro=correct / count,
            prompt_macro=sum(r[desired + '_token_correct'] / r[desired + '_token_count']
                             for r in group) / len(group),
            strict_numerator=sum(r[desired + '_strict'] for r in group),
            strict_denominator=len(group), new_strict_numerator=sum(r['new_strict'] for r in group))
    return result


def scores(adapter, bench, pairs, microbatch):
    """Same full-vocabulary, left-padded teacher-forced evaluation as baseline."""
    import torch

    require(type(microbatch) is int and microbatch > 0, 'INVALID_EVAL_MICROBATCH')
    output = []
    with torch.no_grad():
        for start in range(0, len(pairs), microbatch):
            group = pairs[start:start + microbatch]
            encoded = [bench.evaluation_ids(prompt, target) for prompt, target in group]
            require(all(p and t for p, t in encoded), 'EMPTY_EVAL_TOKENS')
            width = max(len(p) + len(t) - 1 for p, t in encoded)
            require(width <= adapter.model.config.max_position_embeddings,
                    'EVAL_LENGTH_OVERFLOW_NO_TRUNCATION')
            ids = torch.full((len(group), width), bench.tokenizer.pad_token_id,
                             device=adapter.device, dtype=torch.long)
            mask = torch.zeros_like(ids)
            positions = []
            for i, (prompt, target) in enumerate(encoded):
                row = (prompt + target)[:-1]
                offset = width - len(row)
                ids[i, offset:] = torch.tensor(row, device=adapter.device)
                mask[i, offset:] = 1
                positions.append(list(range(offset + len(prompt) - 1, width)))
            hidden = adapter.model.model(input_ids=ids, attention_mask=mask,
                                         use_cache=False).last_hidden_state
            selected = torch.cat([hidden[i, pos] for i, pos in enumerate(positions)])
            logits = adapter.model.lm_head(selected).float()
            logp, pred = logits.log_softmax(-1), logits.argmax(-1)
            cursor = 0
            for prompt, target in encoded:
                target_ids = torch.tensor(target, device=adapter.device)
                lp, prediction = logp[cursor:cursor + len(target)], pred[cursor:cursor + len(target)]
                output.append(dict(nll=float(-lp.gather(1, target_ids[:, None]).mean()),
                                   token_count=len(target), token_correct=int((prediction == target_ids).sum()),
                                   strict=bool((prediction == target_ids).all()),
                                   token_identity=digest([prompt, target])))
                cursor += len(target)
    return output


def observe(adapter, bench, all_records, selected_records, history, endpoint, out,
            microbatch=2, current_ids=None):
    """Write scalar/hash-only rows and verify W/H/guard/hooks. Replay is absent.
    Completed chunks survive a later evaluation failure for partial reporting.
    """
    out = Path(out)
    before = state(adapter, history)
    guard, hooks = adapter.guard(), adapter.hook_signature()
    rng = rng_snapshot(); contexts = digest(bench.contexts)
    started, rows = time.monotonic(), []
    flags = active_flags(all_records)
    try:
        for start in range(0, len(selected_records), 50):
            records = selected_records[start:start + 50]
            specs, pairs = [], []
            for record in records:
                rewrite = record['requested_rewrite']
                for kind, prompts in bench.panels(record).items():
                    require(kind in {'R', 'P', 'N'}, 'UNKNOWN_OBSERVER_KIND')
                    for index, prompt in enumerate(prompts):
                        identity = digest([record['case_id'], kind, index, prompt,
                                           rewrite['target_new']['str'], rewrite['target_true']['str']])
                        specs.append(dict(case_id=record['case_id'], kind=kind, prompt_index=index,
                                          identity=identity, endpoint=endpoint,
                                          active_at_endpoint=flags[record['case_id']]))
                        pairs.extend([(prompt, rewrite['target_new']['str']),
                                      (prompt, rewrite['target_true']['str'])])
            values = scores(adapter, bench, pairs, microbatch)
            require(len(values) == 2 * len(specs), 'OBSERVER_SCORE_COUNT')
            for i, row in enumerate(specs):
                for label, value in zip(('new', 'true'), values[2 * i:2 * i + 2]):
                    row.update({label + '_' + k: v for k, v in value.items()})
                row['margin_true_minus_new'] = row['true_nll'] - row['new_nll']
            reduce_rows(specs)
            rows.extend(specs)
            write(out / f'chunk-{start:04d}.json', dict(schema='jlz-observer-rows-v1',
                  state=before, rows=specs, optimizer_feedback=False))
            print(dict(event='observer', endpoint=endpoint, requests_done=start + len(records)), flush=True)
    finally:
        require(state(adapter, history) == before and adapter.guard() == guard
                and adapter.hook_signature() == hooks and rng_equal(rng) and digest(bench.contexts)==contexts, 'OBSERVER_MUTATION')
    current = set([r['case_id'] for r in selected_records] if current_ids is None else current_ids)
    result = dict(schema='jlz-observer-summary-v1', endpoint=endpoint, state=before,
                  requests=len(selected_records), summary=reduce_rows(rows),
                  current=reduce_rows([r for r in rows if r['case_id'] in current]),
                  row_count=len(rows), row_order=digest([r['identity'] for r in rows]),
                  seconds=time.monotonic() - started, no_mutation=True,
                  replay=False, optimizer_feedback=False)
    write(out / 'summary.json', result)
    return result
