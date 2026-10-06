from .common import require,digest

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
            hidden = adapter.observer_hidden(input_ids=ids, attention_mask=mask)
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
