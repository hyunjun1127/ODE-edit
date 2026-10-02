"""CounterFact is one benchmark adapter, not a method-wide assumption."""
import json
import torch
from project.run_scripts.jlz_pilot.prompts import prepare as native_prepare
from .common import require, digest

class CounterFactAdapter:
    def __init__(self, tokenizer, contexts):
        self.tokenizer, self.contexts = tokenizer, contexts

    def prepare(self, records):
        requests = [r['requested_rewrite'] | {'case_id': r['case_id']} for r in records]
        pack = native_prepare(self.tokenizer, requests, self.contexts, 'cpu')
        pack['record_ids'] = [r['case_id'] for r in records]
        # Exact prefix proof for fusing teacher-forced key captures, per row.
        match = []
        for k, row in enumerate(pack['rw_rows']):
            pos = pack['lookup'][row]
            match.append(pos == pack['key_lookup'][k] and torch.equal(
                pack['tokens']['input_ids'][row, :pos+1], pack['key_tokens']['input_ids'][k, :pos+1]))
        pack['entry_key_prefix_exact'] = all(match)
        return pack

    def panels(self, record):
        r = record['requested_rewrite']
        return {'R': [r['prompt'].format(r['subject'])],
                'P': record['paraphrase_prompts'], 'N': record['neighborhood_prompts']}

    def evaluation_ids(self, prompt, target):
        tok = self.tokenizer
        prompt_ids = list(tok(prompt, add_special_tokens=True)['input_ids'])
        target = target if target.startswith(' ') else ' ' + target
        target_ids = list(tok.encode(target, add_special_tokens=False))
        while target_ids and target_ids[0] in {tok.bos_token_id, tok.unk_token_id}:
            target_ids = target_ids[1:]
        require(prompt_ids and target_ids, 'EMPTY_EVAL_TOKENS')
        return prompt_ids, target_ids

def chunks(spec, microbatch):
    # Stable original request-major schedule common to A/B.
    n = len(spec['row_request'])
    return [list(range(i, min(i+microbatch, n))) for i in range(0, n, microbatch)]

def subset(tokens, rows, device, lengths=None):
    mask = tokens['attention_mask'][rows].clone()
    if lengths is not None:
        for i, n in enumerate(lengths):
            mask[i, n:] = 0
    width = int(mask.sum(1).max())
    require(width > 0, 'EMPTY_MICROBATCH')
    return {'input_ids': tokens['input_ids'][rows, :width].to(device),
            'attention_mask': mask[:, :width].to(device)}

def pool(raw, spec):
    groups = spec['context_group_slices']
    x = raw.reshape(spec['n_requests'], spec['n_rw'], -1)
    return torch.stack([x[:, a:b].mean(1) for a, b in groups]).mean(0).T.contiguous()
