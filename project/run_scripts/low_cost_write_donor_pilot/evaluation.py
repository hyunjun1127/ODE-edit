"""Pilot endpoint evaluation only. Runtime owns W/M/RNG nonmutation guards.

No editor, optimizer, donor-selection or fallback import. CounterFact binds the
historical MB16 path; Wiki and MMLU have distinct layouts and denominators.
"""
from __future__ import annotations

import math
import numpy as np
import torch

from project.run_scripts.baseline_mechanism_first.contracts import ContractBoundary, digest
from project.run_scripts.baseline_mechanism_first.evaluation import evaluate_records
from project.run_scripts.baseline_mechanism_first.performance_schema import normalize
from .panels import validate_wiki


def counterfact(model, tokenizer, records, *, panel, annotations=None):
    result = normalize(evaluate_records(model, tokenizer, records), records)
    annotations = annotations or {}
    for category, metric in result['metrics'].items():
        for row in metric['rows']:
            row['desired_margin'] = ((row['new_nll']-row['true_nll']) if category == 'NS'
                                     else (row['true_nll']-row['new_nll']))
            row['panel'] = panel
            if row['case_id'] in annotations:
                row['historical_status'] = annotations[row['case_id']]
    return dict(result, panel=panel, optimizer_access=0,
                transaction_guard='CALLER_MUST_VALIDATE_ACTUAL_ENDPOINT_W_M_RNG')


@torch.no_grad()
def wiki(model, manifest):
    seal = validate_wiki(manifest)
    device = next(model.parameters()).device
    out = []
    for row in manifest['rows']:
        ids = torch.tensor([row['input_ids']], dtype=torch.long, device=device)
        logits = model(input_ids=ids[:, :-1], attention_mask=torch.ones_like(ids[:, :-1]), use_cache=False).logits.float()
        value = float(torch.nn.functional.cross_entropy(logits.transpose(1, 2), ids[:, 1:]))
        if not math.isfinite(value):
            raise ContractBoundary('NONFINITE_WIKI_NLL')
        out.append(dict(ordinal=row['ordinal'], input_sha256=digest(row['input_ids']),
                        nll=value, predicted_tokens=ids.shape[1]-1))
    return dict(rows=out, numerator_nll_sum=sum(r['nll'] for r in out), denominator=len(out),
                mean_nll=float(np.mean([r['nll'] for r in out])), units='NATS_PER_TOKEN_THEN_SEQUENCE_MEAN',
                seal=seal, layout='SOURCE_E01_ONE_SEQUENCE_FULL_NEXT_TOKEN', optimizer_access=0)


def unique_prediction(probabilities):
    if len(probabilities) != 4 or not all(math.isfinite(x) for x in probabilities):
        raise ContractBoundary('MMLU_NONFINITE_OR_CARDINALITY')
    best = max(probabilities)
    winners = [i for i, x in enumerate(probabilities) if x == best]
    return winners[0] if len(winners) == 1 else -1


def mmlu_prompt(example):
    if type(example['answer']) is not int or example['answer'] not in range(4) or len(example['choices']) != 4:
        raise ContractBoundary('MMLU_SCHEMA')
    return 'Question: '+example['question']+'\n'+''.join(
        '('+label+') '+example['choices'][i]+'\n' for i, label in enumerate('ABCD'))+'Answer:'


@torch.no_grad()
def mmlu_alternative(model, tokenizer, rows, *, expected_count=32):
    """BLUE311b076a alternative branch only, source Llama token offsets intact.

    Extracted from glue_eval/mmlu_eval.py::MMLUEval.evaluate. No generation;
    the exact zero-shot prompt is _create_prompt(..., gen_len=5). Mean suffix
    NLL is exponentiated as in source (underflow/ties remain invalid, not argmin).
    """
    if len(rows) != expected_count or len({digest(r) for r in rows}) != expected_count:
        raise ContractBoundary('MMLU_PANEL_CARDINALITY')
    if 'llama' not in model.config._name_or_path.lower():
        raise ContractBoundary('MMLU_LLAMA_SOURCE_BINDING')
    device = next(model.parameters()).device
    suffixes = [tokenizer(' '+label)['input_ids'][1:] for label in 'ABCD']
    if any(not ids for ids in suffixes):
        raise ContractBoundary('MMLU_EMPTY_SUFFIX')
    out = []
    for example in rows:
        prompt = mmlu_prompt(example)
        prefix_len = len(tokenizer(prompt)['input_ids'])-1
        if prefix_len < 1:
            raise ContractBoundary('MMLU_EMPTY_PREFIX')
        nlls = []
        for label, ids in zip('ABCD', suffixes):
            encoded = tokenizer([prompt+' '+label], return_tensors='pt').to(device)
            logits = model(**encoded).logits[:, 1:, :]
            score = 0.
            for j, token in enumerate(ids):
                score += -torch.nn.functional.log_softmax(logits[0, prefix_len+j-1, :], dim=0)[token].item()
            nlls.append(score/len(ids))
        if not all(math.isfinite(v) for v in nlls):
            raise ContractBoundary('NONFINITE_MMLU_NLL')
        probabilities = [float(np.exp(-x)) for x in nlls]
        predicted = unique_prediction(probabilities)
        out.append(dict(row_sha256=digest(example), gold=example['answer'], prediction=predicted,
                        correct=int(predicted == example['answer']), alternative_nll=nlls,
                        alternative_probability=probabilities, tie_or_underflow_invalid=predicted == -1))
    correct = sum(r['correct'] for r in out)
    return dict(rows=out, correct=correct, denominator=expected_count, accuracy=correct/expected_count,
                invalid=sum(r['prediction'] == -1 for r in out), fewshot=0, prompt_gen_len_reserve=5,
                source='BLUE311b076a glue_eval/mmlu_eval.py alternative branch', generation_calls=0,
                model_forwards=4*expected_count, metric='INTEGER_CORRECT_NOT_WEIGHTED_F1',
                optimizer_access=0, full_mmlu_benchmark=False)
