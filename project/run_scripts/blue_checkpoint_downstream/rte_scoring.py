"""GH-approved scoring-only correction; no evaluator/model/data mutation."""
from collections import Counter
import hashlib
import json

VERSION = 'RTE_LABEL_MAPPING_CORRECTED_V1'
AUTHORITY = 'ODEEDIT-GH-RTE-LABEL-FIX-20260909-R1'


def canonical_prediction(prediction):
    """Preserve unknown/invalid sentinels, instead of indiscriminately using 1-p."""
    if type(prediction) is int and prediction in (0, 1):
        return {1: 0, 0: 1}[prediction]
    return prediction


def prediction_record(prediction, raw_gold):
    assert type(raw_gold) is int and raw_gold in (0, 1)
    valid = type(prediction) is int and prediction in (0, 1)
    canonical = canonical_prediction(prediction)
    return dict(raw_prediction=prediction,
                semantic_prediction={1:'True', 0:'False'}[prediction] if valid else 'INVALID',
                canonical_prediction=canonical, raw_gold=raw_gold, valid=valid,
                correct=valid and canonical == raw_gold, mapping=VERSION)


def summarize(records):
    from sklearn.metrics import f1_score, matthews_corrcoef
    assert records
    gold = [r['raw_gold'] for r in records]
    # Original source produces integer -1 for invalid. Other sentinels remain in
    # the receipt and are represented by the same invalid class for aggregation.
    predicted = [r['canonical_prediction'] if r['valid'] else -1 for r in records]
    correct = sum(r['correct'] for r in records)
    invalid = sum(not r['valid'] for r in records)
    return dict(correct=correct, incorrect=len(records)-correct-invalid, invalid=invalid,
                total=len(records), accuracy=correct/len(records),
                weighted_f1=float(f1_score(gold, predicted, average='weighted')),
                mcc=float(matthews_corrcoef(gold, predicted)),
                support=dict(Counter(str(x) for x in gold)))


def score_source_rows(rows, gold_rows, source_metrics):
    """Score the unchanged source generation and alternative outputs separately.

    source stores alternative prediction as highest_probability_answer, not
    numeric answer_new. Recover exactly that source boolean; do not re-infer
    answers from free text or recalculate model probabilities.
    """
    assert len(rows) == len(gold_rows) and rows
    records = []
    for ordinal, (row, gold) in enumerate(zip(rows, gold_rows, strict=True)):
        assert row['sentence1'] == gold['sentence1'] and row['sentence2'] == gold['sentence2']
        label = gold['label']
        alternative = {'True':1, 'False':0}.get(row['highest_probability_answer'], -1)
        # Existing source alternative branch uses strict > with False on ties.
        if alternative in (0, 1):
            assert alternative == (1 if row['prob_yes'] > row['prob_no'] else 0)
        records.append(dict(ordinal=ordinal,
            record_sha256=hashlib.sha256(json.dumps(gold,sort_keys=True,ensure_ascii=True,separators=(',',':')).encode()).hexdigest(),
            generation=prediction_record(row['answer'], label),
            alternative=prediction_record(alternative, label),
            source_alternative_field=row['highest_probability_answer']))
    generation = summarize([r['generation'] for r in records])
    alternative = summarize([r['alternative'] for r in records])
    return dict(status='RTE_LABEL_MAPPING_CORRECTED', mapping=VERSION, authority=AUTHORITY,
                generation=generation, alternative=alternative, records=records,
                original_bug_diagnostic=dict(source_metrics),
                original_metric_is_main=False, dataset_relabel=0, model_output_change=0,
                f1=generation['weighted_f1'], f1_new=alternative['weighted_f1'])
