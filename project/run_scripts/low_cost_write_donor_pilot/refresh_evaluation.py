"""Write-refresh observations; same MB16 scorer, terminal-only general panels.

The runtime must guard actual W/M/context/RNG and parameter versions. Values
are never exposed to target optimization or policy selection.
"""
import copy
import time

from project.run_scripts.baseline_mechanism_first.contracts import ContractBoundary, digest
from . import evaluation
from .panels import history_status
from .sequential_evaluation import exact_subset, validate_document


def strict_summary(document):
    """Whole-request two-paraphrase strict; never counted as two requests."""
    result = {}
    for tag in ('RS', 'PS'):
        rows = document['metrics'][tag]['rows']
        if not all(type(r.get('new_strict')) is bool for r in rows):
            raise ContractBoundary('STRICT_OBSERVATION_MISSING')
        result[tag] = dict(numerator=sum(r['new_strict'] for r in rows), denominator=len(rows))
    paired = {}
    for row in document['metrics']['PS']['rows']:
        paired.setdefault(row['case_id'], {})[row['prompt_index']] = row['new_strict']
    if any(set(value) != {0, 1} for value in paired.values()):
        raise ContractBoundary('TWO_PARAPHRASE_REQUEST_INVENTORY')
    result['two_P_request'] = dict(numerator=sum(all(v.values()) for v in paired.values()),
                                    denominator=len(paired))
    return result


def evaluate_batch(model, tokenizer, records, batch, historical_ordinals,
                   wiki_panel, dev_mmlu, endpoint_state):
    if type(batch) is not int or batch not in range(51, 61) or len(records) != 10000:
        raise ContractBoundary('REFRESH_EVALUATION_BATCH_OR_DATASET')
    if len({r['case_id'] for r in records}) != 10000:
        raise ContractBoundary('REFRESH_DUPLICATE_CASE')
    if (len(historical_ordinals) != 128 or len(set(historical_ordinals)) != 128
            or any(type(i) is not int or not 0 <= i < 5000 for i in historical_ordinals)):
        raise ContractBoundary('REFRESH_HISTORICAL_PANEL')
    if len(dev_mmlu) != 32:
        raise ContractBoundary('REFRESH_MMLU_DEVELOPMENT_ONLY')
    endpoint = copy.deepcopy(endpoint_state)
    endpoint_hash = digest(endpoint)
    end = batch * 100
    current = records[end-100:end]
    historical = [records[i] for i in historical_ordinals]
    statuses = history_status(records, list(range(end)), entry_n=end)
    annotations = {row['case_id']: row for row in statuses}
    out = dict(batch=batch, endpoint_state_sha256=endpoint_hash, cost_seconds={}, reuse=[],
               optimizer_access=0, audit_evaluations=0, imputation=0,
               general_schedule='TERMINAL_ONLY_NOT_SAME_EVALUATOR_COST_AS_REFERENCE',
               state_guard='CALLER_W_M_P_CONTEXT_RNG_PARAMETER_VERSIONS')

    def call(name, selected):
        begin = time.perf_counter()
        doc = evaluation.counterfact(model, tokenizer, selected, panel=name, annotations=annotations)
        validate_document(doc, selected)
        doc['endpoint_state_sha256'] = endpoint_hash
        out['cost_seconds'][name] = time.perf_counter() - begin
        return doc

    def subset(name, doc, source, selected, label):
        out[name] = exact_subset(doc, source, selected, panel=label,
                                  endpoint_state=endpoint, annotations=annotations)
        out['reuse'].append(dict(panel=label, source_panel=doc['panel'],
                                 requests=len(selected), added_forwards=0,
                                 source_endpoint_state_sha256=endpoint_hash))

    if batch == 60:
        seen = records[:6000]
        out['fullseen'] = call('FullSeen6000', seen)
        for name, selected, label in [
                ('current', current, 'Current'),
                ('suffix', records[5000:6000], 'SuffixSeen1000'),
                ('first_suffix500', records[5000:5500], 'FirstSuffix500'),
                ('entry_old', records[:5000], 'EntryOld5000'),
                ('historical', historical, 'Historical')]:
            subset(name, out['fullseen'], seen, selected, label)
    else:
        if batch == 55:
            suffix = records[5000:5500]
            out['suffix'] = call('SuffixSeen500', suffix)
            subset('first_suffix500', out['suffix'], suffix, suffix, 'FirstSuffix500')
            subset('current', out['suffix'], suffix, current, 'Current')
        else:
            out['current'] = call('Current', current)
        out['historical'] = call('Historical', historical)
    out['strict'] = {name: strict_summary(doc) for name, doc in out.items()
                     if isinstance(doc, dict) and 'metrics' in doc}
    if batch == 60:
        for name, function, args in [
                ('wiki', evaluation.wiki, (model, wiki_panel)),
                ('mmlu', evaluation.mmlu_alternative, (model, tokenizer, dev_mmlu))]:
            begin = time.perf_counter()
            out[name] = function(*args)
            out['cost_seconds'][name] = time.perf_counter() - begin
    else:
        out['general_status'] = 'NOT_SCHEDULED_NONTERMINAL'
    if digest(endpoint_state) != endpoint_hash:
        raise ContractBoundary('REFRESH_ENDPOINT_ARGUMENT_MUTATION')
    return out
