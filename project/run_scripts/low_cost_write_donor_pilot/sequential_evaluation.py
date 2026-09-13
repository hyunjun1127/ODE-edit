"""Read-only sequential endpoint evaluation with exact same-state row reuse.

The caller owns actual W/M/RNG and parameter nonmutation guards. Subsets never
claim a separately rebatched forward: their layout and values inherit the one
larger evaluation at precisely the same endpoint.
"""
from __future__ import annotations

import copy
import time

from project.run_scripts.baseline_mechanism_first.contracts import ContractBoundary, digest
from project.run_scripts.baseline_mechanism_first.case_population import source_digest
from project.run_scripts.baseline_mechanism_first.performance_schema import normalize, from_rows, MULTIPLICITY
from . import evaluation
from .panels import history_status


def validate_document(doc, records):
    normalize(doc, records)
    for tag, metric in doc['metrics'].items():
        rows = metric['rows']
        if (metric['rate'] != metric['numerator']/metric['denominator'] or
                metric['bit_order_sha256'] != source_digest([(r['identity'], r['success']) for r in rows])):
            raise ContractBoundary('SEQUENTIAL_AGGREGATE_OR_BIT_ORDER')
        for row in rows:
            margin = row['true_nll']-row['new_nll']
            if row.get('margin', margin) != margin or row.get('desired_margin',
                    -margin if tag == 'NS' else margin) != (-margin if tag == 'NS' else margin):
                raise ContractBoundary('SEQUENTIAL_MARGIN_DIRECTION')
    return doc


def exact_subset(doc, source_records, selected_records, *, panel, endpoint_state, annotations=None):
    """Select ordered whole requests only after full source identity validation."""
    validate_document(doc, source_records)
    if doc.get('endpoint_state_sha256') != digest(endpoint_state):
        raise ContractBoundary('SUBSET_ENDPOINT_IDENTITY')
    source = {r['case_id']: r for r in source_records}
    ids = [r['case_id'] for r in selected_records]
    if len(source) != len(source_records) or len(set(ids)) != len(ids) or not ids:
        raise ContractBoundary('SUBSET_DUPLICATE_OR_EMPTY')
    for record in selected_records:
        if record['case_id'] not in source or digest(record) != digest(source[record['case_id']]):
            raise ContractBoundary('SUBSET_RECORD_IDENTITY')
    groups = {}
    for tag, multiplicity in MULTIPLICITY.items():
        index = {(r['case_id'], r['prompt_index']): r for r in doc['metrics'][tag]['rows']}
        rows = []
        for case_id in ids:
            for i in range(multiplicity):
                row = dict(index[(case_id, i)], panel=panel)
                row.pop('historical_status', None)
                if annotations is not None:
                    row['historical_status'] = annotations[case_id]
                rows.append(row)
        groups[tag] = {'rows': rows}
    result = from_rows([{'metrics': groups}], selected_records)
    result.update(panel=panel, endpoint_state_sha256=doc['endpoint_state_sha256'],
                  reused_from_panel=doc['panel'], reused_from_request_order=doc['request_order'],
                  evaluator_layout=doc.get('evaluator_layout', 'SOURCE_MB16'),
                  forward_calls_added=0, optimizer_access=0,
                  aggregation_unit='PROMPT_WITH_REQUEST_IDENTITY',
                  reuse_semantics='EXACT_SOURCE_ROWS_NOT_SEPARATE_REBATCHED_FORWARD')
    return result


def evaluate_batch(model, tokenizer, records, batch, historical_ordinals, wiki_panel,
                   dev_mmlu, endpoint_state):
    if type(batch) is not int or batch not in range(51, 61) or len(records) != 10000:
        raise ContractBoundary('SEQUENTIAL_EVALUATION_BATCH_OR_DATASET')
    if len({r['case_id'] for r in records}) != 10000:
        raise ContractBoundary('SEQUENTIAL_EVALUATION_DUPLICATE_CASE')
    if (len(historical_ordinals) != 128 or len(set(historical_ordinals)) != 128
            or any(type(i) is not int or not 0 <= i < 5000 for i in historical_ordinals)):
        raise ContractBoundary('SEQUENTIAL_HISTORICAL_PANEL')
    if len(dev_mmlu) != 32:
        raise ContractBoundary('SEQUENTIAL_MMLU_DEVELOPMENT_ONLY')
    endpoint = copy.deepcopy(endpoint_state)
    endpoint_hash = digest(endpoint)
    end = batch * 100
    current = records[end-100:end]
    historical = [records[i] for i in historical_ordinals]
    annotations = {r['case_id']: r for r in history_status(records, historical_ordinals, entry_n=end)}
    out = dict(batch=batch, endpoint_state_sha256=endpoint_hash, cost_seconds={}, reuse=[],
               optimizer_access=0, audit_evaluations=0, imputation=0,
               state_guard='CALLER_MUST_CHECK_W_M_P_CONTEXT_RNG_AND_PARAMETER_VERSIONS')

    def call(name, selected):
        start = time.perf_counter()
        result = evaluation.counterfact(model, tokenizer, selected, panel=name,
                                        annotations=annotations if name == 'Historical' else None)
        validate_document(result, selected)
        result['endpoint_state_sha256'] = endpoint_hash
        out['cost_seconds'][name] = time.perf_counter()-start
        return result

    def reuse(name, doc, source_records, selected, label):
        result = exact_subset(doc, source_records, selected, panel=label, endpoint_state=endpoint,
                              annotations=annotations if label == 'Historical' else None)
        out[name] = result
        out['reuse'].append(dict(panel=label, source_panel=doc['panel'], requests=len(selected),
                                 source_endpoint_state_sha256=endpoint_hash,
                                 request_order=result['request_order'], added_forwards=0))

    if batch == 60:
        seen = records[:end]
        out['fullseen'] = call('FullSeen6000', seen)
        reuse('current', out['fullseen'], seen, current, 'Current')
        reuse('suffix', out['fullseen'], seen, records[5000:end], 'SuffixSeen1000')
        reuse('entry_old', out['fullseen'], seen, records[:5000], 'EntryOld5000')
        reuse('historical', out['fullseen'], seen, historical, 'Historical')
    else:
        if batch == 55:
            suffix = records[5000:end]
            out['suffix'] = call('SuffixSeen500', suffix)
            reuse('current', out['suffix'], suffix, current, 'Current')
        else:
            out['current'] = call('Current', current)
        out['historical'] = call('Historical', historical)
    for name, function, args in [('wiki', evaluation.wiki, (model, wiki_panel)),
                                  ('mmlu', evaluation.mmlu_alternative, (model, tokenizer, dev_mmlu))]:
        start = time.perf_counter()
        out[name] = function(*args)
        out['cost_seconds'][name] = time.perf_counter()-start
    if digest(endpoint_state) != endpoint_hash:
        raise ContractBoundary('EVALUATION_ENDPOINT_ARGUMENT_MUTATED')
    return out
