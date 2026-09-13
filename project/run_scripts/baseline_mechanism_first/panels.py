"""Outcome-independent E1 panel identities. No write-time use of these panels."""
from __future__ import annotations
from .contracts import Cell, ContractBoundary, INSTRUCTION, digest


def hash_order(indices, namespace):
    return sorted(indices, key=lambda i: (digest([INSTRUCTION, namespace, int(i)]), i))


def historical_ordinals(entry_n):
    if entry_n == 0:
        return []
    if entry_n not in (1000, 5000, 9000):
        raise ContractBoundary('HISTORICAL_ENTRY_OUTSIDE_SCOPE')
    edges = (0, entry_n // 3, 2 * entry_n // 3, entry_n)
    result = []
    for group, count in enumerate((43, 43, 42)):
        pool = range(edges[group], edges[group+1])
        result.extend(hash_order(pool, f'historical:{entry_n}:{group}')[:count])
    if len(result) != 128 or len(set(result)) != 128:
        raise ContractBoundary('HISTORICAL_PANEL_CARDINALITY')
    return result


def panel_manifest(records, entry_n):
    Cell(4, entry_n)  # Domain check; selection is explicitly layer-independent.
    current = list(range(entry_n, entry_n + 100))
    historical = historical_ordinals(entry_n)
    groups = {}
    for name, indices in [('Current', current), ('Historical', historical)]:
        items = []
        for i in indices:
            r = records[i]
            if len(r['paraphrase_prompts']) != 2 or len(r['neighborhood_prompts']) != 10:
                raise ContractBoundary('CANONICAL_PROMPT_CARDINALITY', ordinal=i)
            items.append(dict(ordinal=i, case_id=int(r['case_id']), raw_record_sha256=digest(r)))
        groups[name] = dict(records=items, requests=len(items),
                            prompt_pairs=len(items)*13, candidate_sequences=len(items)*26)
    return dict(entry_n=entry_n, groups=groups, layer_selection_influence=0,
                outcome_selection_influence=0, historical_rule='ordinal thirds43/43/42; canonical JSON SHA order',
                prompt_pairs=sum(x['prompt_pairs'] for x in groups.values()),
                candidate_sequences=sum(x['candidate_sequences'] for x in groups.values()),
                general='SEPARATE_EXISTING_CORPUS_MANIFEST_REQUIRED')
