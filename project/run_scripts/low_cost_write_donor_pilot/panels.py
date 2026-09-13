"""Outcome-independent pilot panels. No runtime optimizer dependency."""
from __future__ import annotations

from project.run_scripts.baseline_mechanism_first.contracts import ContractBoundary, digest

SEED = 20260913


def _unique(records):
    ids = [r['case_id'] for r in records]
    if len(set(ids)) != len(ids):
        raise ContractBoundary('PANEL_DUPLICATE_CASE')


def bind_counterfact_panel(records, ordinals, *, expected_count, expected_identity=None):
    """Require explicit sealed ordinal order; never draw a replacement panel."""
    _unique(records)
    if len(ordinals) != expected_count or len(set(ordinals)) != expected_count:
        raise ContractBoundary('PANEL_CARDINALITY')
    if any(type(i) is not int or not 0 <= i < len(records) for i in ordinals):
        raise ContractBoundary('PANEL_ORDINAL')
    rows = []
    for i in ordinals:
        r = records[i]
        if len(r['paraphrase_prompts']) != 2 or len(r['neighborhood_prompts']) != 10:
            raise ContractBoundary('PANEL_PROMPT_CARDINALITY')
        rows.append(dict(ordinal=i, case_id=r['case_id'], record_sha256=digest(r)))
    identity = digest(rows)
    if expected_identity is not None and identity != expected_identity:
        raise ContractBoundary('PANEL_SEAL_MISMATCH')
    return dict(rows=rows, identity=identity, requests=expected_count,
                R=expected_count, P=2*expected_count, N=10*expected_count,
                outcome_selection=0, optimizer_access=0)


def history_status(records, historical_ordinals, *, entry_n):
    """Exact known subject/relation overwrite only; unresolved keys stay explicit.

    Last event at the entry is active; repeated same target is not called a
    conflicting overwrite. Main evaluator retains every selected historical row.
    """
    last = {}
    for i, r in enumerate(records[:entry_n]):
        w = r['requested_rewrite']
        if w.get('relation_id') is not None:
            last[(w['subject'], w['relation_id'])] = i
    out = []
    for i in historical_ordinals:
        if not 0 <= i < entry_n:
            raise ContractBoundary('HISTORY_OUTSIDE_ENTRY')
        w = records[i]['requested_rewrite']
        key = (w['subject'], w.get('relation_id'))
        newer = last.get(key)
        status = ('UNKNOWN_RELATION' if newer is None else
                  'SUPERSEDED' if records[newer]['requested_rewrite']['target_new']['str'] != w['target_new']['str']
                  else 'ACTIVE_TARGET')
        out.append(dict(ordinal=i, case_id=records[i]['case_id'], status=status,
                        latest_known_event_ordinal=newer, subject_relation_sha256=digest(key),
                        known_exact_key_only=True))
    return out


def seal_audit(records, development_ordinals, *, count=128, seed=SEED):
    """Exclude cases and exact known subject/relation keys before hash selection."""
    _unique(records)
    if len(set(development_ordinals)) != len(development_ordinals) or any(
            type(i) is not int or not 0 <= i < len(records) for i in development_ordinals):
        raise ContractBoundary('AUDIT_DEVELOPMENT_ORDINALS')
    cases = {records[i]['case_id'] for i in development_ordinals}
    keys = {(records[i]['requested_rewrite']['subject'], records[i]['requested_rewrite']['relation_id'])
            for i in development_ordinals if records[i]['requested_rewrite'].get('relation_id') is not None}
    eligible, case_excluded, key_excluded, unknown = [], [], [], []
    for i, r in enumerate(records):
        w = r['requested_rewrite']
        if r['case_id'] in cases:
            case_excluded.append(i)
        elif (w['subject'], w.get('relation_id')) in keys:
            key_excluded.append(i)
        else:
            eligible.append(i)
            if w.get('relation_id') is None:
                unknown.append(i)
    if len(eligible) < count:
        raise ContractBoundary('AUDIT_ELIGIBLE_TOO_SMALL')
    ordered = sorted(eligible, key=lambda i: (digest([seed, 'pilot-audit', records[i]['case_id']]), i))[:count]
    return dict(bind_counterfact_panel(records, ordered, expected_count=count),
                seed=seed, eligible=len(eligible), case_excluded=len(case_excluded),
                subject_relation_excluded=len(key_excluded), missing_relation_eligible=len(unknown),
                known_overlap_policy='EXACT_SUBJECT_RELATION_PAIR',
                unknown_semantic_overlap='NOT_EXCLUDED_NOT_GLOBALLY_BLIND',
                access='ONLY_AFTER_GH_CANDIDATE_POLICY_LOCK')


def split_mmlu(rows, *, seed=SEED):
    """The caller supplies the already sealed original rows[10:110], not corpus."""
    if len(rows) != 100 or len({digest(r) for r in rows}) != 100:
        raise ContractBoundary('MMLU_FIXED100_IDENTITY')
    for r in rows:
        if type(r.get('answer')) is not int or r['answer'] not in range(4) or len(r['choices']) != 4:
            raise ContractBoundary('MMLU_SCHEMA')
    order = sorted(range(100), key=lambda i: (digest([seed, 'pilot-mmlu', digest(rows[i])]), i))
    groups = {}
    for name, indices in [('development', order[:32]), ('audit', order[32:])]:
        groups[name] = dict(indices=indices, row_hashes=[digest(rows[i]) for i in indices],
                            count=len(indices), identity=digest([rows[i] for i in indices]))
    return dict(groups=groups, input100_identity=digest(rows), seed=seed,
                outcome_selection=0, prediction='SOURCE_ALTERNATIVE_UNIQUE_MAX_PROBABILITY',
                metric='INTEGER_CORRECT_COUNT', combined100='DESCRIPTIVE_NOT_INDEPENDENT_AUDIT')


def validate_wiki(manifest, *, count=128):
    """Bind existing tokens and full next-token mask without retokenization."""
    rows = manifest['rows']
    if len(rows) != count or len({r['ordinal'] for r in rows}) != count:
        raise ContractBoundary('WIKI_PANEL_CARDINALITY')
    for row in rows:
        ids = row['input_ids']
        if len(ids) < 2 or any(type(v) is not int or v < 0 for v in ids):
            raise ContractBoundary('WIKI_TOKEN_SCHEMA')
        if row.get('predicted_tokens', len(ids)-1) != len(ids)-1:
            raise ContractBoundary('WIKI_MASK_NOT_FULL_NEXT_TOKEN')
        if row.get('target_mask', [1]*(len(ids)-1)) != [1]*(len(ids)-1):
            raise ContractBoundary('WIKI_MASK_NOT_FULL_NEXT_TOKEN')
    if 'row_identity' in manifest and digest(rows) != manifest['row_identity']:
        raise ContractBoundary('WIKI_TOKEN_IDENTITY')
    return dict(count=count, token_rows_sha256=digest(rows),
                mask='FULL_NEXT_TOKEN_SEQUENCE_NO_RETOKENIZATION',
                predicted_tokens=sum(len(r['input_ids'])-1 for r in rows))
