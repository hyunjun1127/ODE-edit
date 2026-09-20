"""Complete current union with equal request / unique input / token weights.

No official paraphrase or neighborhood inputs enter this module. Identical
byte keys may merge only inside an identical consumed-prefix policy group;
all occurrences and all distinct captured byte variants remain represented.
"""
from collections import defaultdict
import hashlib
import json
import torch
from project.run_scripts.single_layer_edit_preserving_correction.binding import protected_sequences
from project.run_scripts.single_layer_edit_preserving_correction.alltoken import FullWeightLlamaOracle


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False).encode()).hexdigest()


def tokenizer_policy(tok):
    return dict(name_or_path=str(getattr(tok, 'name_or_path', '')),
        padding_side=tok.padding_side, pad_token_id=tok.pad_token_id,
        bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
        add_bos_token=getattr(tok, 'add_bos_token', None),
        add_eos_token=getattr(tok, 'add_eos_token', None),
        physical_batch=1, actual_padding='none', position_ids='explicit_arange')


def complete_packs(base_packs, rows, policies):
    """Expand the legacy ID-only cache dedup into full-policy identity dedup."""
    packs, identities, remapped, lookup = [], [], [], {}
    for row in rows:
        pack = base_packs[row['cache']]
        identity = dict(**{name: pack[name][0].tolist()
                           for name in ('input_ids', 'attention_mask', 'position_ids')},
                        tokenizer=policies[row['kind']])
        sha = digest(identity)
        if sha not in lookup:
            lookup[sha] = len(packs)
            packs.append({k: v.clone() for k, v in pack.items()})
            identities.append(dict(sha256=sha, **identity))
        remapped.append(dict(row, cache=lookup[sha], full_input_sha256=sha))
    return packs, remapped, identities


def weighted_columns(caches, rows, identities, case_ids):
    """Pure CPU mapping; fake caches can exercise weighting independently.

    Duplicate full-input rows in a request receive shares of ONE sequence
    weight. Cross-request duplication receives each request's proper share.
    Native/canonical inputs with differing tokenizer policy stay distinct.
    """
    if not case_ids or len(set(case_ids)) != len(case_ids):
        raise ValueError('NONEMPTY_UNIQUE_CASE_ORDER_REQUIRED')
    if len(caches) != len(identities):
        raise ValueError('CACHE_IDENTITY_CARDINALITY')
    bycase = {case: defaultdict(list) for case in case_ids}
    for row in rows:
        if row['case_id'] not in bycase:
            raise ValueError('UNEXPECTED_CURRENT_CASE')
        bycase[row['case_id']][row['cache']].append(row)
    if any(not groups for groups in bycase.values()):
        raise ValueError('MISSING_CURRENT_REQUEST')
    values, representatives, prefix_groups, location = [], [], {}, {}
    for ci, cache in enumerate(caches):
        identity = identities[ci]
        ids, mask, positions = (identity[k] for k in
                               ('input_ids', 'attention_mask', 'position_ids'))
        if len(ids) != cache.keys.shape[1] or len(mask) != len(ids) or len(positions) != len(ids):
            raise ValueError('KEY_INPUT_POSITION_CARDINALITY')
        for pos in range(len(ids)):
            if not mask[pos]:
                continue
            prefix_sha = digest(dict(ids=ids[:pos+1], mask=mask[:pos+1],
                positions=positions[:pos+1], physical_position=pos,
                tokenizer=identity['tokenizer']))
            group = prefix_groups.setdefault(prefix_sha, [])
            value = cache.keys[0, pos].detach().cpu()
            if value.dtype != torch.float32 or not torch.isfinite(value).all():
                raise ValueError('FINITE_FP32_CURRENT_KEYS_REQUIRED')
            # torch.equal is numeric equality and equates +0/-0. Use exact
            # byte identity for the specified captured-byte dedup contract.
            bits = value.contiguous().view(torch.int32)
            column = next((i for i in group if torch.equal(values[i].view(torch.int32), bits)), None)
            if column is None:
                column = len(values)
                values.append(value.contiguous().clone())
                representatives.append(group[0] if group else column)
                group.append(column)
            location[ci, pos] = (column, prefix_sha)
    weights = torch.zeros(len(values), dtype=torch.float64)
    aliases, request_receipts = [], []
    B = len(case_ids)
    for case in case_ids:
        groups = bycase[case]
        total = 0.
        for ci, occurrences in groups.items():
            mask = identities[ci]['attention_mask']
            valid = [pos for pos, value in enumerate(mask) if value]
            if not valid:
                raise ValueError('EMPTY_CURRENT_SEQUENCE')
            token_weight = 1. / (B * len(groups) * len(valid))
            for row in occurrences:
                # A duplicate row describes the same input; its aliases split
                # that input's weight instead of duplicating it.
                occurrence_weight = token_weight / len(occurrences)
                for pos in valid:
                    column, prefix_sha = location[ci, pos]
                    weights[column] += occurrence_weight
                    total += occurrence_weight
                    aliases.append(dict(case_id=case, sequence_id=row['sequence_id'],
                        kind=row['kind'], branch=row['branch'], context=row.get('context'),
                        cache=ci, position=pos, actual_key_column=column,
                        representative_key_column=representatives[column],
                        full_input_sha256=identities[ci]['sha256'], prefix_sha256=prefix_sha,
                        weight=occurrence_weight, duplicate_input_occurrences=len(occurrences)))
        request_receipts.append(dict(case_id=case, unique_sequences=len(groups),
                                     total_weight=total, expected_weight=1./B))
        if abs(total-1./B) > 1e-12:
            raise ValueError('REQUEST_NESTED_WEIGHT_SUM')
    if not values or not torch.all(weights > 0) or abs(float(weights.sum())-1.) > 1e-12:
        raise ValueError('CURRENT_WEIGHT_MASS')
    K = torch.stack(values, dim=1).contiguous()
    reps = torch.tensor(representatives, dtype=torch.long)
    manifest = dict(schema='EN_ADAPTIVE_CURRENT_PROTECTION_V1',
        case_ids=list(case_ids), request_weights=request_receipts,
        sequence_rows=rows, input_identities=identities, aliases=aliases,
        weighting='equal_request/unique_full_input/valid_input_token',
        key_dedup='identical_consumed_prefix_policy_AND_exact_FP32_bytes',
        numerical_representative='first_captured_byte_column_per_logical_prefix',
        K_shape=list(K.shape), K_dtype=str(K.dtype), weight_sum=float(weights.sum()),
        actual_byte_columns=K.shape[1], logical_prefix_groups=len(prefix_groups),
        original_cache_positions=len(location), original_occurrence_positions=len(aliases),
        byte_variant_columns=K.shape[1]-len(prefix_groups),
        official_P_N=False, current_downstream_candidate_forwards=0)
    manifest['identity_sha256'] = digest(manifest)
    return dict(K=K, weights=weights, representative_indices=reps, manifest=manifest)


def capture_current(model, tok, eval_tok, requests, contexts):
    base_packs, rows, _, legacy = protected_sequences(tok, eval_tok, requests, contexts)
    if legacy['missing_old']:
        raise ValueError('MISSING_SUPPLIED_TRUE_TARGET')
    packs, rows, identities = complete_packs(base_packs, rows,
        dict(native=tokenizer_policy(tok), canonical=tokenizer_policy(eval_tok)))
    oracle = FullWeightLlamaOracle(model, packs)
    result = weighted_columns(oracle.caches, rows, identities, [r['case_id'] for r in requests])
    result.update(oracle=oracle, rows=rows)
    return result
