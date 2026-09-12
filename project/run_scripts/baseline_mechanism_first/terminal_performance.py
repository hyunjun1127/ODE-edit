"""Canonical original/replay performance at the live native endpoint.

Observer only: no native writer, target optimisation or history append. The
whole seen prefix and its last Current100 are distinct evaluation layouts.
"""
import json
import time
from pathlib import Path
import torch
from .contracts import ContractBoundary, digest, member, save
from .fixtures import tensor_sha, capture_rng, restore_rng
from .case_population import source_digest
from .performance_compare import compare_rows
from .cold_analysis import table


def observe_guarded(model, weight, history, expected_weight_sha, expected_history_sha, callback):
    before_w=tensor_sha(weight);before_m=tensor_sha(history)
    if before_w!=expected_weight_sha or before_m!=expected_history_sha:
        raise ContractBoundary('ENDPOINT_PERFORMANCE_WRONG_STATE')
    signatures=[(p,p.data_ptr(),p._version) for p in model.parameters()]
    rng=capture_rng();start=time.monotonic()
    try:
        value=callback()
    finally:
        w_exact=tensor_sha(weight)==before_w;m_exact=tensor_sha(history)==before_m
        pointers_versions=all(p.data_ptr()==ptr and p._version==version for p,ptr,version in signatures)
        rng_exact=digest(capture_rng())==digest(rng)
        restore_rng(rng)
        if not (w_exact and m_exact and pointers_versions and rng_exact):
            raise ContractBoundary('ENDPOINT_PERFORMANCE_OBSERVER_MUTATION',weight=w_exact,history=m_exact,
                                   pointers_versions=pointers_versions,RNG=rng_exact)
    return value,dict(weight_sha256=before_w,history_sha256=before_m,weight_history_bytes_exact=True,
        all_parameter_pointer_versions_exact=True,RNG_exact=True,seconds=time.monotonic()-start,
        writer_calls=0,z_calls=0,history_append=0)


def paired(original, replay, label):
    if original['requests']!=replay['requests'] or original['request_order']!=replay['request_order']:
        raise ContractBoundary('PERFORMANCE_REQUEST_ORDER')
    rows=[]
    for category,multiplicity in [('RS',1),('PS',2),('NS',10)]:
        row=compare_rows(original['metrics'][category]['rows'],replay['metrics'][category]['rows'],category)
        if row['denominator']!=replay['requests']*multiplicity:
            raise ContractBoundary('PERFORMANCE_DENOMINATOR')
        for obj,key in [(original,'original_numerator'),(replay,'replay_numerator')]:
            if obj['metrics'][category]['numerator']!=row[key] or obj['metrics'][category]['denominator']!=row['denominator']:
                raise ContractBoundary('PERFORMANCE_REDUCER_MISMATCH')
        rows.append(dict(panel=label,**row))
    return rows


def initial(model,tok,weight,history,record,original_current,expected_w,expected_m,evaluate):
    """One actual canonical record validates new guarded comparison path only."""
    ref=member(original_current['path'],expected=original_current['sha256'])
    baseline=json.loads(Path(ref['path']).read_text());case=record['case_id'];sub={}
    for category in ('RS','PS','NS'):
        rows=[r for r in baseline['metrics'][category]['rows'] if r['case_id']==case]
        sub[category]=dict(rows=rows,denominator=len(rows),numerator=sum(r['success'] for r in rows))
    baseline=dict(requests=1,request_order=source_digest([case]),metrics=sub)
    replay,guard=observe_guarded(model,weight,history,expected_w,expected_m,lambda:evaluate(model,tok,[record]))
    differences=paired(baseline,replay,'FIRST_NATIVE_CURRENT_ONE_RECORD')
    return dict(status='ACTUAL_PERFORMANCE_REPAIR_INITIAL_VALID',guard=guard,original=ref,
        paired=differences,requests=1,prompt_pairs=13,true_new_sequences=26,
        terminal_fullseen_complete=False,performance_not_correctness_gate=True,trajectory_equivalence=False)


def run(model,tok,weight,history,records,terminal_batch,bindings,expected_w,expected_m,output,evaluate):
    dest=Path(output);n=terminal_batch*100
    if n not in (6000,10000):raise ContractBoundary('TERMINAL_PERFORMANCE_SCOPE')
    sources={key:member(value['path'],expected=value['sha256']) for key,value in bindings.items()}
    metrics=[];receipts=[];start=time.monotonic()
    for label,panel in [('current',records[n-100:n]),('seen-full',records[:n])]:
        baseline=json.loads(Path(sources[label]['path']).read_text())
        if baseline['request_order']!=source_digest([r['case_id'] for r in panel]):
            raise ContractBoundary('ORIGINAL_PERFORMANCE_PANEL_IDENTITY')
        result,guard=observe_guarded(model,weight,history,expected_w,expected_m,lambda:evaluate(model,tok,panel))
        forward_seconds=guard['seconds'];io_start=time.monotonic()
        raw=save(dest/f'{label}.json',result)
        differences=paired(baseline,result,label);metrics.extend(differences)
        # Preserve full original/replay per-prompt distributions through two
        # bound raw files; summary CSV is not a reconstructed distribution.
        receipts.append(dict(panel=label,requests=len(panel),prompt_pairs=13*len(panel),
            raw=raw,original=sources[label],guard=guard,evaluation_seconds=forward_seconds,
            publication_comparison_seconds=time.monotonic()-io_start))
        print('E01_TERMINAL_PERFORMANCE',terminal_batch,label,len(panel),flush=True)
    summary=table(dest/'paired-performance.csv',metrics)
    return save(dest/'receipt.json',dict(status='OBSERVED_CURRENT_AND_FULL_SEEN',batch=terminal_batch,
        actual_materialized_weight_sha256=expected_w,actual_native_history_sha256=expected_m,
        current_requests=100,seen_requests=n,overlapping_current_not_unique_addition=True,
        panels=receipts,summary=summary,seconds=time.monotonic()-start,
        physical_peak_memory_report='PARENT_TERMINAL_MAX_CUDA_ALLOCATED_RESERVED',
        evaluation_layout='CANONICAL_MICROBATCH16_SEPARATE_CURRENT_AND_FULLSEEN',
        native_equivalence_claim=False,scientific_promotion=False))
