"""Controller consumes E/H/D and rewrite strict IDs only; never P/N/Dev."""
import math
from .common import digest

def materialized(entry, native, gate):
    import torch
    if gate not in (0., .5, .75, 1.): raise ValueError('UNREGISTERED_GATE')
    assert entry.device.type == native.device.type == 'cpu'
    assert entry.dtype == native.dtype == torch.float32 and entry.shape == native.shape
    result = entry.clone() if gate == 0 else native.clone() if gate == 1 else entry + gate*(native-entry)
    if not torch.isfinite(result).all(): raise ValueError('NONFINITE_MATERIALIZATION')
    return result

def fact(record):
    rw = record['requested_rewrite']
    return rw['subject'], rw['relation_id']

def event_id(ordinal, record):
    # Fixed ASCII JSON with explicit names; no locale, repr, or future data.
    return digest(dict(ordinal=ordinal, case_id=record['case_id'], subject=fact(record)[0],
                       relation=fact(record)[1], target=record['requested_rewrite']['target_new']['str']))

def past_indices(received, current, records):
    import hashlib
    excluded = {fact(r) for r in current}
    latest = {}
    for ordinal in received: latest[fact(records[ordinal])] = ordinal
    eligible = [o for key, o in latest.items() if key not in excluded]
    def priority(o):
        stable = event_id(o, records[o])
        return hashlib.sha256(('LZ-ALLOC-v1|20260916|past|'+stable).encode('utf-8')).hexdigest(), stable
    return sorted(eligible, key=priority)[:64]

def choose(rows, reference='N4', *, plateau=True, past=True):
    ref = next(r for r in rows if r['candidate_id'] == reference)
    for r in rows:
        if not all(math.isfinite(r[k]) for k in ('E', 'D', 'action_norm')):
            raise ValueError('TECHNICAL_NONFINITE_CANDIDATE')
        if r['H'] is not None and not math.isfinite(r['H']):
            raise ValueError('TECHNICAL_NONFINITE_PAST')
    feasible, observations = [], []
    for r in rows:
        e_limit = max(ref['E'], .05)+1e-4 if plateau else ref['E']+1e-4
        reasons = []
        if r['E'] > e_limit: reasons.append('CURRENT_MEAN')
        if not set(ref['S_cur']).issubset(r['S_cur']): reasons.append('CURRENT_STRICT_IDS')
        if past and ref['H'] is not None:
            if r['H'] is None or r['H'] > ref['H']+1e-4: reasons.append('PAST_MEAN')
            if not set(ref['S_past']).issubset(r['S_past']): reasons.append('PAST_STRICT_IDS')
        observations.append(dict(candidate_id=r['candidate_id'], feasible=not reasons, reasons=reasons,
                                 current_ceiling=e_limit, past_ceiling=None if ref['H'] is None else ref['H']+1e-4))
        if not reasons: feasible.append(r)
    assert feasible, 'OWN_N4_MUST_BE_FEASIBLE'
    minimum = min(r['D'] for r in feasible)
    tied = [r for r in feasible if r['D'] <= minimum+1e-6]
    winner = min(tied, key=lambda r: (r['candidate_id'] != reference, not r['L8_zero'], r['action_norm'], r['candidate_id']))
    return dict(selected=winner['candidate_id'], D_min=minimum, ties=[r['candidate_id'] for r in tied], observations=observations)

def strict_shadow(rows):
    ref = next(r for r in rows if r['candidate_id'] == 'N4')
    valid = [r for r in rows if set(ref['S_cur']).issubset(r['S_cur'])
             and (ref['H'] is None or set(ref['S_past']).issubset(r['S_past']))]
    minimum = min(r['D'] for r in valid)
    return min((r for r in valid if r['D'] <= minimum+1e-6),
               key=lambda r: (r['candidate_id'] != 'N4', not r['L8_zero'], r['action_norm'], r['candidate_id']))['candidate_id']
