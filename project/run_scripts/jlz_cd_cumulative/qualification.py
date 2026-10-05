"""Sealed 2--4-request actual-model, same-candidate technical qualification.

No fit, optimizer update, physical write, observer panels or scientific tuning.
These checks run within the task job; a CPU receipt is never a GPU receipt.
"""
import time

import torch

from project.run_scripts.jlz_realization.common import require, tensor_sha, digest, state, write
from project.run_scripts.jlz_realization.subject import row_logprobs
from project.run_scripts.jlz_realization.writer import rng_snapshot, rng_equal
from project.run_scripts.jlz_realized_writer.capture import cache_identity
from project.run_scripts.jlz_realized_writer.geometry import prepare_metric, solve_realized, action_parity
from project.run_scripts.jlz_writer_coupled.geometry import build_prior_from_npz
from .geometry import projected_action, cost_gradient_u
from .optimize import validate_entry


def _parity(candidate, reference, *, atol=1e-6, rtol=2e-4):
    require(candidate.shape == reference.shape, 'QUALIFICATION_PARITY_SHAPE')
    c, r = candidate.detach().double(), reference.detach().double()
    error = (c - r).abs()
    limit = atol + rtol * r.abs()
    return dict(pass_=bool(torch.isfinite(c).all() and torch.isfinite(r).all()
                          and (error <= limit).all()),
                max_absolute=float(error.max()), max_excess=float((error - limit).max()),
                failed_elements=int((error > limit).sum()), elements=error.numel(),
                atol=atol, rtol=rtol, reduction='elementwise')


def _snapshot_u(a, B):
    # Fixed analytic fixture, not a seed search or an optimizer pilot. Every
    # owner/site including zero-origin sites remains represented.
    result = {}
    for i, l in enumerate(a.sites):
        d = a.dims[l][0]
        coordinates = torch.arange(d * B, dtype=torch.float32, device=a.device).reshape(d, B)
        result[l] = (.02 * torch.sin(coordinates + i + 1) / (d ** .5 * len(a.sites))).requires_grad_()
    return result


def _native_pass(a, entry, groups, geometries, fixture, *, direct=False, backward=True,
                 backward_schedule='groupwise'):
    require(backward_schedule in ('groupwise', 'logical-sum'), 'QUALIFICATION_BACKWARD_SCHEDULE')
    u = {l: value.detach().clone().requires_grad_(backward) for l, value in fixture.items()}
    records, hidden, logprobs = {}, {}, {}
    logical_losses = []
    forward_calls = backward_calls = 0
    for group in groups:
        with torch.set_grad_enabled(backward):
            D = {l: entry['anchors'][l][None, :] * value for l, value in u.items()}
            if direct:
                nh, fh, found = a.native(group, D, True)
            else:
                indices = [row['global_row'] for row in group['rows']]
                Y = {l: projected_action(D[l], geometries[l], indices) for l in a.sites}
                nh, fh, found = a.native_projected(group, Y, True, local_rows=True)
            forward_calls += 1
            probs = row_logprobs(a, group['rows'], nh, fh)
            nll, kl = {}, {}
            for j, (row, lp) in enumerate(zip(group['rows'], probs)):
                r = row['request']
                logprobs[row['global_row']] = lp.detach().cpu().clone()
                hidden[row['global_row']] = {l: h[j].detach().cpu().clone() for l, h in found.items()}
                if row['kind'] == 'rewrite':
                    target = row['target'][row['target'] != -100].to(a.device)
                    nll.setdefault(r, []).append(-lp.gather(1, target[:, None]).mean())
                else:
                    teacher = entry['teachers'][r].to(a.device)
                    kl[r] = (lp.exp() * (lp - teacher)).sum()
            values = {r: (torch.stack(v).mean(), kl[r]) for r, v in nll.items()}
            for r, (nr, kr) in values.items():
                records[r] = torch.stack((nr.detach(), kr.detach())).cpu()
            if backward:
                losses = [nr + .0625 * kr for nr, kr in values.values()]
                if backward_schedule == 'logical-sum':
                    # Keep the original owner-shaped forward graphs, then
                    # differentiate their complete logical SUM in one call.
                    # No new tokens, cache, head batch shape or row order.
                    logical_losses.extend(losses)
                else:
                    torch.stack(losses).sum().backward()
                    backward_calls += 1
    if backward and backward_schedule == 'logical-sum':
        require(bool(logical_losses), 'QUALIFICATION_EMPTY_LOGICAL_SUM')
        torch.stack(logical_losses).sum().backward()
        backward_calls += 1
    require(not backward or all(value.grad is not None for value in u.values()),
            'QUALIFICATION_MISSING_NATIVE_GRADIENT')
    return dict(losses=torch.stack([records[r] for r in sorted(records)]), hidden=hidden, logprobs=logprobs,
                gradients={l: value.grad.detach().clone() if value.grad is not None
                           else torch.zeros_like(value) for l, value in u.items()},
                native_forward_calls=forward_calls, native_backward_calls=backward_calls)


def _validate_original_owner_groups(a, entry, geometries):
    validate_entry(a, entry, geometries)
    B = entry['pack']['n_requests']
    require(len(entry['groups']) == B and all(
        {row['request'] for row in group['rows']} == {owner}
        for owner, group in enumerate(entry['groups'])),
        'QUALIFICATION_FIXED_ORIGINAL_OWNER_GROUPS_ONLY')


def _compare_native(candidate, reference, layers):
    require(set(candidate['hidden']) == set(reference['hidden'])
            and set(candidate['logprobs']) == set(reference['logprobs']),
            'QUALIFICATION_NATIVE_ROW_COVERAGE')
    require(set(candidate['gradients']) == set(reference['gradients']) == set(layers),
            'QUALIFICATION_NATIVE_GRADIENT_COVERAGE')
    return dict(loss=_parity(candidate['losses'], reference['losses']),
                gradient={str(l): _parity(candidate['gradients'][l], reference['gradients'][l]) for l in layers},
                logprobs={str(row): _parity(candidate['logprobs'][row], reference['logprobs'][row])
                          for row in reference['logprobs']},
                subject={str(l): _parity(torch.stack([candidate['hidden'][row][l]
                                                    for row in sorted(reference['hidden'])]),
                                        torch.stack([reference['hidden'][row][l]
                                                    for row in sorted(reference['hidden'])])) for l in layers})


def _all_pass(value):
    if isinstance(value, dict):
        if 'pass_' in value:
            return value['pass_']
        return all(_all_pass(v) for v in value.values())
    return True


def qualify(a, entry, geometries, initial, history, W0, out=None, *, pad_id=0):
    """Qualify exactly one analytic candidate in the original cold model.

    The input must be a freshly prepared 2--4-request native subset, not the
    B100 fit entry. Its fixed entry capture is reused; no fresh capture pass
    is performed here. The original V13 equality solve is evaluated once
    per site to independently check compact response/cost/gradient/action.
    Qualification is restricted to the unchanged one-owner physical groups;
    a joint-backward SUM reference does not qualify the rejected all-row
    forward regrouping, nor reuse the predecessor's GPU evidence.
    """
    started = time.monotonic()
    B, layers = entry['pack']['n_requests'], a.sites
    require(2 <= B <= 4, 'QUALIFICATION_2_TO_4_REQUESTS_ONLY')
    require(a.profile['kl_factor'] == .0625, 'QUALIFICATION_NATIVE_KL')
    require(set(geometries) == set(layers) and set(W0) == set(layers), 'QUALIFICATION_LAYER_COVERAGE')
    _validate_original_owner_groups(a, entry, geometries)
    require(all(torch.equal(a.weights[l].detach().cpu(), W0[l].cpu()) for l in layers)
            and all(not bool(history[l].count_nonzero()) for l in layers), 'QUALIFICATION_OWN_COLD_ENTRY')
    require(all(g.receipt['delta_zero'] and not bool(g.J.count_nonzero()) for g in geometries.values()),
            'QUALIFICATION_COLD_ALIGNMENT_ZERO')
    require([r['global_row'] for r in initial['rows']] == list(range(len(entry['pack']['row_kind']))),
            'QUALIFICATION_NATIVE_ROWS')
    before, cache, rng = state(a, history), cache_identity(entry), rng_snapshot()
    guard, hooks = a.guard(), a.hook_signature()
    route, capture, prior_virtual = a.native_route, a.capture_virtual, dict(a.last_virtual)
    a.capture_virtual = False
    a.last_virtual.clear()
    fixture = _snapshot_u(a, B)
    G = len(entry['groups'])
    require(1 <= G <= B, 'QUALIFICATION_COMPLETE_OWNER_GROUPS')
    compatible = all(g.full_operator_compatible for g in geometries.values())
    # Original-shape cached/full/SUM/direct paths and one no-grad witness.
    # The SUM reference has G unchanged forwards and one joint backward.
    # The rejected all-row forward regrouping is neither retried nor used.
    budget = dict(logical_candidates=1, fit_calls=0, optimizer_updates=0, physical_writes=0,
                  native_forward_max=3 * G + 1 + (G if compatible else 0),
                  native_backward_max=2 * G + 1 + (G if compatible else 0),
                  prefix_calls_max=0, independent_qr_svd_max=len(layers),
                  independent_cholesky_max=len(layers), observer_calls=0)
    counters = dict(native_forward=0, native_backward=0, prefix_calls=0,
                    independent_qr_svd=0, independent_cholesky=0)
    checks = {}
    try:
        a.native_route = 'cached'
        cached = _native_pass(a, entry, entry['groups'], geometries, fixture)
        counters['native_forward'] += cached['native_forward_calls']
        counters['native_backward'] += cached['native_backward_calls']
        a.native_route = 'full'
        full = _native_pass(a, entry, entry['groups'], geometries, fixture)
        counters['native_forward'] += full['native_forward_calls']
        counters['native_backward'] += full['native_backward_calls']
        checks['cached_vs_full_native'] = _compare_native(cached, full, layers)
        a.native_route = 'cached'
        # Independently differentiate the full logical SUM using the exact
        # original physical owner groups. This checks accumulation through
        # the full off-owner S Jacobian, not batch-shape regrouping parity.
        logical_sum = _native_pass(a, entry, entry['groups'], geometries, fixture,
                                   backward_schedule='logical-sum')
        counters['native_forward'] += logical_sum['native_forward_calls']
        counters['native_backward'] += logical_sum['native_backward_calls']
        checks['original_group_logical_SUM'] = _compare_native(cached, logical_sum, layers)
        if compatible:
            direct = _native_pass(a, entry, entry['groups'], geometries, fixture, direct=True)
            counters['native_forward'] += direct['native_forward_calls']
            counters['native_backward'] += direct['native_backward_calls']
            checks['compatible_direct_vs_projected'] = _compare_native(cached, direct, layers)
        else:
            checks['compatible_direct_vs_projected'] = dict(status='NOT_CLAIMED_RANK_INCOMPATIBLE_OPERATOR',
                                                            projected_S_path_required=True)
        probe = _native_pass(a, entry, entry['groups'][:1], geometries, fixture, backward=False)
        counters['native_forward'] += probe['native_forward_calls']
        counters['native_backward'] += probe['native_backward_calls']
        firstowners = sorted({r['request'] for r in entry['groups'][0]['rows']})
        checks['original_no_grad_witness'] = _parity(probe['losses'], cached['losses'][firstowners])
        # Dense old-CD target solve supplies a differentiation reference at
        # this fixed candidate. It is not a causal candidate fit writer.
        dense = {}
        for l in layers:
            g = geometries[l]
            K = initial['keys'][l].to(a.device).double()
            owners = g.owners.to(a.device)
            raw = build_prior_from_npz(entry['stats'][str(l)], history[l], lambda_c=g.lambda_c, device=a.device)
            A, L, _ = prepare_metric(raw)
            counters['independent_cholesky'] += 1
            del raw
            u = fixture[l].detach().clone().requires_grad_()
            D = entry['anchors'][l][None, :] * u
            U, diag = solve_realized(A, L, K, D.double()[:, owners], g.weights.to(a.device), 'equality')
            counters['independent_qr_svd'] += 1
            require(diag['numerical_projection_verified'], 'QUALIFICATION_DENSE_NUMERICAL_PROJECTION')
            Q = (U @ L).square().sum()
            # Actual Wentry==own W0 was checked above. Thus the independent
            # dense alignment and its derivative are exactly zero; no huge
            # d_out-by-n_in by n_in-by-n_in zero GEMM or second C0 is needed.
            c = U.sum() * 0.
            Pi = Q + 2 * g.lambda_c * c.abs()
            dense_grad, = torch.autograd.grad(Pi, u)
            compact_grad, cost = cost_gradient_u(fixture[l], entry['anchors'][l], g, alpha=1)
            ideal = U.detach() @ K
            effective = (a.weights[l].detach() + U.detach().float()).double() - a.weights[l].detach().double()
            dense[str(l)] = dict(projected_response=action_parity(ideal, projected_action(D.detach(), g)),
                FP32_effective_action=action_parity(effective @ K, ideal),
                energy=_parity(Q.detach(), torch.tensor(cost['Q'], dtype=torch.float64, device=a.device),
                               atol=1e-10, rtol=1e-8),
                alignment=_parity(c.detach(), torch.tensor(cost['c'], dtype=torch.float64, device=a.device),
                                  atol=1e-10, rtol=1e-8),
                stored_u_gradient=_parity(compact_grad, dense_grad),
                actual_commit_performed=False, dense_reference_rank=diag['rank'],
                entry_geometry_rank=g.retained_rank, all_native_rows_including_KL=True)
            del K, A, L, U, Q, c, Pi, effective, dense_grad, compact_grad, ideal
        checks['dense_compact_projected_action'] = dense
        require(state(a, history) == before and cache_identity(entry) == cache
                and a.guard() == guard and a.hook_signature() == hooks and rng_equal(rng),
                'QUALIFICATION_STATE_CACHE_RNG_MUTATION')
        require(counters['native_forward'] == budget['native_forward_max']
                and counters['native_backward'] == budget['native_backward_max']
                and counters['prefix_calls'] == budget['prefix_calls_max']
                and counters['independent_qr_svd'] == budget['independent_qr_svd_max']
                and counters['independent_cholesky'] == budget['independent_cholesky_max'],
                'QUALIFICATION_SEALED_CALL_BUDGET')
        passed = _all_pass(checks)
        original_rows = [[r['global_row'] for r in group['rows']] for group in entry['groups']]
        receipt = dict(schema='JLZ_CD_CUMULATIVE_SAME_CANDIDATE_QUALIFICATION_V2', pass_=passed,
            runtime_device=str(a.device), actual_model=True, GPU_qualified=a.device.type == 'cuda' and passed,
            CPU_does_not_qualify_GPU=a.device.type != 'cuda', requests=B,
            native_pack_identity=entry['pack']['identity'], entry_cache_identity=cache,
            source_identity=entry.get('source_identity'), W0=before['W'], H0=before['H'],
            qualification_repair_receipt=entry.get('qualification_repair_receipt'),
            fixture_identity=digest({str(l): tensor_sha(v) for l, v in fixture.items()}),
            geometry_identity={str(l): g.operator_hash for l, g in geometries.items()},
            qualification_scope='FIXED_ORIGINAL_OWNER_GROUPS_ONLY',
            original_group_rows=original_rows, logical_SUM_reference_group_rows=original_rows,
            regrouped_group_rows=[], requests_per_group=1,
            original_group_tokens_cache_order_preserved=True,
            physical_regrouping_qualified=False,
            rejected_physical_regrouping=dict(status='NOT_QUALIFIED', usage='NOT_USED',
                prior_failed_check='logical_SUM_microbatch', old_failure_relabelled_PASS=False,
                immutable_prior_failure_link=entry.get('qualification_repair_receipt'),
                reason='ALL_ROW_FORWARD_BATCH_SHAPE_GRADIENT_PARITY_FAILED_AT_UNCHANGED_TOLERANCE',
                retried=False, qualified_by_original_group_SUM=False),
            checks=checks, sealed_budget=budget, actual_calls=counters,
            native_reductions_preserved=True, teacher_own_entry=True,
            common_candidate_no_fit=True, no_scientific_tuning=True, physical_state_unchanged=True,
            deterministic_nonmutating_witness=passed, seconds=time.monotonic() - started,
            persisted_tensors=False)
        if out is not None:
            write(out / 'qualification.json', receipt)
        require(passed, 'SAME_CANDIDATE_QUALIFICATION_FAILED')
        return receipt
    finally:
        a.native_route, a.capture_virtual = route, capture
        a.last_virtual.clear()
        a.last_virtual.update(prior_virtual)
