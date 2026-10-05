"""Sealed 2--4-request actual-model, same-candidate technical qualification.

No fit, optimizer update, physical write, observer panels or scientific tuning.
These checks run within the task job; a CPU receipt is never a GPU receipt.
"""
import time

import torch

from project.run_scripts.jlz_realization.common import require, tensor_sha, digest, state, write
from project.run_scripts.jlz_realization.inputs import make_rows, batches
from project.run_scripts.jlz_realization.subject import row_logprobs
from project.run_scripts.jlz_realization.writer import rng_snapshot, rng_equal
from project.run_scripts.jlz_realized_writer.capture import cache_identity
from project.run_scripts.jlz_realized_writer.geometry import prepare_metric, solve_realized, action_parity
from project.run_scripts.jlz_writer_coupled.geometry import build_prior_from_npz
from .geometry import projected_action, cost_gradient_u
from .optimize import validate_entry


def _parity(candidate, reference, *, atol=1e-6, rtol=2e-4):
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


def _native_pass(a, entry, groups, geometries, fixture, *, direct=False, backward=True):
    u = {l: value.detach().clone().requires_grad_(backward) for l, value in fixture.items()}
    records, hidden, logprobs = {}, {}, {}
    for group in groups:
        with torch.set_grad_enabled(backward):
            D = {l: entry['anchors'][l][None, :] * value for l, value in u.items()}
            if direct:
                nh, fh, found = a.native(group, D, True)
            else:
                indices = [row['global_row'] for row in group['rows']]
                Y = {l: projected_action(D[l], geometries[l], indices) for l in a.sites}
                nh, fh, found = a.native_projected(group, Y, True, local_rows=True)
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
                torch.stack([nr + .0625 * kr for nr, kr in values.values()]).sum().backward()
    return dict(losses=torch.stack([records[r] for r in sorted(records)]), hidden=hidden, logprobs=logprobs,
                gradients={l: value.grad.detach().clone() if value.grad is not None
                           else torch.zeros_like(value) for l, value in u.items()})


def _compare_native(candidate, reference, layers):
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
    """
    started = time.monotonic()
    B, layers = entry['pack']['n_requests'], a.sites
    require(2 <= B <= 4, 'QUALIFICATION_2_TO_4_REQUESTS_ONLY')
    require(a.profile['kl_factor'] == .0625, 'QUALIFICATION_NATIVE_KL')
    require(set(geometries) == set(layers) and set(W0) == set(layers), 'QUALIFICATION_LAYER_COVERAGE')
    validate_entry(a, entry, geometries)
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
    # Four native paths maximum, a single original no-grad witness, one
    # alternate prefix construction, and one independent solve per site.
    budget = dict(logical_candidates=1, fit_calls=0, optimizer_updates=0, physical_writes=0,
                  native_forward_max=3 * G + 1 + (G if compatible else 0),
                  native_backward_max=2 * G + 1 + (G if compatible else 0),
                  prefix_calls_max=1, independent_qr_svd_max=len(layers),
                  independent_cholesky_max=len(layers), observer_calls=0)
    counters = dict(native_forward=0, native_backward=0, prefix_calls=0,
                    independent_qr_svd=0, independent_cholesky=0)
    checks = {}
    try:
        a.native_route = 'cached'
        cached = _native_pass(a, entry, entry['groups'], geometries, fixture)
        counters['native_forward'] += G
        counters['native_backward'] += G
        a.native_route = 'full'
        full = _native_pass(a, entry, entry['groups'], geometries, fixture)
        counters['native_forward'] += G
        counters['native_backward'] += G
        checks['cached_vs_full_native'] = _compare_native(cached, full, layers)
        a.native_route = 'cached'
        # Change only physical grouping. The logical operator and owner/row
        # mapping remain the same entire narrow batch, never per-group CD.
        rows = make_rows(entry['pack'])
        alt = []
        for grouped, tokens in batches(rows, len(rows), pad_id, a.device):
            alt.append(dict(rows=grouped, tokens={k: v.cpu() for k, v in tokens.items()}, cache=a.prefix(tokens)))
            counters['prefix_calls'] += 1
        regrouped = _native_pass(a, entry, alt, geometries, fixture)
        counters['native_forward'] += 1
        counters['native_backward'] += 1
        checks['logical_SUM_microbatch'] = _compare_native(regrouped, cached, layers)
        if compatible:
            direct = _native_pass(a, entry, entry['groups'], geometries, fixture, direct=True)
            counters['native_forward'] += G
            counters['native_backward'] += G
            checks['compatible_direct_vs_projected'] = _compare_native(cached, direct, layers)
        else:
            checks['compatible_direct_vs_projected'] = dict(status='NOT_CLAIMED_RANK_INCOMPATIBLE_OPERATOR',
                                                            projected_S_path_required=True)
        probe = _native_pass(a, entry, entry['groups'][:1], geometries, fixture, backward=False)
        counters['native_forward'] += 1
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
        require(counters['native_forward'] <= budget['native_forward_max']
                and counters['native_backward'] <= budget['native_backward_max']
                and counters['prefix_calls'] <= budget['prefix_calls_max']
                and counters['independent_qr_svd'] <= budget['independent_qr_svd_max']
                and counters['independent_cholesky'] <= budget['independent_cholesky_max'],
                'QUALIFICATION_SEALED_CALL_BUDGET')
        passed = _all_pass(checks)
        receipt = dict(schema='JLZ_CD_CUMULATIVE_SAME_CANDIDATE_QUALIFICATION_V1', pass_=passed,
            runtime_device=str(a.device), actual_model=True, GPU_qualified=a.device.type == 'cuda' and passed,
            CPU_does_not_qualify_GPU=a.device.type != 'cuda', requests=B,
            native_pack_identity=entry['pack']['identity'], entry_cache_identity=cache,
            source_identity=entry.get('source_identity'), W0=before['W'], H0=before['H'],
            fixture_identity=digest({str(l): tensor_sha(v) for l, v in fixture.items()}),
            geometry_identity={str(l): g.operator_hash for l, g in geometries.items()},
            original_group_rows=[[r['global_row'] for r in group['rows']] for group in entry['groups']],
            regrouped_group_rows=[[r['global_row'] for r in group['rows']] for group in alt],
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
