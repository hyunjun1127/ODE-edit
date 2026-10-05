"""Synchronous projected native SUM fit; no candidate writer or terminal VJP."""
import math
import time

import torch

from project.run_scripts.jlz_realization.common import require, tensor_sha, digest, write
from project.run_scripts.jlz_realization.subject import row_logprobs
from project.run_scripts.jlz_shared_budget.optimizer import EfficiencyAdam, norm_gradient
from project.run_scripts.jlz_realization.writer import rng_snapshot, rng_restore, rng_equal
from .geometry import projected_action, cost_gradient_u


def common_stop(values, candidate, threshold=.05):
    require(0 <= candidate <= 24 and values, 'CANDIDATE_RANGE')
    require(all(math.isfinite(float(v)) for v in values), 'FIT_NONFINITE_OBJECTIVE')
    if all(float(v) < threshold for v in values):
        return 'ZERO_STEP' if candidate == 0 else 'OBJECTIVE_THRESHOLD'
    return 'EVALUATION_BUDGET' if candidate == 24 else None


def validate_entry(a, entry, geometries):
    B = entry['pack']['n_requests']
    require(B > 0 and set(geometries) == set(a.sites), 'FIT_GEOMETRY_COVERAGE')
    rows = [row for group in entry['groups'] for row in group['rows']]
    q = len(entry['pack']['row_kind'])
    require([r['global_row'] for r in rows] == list(range(q)), 'GLOBAL_NATIVE_ROW_ORDER')
    require(set(entry['teachers']) == set(range(B)), 'FRESH_ENTRY_TEACHERS')
    require(a.profile['kl_factor'] == .0625, 'NATIVE_KL_FACTOR')
    expected_owners = [r['request'] for r in rows]
    for l, geometry in geometries.items():
        require(geometry.S.shape == (B, q) and geometry.C.shape == (B, B)
                and geometry.J.shape == (a.dims[l][0], B), 'FIT_GEOMETRY_SHAPE')
        require(geometry.owners.tolist() == expected_owners, 'FIT_GEOMETRY_ROW_OWNERS')
        require(entry['anchors'][l].shape == (B,) and bool((entry['anchors'][l] > 0).all()),
                'FRESH_ENTRY_ANCHORS')
    # The frozen native reduction needs every rewrite and KL row of an owner
    # in one physical graph. It does not assume a fixed seven contexts.
    seen = set()
    for group in entry['groups']:
        owners = {row['request'] for row in group['rows']}
        require(not seen.intersection(owners), 'COMPLETE_OWNER_GROUP_REQUIRED')
        for owner in owners:
            own = [row for row in group['rows'] if row['request'] == owner]
            require(sum(row['kind'] == 'rewrite' for row in own) > 0
                    and sum(row['kind'] == 'kl' for row in own) == 1,
                    'COMPLETE_REQUEST_GRAPH')
        seen.update(owners)
    require(seen == set(range(B)), 'COMPLETE_REQUEST_COVERAGE')


def forward(a, entry, group, u, geometries, capture=True):
    """Use original sentences/readout/NLL and current||own-entry KL exactly."""
    indices = [r['global_row'] for r in group['rows']]
    D = {l: entry['anchors'][l][None, :] * value for l, value in u.items()}
    Y = {l: projected_action(D[l], geometries[l], indices) for l in a.sites}
    nh, fh, hidden = a.native_projected(group, Y, capture, local_rows=True)
    rows = group['rows']
    logps = row_logprobs(a, rows, nh, fh)
    requests = sorted({r['request'] for r in rows})
    nll, kl, captures = {r: [] for r in requests}, {}, {r: {} for r in requests}
    for j, (row, lp) in enumerate(zip(rows, logps)):
        r = row['request']
        if row['kind'] == 'rewrite':
            target = row['target'][row['target'] != -100].to(a.device)
            nll[r].append(-lp.gather(1, target[:, None]).mean())
            if capture and row['global_row'] in entry['pack']['canonical_rows']:
                captures[r] = {l: h[j].detach().cpu().clone() for l, h in hidden.items()}
        else:
            teacher = entry['teachers'][r].to(a.device)
            kl[r] = (lp.exp() * (lp - teacher)).sum()
    require(all(len(nll[r]) == sum(row['kind'] == 'rewrite' and row['request'] == r for row in rows)
                and nll[r] and r in kl for r in requests),
            'COMPLETE_REQUEST_GRAPH')
    if capture:
        require(all(set(captures[r]) == set(a.sites) for r in requests), 'CANONICAL_CAPTURE')
    return {r: (torch.stack(nll[r]).mean(), kl[r]) for r in requests}, captures


def request_values(a, entry, u, values, captures):
    star = entry['anchors'][a.profile['anchor_layer']]
    result = {}
    for r, (nll, kl) in values.items():
        F = nll + a.profile['kl_factor'] * kl
        norms = [float(u[l][:, r].detach().norm()) for l in a.sites]
        price = .5 / float(star[r])
        J = float(F.detach()) + price * sum(norms)
        require(bool(torch.isfinite(F)) and math.isfinite(J), 'FIT_NONFINITE')
        require(all(bool(torch.isfinite(v).all()) for v in captures[r].values()), 'CAPTURE_NONFINITE')
        result[r] = dict(nll=float(nll.detach()), kl=float(kl.detach()), F=float(F.detach()),
                         J=J, price=price, budget=sum(norms), norms=norms, capture=captures[r])
    return result


def _evaluate(a, entry, u, geometries, groups):
    pending = {}
    with torch.no_grad():
        for group in groups:
            values, captures = forward(a, entry, group, u, geometries, True)
            pending.update(request_values(a, entry, u, values, captures))
    return pending


def _native_backward(a, entry, u, geometries, *, evaluated=None, capture=False):
    pending, count = {}, 0
    for group in entry['groups']:
        values, captures = forward(a, entry, group, u, geometries, capture)
        if capture:
            pending.update(request_values(a, entry, u, values, captures))
        if evaluated is not None:
            # Replay is a technical identity check, never a second selection.
            for r, (nll, kl) in values.items():
                require(torch.isclose(nll.detach().cpu(), torch.tensor(evaluated[r]['nll']),
                                      atol=1e-6, rtol=2e-4).item()
                        and torch.isclose(kl.detach().cpu(), torch.tensor(evaluated[r]['kl']),
                                          atol=1e-6, rtol=2e-4).item(), 'NATIVE_REPLAY_PARITY')
        torch.stack([nll + a.profile['kl_factor'] * kl for nll, kl in values.values()]).sum().backward()
        count += 1
    return pending, count


def _emit(events, event, payload, request=None, candidate=None, layer=None):
    if events is not None:
        events.emit(event, payload, request, candidate, layer)


@torch.no_grad()
def _terminal_cost(D, geometry, alpha, allocation_price):
    """Reporting only: terminal candidates have no risk-gradient assembly."""
    target = D.double()
    Q = ((target @ geometry.C.to(D.device)) * target).sum()
    factor_Q = (target @ geometry.N.to(D.device)).square().sum()
    cross = (target * geometry.J.to(D.device)).sum()
    cumulative = 2 * alpha * geometry.lambda_c * cross.abs()
    require(bool(torch.isfinite(Q)) and bool(torch.isfinite(cross)), 'TERMINAL_COST_NONFINITE')
    require(abs(float(Q - factor_Q)) <= 1e-10 + 1e-8 * float(factor_Q),
            'TERMINAL_FACTOR_ENERGY_MISMATCH')
    return dict(Q=float(Q), c=float(cross), cumulative=float(cumulative),
                Pi=float(Q + cumulative), weighted_cost=float(allocation_price * (Q + cumulative)))


def fit(a, entry, geometries, calibration, events=None, out=None, *, alpha=0,
        schedule='full-evaluate-replay'):
    """All requests adopt the same last fully evaluated c0..c24 candidate."""
    started = time.monotonic()
    B, layers = entry['pack']['n_requests'], a.sites
    require(alpha in (0, 1), 'ARM_ALPHA')
    require(schedule in ('full-evaluate-replay', 'qualified-witness'), 'FIT_SCHEDULE')
    if B == 0:
        require(not entry['groups'], 'EMPTY_BATCH_ROWS')
        a.last_virtual.clear()
        receipt = dict(requests=0, logical_evaluations=0, optimizer_updates=0,
                       physical_forward_calls=0, physical_backward_calls=0, empty_noop=True)
        return dict(u={}, D={}, Y={}, z={}, terminal={}), receipt
    validate_entry(a, entry, geometries)
    require(schedule != 'qualified-witness' or entry.get('qualified_witness') is True,
            'WITNESS_SCHEDULE_UNQUALIFIED')
    ids = entry['pack']['record_ids']
    star = entry['anchors'][a.profile['anchor_layer']]
    prices = torch.tensor([.5 / float(value) for value in star],
                          device=a.device, dtype=torch.float64)
    u = {l: torch.zeros((a.dims[l][0], B), device=a.device, dtype=torch.float32,
                        requires_grad=True) for l in layers}
    opts = [EfficiencyAdam([u[l][:, r] for l in layers], star[r]) for r in range(B)]
    a.last_virtual.clear()
    previous_capture = a.capture_virtual
    a.capture_virtual = True
    Fcount = Bcount = probe_count = replay_count = 0
    candidates, terminal = [], {}
    gradient_samples, last_gradient_sample = {}, None
    norm_additions = allocation_additions = 0
    try:
        for candidate in range(25):
            for value in u.values():
                value.grad = None
            rng = rng_snapshot()
            witness = False
            pending = {}
            if schedule == 'qualified-witness' and candidate < 24:
                pending = _evaluate(a, entry, u, geometries, entry['groups'][:1])
                Fcount += 1
                probe_count += 1
                require(rng_equal(rng), 'WITNESS_RNG_MUTATION')
                # Outside the declared FP32 uncertainty band, one complete
                # owner's failure proves that "all owners pass" is false.
                witness = any(p['J'] - (1e-6 + 2e-4 * abs(p['J'])) >= .05
                              for p in pending.values())
            if not witness:
                groups = entry['groups'][1:] if pending else entry['groups']
                pending.update(_evaluate(a, entry, u, geometries, groups))
                Fcount += len(groups)
                reason = common_stop([pending[r]['J'] for r in range(B)], candidate)
            else:
                reason = None
            allocation_price = calibration.resolve(candidate=candidate, geometries=geometries,
                blocks=u, anchors=entry['anchors'], prices=prices,
                delta_zero=entry.get('delta_zero', False))
            if reason is None:
                rng_restore(rng)
                if witness:
                    pending, calls = _native_backward(a, entry, u, geometries, capture=True)
                    require(common_stop([pending[r]['J'] for r in range(B)], candidate) is None,
                            'WITNESS_STOP_PARITY')
                else:
                    _, calls = _native_backward(a, entry, u, geometries, evaluated=pending)
                    replay_count += calls
                Fcount += calls
                Bcount += calls
                # SUM native gradients contain all off-owner S derivatives.
                native = {l: (value.grad.detach().clone() if value.grad is not None
                              else torch.zeros_like(value)) for l, value in u.items()}
                costs, allocation, norm = {}, {}, {}
                for l in layers:
                    grad, cost = cost_gradient_u(u[l].detach(), entry['anchors'][l], geometries[l], alpha=alpha,
                                                 allocation_price=allocation_price or 0.)
                    costs[l], allocation[l] = cost, grad
                    norm[l] = torch.stack([norm_gradient(u[l][:, r].detach(), pending[r]['price'])
                                           for r in range(B)], dim=1)
                allocation_additions += 1
                norm_additions += 1
                components = {str(l): dict(
                    native_u_l2=native[l].double().norm(dim=0).tolist(),
                    norm_u_l2=norm[l].double().norm(dim=0).tolist(),
                    Q_u_l2=costs[l].get('Q_gradient_u_l2'),
                    cumulative_u_l2=costs[l].get('cumulative_gradient_u_l2'),
                    allocation_u_l2=allocation[l].double().norm(dim=0).tolist(),
                    total_u_l2=(native[l] + allocation[l] + norm[l]).double().norm(dim=0).tolist(),
                    component_cast_contract='separate_FP64_D_component_to_FP32_then_anchor_FP32; '
                                            'combined_gradient_cast_is_not_sum_of_separate_casts')
                              for l in layers}
                last_gradient_sample = (candidate, components)
                if candidate == 1:
                    gradient_samples['1'] = components
                # Compute every request proposal against one immutable full u.
                proposals, optimizer_receipts = {}, {}
                for r in range(B):
                    blocks = [u[l][:, r].detach().clone() for l in layers]
                    total = [native[l][:, r] + allocation[l][:, r] + norm[l][:, r] for l in layers]
                    proposals[r], optimizer_receipts[r] = opts[r].step(blocks, total)
            else:
                costs = {l: _terminal_cost(entry['anchors'][l][None, :] * u[l], geometries[l], alpha,
                                          allocation_price or 0.) for l in layers}
                native = allocation = norm = None
            scalar_costs = {str(l): {key: float(cost[key]) for key in ('Q', 'c', 'cumulative', 'Pi', 'weighted_cost')}
                            for l, cost in costs.items()}
            native_sum = sum(p['J'] for p in pending.values())
            weighted_risk = sum(c['weighted_cost'] for c in scalar_costs.values())
            candidate_receipt = dict(candidate=candidate, logical_evaluation=candidate + 1,
                updates_completed=candidate, native_sum=native_sum, objective_mean=(native_sum + weighted_risk) / B,
                lambda_alloc=allocation_price, alpha=alpha, costs=scalar_costs,
                native_J=[pending[r]['J'] for r in range(B)], common_terminal_reason=reason,
                native_gradient_sum=True, norm_added_once=reason is None,
                allocation_added_once=reason is None, witness=witness)
            candidates.append(candidate_receipt)
            _emit(events, 'coupled_candidate', candidate_receipt, candidate=candidate)
            for r in range(B):
                p = pending[r]
                payload = {key: p[key] for key in ('nll', 'kl', 'F', 'J', 'price', 'budget', 'norms')}
                payload.update(evaluation_ordinal=candidate + 1, updates_completed=candidate,
                               will_backward=reason is None, terminal_reason=reason,
                               common_candidate=True, permanent_request_freeze=False, owner=r)
                _emit(events, 'candidate_request', payload, ids[r], candidate)
                for i, l in enumerate(layers):
                    gradients = None if native is None else dict(
                        native_l2=float(native[l][:, r].double().norm()),
                        norm_l2=float(norm[l][:, r].double().norm()),
                        allocation_l2=float(allocation[l][:, r].double().norm()),
                        Q_l2=None if costs[l].get('Q_gradient_u_l2') is None else costs[l]['Q_gradient_u_l2'][r],
                        cumulative_l2=None if costs[l].get('cumulative_gradient_u_l2') is None
                                      else costs[l]['cumulative_gradient_u_l2'][r],
                        total_l2=float((native[l][:, r] + allocation[l][:, r] + norm[l][:, r]).double().norm()))
                    _emit(events, 'candidate_layer', dict(u_l2=p['norms'][i],
                        anchor_l2=float(entry['anchors'][l][r]), gradients=gradients,
                        eligible_for_reentry=True, owner=r), ids[r], candidate, l)
                if reason is not None:
                    terminal[r] = dict(candidate=candidate, reason=reason, J=p['J'])
                    _emit(events, 'terminal_request', dict(accepted_candidate_index=candidate,
                        logical_evaluations=candidate + 1, optimizer_updates=candidate,
                        backward_calls=candidate, accepted_objective_J=p['J'],
                        reason=reason, gradient_kkt_status='NO_BACKWARD_TERMINAL',
                        frozen_after_terminal=False, common_candidate=True, owner=r), ids[r], candidate)
                else:
                    for l in layers:
                        _emit(events, 'optimizer_layer', optimizer_receipts[r] |
                              dict(moments_preserved_after_projection=True, eligible_for_reentry=True, owner=r),
                              ids[r], candidate, l)
            if reason is not None:
                break
            with torch.no_grad():
                for r in range(B):
                    for i, l in enumerate(layers):
                        u[l][:, r].copy_(proposals[r][i])
        require(len(terminal) == B, 'TERMINAL_COMPLETENESS')
        q = len(entry['pack']['row_kind'])
        require(set(a.last_virtual) == set(range(q)), 'ALL_NATIVE_TERMINAL_CAPTURE')
        z = {l: torch.stack([a.last_virtual[row][l] for row in entry['pack']['canonical_rows']], dim=1)
             for l in layers}
        D = {l: (entry['anchors'][l][None, :] * value).detach() for l, value in u.items()}
        Y = {l: projected_action(D[l], geometries[l]).detach() for l in layers}
        accepted = len(candidates) - 1
        if last_gradient_sample is not None:
            sample_index, components = last_gradient_sample
            gradient_samples[str(sample_index)] = components
        receipt = dict(schema='JLZ_CD_CUMULATIVE_FIT_V1', requests=B,
            logical_evaluations=accepted + 1, request_evaluations=B * (accepted + 1),
            optimizer_updates=accepted, request_updates=B * accepted,
            physical_forward_calls=Fcount, physical_backward_calls=Bcount,
            no_grad_probe_calls=probe_count, replay_forward_calls=replay_count,
            terminal_extra_forward=0, terminal_extra_backward=0, terminal_candidate=accepted,
            norm_gradient_additions=norm_additions, allocation_gradient_additions=allocation_additions,
            synchronous_candidate=True, independent_request_freeze=False,
            schedule=schedule, candidates={ids[r]: p for r, p in terminal.items()},
            terminal_by_owner={str(r): dict(record_id=ids[r], **p) for r, p in terminal.items()},
            alpha=alpha, lambda_alloc=calibration.value, calibration_status=calibration.status,
            calibration_receipt=calibration.receipt, candidate_trace=candidates,
            lambda_identity=None if calibration.receipt is None else calibration.receipt['receipt_hash'],
            gradient_samples=gradient_samples, gradient_sample_candidates=sorted(map(int, gradient_samples)),
            terminal_gradient_status='NO_NATIVE_BACKWARD_OR_RISK_GRADIENT_ASSEMBLY',
            projected_Y_identity=digest({str(l): tensor_sha(y) for l, y in Y.items()}),
            seconds=time.monotonic() - started, checkpoint_saved=False)
        if out is not None:
            write(out / 'fit.json', receipt)
        return dict(u={l: value.detach() for l, value in u.items()}, D=D, Y=Y, z=z, terminal=terminal), receipt
    finally:
        a.capture_virtual = previous_capture
