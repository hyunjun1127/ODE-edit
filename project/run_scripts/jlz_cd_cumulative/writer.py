"""Unchanged CD physical writer, with projected-Y and actual-displacement records."""
import time
import torch
from project.run_scripts.jlz_writer_coupled.geometry import build_prior_from_npz
from project.run_scripts.jlz_realization.geometry import mean_keys
from project.run_scripts.jlz_realized_writer.capture import capture_native_sites
from project.run_scripts.jlz_realized_writer.geometry import prepare_metric, solve_realized, owner_weights, action_parity
from project.run_scripts.jlz_realized_writer.telemetry import vector_metrics, shares
from .geometry import load_native_c0, writer_cost
from .common import require, write, tensor_sha, digest

@torch.no_grad()
def apply(a, entry, plan, virtual, initial, H, W0, geometries, allocation, alpha, out):
    started = time.monotonic(); B = entry['pack']['n_requests']
    stored = {}; summaries = {}; calls = 0; guard = a.guard()
    oldH = {l: tensor_sha(h) for l, h in H.items()}
    for l in a.sites:
        begin = time.monotonic(); pre = capture_native_sites(a, entry, (l,)); calls += pre['calls']
        rows = pre['rows']; owners = torch.tensor([r['request'] for r in rows], device=a.device)
        canonical = [j for j, r in enumerate(rows) if r['global_row'] in entry['pack']['canonical_rows']]
        require(len(canonical) == B, 'CANONICAL_COUNT')
        K = pre['keys'][l].to(a.device).double(); mean = pre['mean'][l].to(a.device).double()
        D = plan['D'][l]; T = D[:, owners].double(); weights = owner_weights(owners, B)
        rawA = build_prior_from_npz(entry['stats'][str(l)], H[l], lambda_c=15000., device=a.device)
        A, L, metric = prepare_metric(rawA); del rawA
        U, diag = solve_realized(A, L, K, T, weights, 'equality')
        diag.update(metric=metric, key_hash=tensor_sha(K), target_hash=tensor_sha(T),
                    row_order=digest([(r['global_row'], r['request'], r['kind'], r['lookup']) for r in rows]),
                    branch='CD', upper_actual_capture=True, zero_duplicate_columns_retained=True)
        require(diag['numerical_projection_verified'], 'NUMERICAL_REALIZATION_FAILURE')
        old = a.weights[l].detach().clone(); new = old + U.float()
        require(bool(torch.isfinite(new).all()), 'MATERIALIZED_NONFINITE')
        effective = new.double() - old.double(); cast = U.float().double()
        action = effective @ K; parity = action_parity(action, U @ K)
        require(parity['pass_'], 'FP32_IDEAL_EFFECTIVE_PARITY')
        cov = load_native_c0(entry['stats'][str(l)], H[l].shape, device=a.device)
        cov = (cov + cov.T) * .5
        delta = old.double() - W0[l].to(a.device).double()
        actual_c = float(((delta @ cov) * effective).sum())
        ideal_c = float(((delta @ cov) * U).sum())
        actual_q = float((effective @ L).square().sum())
        cov_q = float(((effective @ cov) * effective).sum())
        g = geometries[l]
        predicted_q = float((D.double() @ g.N).square().sum())
        predicted_c = float((D.double() * g.J).sum())
        predicted = dict(Q=predicted_q, c=predicted_c,
                         cumulative=2 * 15000. * alpha * abs(predicted_c),
                         Pi=predicted_q + 2 * 15000. * alpha * abs(predicted_c))
        key_drift = float((K - initial['keys'][l].to(a.device).double()).norm())
        entry_key_norm = float(initial['keys'][l].double().norm())
        energies = dict(ideal_Q=diag['ideal_Q'], effective_Q=actual_q,
                        ideal_c=ideal_c, ideal_Pi=diag['ideal_Q'] + 2 * 15000. * alpha * abs(ideal_c),
                        actual_covariance_Q=cov_q, actual_c=actual_c,
                        actual_Pi=actual_q + 2 * 15000. * alpha * abs(actual_c),
                        predicted=predicted, allocation_price=allocation,
                        actual_displacement='double(W_entry_FP32)-double(W_initial_FP32)',
                        effective_update_norm=float(effective.norm()), cast_update_norm=float(cast.norm()))
        energies.update(fresh_entry_key_drift=key_drift,
                        fresh_entry_key_relative_drift=key_drift / entry_key_norm if entry_key_norm else None)
        a.weights[l].copy_(new)
        require(tensor_sha(a.weights[l]) == tensor_sha(new), 'EXACT_EVALUATED_COPY')
        anchor = entry['anchors'][l][owners]
        reference = plan['Y'][l].to(a.device).double()
        actions = {name: vector_metrics(update @ K, reference, anchor) for name, update in
                   [('ideal', U), ('cast', cast), ('effective', effective)]}
        requested = vector_metrics(action, T, anchor)
        stored[l] = dict(pre=pre['hidden'][l], effective=action.cpu(), rows=rows, canonical=canonical,
                         actions=actions, requested=requested, mean=(effective @ mean).cpu())
        summaries[l] = dict(**energies, solver=diag, ideal_effective_parity=parity,
                            seconds=time.monotonic() - begin, weight_after=tensor_sha(a.weights[l]))
        write(out / f'layer-{l}-solve.json', summaries[l])
        del K, mean, D, T, A, L, U, old, new, effective, cast, cov, delta, pre
    final = capture_native_sites(a, entry, a.sites); calls += final['calls']
    rowmeta = stored[a.first]['rows']; owners = torch.tensor([r['request'] for r in rowmeta])
    allrows = []; checks = {}; mean_records = {}; canonical_actions = {}; context_actions = {}; mean_actions = {}
    for l in a.sites:
        saved = stored[l]; local = final['hidden'][l].double() - saved['pre'].double()
        checks[l] = action_parity(local, saved['effective'])
        Y = plan['Y'][l].cpu().double(); ownerD = plan['D'][l].cpu().double()[:, owners]
        virtualpost = virtual[l].double(); virtualpre = virtualpost - Y
        inherited = saved['pre'].double() - virtualpre; direct = local - Y
        finalgap = final['hidden'][l].double() - virtualpost
        net = final['hidden'][l].double() - initial['hidden'][l].double()
        require(torch.allclose(inherited + direct, finalgap, atol=1e-12, rtol=1e-12), 'ACTUAL_Y_VECTOR_GAP_IDENTITY')
        anchor = entry['anchors'][l].cpu()[owners]
        actual = vector_metrics(local, Y, anchor); requested = vector_metrics(local, ownerD, anchor)
        net_metrics = vector_metrics(net, Y, anchor)
        rew = [j for j, r in enumerate(rowmeta) if r['kind'] == 'rewrite']; rwrows = [rowmeta[j] for j in rew]
        localmean = mean_keys(local[:, rew].float(), rwrows, entry['pack'])
        plannedmean = mean_keys(Y[:, rew].float(), rwrows, entry['pack'])
        mean_records[l] = dict(actual_vs_Y=vector_metrics(localmean, plannedmean, entry['anchors'][l].cpu()),
                               actual_vs_requested=vector_metrics(localmean, plan['D'][l].cpu(), entry['anchors'][l].cpu()))
        for j, row in enumerate(rowmeta):
            allrows.append(dict(layer=l, owner=row['request'], case_id=entry['pack']['record_ids'][row['request']],
                global_row=row['global_row'], kind=row['kind'], canonical=j in saved['canonical'],
                lookup=row['lookup'],
                reference='ACTUAL_FIT_PROJECTED_Y', ideal=saved['actions']['ideal'][j],
                cast=saved['actions']['cast'][j], effective=saved['actions']['effective'][j],
                actual=actual[j], requested_error=requested[j], net_entry_change=net_metrics[j],
                inherited_gap_norm=float(inherited[:, j].norm()), direct_fit_error_norm=float(direct[:, j].norm()),
                final_virtual_gap_norm=float(finalgap[:, j].norm()),
                inherited_direct_error_dot=float((inherited[:, j] * direct[:, j]).sum())))
        canonical_actions[l] = saved['effective'][:, saved['canonical']]
        context_actions[l] = saved['effective']; mean_actions[l] = saved['mean']
    require(all(v['pass_'] for v in checks.values()), 'ACTUAL_LOCAL_ADDITIVITY_PARITY')
    require(a.guard() == guard, 'NONSELECTED_MUTATION')
    write(out / 'local-additivity.json', dict(checks=checks, all_native_rows=True, no_delta_hook=True))
    write(out / 'actions.json', dict(rows=allrows, mean=mean_records, gap_reference='Y_NOT_OWNER_D'))
    write(out / 'shares.json', dict(currency='effective_FP32_direct_action_over_entry_anchor',
        canonical=shares({l: v.cpu() for l, v in plan['u'].items()}, canonical_actions, entry['anchors'], a.sites),
        mean=shares({l: v.cpu() for l, v in plan['u'].items()}, mean_actions, entry['anchors'], a.sites),
        contexts=shares({l: v.cpu()[:, owners] for l, v in plan['u'].items()}, context_actions,
                        {l: v.cpu()[owners] for l, v in entry['anchors'].items()}, a.sites),
        context_owners=owners.tolist(), context_roles=[r['kind'] for r in rowmeta]))
    history = []
    for l in a.sites:
        k = final['mean'][l]
        require(k.dtype == torch.float32 and k.device.type == 'cpu' and k.shape[1] == B, 'FINAL_NATIVE_KEYS')
        require(tensor_sha(H[l]) == oldH[l], 'PREMATURE_HISTORY')
        native = k.T.contiguous().T; H[l].add_(native @ native.T)
        require(bool(torch.isfinite(H[l]).all()), 'HISTORY_NONFINITE')
        history.append(dict(layer=l, append_count=1, columns=B, key_sha256=tensor_sha(k),
                            before=oldH[l], after=tensor_sha(H[l]), rewrite_only=True, KL_in_history=False, CPU_FP32=True))
    result = dict(branch='CD', layers=summaries, history=history, history_appends=len(history),
                  capture_forward_calls=calls, seconds=time.monotonic() - started, checkpoint_saved=False)
    write(out / 'writer.json', result)
    return result
