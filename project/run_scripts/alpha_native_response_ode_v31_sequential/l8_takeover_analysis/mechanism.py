"""CPU-only mechanism extraction; source-recorded zeros are not missing values.

L8 local records and already-published O/JV scalar tables are kept as separate
provenance classes. No source tensor, evaluator, or model is imported here.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from ..reporting import csvwrite, jwrite, read, sha

NA = 'NOT_RECORDED'
LAYERS = (4, 5, 6, 7, 8)
MODELS = ('llama3-8b-inst', 'qwen2.5-7b-inst')
ARMS = ('O_NATIVE', 'JV_NATIVE', 'L8_ONLY_NATIVE')


def decode(value):
    if value in ('', None):
        return NA
    if not isinstance(value, str):
        return value
    if value in ('True', 'False'):
        return value == 'True'
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return value


def rows(path):
    with Path(path).open(newline='') as f:
        return [{k: decode(v) for k, v in row.items()} for row in csv.DictReader(f)]


def key(row, node=False):
    fields = ('alias', 'arm', 'batch', 'node') if node else ('alias', 'arm', 'batch')
    return tuple(row[k] for k in fields)


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def sum_recorded(values):
    values = list(values)
    return math.fsum(values) if values and all(numeric(v) for v in values) else NA


def safe_ratio(a, b):
    return a / b if numeric(a) and numeric(b) and b != 0 else NA


def quadratic(c, matrix):
    return math.fsum(c[i] * matrix[i][j] * c[j] for i in range(len(c)) for j in range(len(c)))


def controller_quantities(n, accumulated=0.0):
    """Equivalent FP64 Python reductions of the sealed controller matrices.

    B=V+lambda*E; delta-B is NOT the finite-step dissipation defect.
    D=delta-V+h*(response^2+lambda*q)-h^2*response^2/2.
    KKT identity alone does not imply finite-step monotonicity.
    """
    c, g, hessian, metric = n['c'], n['g'], n['full_H'], n['G']
    step, lam = n.get('h', .5), n.get('lambda_value', .1)
    gain = math.fsum(x * y for x, y in zip(c, g))
    response = quadratic(c, hessian)
    q = quadratic(c, metric)
    dv = n['V_after'] - n['V_before']
    work = step * q
    return dict(gain_reconstructed=gain, response_squared_reconstructed=response,
        normalized_velocity_action_reconstructed=q, normalized_native_work=work,
        cumulative_normalized_native_work=accumulated + work,
        continuous_dissipation_residual=gain - response - lam * q,
        observed_delta_V=dv, barrier_increment=dv + lam * work,
        barrier_value=n['V_after'] + lam * (accumulated + work),
        finite_step_defect_reconstructed=dv + step * (response + lam * q) - .5 * step**2 * response,
        linear_prediction_delta_V=-step * gain + .5 * step**2 * response,
        speed_bound=n['V_before'] / (2 * lam),
        speed_excess=q - n['V_before'] / (2 * lam))


def l8_node(meta, n):
    actions = {x['layer']: x for x in n['layer_actions']}
    physical = {x['layer']: x for x in n['actual_physical']}
    if set(physical) != set(LAYERS) or set(actions) - {8}:
        raise ValueError('L8_LAYER_INVENTORY_OR_SUPPORT')
    for layer in LAYERS[:-1]:
        if any(physical[layer][k] != 0 for k in ('actual_step_DeltaW_squared', 'actual_net_DeltaW_squared', 'actual_nonzero')):
            raise ValueError('L8_SUPPORT_VIOLATION')
    layer_rows = []
    for layer in LAYERS:
        p = physical[layer]
        row = dict(meta, node=n['node'], layer=layer,
            step_squared=p['actual_step_DeltaW_squared'], step_norm=math.sqrt(p['actual_step_DeltaW_squared']),
            batch_entry_to_node_squared=p['actual_net_DeltaW_squared'],
            batch_entry_to_node_norm=math.sqrt(p['actual_net_DeltaW_squared']),
            actual_nonzero=p['actual_nonzero'], response_observed=layer in actions,
            W0_to_node_squared=NA, provenance='LOCAL_COMPLETED_L8_RAW',
            V_before=n['V_before'], V_after=n['V_after'], V_ratio=n['V_ratio'],
            model_error_normalized=n['model_error_normalized'], model_error_raw_activation=n['model_error_raw_activation'],
            finite_step_defect=n['finite_step_dissipation_defect'], h=n['h'], qN_ref=n['qN_ref'])
        if layer in actions:
            a = actions[layer]; i = n['active_layers'].index(layer)
            row.update(a)
            row.update(coefficient=n['c'][i], raw_physical_coefficient=n['raw_physical_coefficients'][i],
                H_diagonal=n['full_H'][i][i], g=n['g'][i], signed_g_c=n['predicted_target_contribution'][i],
                supported=n['c'][i] != 0, q=n['q_layers'][i])
            for name in ('raw_native', 'normalized_native', 'history', 'L2', 'frobenius'):
                velocity = 'frobenius_velocity_squared' if name == 'frobenius' else name + '_velocity_action'
                row[name + '_work'] = n['h'] * a[velocity]
        else:
            # The excluded coordinate is exactly unsupported; its response/H/g
            # were not observed. Do not invent zero response measurements.
            row.update(coefficient=0.0, raw_physical_coefficient=0.0, supported=False,
                H_diagonal=NA, g=NA, signed_g_c=0.0, q=NA,
                direction_raw_action=NA, direction_frobenius_squared=NA)
            for name in ('raw_native', 'normalized_native', 'history', 'L2', 'frobenius'):
                row[name + '_work'] = 0.0
        layer_rows.append(row)
    step_sum = math.fsum(x['step_squared'] for x in layer_rows)
    node = dict(meta, **{k: n[k] for k in ('node', 'V_before', 'V_after', 'V_ratio', 'V0',
        'model_error_normalized', 'model_error_raw_activation', 'finite_step_dissipation_defect',
        'KKT_stationarity', 'KKT_complementarity', 'qN_ref', 'node_wall_seconds')},
        total_step_energy=step_sum, batch_entry_to_node_energy=sum(x['batch_entry_to_node_squared'] for x in layer_rows),
        L8_energy_share=safe_ratio(physical[8]['actual_step_DeltaW_squared'], step_sum),
        joint_signed_progress=sum(n['predicted_target_contribution']), L4_7_signed_progress=0.0,
        supported_layers=[l for l, c in zip(n['active_layers'], n['c']) if c != 0],
        provenance='LOCAL_COMPLETED_L8_RAW')
    return node, layer_rows


def cost_row(meta, complete, writer, commit):
    out = dict(meta, provenance='LOCAL_COMPLETED_L8_RAW',
        endpoint_seconds=writer['endpoint_evaluation_seconds'],
        main_JVP=writer['main_jvp_count'], native_direction_builds=writer['dictionary_build_count'],
        native_factor_solves=writer['solve_count'], peak_gpu_bytes=complete['peak_gpu_allocated'],
        source_endpoint_physical_write_count=writer['endpoint']['physical_write_count'],
        persistent_commit_count=1, inner_virtual_euler_nodes=len(writer['nodes']),
        history_seconds=writer['history_finalization_compute']['seconds'])
    scopes = {'target': writer['compute_z'], 'write_including_endpoint': writer['write_including_endpoint'],
        'seen': complete['seen_compute'], 'checkpoint': complete['checkpoint_compute'],
        'commit': commit['compute'], 'full_batch': complete['full_batch_compute']}
    for scope, values in scopes.items():
        for field, value in values.items():
            out[scope + '_' + field] = value
    out['write_excluding_endpoint_seconds'] = out['write_including_endpoint_wall'] - out['endpoint_seconds']
    return out


def published_cost(r):
    out = dict(alias=r['alias'], arm=r['arm'], batch=r['batch'], provenance='SH2_PUBLICATION_SH1_RECOMPUTED',
        endpoint_seconds=r['endpoint_evaluation_seconds'], main_JVP=r['main_jvp_count'],
        native_direction_builds=NA, native_factor_solves=NA, peak_gpu_bytes=r['peak_gpu_bytes'],
        source_endpoint_physical_write_count=r['source_endpoint_physical_write_count'],
        persistent_commit_count=r['persistent_commit_count'], inner_virtual_euler_nodes=r['inner_virtual_euler_nodes'],
        history_seconds=r['history']['seconds'])
    for scope, field in [('target', 'compute_z'), ('write_including_endpoint', 'write_including_endpoint'),
        ('seen', 'seen'), ('checkpoint', 'checkpoint'), ('commit', 'commit_compute'), ('full_batch', 'full_batch')]:
        for k, v in r[field].items():
            out[scope + '_' + k] = v
    out['write_excluding_endpoint_seconds'] = out['write_including_endpoint_wall'] - out['endpoint_seconds']
    return out


def build(l8_root, external_main, sh1_package, output):
    l8_root, external_main, sh1_package, output = map(Path, (l8_root, external_main, sh1_package, output))
    if output.exists():
        raise FileExistsError(output)
    inputs = set()
    def load(p):
        inputs.add(p); return read(p)
    def table(p):
        inputs.add(p); return rows(p)
    # Rehash every available SH1 publication member before using derived tables.
    verification = load(sh1_package / 'analysis-manifest.json')
    member_checked = 0
    for m in verification.get('members', []):
        p = sh1_package / m['path']
        if p.stat().st_size != m['bytes'] or sha(p) != m['sha256']:
            raise ValueError('SH1_MEMBER_MISMATCH:' + str(p))
        inputs.add(p); member_checked += 1
    if member_checked == 0:
        raise ValueError('SH1_NO_MEMBER_BINDINGS')
    batches = table(sh1_package / 'batches.csv')
    nodes = table(sh1_package / 'nodes.csv')
    node_layers = table(sh1_package / 'node-layer.csv')
    batch_layers = table(sh1_package / 'batch-layer.csv')
    scales = table(sh1_package / 'request_normalization_scales.csv')
    normalization = table(sh1_package / 'normalization_batch_summary.csv')
    costs = [published_cost(r) for r in table(sh1_package / 'compute_accounting.csv')]
    allocations = {key(r, True): r for r in table(external_main / 'layer_allocation_nodes.csv')}
    node_index = {key(r, True): r for r in nodes}
    for group in (batches, nodes, node_layers, batch_layers, scales, normalization):
        for r in group:
            r['provenance'] = 'SH2_PUBLICATION_SH1_RECOMPUTED'
    for meta in sorted({key(r) for r in nodes}):
        accumulated = 0.0
        for ordinal in range(4):
            k = meta + (ordinal,); a = allocations[k]; n = node_index[k]
            q = controller_quantities(a, accumulated)
            accumulated = q['cumulative_normalized_native_work']; n.update(q)
            n['KKT_stationarity'] = a['KKT_stationarity']; n['KKT_complementarity'] = NA
            n['source_estimator_class'] = 'FULL_LAYER_TERMINAL_JVP_CURRENT_STATE'
            if not math.isclose(q['finite_step_defect_reconstructed'], a['finite_step_dissipation_defect'], abs_tol=1e-12, rel_tol=1e-8):
                raise ValueError('PUBLISHED_DEFECT_REDUCTION_MISMATCH')
    samples = load(l8_root / 'sample.lock.json')['records']
    sample_batches = {b: [r for r in samples if r['batch_index'] == b] for b in range(1, 11)}
    terminal_costs = table(sh1_package / 'cost.csv')
    for chain in sorted(l8_root.glob('chain-[45]-*')):
        terminal = load(chain / 'terminal-receipt.json')
        if terminal['status'] != 'TERMINAL_VALID' or terminal['completed_batches'] != 10 or terminal['requested'] != 1000:
            raise ValueError('L8_TERMINAL_INCOMPLETE')
        alias, arm = terminal['alias'], terminal['arm']
        terminal_costs.append(dict(alias=alias, arm=arm, process_seconds=terminal['total_seconds'],
            forward=terminal['compute']['forward'], backward=terminal['compute']['backward'],
            native_key_captures=terminal['compute']['native_keys'], solve_instrumented=terminal['compute']['linalg_solve'],
            provenance='LOCAL_COMPLETED_L8_RAW'))
        for b in range(1, 11):
            bd = chain / f'batch-{b:02d}'
            w, complete, commit, target = [load(bd / p) for p in ('writer.json', 'complete.json', 'commit.json', 'target-reference.json')]
            meta = dict(alias=alias, arm=arm, batch=b)
            if len(w['nodes']) != 4 or w['main_jvp_count'] != 4 or w['dictionary_build_count'] != 25:
                raise ValueError('L8_BUILD_JVP_COUNT')
            if target['metric']['capture_count'] != 1 or len(target['metric']['raw_layer_actions']) != 5:
                raise ValueError('L8_ALL_FIVE_ENTRY_REFERENCE')
            if not math.isclose(sum(target['metric']['raw_layer_actions']), target['metric']['qN_ref'], rel_tol=1e-12):
                raise ValueError('L8_ENTRY_REFERENCE_SUM')
            costs.append(cost_row(meta, complete, w, commit))
            accumulated = 0.; current_nodes = []; current_layers = []
            for n in w['nodes']:
                saved = load(bd / 'nodes' / f"node-{n['node']:02d}.json")
                if saved != n:
                    raise ValueError('WRITER_NODE_DUPLICATE_IDENTITY')
                if n['normalization_scales'] != target['normalization_scales'] or n['qN_ref'] != target['metric']['qN_ref']:
                    raise ValueError('L8_FROZEN_NORMALIZATION_REFERENCE')
                node, layers = l8_node(meta, n)
                cq = controller_quantities(n, accumulated); accumulated = cq['cumulative_normalized_native_work']
                node.update(cq, source_estimator_class='L8_ONLY_TERMINAL_JVP_CURRENT_STATE')
                if not math.isclose(cq['finite_step_defect_reconstructed'], n['finite_step_dissipation_defect'], abs_tol=1e-12, rel_tol=1e-8):
                    raise ValueError('L8_DEFECT_REDUCTION_MISMATCH')
                current_nodes.append(node); current_layers.extend(layers)
            nodes.extend(current_nodes); node_layers.extend(current_layers)
            physical = w['actual_physical_action']; by_layer = {r['layer']: r for r in physical['layers']}
            for l in LAYERS:
                p = by_layer[l]; selected = [r for r in current_layers if r['layer'] == l]
                r = dict(meta, layer=l, provenance='LOCAL_COMPLETED_L8_RAW', batch_net_squared=p['frobenius_sq'],
                    batch_net_norm=math.sqrt(p['frobenius_sq']), endpoint_native_raw=p['native_raw'],
                    materialized_nonzero=p['materialized_nonzero'], parameter_elements=p['parameter_elements'],
                    actual_step_squared_sum=sum(x['step_squared'] for x in selected),
                    actual_step_norm_sum=sum(x['step_norm'] for x in selected),
                    signed_predicted_progress=sum(x['h'] * x['signed_g_c'] for x in selected), W0_net_squared=NA)
                for name in ('raw_native', 'normalized_native', 'history', 'L2', 'frobenius'):
                    r[name + '_work'] = sum(x[name + '_work'] for x in selected)
                batch_layers.append(r)
            active_scales = [x for x, active in zip(target['normalization_scales'], target['normalization_active']) if active]
            minimum = min(active_scales); maximum = max(active_scales)
            minids = [r['case_id'] for r, v in zip(sample_batches[b], target['normalization_scales']) if v == minimum]
            for r, scale, active in zip(sample_batches[b], target['normalization_scales'], target['normalization_active']):
                scales.append(dict(meta, case_id=r['case_id'], active=active, source_N0_scale=scale,
                    frozen_row_weight=1 / (math.sqrt(len(active_scales)) * scale) if active else 0.0,
                    source_active_rule='scale > 0; no floor', controller_change_count=0, provenance='LOCAL_COMPLETED_L8_RAW'))
            norm = dict(meta, active_count=len(active_scales), requested=100, minimum_active_scale=minimum,
                maximum_active_scale=maximum, max_over_min_active_scale=maximum/minimum, minimum_scale_case_ids=minids,
                W_sha256=commit['committed_weight_sha256'], actual_endpoint_energy=physical['frobenius_net_sq'],
                node0_KKT_stationarity=w['nodes'][0]['KKT_stationarity'],
                node0_response_Gram_diagonal_max=max(w['nodes'][0]['full_H'][i][i] for i in range(len(w['nodes'][0]['c']))),
                qN_ref=target['metric']['qN_ref'], final_V_ratio=w['nodes'][-1]['V_ratio'],
                materialization_discrepancy_normalized=w['materialization_discrepancy_normalized'],
                max_normalized_model_error=max(n['model_error_normalized'] for n in w['nodes']),
                normalization_sweep_count=0, per_request_response_Gram_decomposition=NA, provenance='LOCAL_COMPLETED_L8_RAW')
            normalization.append(norm)
            net = physical['frobenius_net_sq']
            batchrow = dict(meta, provenance='LOCAL_COMPLETED_L8_RAW', total_batch_net_energy=net,
                total_batch_net_norm=math.sqrt(net), L8_batch_net_energy=by_layer[8]['frobenius_sq'],
                L4_7_batch_net_energy=sum(by_layer[l]['frobenius_sq'] for l in LAYERS[:-1]),
                L8_energy_share=safe_ratio(by_layer[8]['frobenius_sq'], net), final_V_ratio=w['nodes'][-1]['V_ratio'],
                L8_native_work_share=1.0 if sum(r['raw_native_work'] for r in batch_layers if key(r) == key(meta)) > 0 else NA,
                entropy=0.0 if net > 0 else NA, effective_layers=1.0 if net > 0 else NA,
                max_layer=8 if net > 0 else NA,
                signed_predicted_progress=sum(.5*n['joint_signed_progress'] for n in current_nodes), L4_7_signed_progress=0.0,
                W0_net_squared=NA, materialization_discrepancy_normalized=w['materialization_discrepancy_normalized'],
                factor_net_frobenius_squared=w['frobenius_net_sq'],
                materialized_minus_factor_net_squared=net-w['frobenius_net_sq'],
                factor_net_native_raw=w['native_net_raw'],
                **{k: norm[k] for k in ('minimum_active_scale', 'maximum_active_scale', 'minimum_scale_case_ids',
                    'max_over_min_active_scale', 'node0_response_Gram_diagonal_max')})
            bl = [r for r in batch_layers if key(r) == key(meta)]
            for name in ('raw_native', 'normalized_native', 'history', 'L2'):
                batchrow[name + '_work'] = sum(r[name + '_work'] for r in bl)
            batchrow['history_raw_work_ratio'] = safe_ratio(batchrow['history_work'], batchrow['raw_native_work'])
            for metric, value in [('RS', complete['current']['preference']['rewrite']),
                ('PS', complete['current']['preference']['rephrase']), ('NS', complete['current']['locality']['canonical_ns'])]:
                batchrow.update({metric + '_num': value['prompt_success_count'], metric + '_den': value['prompt_denominator'],
                    metric + '_rate': value['prompt_success_rate']})
            batches.append(batchrow)
    # Derive comparable quantities at the exact measurement unit, not a pooled
    # sum of state vectors. Recorded per-layer net is within each batch only.
    for r in batch_layers:
        siblings = [s for s in batch_layers if key(s) == key(r)]
        r['batch_net_magnitude_share'] = safe_ratio(r['batch_net_norm'], sum(s['batch_net_norm'] for s in siblings))
        r['batch_net_squared_share'] = safe_ratio(r['batch_net_squared'], sum(s['batch_net_squared'] for s in siblings))
    for r in batches:
        group = [n for n in nodes if key(n) == key(r)]
        r.update(barrier_positive_increment_count=sum(n['barrier_increment'] > 0 for n in group) if group else NA,
            finite_defect_positive_count=sum(n['finite_step_dissipation_defect'] > 0 for n in group) if group else NA,
            max_abs_continuous_dissipation_residual=max(abs(n['continuous_dissipation_residual']) for n in group) if group else NA,
            max_abs_KKT_stationarity=max(abs(n['KKT_stationarity']) for n in group) if group else NA,
            barrier_end_minus_entry=(group[-1]['barrier_value'] - group[0]['V_before']) if group else NA,
            materialized_net_native_raw=sum_recorded(x['endpoint_native_raw'] for x in batch_layers if key(x) == key(r)))
    for r in terminal_costs:
        part = [c for c in costs if c['alias'] == r['alias'] and c['arm'] == r['arm']]
        if 'provenance' not in r:
            r['provenance'] = 'SH2_PUBLICATION_SH1_RECOMPUTED'
        for field in ('main_JVP', 'native_direction_builds', 'native_factor_solves', 'endpoint_seconds',
            'target_wall', 'write_including_endpoint_wall', 'write_excluding_endpoint_seconds', 'seen_wall',
            'checkpoint_wall', 'commit_wall', 'full_batch_wall', 'full_batch_forward', 'full_batch_backward',
            'full_batch_native_keys', 'full_batch_linalg_solve', 'history_seconds'):
            r[field] = sum_recorded(c.get(field, NA) for c in part)
        r['peak_gpu_bytes'] = max(c['peak_gpu_bytes'] for c in part)
        r['target_seconds'] = r['target_wall']
        r['write_including_endpoint_seconds'] = r['write_including_endpoint_wall']
    for r in terminal_costs:
        ref = next(x for x in terminal_costs if x['alias'] == r['alias'] and x['arm'] == 'O_NATIVE')
        for name in ('process_seconds', 'write_excluding_endpoint_seconds', 'full_batch_wall'):
            r[name + '_over_Official'] = safe_ratio(r[name], ref[name])
        r['cross_server_wall_time_caveat'] = 'O/JV server2 vs L8 server4; hardware/concurrency not controlled'
    groups = {'batch_mechanism': batches, 'node_mechanism': nodes, 'node_layer_actions': node_layers,
        'batch_layer_actions': batch_layers, 'normalization': normalization, 'normalization_scales': scales,
        'cost_by_batch': costs, 'cost_by_arm': terminal_costs}
    expected = {'batch_mechanism':60, 'node_mechanism':160, 'node_layer_actions':800,
        'batch_layer_actions':300, 'normalization':60, 'normalization_scales':6000, 'cost_by_batch':60, 'cost_by_arm':6}
    for name, data in groups.items():
        if len(data) != expected[name]:
            raise ValueError(f'ROW_COUNT:{name}:{len(data)}')
        fields = set().union(*(r.keys() for r in data))
        for r in data:
            for field in fields:
                if field not in r or r[field] is None or r[field] == '':
                    r[field] = NA
        data.sort(key=lambda r: (MODELS.index(r['alias']), ARMS.index(r['arm']), r.get('batch', 0), r.get('node', -1), r.get('layer', -1), r.get('case_id', -1)))
    output.mkdir(parents=True, mode=0o700)
    for name, data in groups.items():
        csvwrite(output / (name + '.csv'), data)
    b10 = [r for r in batches if r['batch'] == 10]
    result = dict(status='MECHANISM_EXTRACTION_COMPLETE', row_counts=expected, batch10=b10, cost=terminal_costs,
        semantics=dict(barrier='B=V+lambda*E; E=sum(h*c^T G c). Its finite increment differs from the continuous KKT/dissipation identity and finite-step defect.',
            normalization='Exact source N0=entry residual norm, scale>0, no floor. Frozen within B; not comparable physical coefficient scale across different B or arms.',
            l8='All five entry native directions set qN_ref; all five directions rebuilt each node before L8 restriction. 25 builds and 4 L8 JVP per B.',
            physical='Actual materialized FP32 net and step differences; sum of step squared norms is not net squared norm. W0-to-W10 net NOT_RECORDED in publication.',
            support='L4–7 actual physical zeros are recorded; missing L4–7 response/H/g in L8 are NOT_RECORDED, not zero.',
            history='Within-state history+L2 native work decomposition; not a controlled history-reset experiment.',
            cost='Timing comparisons are descriptive cross-server/concurrency comparisons, not hardware-controlled speedup.',
            denominator='6 arms x 10 B100, dynamic 4 arms x40 nodes; output rows are batch/node/layer summaries, not independent request trials.',
            boundary='No new science gate or outcome-filtering; positive barrier increments/finite defects are retained observations.'),
        inputs=[dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(inputs)],
        members=[dict(path=p.name, bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(output.iterdir())],
        model_evaluator_gpu_slurm_actions=0)
    jwrite(output / 'mechanism-summary.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for arg in ('l8-root', 'external-main', 'sh1-package', 'output'):
        p.add_argument('--' + arg, type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(build(a.l8_root, a.external_main, a.sh1_package, a.output)['row_counts'], sort_keys=True))
