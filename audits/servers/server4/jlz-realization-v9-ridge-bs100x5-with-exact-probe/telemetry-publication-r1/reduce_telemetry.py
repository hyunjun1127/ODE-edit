"""V9 저장 계측의 CPU 전용 소형 게시. 원자료/실행/환경/Slurm 변경 없음.

현재 실행 source와 별도인 분석 source. 출력 create-once; torch/model import 없음.
"""
import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


def require(ok, message):
    if not ok:
        raise ValueError(message)


def quantile(ordered, p):
    if not ordered:
        return None
    position = (len(ordered)-1)*p
    lo = int(position)
    hi = min(lo+1, len(ordered)-1)
    return ordered[lo] + (ordered[hi]-ordered[lo])*(position-lo)


def stats(values):
    valid = [float(x) for x in values if x is not None]
    require(all(math.isfinite(x) for x in valid), 'NONFINITE_NOT_NULL')
    ordered = sorted(valid)
    return dict(n=len(values), valid=len(valid), null=len(values)-len(valid),
                mean=sum(valid)/len(valid) if valid else None,
                min=ordered[0] if valid else None, q05=quantile(ordered, .05),
                q50=quantile(ordered, .5), q95=quantile(ordered, .95),
                q99=quantile(ordered, .99), max=ordered[-1] if valid else None)


def prefixed(values, prefix, full=False):
    s = stats(values)
    fields = s if full else {k:s[k] for k in ('mean', 'q95', 'min', 'max')}
    return {prefix+'_'+k:v for k,v in fields.items()}


def write_json(path, data):
    with path.open('x') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def write_csv(path, rows):
    require(bool(rows), 'EMPTY_TABLE')
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def file_receipt(path):
    data = path.read_bytes()
    return dict(path=str(path.resolve()), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


class Reader:
    def __init__(self, attempt):
        self.evidence = {}
        self.index = {}
        terminal = self.read(attempt/'report/terminal.json')
        index = self.read(attempt/'report/artifact-index.json')
        require(self.evidence[str((attempt/'report/artifact-index.json').resolve())]['sha256'] ==
                terminal['artifacts']['sha256'], 'COLLECTOR_INDEX_HASH')
        self.index = {x['path']:x for x in index}
        require(terminal['status'] == 'COMPLETED_500_BOTH', 'NOT_COMPLETE')

    def read(self, path):
        path = path.resolve()
        data = path.read_bytes()
        rec = dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        if str(path) in self.index:
            require(rec['sha256'] == self.index[str(path)]['sha256'], 'CHANGED_INPUT '+str(path))
        self.evidence[str(path)] = rec
        return json.loads(data)


def terminal_layer(row, arm, batch, layer):
    data = row['layer'][layer]
    result = dict(arm=arm, batch=batch, layer=layer, candidate=25,
                  zero_D=sum(data['zero_D']), small_rho=sum(data['small_rho']))
    for name in ('gamma','norm_ratio','M_diagonal','rho','realized_rho','orthogonal_relative',
                 'fit_relative','self_norm','cross_norm','self_cross_dot','planned_layer_share','realized_layer_share'):
        result.update(prefixed(data[name], name, full=True))
    for part in ('self_direction','cross_direction'):
        result.update(prefixed(data[part]['gamma'], part+'_gamma', full=True))
    norm2 = [x*x for x in data['self_direction']['planned_norm']]
    denominator = sum(norm2)
    result['gamma_D_energy_weighted'] = (sum(g*n for g,n in zip(data['gamma'],norm2) if g is not None)/denominator
                                         if denominator else None)
    result['Y_to_D_F_norm_ratio'] = math.sqrt(sum(x*x for x in data['Y_norm'])/denominator) if denominator else None
    result['gamma_negative_count'] = sum(x is not None and x < 0 for x in data['gamma'])
    result['gamma_over_one_count'] = sum(x is not None and x > 1 for x in data['gamma'])
    for name in ('M_offdiagonal_F','actual_write_norm','kappa_entry_change',
                 'Y64_vs_materialized_weight_RMS','materialized_weight_vs_Flinear_RMS'):
        result[name] = data[name]
    return result


def decomposition_rows(data, arm, batch, B, n_rw):
    groups = defaultdict(list)
    max_identity = defaultdict(float)
    for chunk in data['decomposition']:
        for layer, d in chunk.items():
            max_identity[layer] = max(max_identity[layer], d['sum_error_max'])
            for i, (request, context) in enumerate(zip(d['request'], d['context_index'])):
                row = dict(request=request, context=context, canonical=d['canonical'][i],
                           norms=d['norms'][i], dots=d['cross_dots'][i],
                           virtual_pre_gap=d['virtual_pre_gap_norm'][i], actual_gap=d['virtual_actual_gap'][i])
                for route in ('ideal_direction','FP32_direction'):
                    row[route] = {name:d[route][name][i] for name in ('gamma','norm_ratio','fit_relative')}
                groups[layer].append(row)
    result = []
    for layer, rows in groups.items():
        require(len(rows) == B*n_rw and len({(x['request'],x['context']) for x in rows}) == B*n_rw,
                'TERMINAL_CONTEXT_COVERAGE')
        require({(x['request'],x['context']) for x in rows} == {(i,j) for i in range(B) for j in range(n_rw)},
                'TERMINAL_CONTEXT_ORDER')
        for scope, subset in (('all_native', rows), ('canonical', [x for x in rows if x['canonical']]),
                              ('augmented', [x for x in rows if not x['canonical']])):
            r = dict(arm=arm, batch=batch, layer=layer, scope=scope, n_context_rows=len(subset),
                     sum_identity_component_abs_max=max_identity[layer])
            for i,name in enumerate(('inherited_gap','mean_key_error','context_key_difference')):
                r.update(prefixed([x['norms'][i] for x in subset], name, full=True))
                r[name+'_mean_square'] = sum(x['norms'][i]**2 for x in subset)/len(subset)
            for i,name in enumerate(('inherited_mean_dot','inherited_context_dot','mean_context_dot')):
                r.update(prefixed([x['dots'][i] for x in subset], name, full=True))
            square = [sum(n*n for n in x['norms'])+2*sum(x['dots']) for x in subset]
            require(min(square) >= -1e-12, 'DECOMPOSITION_NEGATIVE_SQUARE')
            r['sum_parts_ideal_gap_RMSnorm'] = math.sqrt(sum(max(0,x) for x in square)/len(square))
            r.update(prefixed([x['actual_gap'] for x in subset], 'virtual_actual_gap', full=True))
            for route in ('ideal_direction','FP32_direction'):
                for field in ('gamma','norm_ratio','fit_relative'):
                    r.update(prefixed([x[route][field] for x in subset], route+'_'+field))
            result.append(r)
    return result


def bounded_tensor_check(path, layer_data, expected):
    import numpy as np
    rec = file_receipt(path)
    require(rec['sha256'] == expected['sha256'], 'SENTINEL_TENSOR_HASH')
    with np.load(path, allow_pickle=False) as z:
        require(set(z.files) == {'M','planned_D','realized_Y'}, 'TENSOR_ALLOWLIST')
        d, m, y = z['planned_D'].astype('float64'), z['M'], z['realized_Y']
        require(np.allclose(d@m,y,atol=1e-9,rtol=1e-8), 'SENTINEL_Y')
        n2 = (d*d).sum(0)
        gamma = (d*y).sum(0)/n2
        error = float(np.max(np.abs(gamma-np.asarray(layer_data['gamma']))))
        require(np.allclose(gamma,layer_data['gamma'],atol=1e-9,rtol=1e-8), 'SENTINEL_GAMMA')
        require(np.allclose(np.diag(m),layer_data['M_diagonal'],atol=1e-12,rtol=1e-10), 'SENTINEL_M')
        rec.update(Y_check=True, gamma_abs_error_max=error, M_diagonal_check=True)
    return rec


def run(attempt, out):
    attempt, out = attempt.resolve(), out.resolve()
    require(not out.exists(), 'OUTPUT_CREATE_ONCE')
    reader = Reader(attempt)
    tables = defaultdict(list)
    terminal_rows, sentinels, inputs_inventory = {}, [], []
    lock = reader.read(attempt/'execution.lock.json')
    for arm in ('A','B'):
        lane = attempt/('main-'+arm)
        runtime = reader.read(lane/'runtime.json')
        tables['hardware'].append(dict(arm=arm, job_id=runtime['job_id'], device=runtime['device'],
            torch=runtime['versions']['torch'], transformers=runtime['versions']['transformers'],
            GPU_UUID='NOT_RECORDED', per_GPU_clocks_utilization='NOT_RECORDED'))
        for batch in range(1,6):
            fit = lane/f'batch-{batch:02d}'/'fit'
            commit = reader.read(fit.parent/'commit.json')
            B = commit['actual_B']
            require(commit['source'] == lock['source_commit'] and B == 100, 'COMMIT_SOURCE_INPUT')
            candidates = sorted(fit.glob('candidate-*.json'))
            require(len(candidates) == 25, 'CANDIDATE_COUNT')
            first_clip = {str(l):None for l in range(4,9)}
            pulse_ids = []
            for k, path in enumerate(candidates,1):
                d = reader.read(path)
                require(d['candidate'] == k and d['Adam_updates_after'] == min(k,24), 'BUDGET')
                require(d['gradient_measured'] == (k < 25), 'TERMINAL_GRADIENT')
                require(d['past_rows'] == d['past_forward_count'] == d['past_loss'] == 0, 'REPLAY')
                native_mean = d['native_nll_mean']+.0625*d['native_kl_mean']+d['native_norm_mean']
                loss = dict(arm=arm,batch=batch,candidate=k,updates_before=k-1,gradient_measured=k<25,
                    native_nll_mean=d['native_nll_mean'],native_kl_raw_mean=d['native_kl_mean'],
                    native_kl_weighted_mean=.0625*d['native_kl_mean'],native_norm_weighted_mean=d['native_norm_mean'],
                    native_total_mean=native_mean,policy_weighted_mean=d['policy_mean'],
                    policy_to_native_loss_ratio=d['policy_mean']/native_mean if native_mean else None,
                    component_gradient_norm='NOT_RECORDED',native_seconds=d['native_seconds'],
                    builder_seconds=d['builder_seconds'],candidate_seconds_inclusive=d['seconds'],
                    terminal_actual_seconds=d['terminal_actual_seconds'])
                aux = d['auxiliary']
                require(bool(aux) == (k in (5,10,15,20)), 'PULSE_SCHEDULE')
                loss.update(pulse_current_count=len(aux['current']) if aux else 0,
                            pulse_loss_sum=aux['loss_sum'] if aux else 0,
                            pulse_seconds=aux['seconds'] if aux else None)
                for name,coefficient in (('subject',.1),('distillation',.1),('current_kl',.0625)):
                    loss['pulse_'+name+'_weighted_mean'] = coefficient*aux['stats'][name]/B if aux else 0
                pulse_mean = sum(loss['pulse_'+name+'_weighted_mean'] for name in ('subject','distillation','current_kl'))
                require(math.isclose(pulse_mean,loss['pulse_loss_sum']/B,abs_tol=1e-6,rel_tol=1e-6), 'PULSE_REDUCTION')
                loss['pulse_weighted_mean'] = pulse_mean
                loss['objective_mean'] = native_mean+d['policy_mean']+pulse_mean
                loss['objective_SUM'] = B*loss['objective_mean']
                if aux:
                    pulse_ids.extend(aux['current'])
                c = [e['c'] for e in d['energy'].values()]
                policy = .1*(sum(c) if arm=='A' else math.sqrt(sum(x*x for x in c)))
                require(math.isclose(policy,d['policy_mean'],abs_tol=1e-12,rel_tol=1e-10), 'POLICY_REDUCTION')
                loss['total_D_gradient_F'] = math.sqrt(sum(x['D']**2 for x in d['gradient_norm'].values())) if k<25 else None
                loss['total_q_gradient_F'] = math.sqrt(sum(x['q']**2 for x in d['gradient_norm'].values())) if k<25 else None
                tables['candidate-objectives'].append(loss)
                for j in range(B):
                    for share in ('planned_layer_share','realized_layer_share'):
                        values = [v[share][j] for v in d['layer'].values()]
                        require(all(v is None for v in values) or
                                (all(v is not None for v in values) and abs(sum(values)-1)<1e-10), 'SHARE_SUM')
                for layer, v in d['layer'].items():
                    require(all(len(v[name]) == B for name in ('rho','gamma','M_diagonal','planned_layer_share')), 'REQUEST_DENOMINATOR')
                    t = dict(arm=arm,batch=batch,candidate=k,layer=layer,gradient_measured=k<25,
                             gamma_valid=sum(x is not None for x in v['gamma']),zero_D=sum(v['zero_D']))
                    for name in ('rho','realized_rho','gamma','norm_ratio','M_diagonal','planned_layer_share','realized_layer_share'):
                        t.update(prefixed(v[name],name))
                    t.update(d['energy'][layer]);t.pop('roundoff')
                    t.update(actual_write_norm=v['actual_write_norm'],kappa_entry_change=v['kappa_entry_change'],
                             D_gradient_F=d['gradient_norm'][layer]['D'] if k<25 else None,
                             q_gradient_F=d['gradient_norm'][layer]['q'] if k<25 else None)
                    tables['candidate-layer-trajectory'].append(t)
                    if k<25:
                        cl = d['post_update_clamp'][layer]
                        require(cl['clipped'] == sum(x<1 for x in cl['radial_scale']), 'CLAMP_COUNT')
                        require(max(cl['post_rho']) < .75001, 'STORED_CLAMP_RADIUS')
                        if cl['clipped'] and first_clip[layer] is None:
                            first_clip[layer] = k
                        row = dict(arm=arm,batch=batch,update=k,layer=layer,groups=B,
                                   clipped=cl['clipped'],clipped_fraction=cl['clipped']/B)
                        for name in ('pre_rho','post_rho','radial_scale'):
                            row.update(prefixed(cl[name],name))
                        for name in ('projection_discarded_norm','proposal_radial'):
                            row.update(prefixed(d[name][layer],name))
                        row.update(prefixed(d['gradient_norm'][layer]['radial'],'total_gradient_radial'))
                        tables['clamp-trajectory'].append(row)
                if k==25:
                    terminal_rows[arm,batch] = d
                    for layer in d['layer']:
                        t = terminal_layer(d,arm,batch,layer)
                        t['first_clipped_update'] = first_clip[layer]
                        tables['terminal-layer-realization'].append(t)
                    # 사전 지정한 4개 terminal tensor만 CPU 독립 대조; 전체 재해시 없음.
                    if batch in (1,5):
                        layer = '4' if batch==1 else '8'
                        npz = fit/'diagnostics'/f'candidate-25-L{layer}.npz'
                        sentinels.append(bounded_tensor_check(npz,d['layer'][layer],reader.index[str(npz)]))
            require(sorted(pulse_ids) == list(range(B)), 'PULSE_PARTITION_COVERAGE')
            terminal = reader.read(fit/'terminal-actual.json')
            n_rw = len(terminal['context_nll'][0])
            tables['terminal-context-decomposition'].extend(decomposition_rows(terminal,arm,batch,B,n_rw))
            means = dict(arm=arm,batch=batch,n_requests=B,n_contexts=B*n_rw)
            for name in ('context_nll','virtual_context_nll','actual_target_kl_from_virtual'):
                means.update(prefixed([x for row in terminal[name] for x in row],name,full=True))
            for name in ('native_kl','virtual_kl'):
                means.update(prefixed(terminal[name],name,full=True))
            tables['terminal-native-observations'].append(means)
            manifest = reader.read(fit/'diagnostic-manifest.json')
            for item in manifest:
                st = Path(item['path']).stat()
                require(st.st_size == item['bytes'] and st.st_mtime_ns == item['mtime_ns'], 'TENSOR_STAT_CHANGED')
                inputs_inventory.append(dict(path=item['path'],bytes=item['bytes'],sha256=item['sha256'],
                    hash_evidence='SEALED_COLLECTOR_AND_RUNTIME_PRIOR_SHA_CURRENT_STAT',
                    purpose=item['purpose'],tensors=item['tensors']))
    q2 = reader.read(attempt/'main-A/batch-01/Q2/frozen-ridge-K-operators.json')
    for layer, data in q2.items():
        for writer in ('ridge_operator','exact_operator'):
            v = data[writer]
            row = dict(layer=layer,writer=writer,geometry='same_frozen_ridge_K',
                       preservation_energy=v['preservation_energy'],update_norm=v['update_norm'],M_offdiagonal_F=v['M_offdiagonal_F'])
            for name in ('gamma','norm_ratio','fit_relative','orthogonal_relative'):
                row.update(prefixed(v['mean'][name],name,full=True))
            row.update(prefixed(v['M_diagonal'],'M_diagonal',full=True))
            for part in ('self_action','cross_action'):
                row.update(prefixed(v[part]['gamma'],part+'_gamma',full=True))
            tables['Q2-frozen-operator'].append(row)
    for batch in range(1,6):
        a,b = terminal_rows['A',batch],terminal_rows['B',batch]
        for layer in a['layer']:
            row = dict(batch=batch,layer=layer,alignment='same offered request position; own trajectories')
            for name in ('gamma','rho','planned_layer_share','realized_layer_share','M_diagonal'):
                diffs = [x-y for x,y in zip(a['layer'][layer][name],b['layer'][layer][name])]
                row[name+'_A_minus_B_mean'] = sum(diffs)/len(diffs)
                row[name+'_A_minus_B_abs_max'] = max(abs(x) for x in diffs)
            tables['AB-terminal-differences'].append(row)
    expected = {'candidate-objectives':250,'candidate-layer-trajectory':1250,'clamp-trajectory':1200,
                'terminal-layer-realization':50,'terminal-context-decomposition':150,
                'terminal-native-observations':10,'Q2-frozen-operator':10,'AB-terminal-differences':25,'hardware':2}
    require({k:len(v) for k,v in tables.items()} == expected, 'OUTPUT_COMPLETENESS')
    out.mkdir(parents=True)
    for name,rows in tables.items():
        write_csv(out/(name+'.csv'),rows)
    write_json(out/'input-manifest.json',dict(runtime_source=lock['source_commit'],
        analysis_source=file_receipt(Path(__file__)),read_scalar_inputs=list(reader.evidence.values()),
        local_tensor_inventory=inputs_inventory,bounded_tensor_rechecks=sentinels,
        raw_and_tensor_in_Git=False,model_CP_rehash=False,new_model_forward=0,new_Slurm_calls=0))
    write_json(out/'validation.json',dict(status='PASS_BOUNDED_OWNER_CPU_REDUCTION',table_rows=expected,
        tensor_sentinels=4,component_gradients='NOT_RECORDED_NOT_RECONSTRUCTED',
        terminal_gradient='NOT_MEASURED_NULL',quantile='linear interpolation at (n-1)*p',
        null_policy='zero-D ratios/shares remain null; finite negative or >1 gamma retained',
        count_semantics='request-layer/candidate counts are not independent experiment repeats',
        decomposition='signed cross dots retained; norms are not additive causal contribution fractions',
        loss_units='reported mean plus whole-B SUM; policy already includes lambda .1; KL coefficient .0625',
        independent_reviewer=False,monitoring_active=False,automatic_resume=False))
    print(json.dumps(dict(status='PASS',table_rows=expected,out=str(out))))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--attempt',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    run(a.attempt,a.out)
