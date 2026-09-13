"""Read-only CPU audit of the six completed sequential outputs. No model imports."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import stat
import torch

POLICIES = {'N4': (1., None), 'RES8': (.75, 8), 'S875': (.875, None),
            'S75': (.75, None), 'FULL8': (1., 8), 'REFIT4': (.75, 4)}

def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def tensor_sha(t, header=True):
    t = t.detach().cpu().contiguous()
    h = hashlib.sha256(str((str(t.dtype), list(t.shape))).encode() if header else b'')
    raw = t.reshape(-1).view(torch.uint8).numpy()
    for i in range(0, raw.size, 8 << 20):
        h.update(memoryview(raw[i:i+(8 << 20)]))
    return h.hexdigest()

def norm(t, other=None):
    total = 0.
    a = t.reshape(-1)
    b = other.reshape(-1) if other is not None else None
    for i in range(0, a.numel(), 1 << 20):
        x = a[i:i+(1 << 20)].double()
        if b is not None:
            x -= b[i:i+(1 << 20)].double()
        total += x.square().sum().item()
    return total ** .5

def tensor_inventory(x, key=''):
    out = []
    if isinstance(x, torch.Tensor):
        finite = all(torch.isfinite(v).all().item() for v in x.reshape(-1).split(1 << 20))
        assert finite, ('NONFINITE_TENSOR', key)
        out.append(dict(key=key, shape=list(x.shape), dtype=str(x.dtype),
                        sha256=tensor_sha(x), finite=True, numel=x.numel()))
    elif isinstance(x, dict):
        for k, v in x.items():
            out.extend(tensor_inventory(v, f'{key}.{k}' if key else str(k)))
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            out.extend(tensor_inventory(v, f'{key}[{i}]'))
    return out

def file_member(p):
    a = p.lstat()
    assert stat.S_ISREG(a.st_mode) and not p.is_symlink(), ('FILE_TYPE', str(p))
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    b = p.stat()
    assert (a.st_dev, a.st_ino, a.st_size, a.st_mtime_ns) == (b.st_dev, b.st_ino, b.st_size, b.st_mtime_ns), ('UNSTABLE_FILE', str(p))
    return dict(path=str(p), bytes=a.st_size, sha256=h.hexdigest(), stable_stat=True)

def load(p):
    return json.loads(p.read_text())

def csv_write(p, rows):
    with p.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def validate_commit(c, entry, previous, alpha, second):
    assert c['entry'] == entry['state'] == previous
    assert entry['previous_commit_exact'] and c['evaluation_nonmutation']
    assert c['materialization']['alpha'] == alpha
    assert c['first_fit']['layer'] == 4
    fits = [c['first_fit']] + ([c['second_fit']] if second is not None else [])
    if second is None:
        assert c['second_fit'] is None
    else:
        assert c['second_fit']['layer'] == second
        assert c['second_fit']['actual_partial_state'] == c['partial_state']
    for fit in fits:
        assert fit['compute_z'] == 100 and fit['solve'] == 1 and fit['history_append'] == 0
        assert fit['target_mode'] == 'same-host-fresh-current-state'
        assert fit['source']['optimizer_or_solve_equation_changes'] == 0
    for k in ('M4', 'M8', 'P4', 'P8', 'contexts'):
        assert c['partial_state'][k] == previous[k]
    layers = [4, 8] if second == 8 else [4]
    assert [r['layer'] for r in c['history']] == layers
    assert all(r['history_append'] == 1 for r in c['history'])
    return fits, layers

def capture_and_raw_edges(attempt):
    """Check stored capture cardinalities and raw-byte fit/finalizer chains."""
    counts = {'compute_z': 0, 'compute_ks': 0, 'get_module_input_output_at_words': 0}
    lock = load(attempt/'execution.lock.json')
    batch_locks = {r['batch']:r for r in lock['batch_locks']}
    prepared = torch.load(lock['prepared']['path'],map_location='cpu',weights_only=True,mmap=True)
    for cell, (arm, (_, second)) in enumerate(POLICIES.items()):
        previous = None
        common = load(attempt/'output'/f'cell-{cell}'/'terminal.json')['entry_state']
        for batch in range(51,61):
            d = attempt/'output'/f'cell-{cell}'/f'B{batch:03d}'
            c = load(d/'commit.json')
            assert load(d/'entry.json')['order'] == batch_locks[batch]
            if second != 8:
                assert c['endpoint']['M8'] == common['M8']
                assert c['endpoint']['weights']['8'] == common['weights']['8']
            if previous is not None:
                assert c['first_fit']['entry_weight_sha256'] == previous[4]['weight_sha256']
                assert c['first_fit']['history_sha256'] == previous[4]['after_sha256']
            finals = {r['layer']: r for r in c['history']}
            assert finals[4]['before_sha256'] == c['first_fit']['history_sha256']
            if c['materialization']['alpha'] == 1.:
                assert c['materialization']['weight_sha256'] == c['first_fit']['endpoint_weight_sha256']
            if second is not None:
                fit = c['second_fit']
                assert fit['history_sha256'] == finals[second]['before_sha256']
                if second == 4:
                    assert fit['entry_weight_sha256'] == c['materialization']['weight_sha256']
                elif previous is not None:
                    assert fit['entry_weight_sha256'] == previous[8]['weight_sha256']
                    assert fit['history_sha256'] == previous[8]['after_sha256']
                assert fit['endpoint_weight_sha256'] == finals[second]['weight_sha256']
            if second != 4:
                assert c['materialization']['weight_sha256'] == finals[4]['weight_sha256']
            if batch in (51,55,60):
                cp = torch.load(d/'checkpoint.pt',map_location='cpu',weights_only=True,mmap=True)
                for layer in cp['weights']:
                    assert cp['weights'][layer].shape == prepared['weights'][layer].shape
                    assert cp[f'M{layer}'].shape == prepared[f'M{layer}'].shape
                maps = cp['metadata']['P_mapping']
                assert [(p['physical_layer'],p['source_index'],p['local_index']) for p in maps] == [(4,0,0),(8,4,0)]
                assert all(p['source_tensor_sha256'] == p['selected_tensor_sha256'] for p in maps)
                assert maps[0]['selected_tensor_sha256'] == c['first_fit']['projector_sha256']
                if second == 8:
                    assert maps[1]['selected_tensor_sha256'] == c['second_fit']['projector_sha256']
                if batch == 51:
                    inc = torch.load(d/'actual-increments.pt',map_location='cpu',weights_only=True,mmap=True)
                    for layer, delta in inc['deltas'].items():
                        a, b, v = cp['weights'][layer].reshape(-1), prepared['weights'][layer].reshape(-1), delta.reshape(-1)
                        for i in range(0,a.numel(),1 << 20):
                            assert torch.equal(a[i:i+(1 << 20)]-b[i:i+(1 << 20)],v[i:i+(1 << 20)])
                    del inc
                del cp
            for prefix in (['first','second'] if second is not None else ['first']):
                x = torch.load(d/f'{prefix}-target-key-readout.pt',map_location='cpu',weights_only=True,mmap=True)
                for key, expected in [('compute_z',100),('compute_ks',1),('get_module_input_output_at_words',1)]:
                    assert len(x[key]) == expected
                    counts[key] += len(x[key])
                assert all(t.dtype == torch.float32 for values in x.values() for t in values)
                del x
            previous = finals
    assert counts == {'compute_z':9000,'compute_ks':90,'get_module_input_output_at_words':90}
    return dict(status='PASS',captured_calls=counts,raw_fit_finalizer_chain_edges='PASS',
                B51_actual_increment_vs_checkpoint_entry='FP32_TORCH_EQUAL',
                fresh_target_forward_reexecution='NOT_TESTED')

def run(attempt, package):
    torch.set_num_threads(1)
    package.mkdir(parents=True, exist_ok=True)
    root = attempt / 'output'
    lock = load(attempt / 'execution.lock.json')
    common = lock['common_state']
    capture_edges = capture_and_raw_edges(attempt)
    prepared = torch.load(lock['prepared']['path'], map_location='cpu', weights_only=True, mmap=True)
    members, cps, states, actions = [], [], [], []
    z = solves = links = history = 0
    orders = {}
    references = {}
    for cell, (arm, (alpha, second)) in enumerate(POLICIES.items()):
        folder = root / f'cell-{cell}'
        term = load(folder / 'terminal.json')
        assert term['status'] == 'TEN_SEQUENTIAL_BATCHES_COMPLETE' and term['arm'] == arm
        assert not (folder / 'failure.json').exists()
        assert term['entry_state'] == common and term['source_lock_sha256'] == file_member(attempt / 'execution.lock.json')['sha256']
        previous = common
        path_work = {4: 0., 8: 0.}
        for batch in range(51, 61):
            d = folder / f'B{batch:03d}'
            c, entry = load(d / 'commit.json'), load(d / 'entry.json')
            fits, layers = validate_commit(c, entry, previous, alpha, second)
            assert c['arm'] == arm and c['batch'] == batch
            if batch in orders:
                assert entry['order'] == orders[batch]
            else:
                orders[batch] = entry['order']
            assert entry['order']['ordinal_start'] == (batch - 1) * 100
            assert entry['order']['ordinal_end'] == batch * 100
            assert len(set(entry['order']['case_ids'])) == 100
            assert entry['order']['request_order_sha256'] == digest(entry['order']['case_ids'])
            assert c['history_counts'] == {'4': batch-50, '8': batch-50 if second == 8 else 0}
            ev = load(d / 'evaluation.json')
            assert ev['endpoint_state'] == c['endpoint'] and ev['evaluation_nonmutation']
            for key in ('checkpoint', 'increments', 'evaluation'):
                if c[key]:
                    references[c[key]['path']] = c[key]
            increments = torch.load(d / 'actual-increments.pt', map_location='cpu', weights_only=True, mmap=True)
            assert increments['entry'] == previous and increments['endpoint'] == c['endpoint']
            assert set(increments['deltas']) == set(layers)
            for layer, delta in increments['deltas'].items():
                assert delta.dtype == torch.float32 and list(delta.shape) == [4096, 14336]
                n = norm(delta)
                path_work[layer] += n
                actions.append(dict(arm=arm, batch=batch, layer=layer,
                                    incremental_frobenius=n, cumulative_path_frobenius_sum=path_work[layer],
                                    checkpoint_net_from_W50='NOT_CHECKPOINT_BATCH', M_frobenius='NOT_CHECKPOINT_BATCH',
                                    M_delta_from_entry='NOT_CHECKPOINT_BATCH'))
            del increments
            if c['checkpoint']:
                cp = torch.load(d / 'checkpoint.pt', map_location='cpu', weights_only=True, mmap=True)
                assert set(cp['weights']) == set(layers)
                assert cp['metadata']['state'] == c['endpoint']
                assert {str(k):v for k,v in cp['metadata']['history_counts'].items()} == c['history_counts']
                assert cp['metadata']['batch'] == batch and cp['metadata']['next_batch'] == batch + 1
                assert len(cp['metadata']['seen_ids']) == batch * 100
                assert cp['metadata']['source_lock_sha256'] == term['source_lock_sha256']
                assert cp['metadata']['policy'] == dict(alpha=alpha, second_layer=second)
                assert digest(cp['contexts']) == c['endpoint']['contexts']
                assert digest(cp['rng']) == c['endpoint']['rng']
                for layer in layers:
                    w, m = cp['weights'][layer], cp[f'M{layer}']
                    assert w.dtype == m.dtype == torch.float32
                    assert tensor_sha(w) == c['endpoint']['weights'][str(layer)]
                    assert tensor_sha(m) == c['endpoint'][f'M{layer}']
                    assert tensor_sha(w, False) == next(r for r in c['history'] if r['layer'] == layer)['weight_sha256']
                    assert tensor_sha(m, False) == next(r for r in c['history'] if r['layer'] == layer)['after_sha256']
                    row = next(r for r in reversed(actions) if r['arm'] == arm and r['batch'] == batch and r['layer'] == layer)
                    row.update(checkpoint_net_from_W50=norm(w, prepared['weights'][layer]),
                               M_frobenius=norm(m), M_delta_from_entry=norm(m, prepared[f'M{layer}']))
                info = tensor_inventory(cp)
                cps.append(dict(arm=arm, batch=batch, path=str(d/'checkpoint.pt'),
                                selected_layers=','.join(map(str,layers)), tensor_count=len(info),
                                tensors_json=json.dumps(info, sort_keys=True),
                                contexts_rng_endpoint_exact=True, gpu_continuation_replay='NOT_TESTED'))
                del cp
            states.append(dict(arm=arm,batch=batch,alpha=alpha,second_layer=second,
                               prior_commit_entry_exact=True, same_order=True, fresh_fit_counters=True,
                               fit_history_append=0, final_history_appends=len(layers),
                               evaluation_nonmutation_recorded=True, request_z=sum(f['compute_z'] for f in fits),
                               solve=sum(f['solve'] for f in fits), entry_sha=digest(previous), endpoint_sha=digest(c['endpoint'])))
            z += sum(f['compute_z'] for f in fits)
            solves += sum(f['solve'] for f in fits)
            history += len(layers)
            links += int(batch > 51)
            previous = c['endpoint']
        assert previous == term['terminal_state']
        assert term['request_z_total'] == (2000 if second is not None else 1000)
        assert term['fit_solve_total'] == (20 if second is not None else 10)
        assert term['M8_reconstruct_executions'] == 0 and not term['audit_executed']
        restore = load(folder/'process-restore.json')
        assert restore['selected_W0_exact'] and restore['RNG_restored']
        for r in term['commits']:
            references[r['path']] = r
        for p in sorted(folder.rglob('*')):
            if not p.is_file():
                continue
            member = file_member(p)
            member['preexisting_runtime_reference'] = str(p) in references
            if str(p) in references:
                ref = references[str(p)]
                assert (member['bytes'], member['sha256']) == (ref['bytes'], ref['sha256'])
            if p.suffix == '.pt':
                obj = torch.load(p, map_location='cpu', weights_only=True, mmap=True)
                tensors = tensor_inventory(obj)
                member['tensor_count'] = len(tensors)
                member['tensor_inventory_sha256'] = digest(tensors)
                member['tensor_finite'] = True
                del obj
            members.append(member)
        print('STATE_AUDIT_CELL_COMPLETE', cell, arm, flush=True)
    assert (len(states),len(cps),links,z,solves,history) == (60,18,54,9000,90,80)
    csv_write(package/'state-history.csv',states)
    csv_write(package/'checkpoint-inventory.csv',cps)
    csv_write(package/'layer-action.csv',actions)
    (package/'raw-member-inventory.json').write_text(json.dumps(dict(members=members,count=len(members),bytes=sum(r['bytes'] for r in members)),indent=2)+'\n')
    summary = dict(status='STORED_STATE_AND_CPU_TENSOR_CHECKS_PASS',batches=60,checkpoints=18,links=54,
                   request_z=z,fit_solve=solves,history_append=history,L4_history_append=60,L8_history_append=20,
                   full_sha_members=len(members),bytes=sum(r['bytes'] for r in members),
                   full_sha_tensor_finite=True, GPU_continuation_replay='NOT_TESTED',
                   model_level_observer_off_on_parity='NOT_TESTED',incremental_exact_replay='NOT_TESTED',
                   state_nonmutation='RECORDED_RUNTIME_GUARDS_AND_CPU_BOUND_SNAPSHOTS_NOT_NEW_GPU_TEST',
                   capture_freshness='PER_BATCH_DISTINCT_CAPTURE_PATH_AND_SOURCE_COUNTER_BOUND_NOT_NEW_FORWARD',
                   intermediate_alpha_reconstruction='SOURCE_AND_RECORDED_MATERIALIZATION_BOUND_NOT_NEW_NATIVE_CANDIDATE_RECOMPUTATION',
                   nonselected_preservation='RUNTIME_TERMINAL_FULL_PARAMETER_GUARD_NOT_NEW_FULL_MODEL_LOAD')
    summary['capture_and_raw_edges'] = capture_edges
    (package/'state-verification.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--attempt',type=Path,required=True)
    p.add_argument('--package',type=Path,required=True)
    p.add_argument('--capture-supplement-only',action='store_true',help='Recheck lightweight capture/P-map edges and bind to existing full audit.')
    args = p.parse_args()
    if args.capture_supplement_only:
        torch.set_num_threads(1)
        target = args.package/'state-verification.json'
        summary = load(target)
        summary['capture_and_raw_edges'] = capture_and_raw_edges(args.attempt)
        summary['intermediate_alpha_reconstruction'] = 'SOURCE_AND_RECORDED_MATERIALIZATION_BOUND_NOT_NEW_NATIVE_CANDIDATE_RECOMPUTATION'
        inventory_path = args.package/'raw-member-inventory.json'
        inventory = load(inventory_path)
        refs = set()
        for cell in range(6):
            folder = args.attempt/'output'/f'cell-{cell}'
            refs.update(r['path'] for r in load(folder/'terminal.json')['commits'])
            for batch in range(51,61):
                c = load(folder/f'B{batch:03d}'/'commit.json')
                refs.update(c[k]['path'] for k in ('checkpoint','increments','evaluation') if c[k])
        for member in inventory['members']:
            member['preexisting_runtime_reference'] = member['path'] in refs
        inventory_path.write_text(json.dumps(inventory,indent=2)+'\n')
        target.write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(summary,indent=2))
    else:
        print(json.dumps(run(args.attempt,args.package),indent=2))
