"""CPU file/hash/stat review only. Never imports model or execution modules."""
import argparse, csv, hashlib, json
from pathlib import Path

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--attempt', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    lock_path, config_path = a.attempt/'execution.lock.json', a.attempt/'config.json'
    assert sha(lock_path) == 'a399d80257c38182cb67498d4aef8be12729f0e8fffe118f8d7f1e8b2701351c'
    assert sha(config_path) == 'b2745b667e011f2a27e81b5f5f275cb731b9f418342b20cf741855c2149fd429'
    lock, c = json.loads(lock_path.read_text()), json.loads(config_path.read_text())
    assert lock['source_commit'] == '2ff0ecc63dda2c830fbd68e6a2d76f3317a446f7'
    assert lock['host'] == 'server4' and lock['owner'] == 'janghj'
    groups = {k: lock[k] for k in ('source_members', 'runtime_sources', 'dependency_sources', 'native_reference', 'launchers')}
    groups['archive_hparams'] = [lock['archive'], lock['native_hparams']]
    groups['input_and_prior_receipts'] = (c['authority_members'] + c['qualification_reuse']['receipts']
        + [c['observer_identity'], c['native_input_alignment'], c['cpu_preflight']])
    for arm in ('MD','CD'):
        groups['actual_imports_'+arm] = json.loads((a.attempt/('main-'+arm)/'actual-imports.json').read_text())['files']
    checked = []
    for kind, records in groups.items():
        for r in records:
            path = Path(r['path'])
            assert path.stat().st_size == r['bytes'], (kind, str(path), 'size')
            assert sha(path) == r['sha256'], (kind, str(path), 'SHA')
            checked.append({'group':kind, 'path':str(path), 'bytes':r['bytes'], 'sha256':r['sha256']})
    stats = []
    for r in c['assets']:
        s = Path(r['path']).stat()
        actual = (s.st_size, s.st_ino, s.st_mtime_ns)
        expected = (r['bytes'], r['inode'], r['mtime_ns'])
        assert actual == expected, (r['path'], 'asset_stat_changed')
        stats.append({'path':r['path'],'bytes':r['bytes'],'prior_full_sha256':r['sha256'],
            'fresh_stat_matches':True,'fresh_full_hash':False})
    records = json.loads(Path(c['stream']).read_text())[:2000]
    ids = [r['case_id'] for r in records]
    id_sha = hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
    assert id_sha == c['ordered_ids_sha256'] == '0b912d11659eb087ee71a391b7bea1e02ecc9d999a48254eb8559434965640f4'
    assert len(c['packs']) == 20 and [i for p in c['packs'] for i in p['ids']] == ids
    pack_counts = [len(p['ids']) for p in c['packs']]
    assert pack_counts == [100]*20
    observer = json.loads(Path(c['observer_identity']['path']).read_text())
    assert len(observer['rows']) == 26000
    runtime = {}
    history_lists = {}
    for arm, job in (('MD','58442'),('CD','58443')):
        r = json.loads((a.attempt/('main-'+arm)/'runtime.json').read_text())
        assert r['job'] == job and r['arm'] == arm and r['source'] == lock['source_commit']
        assert r['cold_W0_H0'] == c['qualification_reuse']['cold_W0_H0']
        assert r['torch'] == c['runtime']['torch'] and r['transformers'] == c['runtime']['transformers']
        assert r['FP32'] and r['geometry_FP64'] and r['eager'] and not r['TF32'] and not r['autocast']
        runtime[arm] = r
        terminal = json.loads((a.attempt/('main-'+arm)/'terminal.json').read_text())
        assert terminal['source'] == lock['source_commit'] and terminal['job'] == job and terminal['arm'] == arm
        assert terminal['status'] == 'W20_COMPLETE' and terminal['commits'] == 20
        assert terminal['checkpoint_saved'] is False and terminal['no_B21'] is True
        lengths, appends = [], 0
        for b in range(1,21):
            folder = a.attempt/('main-'+arm)/f'batch-{b:02d}'
            commit = json.loads((folder/'commit.json').read_text())
            wr = commit['writer']
            assert sha(wr['path']) == wr['sha256'] and Path(wr['path']).stat().st_size == wr['bytes']
            writer = json.loads(Path(wr['path']).read_text())
            history = writer['history']
            layers = [item['layer'] for item in history]
            assert len(history) == len(c['profile']['eligible_layers']) == len(set(layers))
            assert set(layers) == set(c['profile']['eligible_layers'])
            assert all(item['append_count'] == 1 for item in history)
            lengths.append(len(history))
            appends += sum(item['append_count'] for item in history)
        assert appends == 100
        history_lists[arm] = {'lists':20,'lengths':lengths,'duplicates':0,'actual_append_count_sum':appends}
    collector_terminal = json.loads((a.attempt/'cpu-report/terminal.json').read_text())
    assert collector_terminal['status'] == 'COMPLETED'
    for key in ('inventory','report','summary'):
        item = collector_terminal[key]
        assert sha(item['path']) == item['sha256'] and Path(item['path']).stat().st_size == item['bytes']
    result = dict(schema=1,kind='CPU_SOURCE_INPUT_RUNTIME_BINDING_NOT_TENSOR_REPLAY',passed=True,
        execution_source=lock['source_commit'],source_tree=lock['source_tree'],
        lock_sha256=sha(lock_path),config_file_sha256=sha(config_path),
        config_decoded_digest=runtime['MD']['config'], groups={k:len(v) for k,v in groups.items()},
        fresh_hash_records=checked,prior_full_SHA_fresh_stat_assets=stats,
        ordered2000_sha256=id_sha,packs=20,pack_counts=pack_counts,observer_identity_rows=26000,
        runtime=runtime,exact_history_list_cardinality=history_lists,
        collector_terminal_targets_SHA_verified=True,new_GPU=0,model_load=0,forward=0,tokenizer_reexecution=0)
    assert not a.out.exists(), 'CREATE_ONCE_REVIEW_BINDING'
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('fresh_hash_records','prior_full_SHA_fresh_stat_assets','runtime')}))

if __name__ == '__main__':
    main()
