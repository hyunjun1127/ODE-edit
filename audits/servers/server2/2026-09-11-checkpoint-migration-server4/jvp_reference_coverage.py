"""JVP source-pinned snapshots의 모든 index shard와 YAML 복원 참조 검산."""
import json
from pathlib import Path
from verify_initial import CONTROL, ARCHIVE, REPO, save, sha, stable_verify


def main():
    refs_path = CONTROL / 'jvp1k-v1-reference-closure.json'
    refs = json.loads(refs_path.read_text())
    assert refs['status'] == 'REFERENCE_CLOSURE_PASS'
    verified = {v['destination']['path']: v['destination'] for v in refs['verified']}
    stage = ARCHIVE / 'jvp1k-v1.partial'
    assets_path = stage / 'closure/local/state/alpha-jv-migration-server4-20260907/tech-r1/assets.lock.json'
    assets = json.loads(assets_path.read_text())
    snapshots, hparams = [], []
    for name, model in assets['models'].items():
        root = Path(model['snapshot'].replace('/data/janghj/', '/mnt/raid5/janghj/', 1))
        assert root.name == model['revision'] and root.is_dir()
        index_path = root / 'model.safetensors.index.json'
        index = json.loads(index_path.read_text())
        assert index['weight_map']
        shards = []
        for filename in sorted(set(index['weight_map'].values())):
            assert Path(filename).name == filename
            p = root / filename
            v = verified[str(p)]  # Missing shards, or shards not full-SHA verified, fail closed.
            s = p.stat()
            assert (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns) == (v['dev'], v['inode'], v['bytes'], v['mtime_ns'])
            shards.append(v)
        small = []
        for filename in ['model.safetensors.index.json', 'config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'generation_config.json']:
            p = root / filename
            if filename == 'special_tokens_map.json' and not p.exists():
                # Optional standalone file: source pins tokenizer.json; tokenizer_config
                # carries the special-token settings for snapshots without this file.
                small.append(dict(path=str(p), status='OPTIONAL_FILE_ABSENT', settings_in='tokenizer_config.json'))
                continue
            assert p.is_file(), ('MISSING_SNAPSHOT_METADATA', str(p))
            # Source config/tokenizer hashes have already been independently checked.
            small.append(dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p)))
        snapshots.append(dict(model=name, revision=model['revision'], shards=shards, metadata=small,
                              source_config_sha256=model['snapshot_config_sha256'], source_tokenizer_sha256=model['snapshot_tokenizer_sha256'],
                              boundary='Source pins revision/config/tokenizer; local shard full SHA verified against HF content-addressed blobs. No new remote source-shard parity or model replay.'))
        for family, m in model['hparams'].items():
            p = REPO / 'project/run_scripts/fixed_z_nonuniqueness/config' / Path(m['absolute_path']).name
            hparams.append(dict(model=name, family=family, source_path=m['absolute_path'], destination=stable_verify(p, m)))
    save(CONTROL / 'jvp1k-v1-reference-coverage.json', dict(status='REFERENCE_COVERAGE_PASS',
         reference_receipt_sha256=sha(refs_path), assets_lock_sha256=sha(assets_path), snapshots=snapshots,
         hparams=hparams, gpu_model_replay=0))
    print('JVP_REFERENCE_COVERAGE_PASS', len(snapshots), len(hparams), flush=True)


if __name__ == '__main__':
    main()
