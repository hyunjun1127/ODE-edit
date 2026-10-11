"""Index completed native FE contrasts and independently reaggregated metrics.

This is a read-only evidence collector, not an experiment or metric runner.
It refuses to publish a partial five-arm comparison.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess


ARMS = {
    'qwen_current': (63385, 63387, 'native-r2/configs/qwen25.json',
                     'native-eval-r1/qwen-independent-metrics.json'),
    'qwen_online_w0': (63391, 63392, 'native-causal-r1/configs/qwen_online_w0.json',
                       'native-causal-r1/qwen-online-independent-metrics.json'),
    'gptj_current': (63386, 63388, 'native-r2/configs/gptj.json',
                     'native-eval-r1/gptj-independent-metrics.json'),
    'gptj_repaired': (63389, 63390, 'native-causal-r1/configs/gptj_repaired.json',
                      'native-causal-r1/gptj-repaired-independent-metrics.json'),
    'gptj_online_w0_repaired': (63425, 63426,
        'native-causal-r2/configs/gptj_online_w0_repaired.json',
        'native-causal-r2/gptj-online-w0-independent-metrics.json'),
}


def read(path):
    data = path.read_bytes()
    return json.loads(data), {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(),
                              'bytes': len(data)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'OUTPUT_EXISTS'
    ids = [str(j) for row in ARMS.values() for j in row[:2]]
    sacct = subprocess.check_output([
        'sacct', '-n', '-X', '-j', ','.join(ids),
        '--format=JobID,State,ExitCode,Start,End,Elapsed', '-P'], text=True)
    statuses = {row[0]: row for line in sacct.splitlines() if (row := line.split('|'))[0] in ids}
    assert set(statuses) == set(ids)
    assert all(row[1:3] == ['COMPLETED', '0:0'] for row in statuses.values()), statuses
    result = {
        'schema': 'native-fe-cf-2k-contrasts-v1',
        'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'collector_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'sacct_times_timezone': 'Asia/Seoul',
        'scope': 'One identical ordered CF 2K stream, 20 batches of 100; not a multi-seed paper replication',
        'training': 'Pinned author main.py, author hparams, BF16, native writer and persistent history',
        'target_request_scope': 'Current 100 only; native loader still reads the complete dataset JSON',
        'metrics': 'Common CF target-mean-NLL preference, request macro mean, harmonic Score; percentages',
        'baseline_adoption': 'Explicit contrasts; initial-model target reference does not replace batch-entry by default',
        'sacct': sacct,
        'arms': {},
    }
    stream_shas = set()
    w0_by_model = {}
    for name, (training_job, evaluation_job, config_path, metrics_path) in ARMS.items():
        config, config_receipt = read(args.root / config_path)
        metrics, metric_receipt = read(args.root / metrics_path)
        assert config_receipt['sha256'] == metrics['native_config_sha256']
        assert config['identity_sha256'] == metrics['identity_sha256']
        assert set(metrics['checkpoints']) == {'0', '5', '10', '15', '20'}
        assert all(row['independent_reaggregation'] == 'PASS' for row in metrics['checkpoints'].values())
        assert metrics['checkpoints']['20']['requests'] == 2000
        stream_shas.add(metrics['stream_sha256'])
        w0 = metrics['checkpoints']['0']['summary']
        if config['model'] in w0_by_model:
            assert w0 == w0_by_model[config['model']], (name, 'W0_METRICS_DIFFER')
        w0_by_model[config['model']] = w0
        checkpoint_receipts = {}
        for batch in (0, 5, 10, 15, 20):
            path = Path(config['output']) / 'checkpoints' / f'B{batch:02d}.json'
            receipt, metadata = read(path)
            assert receipt['completed_edits'] == batch * 100
            assert receipt['identity_sha256'] == config['identity_sha256']
            assert Path(receipt['path']).is_file()
            assert Path(receipt['path']).stat().st_size == receipt['bytes']
            checkpoint_receipts[str(batch)] = dict(receipt=receipt, metadata=metadata)
        result['arms'][name] = {
            'training_job': training_job, 'evaluation_job': evaluation_job,
            'model': config['model'], 'config': config_receipt,
            'identity_sha256': config['identity_sha256'],
            'source_commit': config['ode_patch_source'], 'author_commit': config['author_commit'],
            'z_reference': config.get('z_reference', 'batch_entry'),
            'covariance_repair': config['covariance_repair'],
            'independent_metrics': metric_receipt,
            'milestones': {b: row['summary'] for b, row in metrics['checkpoints'].items()},
            'checkpoints': checkpoint_receipts,
        }
    assert len(stream_shas) == 1
    result['stream_sha256'] = next(iter(stream_shas))
    result['all_ten_jobs_completed_successfully'] = True
    result['all_25_milestones_independently_reaggregated'] = True
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({name: row['milestones']['20'] for name, row in result['arms'].items()}))


if __name__ == '__main__':
    main()
