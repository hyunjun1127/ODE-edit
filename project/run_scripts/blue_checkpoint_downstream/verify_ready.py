"""Independent, read-only asset verification; create-once raw-free receipt."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from audit_inputs import audit, sha, verify_members
from rte_scoring import VERSION, AUTHORITY


def file_sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_once(path, value):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imports', type=Path, required=True)
    parser.add_argument('--dataset-receipts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--payload', action='store_true')
    args = parser.parse_args()
    assert not args.output.exists(), args.output
    started = time.monotonic()
    original = audit(args.imports, args.dataset_receipts)
    here = Path(__file__).resolve().parent
    subprocess.run([sys.executable, '-B', str(here/'test_rte_scoring.py'), '-q'], check=True)
    after = audit(args.imports, args.dataset_receipts)
    assert after == original, 'SOURCE_OR_DATA_CHANGED_DURING_TEST'
    supplement = args.imports/'evaluator-import-supplement'
    assert sha(supplement/'manifest.json') == '8184c373ca4d8e4d5f60a78923243cdab9a0b1b24fcecb8ef116f7025ec356cc'
    extra = json.loads((supplement/'manifest.json').read_text())
    verify_members(supplement, extra['members'])
    ready = json.loads((args.imports/'transfer-ready.json').read_text())
    assert ready['status'] == 'READY' and ready['checkpoints'] == 72
    entries = json.loads((args.imports/'source/checkpoint-manifest.json').read_text())['checkpoints']
    checked = []
    if args.payload:
        for entry in entries:
            member = entry['file']
            relative = Path(member['destination_relative'])
            assert relative.parts[0] == 'payload' and '..' not in relative.parts
            path = args.imports/relative
            assert path.is_file() and not path.is_symlink()
            assert path.stat().st_size == member['bytes']
            assert file_sha(path) == member['sha256'], path
            checked.append(dict(relative=str(relative), bytes=member['bytes'], sha256=member['sha256']))
        assert len(checked) == 72 and sum(x['bytes'] for x in checked) == 62011141768
    result = dict(status='RTE_SCORING_GATE_PASS', mapping=VERSION, authority=AUTHORITY,
                  original_source_diagnostic=original, source_data_unchanged=True,
                  test_count=6, tests='PASS', evaluator_supplement_members=2,
                  adapter_sha256=sha(here/'rte_scoring.py'), test_sha256=sha(here/'test_rte_scoring.py'),
                  checkpoint_manifest_sha256=sha(args.imports/'source/checkpoint-manifest.json'),
                  transfer_ready_sha256=sha(args.imports/'transfer-ready.json'),
                  payload_file_verification='PASS' if args.payload else 'NOT_RUN',
                  verified_payload=checked, physical_restore_gate='NOT_RUN',
                  eval_rows_per_task=100, eval_slice=[10,110], fewshot=0, generation_length=5,
                  mmlu_parser='SOURCE_EXACT_UNCHANGED', model_load=0, gpu=0, slurm=0,
                  evaluation_valid_claim=False, scientific_promotion=False,
                  elapsed_seconds=time.monotonic()-started)
    write_once(args.output, result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('verified_payload','original_source_diagnostic')}))


if __name__ == '__main__':
    main()
