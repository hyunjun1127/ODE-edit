"""Six cold CF pipelines, actual resume proof, final checkpoint, generation OFF.

Each Slurm allocation uses separate model processes for qualification and the
cold scientific chain. The native math and checkpoint implementation remain
the official ones. No periodic polling, retry, new submission or checkpoint
archive/delete is performed by this runner or its CPU reducer.
"""
import argparse
from pathlib import Path
import subprocess
import time

from official.experiments.prepare import read, write_new
from official.runners.server2 import collect, run
from official.runners.server2.checkpoint_profile import deferred, METHODS


def verify_qualification(folder, manifest, method):
    value = read(folder/'qualification.json')
    collect.validate_qualification(value, manifest, method)
    collect.validate_qualification_calls(value, manifest, method)
    collect.validate_owner_formula_parity(value['native_owner_formula_parity'], manifest, method)
    records = read(manifest['streams']['cf']['path'])[200:300]
    continuous = read(collect.verify_member(value['continuous_metric']))
    resumed = read(collect.verify_member(value['resumed_metric']))
    for observed in (continuous, resumed):
        collect.validate_factual(observed, 'cf', records, manifest=manifest)
    run.require(continuous['cases'] == resumed['cases'] and continuous['summary'] == resumed['summary'],
                'ACTUAL_QUALIFICATION_METRICS_NOT_EQUAL')
    saved = collect.validate_checkpoint_metadata(folder/'checkpoints', value['checkpoint_identity'], final_batch=3)
    run.require(saved['saved_payload_sha256'] == value['checkpoint']['sha256'], 'ACTUAL_B3_CP_POINTER')
    return run.member(folder/'qualification.json')


def execute(manifest_path, method, out, *, invoke=subprocess.run):
    manifest_path, out = Path(manifest_path).resolve(), Path(out).resolve()
    manifest = read(manifest_path)
    run.require(deferred(manifest) and method in METHODS
        and manifest['registration_stage'] == 'cf_checkpoint' and out == manifest_path.parent/method,
        'EXACT_CHECKPOINT_PIPELINE_BINDING')
    out.mkdir(exist_ok=True)
    run.require(not (out/'pipeline-start.json').exists(), 'NO_PIPELINE_AUTOMATIC_RETRY')
    write_new(out/'pipeline-start.json', dict(method=method, actual_qualification='NOT_OBSERVED',
        generation='DEFERRED_NOT_MEASURED', source=manifest['code_commit']))
    started = time.monotonic()
    def phase(mode, destination):
        argv = [manifest['runtime']['python'], '-u', '-m', 'official.runners.server2.run',
            '--manifest', str(manifest_path), '--method', method, '--dataset', 'cf',
            '--mode', mode, '--out', str(destination)]
        result = invoke(argv, check=False)
        run.require(result.returncode == 0, 'PIPELINE_PHASE_FAILED:'+mode)
    try:
        if method == METHODS[0]:
            phase('w0', manifest_path.parent/'W0_CF')
        run.read_w0(manifest, 'cf', read(manifest['streams']['cf']['path']))
        phase('qualification', out/'qualification')
        proof = verify_qualification(out/'qualification', manifest, method)
        write_new(out/'qualification-verified.json', dict(member=proof, actual_GPU=True))
        phase('chain', out/'chain')
        result = read(out/'chain/result.json')
        run.require(result['batches'] == 20 and result['generation_status'] == 'DEFERRED_NOT_MEASURED',
                    'CHECKPOINT_PIPELINE_FINAL_SCOPE')
        write_new(out/'pipeline-terminal.json', dict(status='EDIT_FACTUAL_CHECKPOINT_COMPLETE',
            method=method, qualification=proof, result=run.member(out/'chain/result.json'),
            generation='DEFERRED_NOT_MEASURED', final_checkpoint_KEEP=True,
            future_evaluation_consumer_pending=True, elapsed_seconds=time.monotonic()-started))
    except BaseException as error:
        write_new(out/'pipeline-failure.json', dict(status='TECHNICAL_FAILURE',
            exception_type=type(error).__name__, code=str(error)[:200],
            elapsed_seconds=time.monotonic()-started, existing_checkpoint_KEEP=True, automatic_retry=False))
        raise


def reduce(attempt):
    attempt = Path(attempt).resolve()
    manifest, submission = read(attempt/'manifest.json'), read(attempt/'submission.json')
    run.require(deferred(manifest) and submission['stage'] == 'cf_checkpoint'
        and set(submission['jobs']) == set(METHODS)|{'collector'}, 'CHECKPOINT_REDUCER_DAG')
    assets = read(manifest['asset_manifest'])
    run.require(run.member(manifest['asset_manifest'])['sha256'] == manifest['asset_manifest_sha256'],
                'CHECKPOINT_REDUCER_ASSET_BINDING')
    records = read(manifest['streams']['cf']['path'])
    results, failures = {}, {}
    for method in METHODS:
        try:
            proof = verify_qualification(attempt/method/'qualification', manifest, method)
            value = read(attempt/method/'chain/result.json')
            result = collect.validate_chain(value, manifest, assets, records, method, 'cf')
            results[method] = dict(result, actual_qualification=proof,
                completion_scope='EDIT_FACTUAL_CHECKPOINT_ONLY', generation_evaluated=False,
                checkpoint_evaluation_consumer_pending=True)
        except Exception as error:
            failures[method] = dict(status='INCOMPLETE_OR_INVALID', error_type=type(error).__name__,
                                    code=str(error)[:240])
    try:
        accounting = collect.accounting(submission['jobs'], 'cf_checkpoint')
        accounting.pop('raw', None)
    except Exception as error:
        accounting = dict(status='ACCOUNTING_NOT_RECORDED', error_type=type(error).__name__)
    scheduler_failures = {method: accounting.get('jobs', {}).get(method, {}).get('state', 'NOT_OBSERVED')
        for method in METHODS if accounting.get('jobs', {}).get(method, {}).get('state') != 'COMPLETED'}
    terminal = dict(status='EDIT_FACTUAL_CHECKPOINT_COMPLETE' if not failures and not scheduler_failures else 'PARTIAL_OR_FAILED',
        methods=results, failures=failures, accounting=accounting,
        scheduler_failures=scheduler_failures,
        generation='DEFERRED_NOT_MEASURED', checkpoint_archive_or_delete=False,
        source=manifest['code_commit'], official_tree=manifest['official_tree_sha256'])
    write_new(attempt/'collector/terminal.json', terminal)
    return terminal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest')
    parser.add_argument('--method', choices=METHODS)
    parser.add_argument('--out')
    parser.add_argument('--attempt')
    args = parser.parse_args()
    if args.attempt:
        reduce(args.attempt)
    else:
        execute(args.manifest, args.method, args.out)


if __name__ == '__main__':
    main()
