"""One model W0/smoke, six cold zsRE chains, fail-closed CPU receipts."""
import argparse
from pathlib import Path
import subprocess
from official.experiments.prepare import read, write_new
from official.runners.server2 import collect, run
from official.runners.server2.zsre_profile import enabled, METHODS


def verify_smoke(attempt, manifest):
    folder = Path(attempt)/'ZSRE_SMOKE'
    value = read(folder/'smoke.json')
    assets = read(manifest['asset_manifest'])
    run.require(run.member(manifest['asset_manifest'])['sha256'] == manifest['asset_manifest_sha256'],
                'ZSRE_SMOKE_ASSET_BINDING')
    records = read(manifest['streams']['zsre']['path'])
    verified = collect.validate_smoke(value, manifest, assets, records, folder)
    observed = read(collect.verify_member(value['factual_endpoints']['W1']))
    collect.validate_owner_formula_parity(observed['native_owner_formula_parity'], manifest,
                                         'MEMIT', dataset='zsre')
    return dict(member=run.member(folder/'smoke.json'), W0member=verified['W0member'],
                actual_GPU=True, native_batch_calls=1, native_request_applications=100)


def execute(manifest_path, method, out, *, invoke=subprocess.run):
    manifest_path, out = Path(manifest_path).resolve(), Path(out).resolve()
    manifest = read(manifest_path)
    run.require(enabled(manifest) and method in METHODS
        and manifest['registration_stage']=='zsre_pipeline' and out==manifest_path.parent/method,
        'EXACT_ZSRE_PIPELINE_BINDING')
    out.mkdir(exist_ok=True)
    run.require(not (out/'pipeline-start.json').exists(), 'NO_PIPELINE_AUTOMATIC_RETRY')
    write_new(out/'pipeline-start.json', dict(method=method, source=manifest['code_commit'],
        actual_smoke='NOT_OBSERVED', dataset='zsre'))
    def phase(mode, destination):
        argv=[manifest['runtime']['python'], '-u', '-m', 'official.runners.server2.run',
              '--manifest',str(manifest_path),'--method',method,'--dataset','zsre',
              '--mode',mode,'--out',str(destination)]
        result=invoke(argv, check=False)
        run.require(result.returncode==0,'ZSRE_PIPELINE_PHASE_FAILED:'+mode)
    try:
        if method==METHODS[0]:
            phase('w0',manifest_path.parent/'W0_ZSRE')
            phase('smoke',manifest_path.parent/'ZSRE_SMOKE')
            write_new(manifest_path.parent/'zsre-smoke-verified.json',verify_smoke(manifest_path.parent,manifest))
        proof=verify_smoke(manifest_path.parent,manifest)
        run.require(read(manifest_path.parent/'zsre-smoke-verified.json')==proof,'ZSRE_ACTUAL_SMOKE_BINDING')
        # New process/model: smoke RAM is never the cold scientific chain.
        phase('chain',out/'chain')
        value=read(out/'chain/result.json')
        run.require(value['dataset']=='zsre' and value['batches']==20,'ZSRE_FINAL_SCOPE')
        write_new(out/'pipeline-terminal.json',dict(status='SCIENTIFIC_COMPLETE',
            result=run.member(out/'chain/result.json'),smoke=proof,checkpoint_KEEP=True))
    except BaseException as error:
        write_new(out/'pipeline-failure.json',dict(status='TECHNICAL_FAILURE',
            exception_type=type(error).__name__,code=str(error)[:200],automatic_retry=False,
            source_raw_checkpoint_KEEP=True))
        raise


def reduce(attempt):
    attempt=Path(attempt).resolve()
    manifest,submission=read(attempt/'manifest.json'),read(attempt/'submission.json')
    run.require(enabled(manifest) and set(submission['jobs'])==set(METHODS)|{'collector'},'ZSRE_REDUCER_SCOPE')
    results,failures={},{}
    for method in METHODS:
        try:
            proof=verify_smoke(attempt,manifest)
            value=read(attempt/method/'chain/result.json')
            results[method]=collect.validate_chain(value,manifest,read(manifest['asset_manifest']),
                read(manifest['streams']['zsre']['path']),method,'zsre')
        except Exception as error:
            failures[method]=dict(error_type=type(error).__name__,code=str(error)[:200])
    try:
        accounting=collect.accounting(submission['jobs'],'zsre_pipeline');accounting.pop('raw',None)
    except Exception as error:
        accounting=dict(status='NOT_RECORDED',error_type=type(error).__name__)
    completed=all(accounting.get('jobs',{}).get(m,{}).get('state')=='COMPLETED' for m in METHODS)
    result=dict(status='SCIENTIFIC_COMPLETE' if not failures and completed else 'PARTIAL_OR_FAILED',
        dataset='zsre',methods=results,failures=failures,accounting=accounting,
        generation='NOT_APPLICABLE',checkpoint_KEEP=True,source=manifest['code_commit'])
    write_new(attempt/'collector/terminal.json',result)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest');parser.add_argument('--method',choices=METHODS)
    parser.add_argument('--out');parser.add_argument('--attempt')
    args=parser.parse_args()
    if args.attempt: reduce(args.attempt)
    else: execute(args.manifest,args.method,args.out)


if __name__=='__main__': main()
