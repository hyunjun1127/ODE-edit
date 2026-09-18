"""One bounded snapshot of this M array. No loop, callback, mutations or T access."""
import argparse
import datetime
import json
from pathlib import Path
from .common import ROOT,member,write,tensor_sha
from .control import call


def stable_json(path):
    if not path.exists():return None
    before=path.stat()
    try:value=json.loads(path.read_text())
    except json.JSONDecodeError:return None
    after=path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):return None
    return value


def snapshot(index,attempt='attempt-skip-t-v1',receipt_namespace='storage-waiver-r1'):
    if Path(attempt).name!=attempt or Path(receipt_namespace).name!=receipt_namespace:raise ValueError('LOCAL_NAMESPACE')
    parent=ROOT/'M'/attempt;submission=json.loads((parent/'submission.json').read_text())
    job=submission['job'];lock=json.loads((parent/'execution.lock.json').read_text())
    queue=call(['squeue','-h','-j',job,'-o','%i|%j|%T|%R|%b|%N'])
    episodes=[];initials=[]
    for i in range(10):
        path=Path(lock['M_root'])/f'b{i+1:03d}'/'attempt-v1'
        names=sorted(str(p.relative_to(path)) for p in path.glob('**/*.json')) if path.exists() else []
        failure=stable_json(path/'failure.json')
        initial=stable_json(path/'M_INITIAL_VALID_WITH_T_SKIPPED.json')
        native=stable_json(path/'native/native-binding.json')
        episode=dict(index=i,episode=f'b{i+1:03d}',json_members=names,
            native_new=None if native is None else native['native_fit_new_calls'],
            initial_observed=initial is not None,
            failure=None if failure is None else dict(stage=failure['stage'],error=failure['error'],receipt=member(path/'failure.json')))
        if initial is not None:
            if not (initial['validation']=='SKIPPED_USER_DIRECTED' and
                    initial['full_numerical_validation']=='NOT_ESTABLISHED' and
                    initial['observer_nonmutation'] and initial['independent_W0_reset']['W0_exact'] and
                    initial['independent_W0_reset']['M0_exact'] and initial['M_history']==0 and
                    initial['S_R_L_started'] is False and initial['efficacy_PASS'] is False):
                raise ValueError('INITIAL_RECEIPT_SCOPE_OR_STATE')
            endpoint=initial['endpoint_retained']
            if member(endpoint['path'])!=endpoint:raise ValueError('INITIAL_ENDPOINT_FILE_IDENTITY')
            import torch
            torch.set_num_threads(8)
            saved=torch.load(endpoint['path'],weights_only=True,mmap=True,map_location='cpu')
            weight=saved['weight']
            if weight.shape!=(4096,14336) or weight.dtype!=torch.float32 or not torch.isfinite(weight).all():
                raise ValueError('INITIAL_ENDPOINT_SHAPE_DTYPE_FINITE')
            from .alltoken import tensor_sha256
            if tensor_sha256(weight)!=initial['EN_F']['selected_weight_sha256']:
                raise ValueError('INITIAL_ENDPOINT_TENSOR_HASH')
            seal=initial['selection_before_observer']
            if member(seal['path'])!=seal:raise ValueError('INITIAL_SELECTION_SEAL_IDENTITY')
            seals=json.loads(Path(seal['path']).read_text())
            if seals['completed_optimizers']!=8 or seals['official_P_N_access_so_far']!=0:
                raise ValueError('INITIAL_SELECTION_OBSERVER_ORDER')
            initials.append(dict(episode=episode['episode'],marker=member(path/'M_INITIAL_VALID_WITH_T_SKIPPED.json'),
                endpoint=endpoint,actual_weight_raw_sha=tensor_sha(weight),
                stop_reason=initial['EN_F']['stop_reason'],accepted_rounds=initial['EN_F']['counters']['accepted_rounds'],
                CPU_endpoint_reload='shape/FP32/finite/fileSHA/tensorSHA only; not GPU continuation',
                scope=initial['scope'],no_numerical_or_efficacy_PASS=True))
        episodes.append(episode)
    result=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),job=job,queue=queue,
        source=lock['execution']['head'],lock=member(parent/'execution.lock.json'),
        episodes=episodes,initials=initials,monitor_snapshot_only=True,new_GPU=0,
        T='SKIPPED_USER_DIRECTED',full_numerical_validation='NOT_ESTABLISHED')
    ref=write(ROOT/'receipts'/receipt_namespace/f'observation-{index}.json',result)
    print(json.dumps(dict(receipt=ref,time=result['time'],queue=queue,initials=initials,
        stages=[dict(episode=e['episode'],native_new=e['native_new'],json_count=len(e['json_members']),
            phase=('initial' if e['initial_observed'] else 'observers' if 'ALL_SELECTIONS_SEALED.json' in e['json_members']
                else 'controllers' if any(n.startswith('arms/') for n in e['json_members'])
                else 'gradient_diagnostics' if 'native-objective.json' in e['json_members']
                else 'geometry_or_anchor' if 'protected-provenance.json' in e['json_members']
                else 'native' if 'nonselected-before.json' in e['json_members'] else 'load'),
            initial=e['initial_observed'],failure=e['failure']) for e in episodes if e['json_members']])))


def main():
    p=argparse.ArgumentParser();p.add_argument('--index',required=True)
    p.add_argument('--attempt',default='attempt-skip-t-v1');p.add_argument('--receipt-namespace',default='storage-waiver-r1')
    a=p.parse_args();snapshot(a.index,a.attempt,a.receipt_namespace)


if __name__=='__main__':main()
