"""Task-local metadata repair control under explicit report-then-repair recall."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
from .common import ROOT,member,write
from .control import call
from .paired_stop import checked_targets
from .common import tensor_sha
from .retained_native import validate_native


def prepare():
    import torch
    torch.set_num_threads(8)
    parent=ROOT/'M/attempt-skip-t-v1';out=ROOT/'receipts/metadata-repair-r1'
    q=subprocess.run(['squeue','-h','-j','50021','-o','%i|%T|%R'],capture_output=True,text=True)
    if q.returncode and 'Invalid job id specified' not in q.stderr:raise RuntimeError(q.stderr)
    queue=q.stdout.strip()
    if queue.strip():raise ValueError('OLD_REPAIR_ARRAY_STILL_LIVE')
    accounting=call(['sacct','-n','-P','-X','-j','50021','--format=JobID,State,ExitCode,ElapsedRaw,AllocTRES,Start,End'])
    if not accounting or any(x.split('|')[1].split()[0] not in ('FAILED','CANCELLED') for x in accounting.splitlines()):
        raise ValueError('OLD_ARRAY_NOT_TERMINAL_FAILURE')
    terminal=write(out/'old-attempt-terminal.json',dict(queue=queue,accounting=accounting,
        time=datetime.datetime.now(datetime.timezone.utc).isoformat(),queue_stderr=q.stderr,allocated_GPU_seconds=2301,
        prior_T49928_M49973_GPU_seconds=3231,step_extern_double_count=False,
        cancellation=member(out/'cancel-result.json'),files_deleted=0,rollback='NOT_VERIFIED'))
    lockpath=parent/'execution.lock.json';lock=json.loads(lockpath.read_text());specs={}
    for episode in (1,2):
        ep=parent/'episodes'/f'b{episode+1:03d}'/'attempt-v1'
        bindingpath=ep/'native/native-binding.json';binding=json.loads(bindingpath.read_text())
        native=member(binding['source']['path'])
        if native!=binding['source']:raise ValueError('NATIVE_SAVED_FILE_IDENTITY')
        unsafe=torch.serialization.get_unsafe_globals_in_checkpoint(native['path'])
        if unsafe:raise ValueError('RETAINED_NATIVE_UNSAFE_GLOBALS:'+str(unsafe))
        result=torch.load(native['path'],weights_only=True,mmap=True,map_location='cpu')
        identity=json.loads((ep/'runtime-load.json').read_text())['identity']
        validate_native(result,binding,identity,lock['sample_order'][episode*100:(episode+1)*100])
        if result['weight'].shape!=(4096,14336):raise ValueError('ACTUAL_L4_SHAPE')
        specs[str(episode)]=dict(native=native,binding=member(bindingpath),runtime=member(ep/'runtime-load.json'),
            prior_lock=member(lockpath),CPU_check='weights_only/mmap/finite/hash/order/cold/P/receipt; not GPU continuation')
    ref=write(out/'retained-native-plan.json',dict(status='REUSE_NATIVE_B1_B2_B3',retained=specs,
        old_terminal=terminal,user_recall=member(out/'user-recall.txt'),source_lock=member(lockpath),
        new_native_fits_max=7,new_native_targets_max=700,
        B4_partial='NO_COMPLETE_CAPSULE; original partial work/cost preserved',
        completed_M_endpoints=0,new_T=0,geometry_reuse=False,all_final_endpoints_to_save=80))
    print(json.dumps(ref))


def retire():
    parent=ROOT/'M/attempt-skip-t-v1';out=ROOT/'receipts/metadata-repair-r1'
    authority=out/'user-recall.txt'
    if not authority.exists():raise ValueError('NO_REPAIR_RECALL')
    failure=parent/'episodes/b001/attempt-v1/failure.json'
    if 'TorchVersion' not in json.loads(failure.read_text())['error']:raise ValueError('WRONG_FAILURE')
    submission=json.loads((parent/'submission.json').read_text())
    if submission['job']!='50021':raise ValueError('WRONG_ATTEMPT')
    if (out/'cancel-request.json').exists():raise ValueError('NO_DUPLICATE_CANCEL')
    queue=call(['squeue','-h','-r','-j','50021','-o','%i|%u|%T|%j|%b|%R'])
    ids=checked_targets(queue,['50021_'+str(i) for i in range(10)])
    request=write(out/'cancel-request.json',dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        user_recall=member(authority),reported_failure=member(failure),queue=queue,exact_live_ids=ids,
        completed_excluded=True,other_jobs_changed=0,old_T_unrelated=True,rollback='NOT_VERIFIED',files_deleted=0))
    result=subprocess.run(['scancel',*ids],capture_output=True,text=True) if ids else None
    ref=write(out/'cancel-result.json',dict(request=request,ids=ids,returncode=0 if result is None else result.returncode,
        stdout='' if result is None else result.stdout,stderr='' if result is None else result.stderr))
    if result is not None and result.returncode:raise RuntimeError('EXACT_REPAIR_CANCEL_FAILED')
    print(json.dumps(ref))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['retire','prepare']);a=p.parse_args()
    retire() if a.action=='retire' else prepare()
