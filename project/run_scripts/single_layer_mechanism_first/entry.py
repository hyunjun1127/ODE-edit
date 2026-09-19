"""Immutable task job entry; no scheduler writes or automatic submissions."""
import argparse
import json
from pathlib import Path
import resource
import time
import traceback
from .config import check_lock
from project.run_scripts.single_layer_edit_preserving_correction.common import write, sha


def execute(lock):
    check_lock(lock)
    for item in lock['execution']['members']:
        if Path(item['path']).stat().st_size!=item['bytes'] or sha(item['path'])!=item['sha256']:
            raise ValueError('EXECUTABLE_SOURCE_CHANGED:'+item['path'])
    for item in lock['external_members']:
        if Path(item['path']).stat().st_size!=item['bytes'] or sha(item['path'])!=item['sha256']:
            raise ValueError('EXTERNAL_IMPORT_CHANGED:'+item['path'])
    out=Path(lock['output']);out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();stage='model_load'
    try:
        write(out/'execution-entry.json',dict(lock=lock,status='ENTERED_NOT_COMPLETE'))
        from .model import Runtime
        rt=Runtime(lock,out)
        if lock['stage']=='T0' and lock['phase']=='HOOK':
            from .technical import hook_checks
            stage='native_hook_actual_T0'
            result,_=hook_checks(rt,out/'technical-hook')
            write(out/'terminal.json',dict(status='T0_HOOK_PART_COMPLETE_NOT_FULL_T0_READY',result=result,
                source=lock['execution']['commit'],seconds=time.monotonic()-start,
                peak_host_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                scientific_batches_completed=0,full_T0_ready=False))
        else:
            raise ValueError('UNIMPLEMENTED_ENTRY_ROUTE_NOT_ADMISSIBLE')
    except BaseException as exc:
        write(out/'failure.json',dict(status='TECHNICAL_FAILURE',stage=stage,error=repr(exc),
            traceback=traceback.format_exc(),seconds=time.monotonic()-start,
            source=lock['execution']['commit'],partial_preserved=True,automatic_resubmit=False))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lock',type=Path,required=True)
    a=p.parse_args();execute(json.loads(a.lock.read_text()))
