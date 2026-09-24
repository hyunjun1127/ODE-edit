"""Exact pending CPU collector replacement only; never changes GPU jobs."""
import shutil
import subprocess
from .common import *

def cmd(argv):return subprocess.check_output(argv,text=True).strip()
def main():
    attempt=ROOT/'attempt-v1';old=read(attempt/'submission.json');assert old['collector']=='53178'
    d=attempt/'collector-repair-r1';d.mkdir(exist_ok=False)
    info=cmd(['scontrol','show','job','-o','53178'])
    assert 'UserId=janghj(' in info and 'JobState=PENDING' in info and 'RunTime=00:00:00' in info and str(attempt/'collector.sbatch') in info
    assert 'afterany:53176' in info and 'afterany:53177' in info
    source=d/'source'
    for name in ('common.py','reduce.py'):
        p=source/'project/run_scripts/historical_update_timeaxis'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(Path(__file__).parent/name,p)
    source_members=[record(p) for p in sorted(source.rglob('*.py'))]
    source_commit=cmd(['git','rev-parse','HEAD']);save(d/'source.json',dict(source_commit=source_commit,members=source_members,GPU_execution_lock=record(attempt/'execution.lock.json'),old_collector=53178,pre_cancel_inspection=info))
    script=d/'collector.sbatch';text=(attempt/'collector.sbatch').read_text().replace('cd '+str(attempt/'source'),'cd '+str(source));script.write_text(text)
    subprocess.run(['scancel','53178'],check=True)
    save(d/'cancel-request.json',dict(exact_job_id=53178,reason='CPU preexecution terminal-order bug; source preserved',GPU_jobs_unchanged=[53176,53177]))
    job=cmd(['sbatch','--parsable','--hold','--job-name=odeedit_hist_T4r1_s4','--dependency=afterany:53176:53177','--output='+str(d/'%j.out'),'--error='+str(d/'%j.err'),str(script)]).split(';')[0]
    save(d/'submitted.json',dict(job_id=job,source_commit=source_commit,source_members=source_members,argv_script=record(script),held=True))
    new=cmd(['scontrol','show','job','-o',job]);assert 'UserId=janghj(' in new and 'JobState=PENDING' in new and 'NumCPUs=8' in new and 'MinMemoryNode=24G' in new and 'Requeue=0' in new
    assert 'afterany:53176' in new and 'afterany:53177' in new
    subprocess.run(['scontrol','release',job],check=True)
    save(d/'receipt.json',dict(old_collector=53178,new_collector=int(job),GPU_jobs_unchanged=[53176,53177],held_inspection=new,source_commit=source_commit,source_members=source_members,status='RELEASED'))
    print(job)
if __name__=='__main__':main()
