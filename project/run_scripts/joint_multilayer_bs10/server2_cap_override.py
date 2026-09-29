"""Apply the explicit current USER cap3 to this exact existing array only."""
import datetime
import subprocess
from .server2_entry import ROOT,NONCE
from .common import save,read,record,require

def main():
    p=ROOT/'attempt-s2-r1';require(not (p/'user-cap3.json').exists(),'ALREADY_APPLIED')
    sub=read(p/'submission.json');require(sub['array_job']=='55116','EXACT_ARRAY')
    get=lambda a:subprocess.check_output(a,text=True)
    queue=get(['squeue','-h','-u','janghj','-o','%i|%j|%T|%b|%R'])
    before=get(['scontrol','show','job','-o','55116'])
    require('ArrayTaskThrottle=2' in before and 'UserId=janghj(' in before and 'JobName=odeedit_joint_bs1_s2' in before,'ARRAY_OWNER')
    # Original admission had no other owner job on S2. Explicitly bound all owner
    # pending jobs now, not just rows with already allocated node server2.
    rows=[line.split('|') for line in queue.splitlines()]
    require(all(r[0].startswith('55116') or r[0]=='55117' for r in rows),'OTHER_ADMISSION_REQUIRES_RECOUNT')
    argv=['scontrol','update','JobId=55116','ArrayTaskThrottle=3']
    node=get(['scontrol','show','node','server2'])
    subprocess.run(argv,check=True)
    after=get(['scontrol','show','job','-o','55116']);require('ArrayTaskThrottle=3' in after,'CAP3_NOT_APPLIED')
    save(p/'user-cap3.json',dict(instruction_id=NONCE,user_text='CAP 3으로 해',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        previous_project_task_cap=2,project_cap=3,task_cap=3,array='55116_[0-8]%3',argv=argv,before=before,after=after,
        owner_queue=queue,node=node,execution_lock=record(p/'execution.lock.json'),frozen_source_changed=False,
        per_process_gpu=1,cpus=8,host_mem_mib=60416,science_paths=9,additional_jobs=0,
        scope='explicit latest USER supersedes same-task throttle prohibition only; no other job mutation'))
    print('CAP3_APPLIED_55116')

if __name__=='__main__':main()
