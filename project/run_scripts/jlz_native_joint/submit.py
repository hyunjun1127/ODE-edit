"""Exact cap2 DAG, held inspection then release; no background monitoring."""
import argparse
import getpass
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
from .common import require,write,sha

def command(argv):
    result=subprocess.run(argv,capture_output=True,text=True)
    require(result.returncode==0,repr(argv)+':'+result.stderr+result.stdout)
    return result.stdout.strip()

def admission(exclude=()):
    raw=command(['squeue','-h','-r','-w','server4','-o','%i|%u|%j|%T|%b|%R'])
    jobs=[]
    for line in raw.splitlines():
        job,user,name,status,gres,reason=line.split('|',5)
        if job in exclude:continue
        if user!=getpass.getuser() and not name.startswith(('odeedit_','bfode_','motivation_','session01_')):continue
        detail=command(['scontrol','show','job',job,'--oneliner'])
        req=re.search(r'\bReqTRES=([^ ]+)',detail);require(req,'UNKNOWN_RESOURCE')
        gpu=re.search(r'gres/gpu=(\d+)',req[1])
        if gpu:jobs.append(dict(job=job,user=user,name=name,state=status,gpus=int(gpu[1]),reason=reason))
    return dict(jobs=jobs,scope='resource-only current owner/project server4; no mutation',raw=raw)

def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True);args=p.parse_args();root=args.attempt.resolve()
    require(not list(root.glob('submitted-*.json')) and not (root/'submission.json').exists(),'NO_DUPLICATE_SUBMISSION')
    lock=json.loads((root/'execution.lock.json').read_text())
    require(sha(root/'config.json')==lock['config_sha256'],'CONFIG_CHANGED')
    for item in lock['launchers']+lock['source_members']:require(sha(item['path'])==item['sha256'],'FROZEN_SOURCE_CHANGED')
    require(shutil.disk_usage(root).free>=lock['resources']['reserve_bytes'],'STORAGE_RESERVE')
    before=admission();external=[j['job'] for j in before['jobs']]
    names=['shared-SHARED','pilot-JLZ_A','pilot-JLZ_B','main-JLZ_A','main-JLZ_B','collector'];ids={};inspected=[]
    for name in names:
        gpu=name!='collector';dep=None;tail=[]
        if name.startswith('shared') and external:dep='afterany:'+':'.join(external)
        elif name.startswith('pilot'):dep='afterok:'+ids['shared-SHARED']
        elif name.startswith('main'):dep='afterok:'+':'.join(ids[x] for x in ['pilot-JLZ_A','pilot-JLZ_B'])
        elif name=='collector':dep='afterany:'+':'.join(ids.values());tail=[','.join(ids.values())]
        script=root/(name+'.sh');jobname='odeedit_jlz_v4_s4_'+name.replace('-','_')
        argv=['sbatch','--parsable','--hold','--partition=gpu','--qos=lab_gpu_s4','--nodelist=server4','--nodes=1',
              '--ntasks=1','--cpus-per-task=8','--export=NONE','--no-requeue','--job-name='+jobname,
              '--chdir='+str(root/'source'),'--mem='+('60416M' if gpu else '24576M'),
              '--time='+('7-00:00:00' if gpu else '04:00:00'),
              '--output='+str(root/(name+'-%j.out')),'--error='+str(root/(name+'-%j.err'))]
        if gpu:argv+=['--gres=gpu:1']
        if dep:argv+=['--dependency='+dep]
        if dep and dep.startswith('afterok'):argv+=['--kill-on-invalid-dep=yes']
        argv += [str(script),*tail]
        job=command(argv).split(';')[0];require(job.isdigit(),'INVALID_JOB_ID');ids[name]=job
        write(root/('submitted-'+name+'.json'),dict(job=job,argv=argv,status='HELD'))
        detail=command(['scontrol','show','job',job,'--oneliner'])
        for term in [f'JobId={job} ',f'JobName={jobname} ','UserId='+getpass.getuser()+'(',
                     'JobState=PENDING ','Reason=JobHeldUser ','Requeue=0 ','NumCPUs=8 ',
                     'ReqNodeList=server4 ','Partition=gpu ','QOS=lab_gpu_s4 ',
                     'Command='+str(script),'WorkDir='+str(root/'source')]:require(term in detail,'HELD_INSPECTION:'+term)
        require(('TresPerNode=gres/gpu:1' in detail) if gpu else 'gres/gpu' not in detail,'HELD_GPU')
        mem=60416 if gpu else 24576
        require(f'mem={mem}M' in detail or f'mem={mem//1024}G' in detail,'HELD_MEMORY')
        require('TimeLimit='+('7-00:00:00' if gpu else '04:00:00') in detail,'HELD_WALL')
        field=re.search(r'\bDependency=([^ ]+)',detail);require(field,'HELD_DEPENDENCY_FIELD')
        expected={(dep.split(':')[0],i) for i in dep.split(':')[1:]} if dep else set()
        require(set(re.findall(r'(afterok|afterany):([0-9]+)',field[1]))==expected,'HELD_DEPENDENCY_EXACT')
        submit=re.search(r'\bSubmitLine=(.*?)(?= WorkDir=|$)',detail)
        require(submit and shlex.split(submit[1])==argv,'HELD_FULL_ARGV')
        require(command(['scontrol','write','batch_script',job,'-']).strip()==script.read_text().strip(),'HELD_SCRIPT_BYTES')
        inspected.append(dict(job=job,name=name,argv=argv,scontrol=detail,script_sha256=sha(script)))
    after=admission(tuple(ids.values()))
    require({j['job'] for j in after['jobs']}<=set(external),'ADMISSION_RACE_KEEP_NEW_HELD')
    write(root/'held-inspection.json',dict(jobs=inspected,before=before,prerelease=after,external_afterany=external,
         graph='shared -> pilot A/B -> timing+main A/B -> CPU afterany all',max_concurrent_gpu=2,lock_sha256=sha(root/'execution.lock.json')))
    for name in reversed(names):
        output=command(['scontrol','release',ids[name]])
        write(root/('released-'+name+'.json'),dict(job=ids[name],command_succeeded=True,output=output))
    write(root/'submission.json',dict(status='RELEASED',jobs=ids,held_inspection=sha(root/'held-inspection.json'),
         lock_sha256=sha(root/'execution.lock.json'),initial='NOT_OBSERVED',no_other_job_mutation=True))
    print(json.dumps(dict(status='RELEASED',jobs=ids,initial='NOT_OBSERVED')))

if __name__=='__main__':main()
