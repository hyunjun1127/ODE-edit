"""Freeze a tiny committed CPU logger closure and launch once, not a Slurm job."""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile

FILES=[
 'project/run_scripts/jlz_interference_l1/comparison_bridge.py',
 'project/run_scripts/jlz_interference_l1/__init__.py',
 'project/run_scripts/jlz_realization/__init__.py',
 'project/run_scripts/jlz_realization/common.py',
 'project/run_scripts/jlz_realization/observe.py',
 'project/run_scripts/jlz_shared_budget/__init__.py',
 'project/run_scripts/jlz_shared_budget/common.py',
 'audits/servers/server4/wandb-model-views/comparison-bindings.json']

def main():
    root=Path('/data/janghj/ODE-edit/local/wandb-comparison-bridge')
    receipt=root/'launch.json'
    if receipt.exists():raise RuntimeError('ALREADY_LAUNCHED_NO_DUPLICATE')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    source=root/('source-'+commit[:12]);source.mkdir(exist_ok=False)
    data=subprocess.check_output(['git','archive','--format=tar',commit,*FILES])
    members=[]
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        for member in archive.getmembers():
            path=Path(member.name)
            if path.is_absolute() or '..' in path.parts:raise RuntimeError('UNSAFE_SOURCE')
            if member.isdir():continue
            if not member.isfile() or member.name not in FILES:raise RuntimeError('NONREGULAR_SOURCE')
            value=archive.extractfile(member).read();dest=source/path;dest.parent.mkdir(parents=True,exist_ok=True)
            with dest.open('xb') as f:f.write(value)
            members.append(dict(path=member.name,bytes=len(value),sha256=hashlib.sha256(value).hexdigest()))
    assert len(members)==len(FILES)
    keys=('PATH','HOME','USER','LANG','LC_ALL','SSL_CERT_FILE','REQUESTS_CA_BUNDLE',
          'NETRC','WANDB_API_KEY','WANDB_IDENTITY_TOKEN_FILE','WANDB_CREDENTIALS_FILE','WANDB_CONFIG_DIR')
    env={k:os.environ[k] for k in keys if k in os.environ}
    env.update(WANDB_CONSOLE='off',WANDB_SILENT='true',WANDB_DISABLE_CODE='true',WANDB_DISABLE_GIT='true',
        CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',GOMAXPROCS='2',
        PYTHONPATH=str(source),PYTHONDONTWRITEBYTECODE='1')
    argv=['/data/janghj/ODE-edit/local/wandb-setup/sdk/bin/python','-u','-m',
        'project.run_scripts.jlz_interference_l1.comparison_bridge','--bindings',str(source/FILES[-1]),
        '--out',str(root),'--publish','--watch']
    cpus=sorted(os.sched_getaffinity(0))[:2]
    def limits():os.nice(10);os.sched_setaffinity(0,cpus)
    with (root/'bridge.log').open('ab') as output:
        proc=subprocess.Popen(argv,cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=output,stderr=output,
            start_new_session=True,preexec_fn=limits)
    obj=dict(pid=proc.pid,argv=argv,source_commit=commit,source_members=members,
        GPU=0,new_Slurm_jobs=0,CPU_affinity=cpus,nice=10,max_hours=168,
        user_authority='실시간 자동 동기화',original_experiments_modified=False)
    with receipt.open('x') as f:json.dump(obj,f,indent=2)
    print(json.dumps(dict(pid=proc.pid,source_commit=commit,receipt=str(receipt))))

if __name__=='__main__':main()
