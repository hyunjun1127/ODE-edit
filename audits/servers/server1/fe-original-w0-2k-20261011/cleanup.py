"""Explicit FE config-bound payload removal. Raw/config/metadata retained."""
import os,json,stat,subprocess
from pathlib import Path
from official.runners.server1.common import read,member
from official.experiments.prepare import write_new
O=Path(__file__).parent;root=Path('/mnt/raid5/janghj/ODE-edit/local/official-baselines/server1')
methods={'MEMIT_FE','MEMIT_FE_HISTORY'};bound={};configs=[]
for p in root.glob('**/configs/*.json'):
    try:c=read(p)
    except (ValueError,OSError):continue
    if c.get('method') not in methods or not c.get('output'):continue
    output=Path(c['output'])
    if not output.is_relative_to(root):continue
    configs.append(dict(config=member(p),method=c['method'],output=str(output)))
    for directory in [output/'checkpoint',output/'continuous/checkpoint',output/'split/checkpoint']:
        if not directory.is_dir():continue
        for f in directory.iterdir():
            if f.suffix not in ('.pt','.pth','.tmp'):continue
            assert not f.is_symlink() and stat.S_ISREG(f.lstat().st_mode)
            for parent in f.parents:
                if parent==root:break
                assert not parent.is_symlink()
            s=f.stat();bound[str(f)]=dict(path=str(f),bytes=s.st_size,device=s.st_dev,inode=s.st_ino,mtime_ns=s.st_mtime_ns,nlink=s.st_nlink,config=member(p),method=c['method'])
assert bound,'NO_BOUND_FE_PAYLOADS'
state=subprocess.check_output(['sacct','-n','-X','-P','-j','62530','--format=JobID,State'],text=True)
assert 'CANCELLED' in state,'WRITER_NOT_TERMINAL'
write_new(O/'checkpoint-delete-manifest.json',dict(authorization='USER-FE-ORIGINAL-W0-RESET-20261011-R1',writer_state=state,configs=configs,files=list(bound.values()),count=len(bound),bytes=sum(x['bytes'] for x in bound.values())))
before=os.statvfs(root);deleted=[]
for p,m in bound.items():
    f=Path(p);s=f.lstat()
    assert stat.S_ISREG(s.st_mode) and (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)==(m['device'],m['inode'],m['bytes'],m['mtime_ns'])
    f.unlink();assert not f.exists();deleted.append(p)
after=os.statvfs(root)
write_new(O/'checkpoint-deleted.json',dict(deleted=deleted,count=len(deleted),logical_bytes=sum(x['bytes'] for x in bound.values()),filesystem_available_delta=(after.f_bavail-before.f_bavail)*after.f_frsize,all_exact_targets_absent=True,backup_created=False,raw_logs_config_retained=True,remaining_inventory_scope='config-bound official server1; other legacy/replica roots still being audited'))
print('DELETED',len(deleted),sum(x['bytes'] for x in bound.values()))
