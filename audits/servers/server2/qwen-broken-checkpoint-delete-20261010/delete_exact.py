"""Explicit USER-authorized three-file deletion, not a generic cleanup helper."""
import hashlib,json,os,stat,subprocess
from datetime import datetime,timezone
from pathlib import Path

BASE=Path('/mnt/raid5/janghj/ODE-edit/local')
OLD=BASE/'qwen-baselines-server2-20261009/registration-native-eval-r2'
NEW=BASE/'qwen-baseline-mask-cold-rerun-20261010/registration-r1'
GJ=BASE/'official-baselines/server2/zsre-2k-reeval-20261009/registration-r1'
OUT=Path(__file__).resolve().parent
SOURCE='5503935821b0ececb4aef09a5bccb5308879a6b5'
TARGETS=[('61956','qwen25-zsre-memit','MEMIT',20),('61960','qwen25-zsre-alphaedit','ALPHAEDIT',20),('61968','qwen25-zsre-memit_fe','MEMIT_FE',11)]
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def write(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True)
def require(x,c):
    if not x:raise RuntimeError(c)
def identity(s):return dict(dev=s.st_dev,inode=s.st_ino,nlink=s.st_nlink,mode=s.st_mode,uid=s.st_uid,
    bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,blocks=s.st_blocks)
def terminal():
    ids=['61956','61960','61968','61957','61961','61969','61974']
    text=subprocess.check_output(['sacct','-X','-n','-P','-j',','.join(ids),
        '--format=JobIDRaw,JobName%100,User,State,NodeList,WorkDir%400'],text=True)
    rows={p[0]:p for p in (l.split('|') for l in text.splitlines())}
    registered={j['job_id']:j for j in read(OLD/'released.json')['jobs']}
    for j in ids:
        p=rows[j];require(p[1]==registered[j]['name'] and p[2]=='janghj' and p[5]==str(OLD),'TERMINAL_IDENTITY')
        require(p[3].split()[0] in ('COMPLETED','CANCELLED','FAILED','TIMEOUT','OUT_OF_MEMORY'),'WRITER_CONSUMER_ACTIVE')
    return text
def consumers(targets):
    text=subprocess.check_output(['scontrol','show','job','-o'],text=True)
    live=[]
    registry={root:{j['job_id']:j for j in read(root/'released.json')['jobs']} for root in (OLD,NEW,GJ)}
    for line in text.splitlines():
        f=dict(s.split('=',1) for s in line.split() if '=' in s)
        if not(f.get('UserId','').startswith('janghj(') and (f.get('ReqNodeList')=='server2' or f.get('NodeList')=='server2') and f.get('JobState') in ('PENDING','RUNNING','COMPLETING','CONFIGURING')):continue
        root=Path(f['WorkDir']);require(root in registry,'UNKNOWN_LIVE_CONSUMER_ROOT')
        j=registry[root][f['JobId']]
        script=(j['method']+'.sh') if root==GJ else (j['cell']+'-'+j['kind']+'.sh')
        require(f['Command']==str(root/'scripts'/script) and f['JobName']==j['name'],'LIVE_COMMAND_MISMATCH')
        if root==OLD:require(j['job_id'] in ('61962','61963','61964','61965'),'OLD_ACTIVE_POSSIBLE_CONSUMER')
        # New main is cold; its kept rows are FT/BLUE only. FT eval uses61900.
        members=['kept.json','w0-parent.json','ft-eval-inputs.json'] if root==NEW else ['inputs.json'] if root==GJ else []
        texts=[(root/'scripts'/script).read_text()]+[(root/m).read_text() for m in members]
        for target in targets:
            require(all(str(target['path']) not in s and target['sha256'] not in s for s in texts),'ACTIVE_CP_CONSUMER')
        live.append({k:f.get(k) for k in ('JobId','JobName','JobState','Command','WorkDir','Dependency')})
    kept=read(NEW/'kept.json')
    require(set(kept)=={'qwen25-cf-ft','qwen25-zsre-ft','qwen25-cf-alphaedit_blue','qwen25-zsre-alphaedit_blue'},'COLLECTOR_OLD_SCOPE')
    # Check open descriptors/mappings of same-UID local processes, without argv/env.
    scanned=0;daemon_exclusions=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try:
            if proc.stat().st_uid!=os.getuid():continue
            scanned+=1
            for fd in (proc/'fd').iterdir():
                try:s=fd.stat()
                except FileNotFoundError:continue
                require(all((s.st_dev,s.st_ino)!=(t['stat']['dev'],t['stat']['inode']) for t in targets),'OPEN_PAYLOAD_FD')
            maps=(proc/'maps').read_text()
            require(all(str(t['path']) not in maps for t in targets),'MAPPED_PAYLOAD')
        except PermissionError:
            # Linux non-dumpable authentication processes are not science
            # consumers. Do not elevate, read credentials, or claim a full
            # /proc scan. Scheduler + sealed consumer inputs above are the
            # primary consumer proof; any other unreadable process blocks.
            name=(proc/'comm').read_text().strip()
            require(name in ('(sd-pam)','sshd'),'UNCLASSIFIED_UNREADABLE_PROCESS')
            daemon_exclusions.append(dict(pid=proc.name,name=name,reason='AUTHENTICATION_DAEMON_FD_ACCESS_DENIED'))
        except (FileNotFoundError,ProcessLookupError):continue
    return dict(at=now(),live=live,local_same_uid_processes_scanned=scanned,auxiliary_fd_scan_exclusions=daemon_exclusions,
        new_cold_runs_and_collectors_do_not_consume_old_targets=True,
        frozen_collector_source_sha256=sha(NEW/'source/project/run_scripts/server2_qwen_finish.py'))

def run():
    require(not (OUT/'deletion.json').exists(),'NO_REPEAT_DELETION')
    require(read(OLD/'source-lock.json')['code_commit']==SOURCE,'SOURCE_IDENTITY')
    jobs={j['job_id']:j for j in read(OLD/'released.json')['jobs']};targets=[]
    for jid,cell,method,batch in TARGETS:
        config=read(OLD/'configs'/f'{cell}.json');require(config['method']==method and config['model']=='qwen25' and config['dataset']=='zsre','METHOD')
        require(jobs[jid]['cell']==cell and jobs[jid]['source']==SOURCE,'JOB_BINDING')
        folder=OLD/'runs'/cell/'checkpoint';pointer=folder/'latest.json';p=read(pointer)
        require(p['batch']==batch and Path(p['file']).name==p['file'],'POINTER_SCOPE')
        path=folder/p['file'];s=path.lstat()
        require(stat.S_ISREG(s.st_mode) and s.st_nlink==1 and s.st_uid==os.getuid() and path.resolve()==path,'REGULAR_UNSHARED_EXACT_PATH')
        commit=OLD/'runs'/cell/'commits'/f'b{batch:02d}.json';c=read(commit)
        require(c['checkpoint_sha256']==p['sha256'] and c['checkpoint_identity']['code_commit']==SOURCE and
            c['checkpoint_identity']['config_sha256']==config['config_sha256'] and c['checkpoint_identity']['model_revision']=='a09a35458c702b33eeacc393d103063234e8bc28','COMMIT_CP_BINDING')
        require(sha(path)==p['sha256'],'FULL_SHA_MISMATCH')
        targets.append(dict(job_id=jid,cell=cell,method=method,batch=batch,path=str(path),realpath=str(path.resolve()),
            sha256=p['sha256'],stat=identity(s),pointer_path=str(pointer),pointer_sha256=sha(pointer),
            commit_path=str(commit),commit_sha256=sha(commit),source=SOURCE,checkpoint_identity=c['checkpoint_identity']))
    initial=terminal();consumer=consumers(targets)
    write(OUT/'predelete.json',dict(at=now(),targets=targets,accounting=initial,consumers=consumer,
        authorization='USER-GH-SH2-QWEN-BROKEN-WEIGHTS-DELETE-20261010-R1',verified_recovery_copy='NOT_KNOWN_NO_RECOVERY_CLAIM'))
    before=os.statvfs(OLD);deleted=[]
    terminal();consumers(targets)
    for target in targets:
        path=Path(target['path']);require(identity(path.lstat())==target['stat'],'PRE_UNLINK_STAT_CHANGED')
        require(sha(target['pointer_path'])==target['pointer_sha256'] and sha(target['commit_path'])==target['commit_sha256'],'MANIFEST_CHANGED')
        fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
        try:
            require(identity(os.fstat(fd))==target['stat'],'PRE_UNLINK_FD_IDENTITY_CHANGED')
            require(identity(path.lstat())==target['stat'],'PRE_UNLINK_PATH_CHANGED')
            os.unlink(path)
        finally:os.close(fd)
        require(not path.exists() and not path.is_symlink(),'UNLINK_POSTCHECK')
        tombstone=dict(**target,deleted_at_utc=now(),absent=True,verified_recovery_copy='NOT_KNOWN',
            latest_metadata_preserved_but_payload_deleted=True)
        write(OUT/f"deleted-{target['job_id']}.json",tombstone);deleted.append(tombstone)
    after=os.statvfs(OLD)
    result=dict(at=now(),deleted=deleted,payload_count=len(deleted),deleted_logical_bytes=sum(t['stat']['bytes'] for t in targets),
        deleted_allocated_bytes=sum(t['stat']['blocks']*512 for t in targets),
        filesystem_available_delta_bytes=(after.f_bavail*after.f_frsize-before.f_bavail*before.f_frsize),
        filesystem_delta_not_exclusive_attribution=True,raw_source_context_config_manifests_KEEP=True,
        jobs_mutated=0,other_payloads_deleted=0,verified_recovery_copy='NOT_KNOWN_NO_RECOVERY_CLAIM')
    write(OUT/'deletion.json',result);print(json.dumps({k:v for k,v in result.items() if k!='deleted'}))

if __name__=='__main__':run()
