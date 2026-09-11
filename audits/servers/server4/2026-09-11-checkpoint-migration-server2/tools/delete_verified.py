"""Explicit receipt-bound individual unlink. Never recursive/glob deletion."""
from inventory import *
import argparse
def used(paths):
 hits=[]; denied=[];daemon_boundary=[];checked=0
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit():continue
  try:
   if proc.stat().st_uid!=os.getuid():continue
   checked+=1
   for fd in (proc/'fd').iterdir():
    try:
     target=os.readlink(fd)
     if target in paths or target.removesuffix(' (deleted)') in paths:hits.append(dict(pid=proc.name,kind='fd',path=target))
    except FileNotFoundError:pass
   for line in (proc/'maps').read_text().splitlines():
    part=line.split(None,5)
    if len(part)==6 and part[5] in paths:hits.append(dict(pid=proc.name,kind='mmap',path=part[5]))
  except (FileNotFoundError,ProcessLookupError):pass
  except PermissionError:
   try:comm=(proc/'comm').read_text().strip()
   except FileNotFoundError:continue
   # Login/session-manager daemons are not experiment consumers. Their user
   # children are scanned independently; unknown inaccessible processes HOLD.
   if comm in ['systemd','(sd-pam)','sshd']:
    daemon_boundary.append(dict(pid=proc.name,comm=comm,scope='OS_LOGIN_DAEMON_NOT_EXPERIMENT_CONSUMER; fd/mmap not accessible'))
   else:denied.append(proc.name)
 return dict(hits=hits,denied_owned_pids=denied,excluded_login_daemon_visibility=daemon_boundary,checked_owned_processes=checked)
def identity(e):
 p=Path(e['path']);s=p.lstat()
 assert p.is_absolute() and str(p.resolve())==e['realpath']==str(p)
 assert p.name in ['W-method-state.pt','W-M.pt'] or (p.parent.name=='checkpoints' and p.name in ['W1-M1.pt','W5-M5.pt','W10-M10.pt'])
 assert stat.S_ISREG(s.st_mode) and s.st_uid==os.getuid()==e['uid'] and s.st_nlink==e['nlink']==1
 assert (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)==(e['dev'],e['inode'],e['bytes'],e['mtime_ns']),str(p)
 return p
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--bundle',choices=['initial72'],required=True);ap.add_argument('--execute-user-approved',action='store_true');a=ap.parse_args()
 manifest=read(C/'source-manifest.json');es=[e for e in manifest['members'] if e['reuse']]
 receipt=C/'initial72-VERIFIED_DESTINATION-v2.json';cat=C/'initial72-retention-catalog.json'
 assert sha(receipt)=='209c6d506f3fdb9cad6255b63e761067e4935d510167bdfd51a1592c69391159'
 assert sha(cat)=='553d4155aeb0a12f0568b8f653b81bc1f13a2a4f78040e90250f43d1ebb3c0c8'
 v=read(receipt); assert v['status']=='VERIFIED_DESTINATION' and v['full_sha256'] and v['count']==len(es)==72
 dest={m['source_path']:m for m in read(cat)['members']};assert set(dest)=={e['path'] for e in es}
 assert read(C/'scheduler-scope.json')['target_squeue_rows']==0
 paths={e['path'] for e in es};check=used(paths)
 assert not check['hits'] and not check['denied_owned_pids'],check
 for e in es:
  d=dest[e['path']]['destination'];assert d['path']==e['retained_path'] and (d['bytes'],d['sha256'])==(e['bytes'],e['sha256'])
  p=identity(e);assert sha(p)==e['sha256'];identity(e)
 check2=used(paths);assert not check2['hits'] and not check2['denied_owned_pids'],check2
 for e in es:identity(e)
 plan=dict(bundle=a.bundle,source_manifest=member(C/'source-manifest.json'),receiver_receipt=member(receipt),retention_catalog=member(cat),members=es,in_use_before=check,in_use_immediately_before=check2,logical_bytes=sum(e['bytes'] for e in es),source_permanent_unlink=True,restore='copy verified destination retained_path back to original path after separate restore instruction',time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
 save(a.bundle+'-deletion-plan.json',plan)
 if not a.execute_user_approved:print('PLAN_ONLY');return
 fs=os.statvfs(R);before=fs.f_bavail*fs.f_frsize;removed=[]
 with (C/(a.bundle+'-unlink-journal.jsonl')).open('x') as journal:
  for e in es:
   p=identity(e);os.unlink(p)
   row=dict(source=e['path'],destination=e['retained_path'],sha256=e['sha256'],bytes=e['bytes'],removed=not p.exists(),reason='USER_APPROVED_VERIFIED_SERVER2_RETAINED_COPY')
   journal.write(json.dumps(row)+'\n');journal.flush();os.fsync(journal.fileno());removed.append(row)
 fs=os.statvfs(R);after=fs.f_bavail*fs.f_frsize
 save(a.bundle+'-deletion-receipt.json',dict(status='SOURCE_EXACT_FILES_REMOVED_DESTINATION_PRESERVED',removed=removed,count=len(removed),logical_bytes=sum(e['bytes'] for e in es),filesystem_available_before=before,filesystem_available_after=after,observed_available_delta=after-before,delta_caveat='shared filesystem concurrent activity; not exclusively attributed',receiver_receipt_sha256=sha(receipt),source_plan_sha256=sha(C/(a.bundle+'-deletion-plan.json')),recursive_delete=0,other_assets_changed=0,GPU=0))
 print('DELETED',len(removed),sum(e['bytes'] for e in es),'AVAILABLE_DELTA',after-before,flush=True)
if __name__=='__main__':main()
