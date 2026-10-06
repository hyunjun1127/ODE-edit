"""Bounded lstat-only project inventory; never opens scientific payloads or follows links."""
import collections,datetime,json,os,stat,subprocess,time,sys
from pathlib import Path
ROOT=Path('/data/janghj/ODE-edit')
OUT=ROOT/'local/price-storage-inventory-20261007'
WT=OUT/'worktree'
MAX_ENTRIES=400000
MAX_SECONDS=120
EXCLUDED_NAMES={'.git','.ssh','.codex','.config','.netrc','_netrc','.env','credentials','credentials.json','settings','secrets','token','tokens','__pycache__','node_modules','site-packages','venv','.venv','sdk-env','uv-cache'}
EXT={'.pt','.pth','.ckpt','.bin','.safetensors','.npy','.npz','.mmap','.memmap','.tensor'}
def main():
 os.nice(15)
 os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:1])
 begin=time.monotonic(); now=datetime.datetime.now(datetime.timezone.utc).isoformat()
 ready=json.loads((WT/'agents/server3/experiment-ready-paths-20260919-v1.json').read_text())
 supplement='--bootstrap-supplement' in sys.argv
 roots=[(ROOT,'repository_root')]
 for m in ready['models'].values():
  roots.append((Path(m['snapshot']).parents[1],'registered_HF_model_cache'))
  a=m['assets']; roots.append((Path(a['projector']['path']),'registered_projector'))
  roots.append((Path(a['canonical_hparams']['path']),'registered_hparams'))
  for c in a['covariance'].values():roots.append((Path(c['path']),'registered_covariance'))
 wts=[Path(line[9:]) for line in subprocess.check_output(['git','worktree','list','--porcelain'],cwd=ROOT,text=True).splitlines() if line.startswith('worktree ')]
 for w in wts:
  if not w.is_relative_to(ROOT):roots.append((w,'registered_external_worktree'))
 if supplement:
  roots=[(Path('/data/janghj/.cache/huggingface/hub/models--EleutherAI--gpt-j-6b'),'bootstrap_registered_model_cache'),(Path('/data/janghj/EasyEdit/examples/null_space_project_gpt-j-6b.pt'),'bootstrap_registered_projector')]
  wts=[]
 coverage={'started_utc':now,'roots':[{'path':str(p),'reason':k} for p,k in roots],'worktrees':[str(w) for w in wts],'no_symlink_follow':True,'limits':{'entries':MAX_ENTRIES,'seconds':MAX_SECONDS},'excluded_names':sorted(EXCLUDED_NAMES),'excluded':[],'errors':[],'truncated':False,'directories':0,'regular_paths':0,'symlink_paths':0,'other_paths':0}
 visited=set(); seen_dirs=set(); entries=0
 with (OUT/('supplement-stat.jsonl' if supplement else 'full-stat.jsonl')).open('x') as f:
  for base,reason in roots:
   stack=[base]
   while stack:
    if entries>=MAX_ENTRIES or time.monotonic()-begin>MAX_SECONDS:coverage['truncated']=True;break
    p=stack.pop(); entries+=1
    if p==OUT or p.name in EXCLUDED_NAMES or (p.parent.name=='servers' and p.name=='local') or p.name.startswith('.env.'):
     coverage['excluded'].append({'path':str(p),'reason':'task-output/self or private/config/package infrastructure'});continue
    if str(p) in visited:continue
    visited.add(str(p))
    try:s=p.lstat()
    except OSError as e:coverage['errors'].append({'path':str(p),'error':type(e).__name__});continue
    if stat.S_ISDIR(s.st_mode):
     di=(s.st_dev,s.st_ino)
     if di in seen_dirs:continue
     seen_dirs.add(di);coverage['directories']+=1
     try:
      with os.scandir(p) as it:stack.extend(Path(e.path) for e in it)
     except OSError as e:coverage['errors'].append({'path':str(p),'error':type(e).__name__})
     continue
    if stat.S_ISREG(s.st_mode):kind='regular';coverage['regular_paths']+=1
    elif stat.S_ISLNK(s.st_mode):kind='symlink';coverage['symlink_paths']+=1
    else:coverage['other_paths']+=1;continue
    tensor=p.suffix.lower() in EXT or (not p.suffix and any(x in p.name.lower() for x in ['weight','tensor','memmap','teacher','history','activation']))
    row={'path':str(p),'type':kind,'logical_bytes':s.st_size,'allocated_bytes':s.st_blocks*512,'dev':s.st_dev,'inode':s.st_ino,'nlink':s.st_nlink,'mtime_ns':s.st_mtime_ns,'mtime_utc':datetime.datetime.fromtimestamp(s.st_mtime,datetime.timezone.utc).isoformat(),'uid':s.st_uid,'registered_root':str(base),'root_reason':reason,'tensor_extension_or_name':tensor}
    f.write(json.dumps(row,separators=(',',':'))+'\n')
   if coverage['truncated']:break
 v=os.statvfs(ROOT)
 coverage.update(finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),duration_seconds=time.monotonic()-begin,entries=entries,df={'block_size':v.f_frsize,'total_bytes':v.f_blocks*v.f_frsize,'free_including_reserved_bytes':v.f_bfree*v.f_frsize,'available_bytes':v.f_bavail*v.f_frsize,'inodes':v.f_files,'available_inodes':v.f_favail},content_reads='none: lstat/scandir/statvfs only; tiny existing readiness manifest read to establish roots',directory_block_bytes_included=False)
 (OUT/('supplement-coverage.json' if supplement else 'scan-coverage.json')).write_text(json.dumps(coverage,indent=2)+'\n')
 print(json.dumps({k:coverage[k] for k in ['regular_paths','symlink_paths','directories','truncated','duration_seconds','df']}))
if __name__=='__main__':main()
