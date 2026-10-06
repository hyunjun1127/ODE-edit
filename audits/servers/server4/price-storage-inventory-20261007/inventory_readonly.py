"""Bounded metadata-only inventory. Never imports/loads tensor content or changes inputs."""
import csv
import datetime as dt
import hashlib
import heapq
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import time
from collections import Counter, defaultdict

ROOT = Path('/data/janghj/ODE-edit')
WT = Path(__file__).resolve().parents[4]
TASK = 'price-storage-inventory-20261007'
OUT = ROOT / 'local' / TASK / 'metadata'
AUDIT = WT / 'audits/servers/server4' / TASK
CLASSES = ['PRICE_REQUIRED','OTHER_TASK_REQUIRED','REPRODUCTION_KEEP','UNREFERENCED_CANDIDATE','UNKNOWN']
CURRENT = ['jlz-price-cap-base-repair-2k','jlz-price-alpha-writer-2k']
SENSITIVE = re.compile(r'(^|[/_.-])(secret|credential|private|netrc|ssh|auth|token|connection)([/_.-]|$)', re.I)
def sensitive(p):
    # Tokenizer and token-identity assets are public experiment input, not credentials.
    return bool(SENSITIVE.search(str(p))) or str(p).endswith('.env') or '/.codex/' in str(p)
def save(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f: json.dump(x, f, ensure_ascii=False, indent=2); f.write('\n')
def objects(x):
    if isinstance(x, dict):
        yield x
        for k,v in x.items():
            if not SENSITIVE.search(k): yield from objects(v)
    elif isinstance(x,list):
        for v in x: yield from objects(v)
def paths(x):
    if isinstance(x, str) and x.startswith('/data/janghj/') and not sensitive(x): yield x
    elif isinstance(x,dict):
        for k,v in x.items():
            if not SENSITIVE.search(k): yield from paths(v)
    elif isinstance(x,list):
        for v in x: yield from paths(v)

def main():
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    refs = defaultdict(set); historical = defaultdict(set); binding=[]; docs=[]; external=set()
    for task in CURRENT:
        for name in ['config.json','execution.lock.json']:
            p=ROOT/'local'/task/'attempt'/name
            raw=p.read_bytes(); data=json.loads(raw)
            docs.append({'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
            for q in paths(data): refs[q].add(task+'/'+name)
            for a in objects(data):
                q=a.get('path')
                if isinstance(q,str) and q.startswith('/data/janghj/') and not sensitive(q) and 'bytes' in a:
                    try:
                        s=os.lstat(q)
                        binding.append({'path':q,'reference':task+'/'+name,'prior_sha256':a.get('sha256'),
                            'size_match':s.st_size==a['bytes'],'inode_match':None if 'inode' not in a else s.st_ino==a['inode'],
                            'mtime_match':None if 'mtime_ns' not in a else s.st_mtime_ns==a['mtime_ns'],
                            'sha_recomputed':False,'symlink':stat.S_ISLNK(s.st_mode)})
                    except OSError as e: binding.append({'path':q,'error':type(e).__name__})
    # Only repository receipts, not raw results/private settings, are read for historical references.
    candidates=[]
    for base,pat in [('runs','**/*submission*.json'),('tasks/status','**/server4.json'),('audits/servers/server4','**/*inventory*.json')]:
        candidates.extend((WT/base).glob(pat))
    total_read=0
    for p in sorted(set(candidates)):
        if TASK in str(p.relative_to(WT)) or sensitive(p.relative_to(WT)) or p.is_symlink(): continue
        size=p.stat().st_size
        if size>8*1024**2 or total_read+size>32*1024**2: continue
        try: data=json.loads(p.read_bytes())
        except (OSError,ValueError): continue
        total_read+=size; docs.append({'path':str(p.relative_to(WT)),'bytes':size,'role':'historical_reference_only'})
        for q in paths(data):
            if q.startswith(str(ROOT)+'/local/'):
                family=q[len(str(ROOT)+'/local/'):].split('/')[0]
                if family not in ['worktrees','state','logs','cache','analysis','datasets']: historical[family].add(str(p.relative_to(WT)))
            # External paths only explicit checkpoint/tensor/cache files; never arbitrary home scans.
            elif q.startswith('/data/janghj/EasyEdit/') and Path(q).suffix.lower() in ['.pt','.pth','.ckpt','.npy','.npz','.safetensors','.bin']:
                external.add(q)
    for q in refs:
        if q.startswith('/data/janghj/EasyEdit/examples/') or '/.cache/huggingface/hub/' in q: external.add(q)
    worktrees=[x[9:] for x in subprocess.check_output(['git','worktree','list','--porcelain'],cwd=ROOT,text=True).splitlines() if x.startswith('worktree ')]
    wts=set(worktrees)-{str(ROOT)}
    scan_roots=[str(ROOT)]
    for w in sorted(wts):
        if w == str(WT): continue
        for d in ['local','outputs','output','results']:
            p=Path(w)/d
            if p.is_dir() and not p.is_symlink(): scan_roots.append(str(p))
    seen_paths=set(); seen_inodes=set(); sums={c:Counter() for c in CLASSES}; directories=defaultdict(Counter)
    types=defaultdict(Counter); top=[]; tensor_top=[]; links=[]; errors=[]; skips=Counter(); symlinks=[]
    dirs=0; files=0; exhausted=False; remaining=[]
    suffixes={'.pt','.pth','.ckpt','.bin','.safetensors','.npy','.npz','.mmap','.memmap'}
    def classify(p):
        if p in refs: return 'PRICE_REQUIRED',';'.join(sorted(refs[p])),'현재 봉인 입력/lock의 exact path'
        for task in CURRENT:
            if p.startswith(str(ROOT/'local'/task)+'/'): return 'PRICE_REQUIRED',task,'현재 실행/대기 task namespace: source/raw/config 보존'
        if '/wandb' in p.lower() or '/tracking/' in p:
            return 'OTHER_TASK_REQUIRED','W&B policy / storage inventory authority','SDK/spool 보호; 동기화 여부 미검증'
        rel=p[len(str(ROOT)+'/local/'):] if p.startswith(str(ROOT)+'/local/') else ''
        family=rel.split('/')[0]
        if family in historical:
            return 'REPRODUCTION_KEEP',sorted(historical[family])[0],'기존 submission/inventory/status가 참조하는 task family; 보수적 재현 보존'
        if p in external: return 'REPRODUCTION_KEEP','기존 inventory의 exact 외부 tensor path','기존 보호 자산/재현 참조; current use 미검증'
        return 'UNKNOWN','','참조/소유 용도 불완전: 보존, 미참조 확정 아님'
    fields=['server','absolute_path','artifact_kind','logical_bytes','allocated_bytes','count','device_inode_or_hardlink_group','nlink','mtime','referenced_task_or_lock','classification','classification_evidence','uniqueness_or_reproducibility','uncertainty','delete_authorized']
    full=OUT/'inventory-full.csv'
    with full.open('x',newline='') as f:
        writer=csv.DictWriter(f,fields); writer.writeheader()
        def record(p,s):
            nonlocal files
            if p in seen_paths:return
            seen_paths.add(p); files+=1
            c,ref,why=classify(p); inode=(s.st_dev,s.st_ino); allocated=s.st_blocks*512
            kind='tensor_or_checkpoint_by_extension' if Path(p).suffix.lower() in suffixes else ('large_extensionless_or_other' if s.st_size>=1024**3 else 'ordinary_file')
            if '/blobs/' in p and p in refs: kind='protected_HF_blob'
            r=dict(zip(fields,['server4',p,kind,s.st_size,allocated,1,f'{s.st_dev}:{s.st_ino}',s.st_nlink,dt.datetime.fromtimestamp(s.st_mtime,dt.timezone.utc).isoformat(),ref,c,why,'byte equality not checked; inode identity only','동시 실행 snapshot; active use/byte duplicate 미확정',False]))
            writer.writerow(r); sums[c]['path_count']+=1
            if inode in seen_inodes:
                sums[c]['duplicate_hardlink_paths']+=1
                return
            seen_inodes.add(inode); sums[c]['unique_files']+=1; sums[c]['logical_bytes']+=s.st_size; sums[c]['allocated_bytes']+=allocated
            types[kind]['count']+=1;types[kind]['logical_bytes']+=s.st_size;types[kind]['allocated_bytes']+=allocated
            if p.startswith(str(ROOT)+'/local/'): group=str(ROOT/'local'/p[len(str(ROOT)+'/local/'):].split('/')[0])
            elif p.startswith(str(ROOT)+'/'):group=str(ROOT)+'/[non-local]'
            else:group='EXTERNAL_EXACT_REFERENCES'
            directories[group]['files']+=1;directories[group]['logical_bytes']+=s.st_size;directories[group]['allocated_bytes']+=allocated
            item=(s.st_size,p,r)
            heapq.heappush(top,item)
            if len(top)>20:heapq.heappop(top)
            if kind!='ordinary_file':
                heapq.heappush(tensor_top,item)
                if len(tensor_top)>100:heapq.heappop(tensor_top)
            if s.st_nlink>1 and len(links)<100:links.append(r)
        # Exact current assets first gives PRICE classification priority for shared inodes.
        for p in sorted(set(refs)|external):
            if not (p.startswith(str(ROOT)+'/') or p in external) or sensitive(p):continue
            try:
                s=os.lstat(p)
                if stat.S_ISREG(s.st_mode): record(p,s)
                elif stat.S_ISLNK(s.st_mode):symlinks.append(p)
            except OSError as e:
                if len(errors)<200:errors.append({'path':p,'error':type(e).__name__})
        for scan_root in scan_roots:
            stack=[scan_root]
            while stack:
                if files>=1000000 or time.time()-started>180:
                    exhausted=True;remaining.extend(stack);break
                d=stack.pop();dirs+=1
                try:
                    with os.scandir(d) as it:
                        for e in it:
                            p=e.path
                            if sensitive(p) or e.name in ['.git','.codex','.ssh','.gnupg']:
                                skips['private_or_git']+=1;continue
                            if p==str(ROOT/'local'/TASK):skips['own_inventory_output']+=1;continue
                            if p in wts:skips['registered_worktree_source']+=1;continue
                            try:
                                s=e.stat(follow_symlinks=False)
                                if stat.S_ISLNK(s.st_mode):
                                    skips['symlink_not_followed']+=1
                                    if len(symlinks)<300:symlinks.append(p)
                                elif s.st_uid!=os.getuid():skips['foreign_uid']+=1
                                elif stat.S_ISDIR(s.st_mode):stack.append(p)
                                elif stat.S_ISREG(s.st_mode):record(p,s)
                                else:skips['special_file']+=1
                            except OSError as ex:
                                if len(errors)<200:errors.append({'path':p,'error':type(ex).__name__})
                except OSError as ex:
                    if len(errors)<200:errors.append({'path':d,'error':type(ex).__name__})
                if dirs%100==0: time.sleep(.005)
            if exhausted:break
    fs=os.statvfs(ROOT)
    summary={'task_id':TASK,'snapshot_started_utc':dt.datetime.fromtimestamp(started,dt.timezone.utc).isoformat(),'elapsed_seconds':time.time()-started,
        'delete_authorized':False,'classification_totals':sums,'kind_totals':types,'top20':[x[2] for x in sorted(top,reverse=True)],
        'directories':dict(sorted(directories.items(),key=lambda x:-x[1]['allocated_bytes'])),'observed_hardlinks':links,
        'total_unique_files':len(seen_inodes),'total_paths':files,'full_inventory_path':str(full),
        'filesystem':{'total_bytes':fs.f_blocks*fs.f_frsize,'available_bytes':fs.f_bavail*fs.f_frsize,'free_bytes_including_reserved':fs.f_bfree*fs.f_frsize,'free_inodes':fs.f_favail},
        'confirmed_reclaimable_bytes':None,'candidate_reclaimable_bytes':None,'reason':'삭제 승인 없음; 참조 없는 파일/동일 byte 중복을 확정하지 않음',
        'scheduler_queries':0,'tensor_content_reads':0,'model_loads':0,'large_SHA_recomputations':0,'reviewer':'owner metadata audit; independent reviewer not used'}
    coverage={'roots':scan_roots,'registered_worktrees':worktrees,'external_mode':'기존 receipt의 exact file lstat only; no directory traversal',
        'external_exact_paths':sorted(external),'directories_visited':dirs,'limits':{'files':1000000,'seconds':180},'exhausted':exhausted,
        'unvisited_stack':remaining,'skips':skips,'errors':errors,'symlink_samples_no_follow':sorted(set(symlinks))[:300],
        'notes':['등록 WT source 전체는 제외하고 local/output/outputs/results 실제 디렉터리만 포함','민감 경로는 이름 단계에서 제외; 내용 읽기 없음','현재 config/lock 및 제한된 tracked receipt만 JSON 내용 읽음','snapshot이며 동시 실행 write의 일관된 atomic filesystem snapshot 아님','UNKNOWN은 삭제후보 아님; live process/openfile 전수검사 안함','quota 미확인; df available은 프로젝트 할당량/보장 여유 아님','다른 사용자와 home 전체/비등록 외부 cache 스캔 없음'],
        'reference_documents':docs,'historical_reference_families':{k:sorted(v)[:3] for k,v in historical.items()},'asset_binding':binding}
    save(AUDIT/'summary.json',summary);save(AUDIT/'coverage.json',coverage)
    selected={x[1]:x[2] for x in top+tensor_top}
    with (AUDIT/'inventory.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(sorted(selected.values(),key=lambda r:-r['logical_bytes']))
    save(OUT/'full-inventory-receipt.json',{'path':str(full),'bytes':full.stat().st_size,'sha256':hashlib.sha256(full.read_bytes()).hexdigest(),'rows':files,'delete_authorized':False})
    print(json.dumps({'paths':files,'unique':len(seen_inodes),'seconds':summary['elapsed_seconds'],'exhausted':exhausted,'classes':sums,'top20':[(x[1],x[0]) for x in sorted(top,reverse=True)],'directories':summary['directories']},ensure_ascii=False))

if __name__=='__main__':main()
