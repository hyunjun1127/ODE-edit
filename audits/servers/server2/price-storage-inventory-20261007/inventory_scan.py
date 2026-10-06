"""읽기전용 bounded metadata inventory. Tensor load/hash/remote/scheduler 없음."""
import collections,csv,datetime,hashlib,json,os,re,resource,stat,subprocess,time
from pathlib import Path

REPO=Path('/mnt/raid5/janghj/ODE-edit')
WT=Path(__file__).resolve().parents[4]
NAME='price-storage-inventory-20261007'
LOCAL=REPO/'local'/NAME
AUDIT=WT/'audits/servers/server2'/NAME
REPORT=WT/'experiment-reports/servers/server2'/NAME
TENSOR={'.pt','.pth','.ckpt','.bin','.safetensors','.npy','.npz','.mmap','.memmap'}
SKIP={'.git','.ssh','.gnupg','__pycache__','.venv','venv','sdk-env','site-packages','node_modules'}
SECRET=re.compile(r'(credential|secret|private[-_]?(settings|connection)|\.netrc|\.env$)',re.I)
CLASSES=['PRICE_REQUIRED','OTHER_TASK_REQUIRED','REPRODUCTION_KEEP','UNREFERENCED_CANDIDATE','UNKNOWN']

def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n')
def paths(obj):
    if isinstance(obj,dict):
        for k,v in obj.items():
            if not SECRET.search(k):yield from paths(v)
    elif isinstance(obj,list):
        for v in obj:yield from paths(v)
    elif isinstance(obj,str) and obj.startswith('/mnt/raid5/janghj/') and '\n' not in obj:yield obj
def read_meta(p):
    s=p.lstat()
    if not stat.S_ISREG(s.st_mode) or s.st_size>2*1024**2 or SECRET.search(p.name):return None
    return json.loads(p.read_text())
def main():
    started=time.monotonic();os.nice(15);os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
    resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3))
    LOCAL.mkdir(exist_ok=False);REPORT.mkdir(parents=True,exist_ok=True)
    raw=subprocess.check_output(['git','worktree','list','--porcelain'],cwd=REPO,text=True)
    wts=[Path(s[9:]) for s in raw.splitlines() if s.startswith('worktree ')]
    roots=[REPO/'local'];omissions=[];refs={};meta_reads=[];external=set();migration={}
    # Metadata only from already registered FE inputs; no live science result read.
    fe_config=REPO/'local/fe-sequential-2k/attempt-online-10k-r1/config.json'
    fe=read_meta(fe_config)
    for p in paths(fe):
        refs[p]=('OTHER_TASK_REQUIRED',str(fe_config))
        if p.startswith('/mnt/raid5/janghj/EasyEdit/') and Path(p).suffix in TENSOR:external.add(p)
        if '/.cache/huggingface/hub/models--' in p:
            parts=Path(p).parts;i=next(i for i,v in enumerate(parts) if v.startswith('models--'))
            external.add(str(Path(*parts[:i+1])))
    for p in [fe['stream'],fe['contexts']]:refs[p]=('OTHER_TASK_REQUIRED',str(fe_config))
    meta_reads.append(str(fe_config))
    # Explicit protected projector references; not a tensor-content/parity check.
    for model in ('Meta-Llama-3-8B-Instruct','Qwen2.5-7B-Instruct'):
        p=f'/mnt/raid5/janghj/EasyEdit/examples/null_space_project_{model}.pt'
        external.add(p);refs[p]=('REPRODUCTION_KEEP','USER protected projector; project/run_scripts/jlz_price_alpha_writer/prepare.py (S4 input, S2 analogue only)')
    mapping=WT/'transfers/verifications/2026-09-11-checkpoint-migration-server2-destination/migration-map.csv'
    with mapping.open() as f:
        for r in csv.DictReader(f):migration[r['destination']]=r
    meta_reads.append(str(mapping))
    price=[]
    for task in ('jlz-price-cap-base-repair-2k','jlz-price-alpha-writer-2k'):
        p=WT/'runs'/task/'submission.json';d=read_meta(p)
        price.append(dict(task=task,source=d['source'],published_lock=d['lock'],server='server4',
            local_input_claim=False,remote_read=False,scheduler_query=False))
        meta_reads.append(str(p))
        for v in paths(d):refs[v]=('PRICE_REQUIRED',str(p))
    for wt in wts:
        for folder in ('local','output','outputs','results','artifacts','cache'):
            p=wt/folder
            if p.is_symlink():omissions.append(dict(path=str(p),reason='SYMLINK_ROOT_NOT_FOLLOWED'))
            elif p.is_dir() and not p.is_relative_to(REPO/'local') and p not in roots:roots.append(p)
    roots += [Path(p) for p in sorted(external)]
    dump(LOCAL/'input-evidence.json',dict(worktrees=[str(x) for x in wts],roots=[str(x) for x in roots],
        metadata_read=meta_reads,price=price,FE_config=str(fe_config),delete_authorized=False))
    files=[];seen_dirs=set();errors=[];skips=collections.Counter();scanned=0;deadline=started+180
    def walk(root):
        nonlocal scanned
        stack=[root]
        while stack:
            if time.monotonic()>deadline or scanned>=400000:
                omissions.append(dict(path=str(root),reason='TIME180S_OR_FILE400K_BOUND',pending_directories=len(stack)));return
            p=stack.pop()
            try:
                s=p.lstat()
                if s.st_uid!=os.getuid():skips['OTHER_OWNER']+=1;continue
                if p==LOCAL or p.name in SKIP or SECRET.search(p.name):skips['PRIVATE_ENV_CODECACHE_OR_OWN_OUTPUT']+=1;continue
                if stat.S_ISLNK(s.st_mode):
                    files.append(dict(path=str(p),type='symlink',size=s.st_size,allocated=s.st_blocks*512,dev=s.st_dev,ino=s.st_ino,nlink=s.st_nlink,mtime=s.st_mtime_ns,root=str(root)))
                    skips['SYMLINK_NOT_FOLLOWED']+=1;continue
                if stat.S_ISDIR(s.st_mode):
                    key=(s.st_dev,s.st_ino)
                    if key in seen_dirs:continue
                    seen_dirs.add(key)
                    with os.scandir(p) as it:stack.extend(Path(e.path) for e in it)
                elif stat.S_ISREG(s.st_mode):
                    scanned+=1
                    files.append(dict(path=str(p),type='regular',size=s.st_size,allocated=s.st_blocks*512,dev=s.st_dev,ino=s.st_ino,nlink=s.st_nlink,mtime=s.st_mtime_ns,root=str(root)))
            except OSError as e:errors.append(dict(path=str(p),error=type(e).__name__))
    for root in roots:walk(root)
    # Root top-level tensor files only; do not traverse unrelated repositories/home.
    for p in REPO.iterdir():
        if p.suffix.lower() in TENSOR and not p.is_symlink():walk(p)
    # Frozen project config/input locks only, bounded content reads; never metrics/logs.
    metadata_budget=0
    for row in files:
        p=Path(row['path']);n=p.name.lower()
        if row['type']!='regular' or p.suffix!='.json' or row['size']>2*1024**2:continue
        if not (n in ('config.json','configuration.json','configuration-final.json','input.lock.json','execution.lock.json')):continue
        if '/source/' in str(p) or '/upstream/' in str(p):continue
        if metadata_budget+row['size']>24*1024**2:omissions.append(dict(path='metadata_index',reason='24MiB_CONTENT_BOUND'));break
        try:d=read_meta(p)
        except (OSError,ValueError):continue
        if d is None:continue
        metadata_budget+=row['size'];meta_reads.append(str(p))
        cls='PRICE_REQUIRED' if any(t in str(p) for t in ('jlz-price-cap-base-repair','jlz-price-alpha-writer')) else 'REPRODUCTION_KEEP'
        for v in paths(d):refs.setdefault(v,(cls,str(p)))
    # Exact path/directory references. No unreferenced=>safe-delete inference.
    dir_refs={p:v for p,v in refs.items() if any(x['path'].startswith(p.rstrip('/')+'/') for x in files[:0])}
    # Avoid filesystem-following is_dir on references. Prefix lookup uses lexical ancestors.
    def classify(p):
        name=str(p)
        for parent in [p,*p.parents]:
            if str(parent) in refs:return refs[str(parent)]
        if name in migration:return 'REPRODUCTION_KEEP',str(mapping)
        if '/fe-sequential-2k/' in name or '/fe-live-metrics' in name:return 'OTHER_TASK_REQUIRED','FE sealed config / raw & telemetry retained; current progress not queried'
        if '/wandb' in name or '/tracking/' in name or p.suffix=='.wandb':return 'OTHER_TASK_REQUIRED','USER protected W&B spool; sync status not inspected'
        if '/checkpoint-archives/' in name or '/blue-checkpoint-downstream/' in name:return 'REPRODUCTION_KEEP','S4->S2 archive retention authority and source removal receipts'
        if '/.cache/huggingface/hub/models--' in name:return 'REPRODUCTION_KEEP','Explicit registered base-model cache; protected, no content read'
        if '/datasets/' in name:return 'OTHER_TASK_REQUIRED','USER protected fixed dataset/source input'
        if p.suffix.lower() in ('.json','.jsonl','.csv','.log','.out','.err','.md','.png','.pdf','.tar','.gz'):
            return 'REPRODUCTION_KEEP','Project source/measurement/receipt preservation; no science content read'
        return 'UNKNOWN','No complete path-use evidence in bounded metadata index; KEEP'
    inode_paths=collections.defaultdict(list)
    for r in files:inode_paths[(r['dev'],r['ino'])].append(r)
    unique=[];aliases=0
    rank={k:i for i,k in enumerate(CLASSES)}
    for key,group in inode_paths.items():
        group.sort(key=lambda r:(rank[classify(Path(r['path']))[0]],r['path']))
        r=group[0];aliases+=len(group)-1;p=Path(r['path']);cls,evidence=classify(p)
        kind='tensor_extension' if p.suffix.lower() in TENSOR else ('large_or_extensionless' if r['size']>=64*1024**2 or not p.suffix else 'other_metadata_raw_source')
        if r['type']=='symlink':kind='symlink_not_followed'
        old=migration.get(str(p));oldsha=old['sha256'] if old else ''
        unique.append(dict(server='server2',absolute_path=str(p),artifact_kind=kind,logical_bytes=r['size'],allocated_bytes=r['allocated'],count=1,
            device_inode_or_hardlink_group=f'{key[0]}:{key[1]}',nlink=r['nlink'],observed_alias_count=len(group),
            mtime=datetime.datetime.fromtimestamp(r['mtime']/1e9,datetime.timezone.utc).isoformat(),root=r['root'],
            referenced_task_or_lock=evidence,classification=cls,classification_evidence=evidence,
            historical_sha256=oldsha,historical_size_matches=(int(old['bytes'])==r['size']) if old else '',
            uniqueness_or_reproducibility='dev/inode deduplicated; byte equivalence of distinct inodes NOT_VERIFIED',
            uncertainty='metadata-only; no fd/no-use proof; historical SHA not rehashed; concurrent file growth possible',delete_authorized=False))
    fields=list(unique[0])
    def csv_out(path,rows):
        with path.open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    csv_out(LOCAL/'full-inventory.csv',unique)
    dump(LOCAL/'hardlinks.json',[dict(device_inode=f'{k[0]}:{k[1]}',paths=[r['path'] for r in v]) for k,v in inode_paths.items() if len(v)>1])
    tensor=[r for r in unique if r['artifact_kind']=='tensor_extension']
    def totals(rows):return dict(unique_files=len(rows),logical_bytes=sum(r['logical_bytes'] for r in rows),allocated_bytes=sum(r['allocated_bytes'] for r in rows))
    categories={c:totals([r for r in unique if r['classification']==c]) for c in CLASSES}
    tc={c:totals([r for r in tensor if r['classification']==c]) for c in CLASSES}
    top=sorted(unique,key=lambda r:r['logical_bytes'],reverse=True)[:20]
    groups=collections.defaultdict(list)
    for r in unique:
        p=Path(r['absolute_path'])
        if p.is_relative_to(REPO/'local'):group=str(REPO/'local'/p.relative_to(REPO/'local').parts[0])
        else:group=r['root']
        groups[group].append(r)
    aggregates=[dict(path=k,**totals(v)) for k,v in groups.items()]
    compact={r['absolute_path']:r for r in top}
    for cls in CLASSES:
        for r in sorted((r for r in tensor if r['classification']==cls),key=lambda r:r['logical_bytes'],reverse=True)[:20]:compact[r['absolute_path']]=r
    csv_out(AUDIT/'inventory.csv',list(compact.values()))
    observed={r['path']:r for r in files}
    retained=dict(expected=len(migration),present=sum(p in observed for p in migration),
        size_matches=sum(p in observed and observed[p]['size']==int(v['bytes']) for p,v in migration.items()),
        historical_bytes=sum(int(v['bytes']) for v in migration.values()),hash_recomputed=False)
    vfs=os.statvfs(REPO)
    summary=dict(status='READONLY_INVENTORY_COMPLETE',nonce='USER-GH-ALL-SH-PRICE-STORAGE-INVENTORY-20261007-SERVER2',
        host=os.uname().nodename,session='01a0493a-074c-7f91-9a13-769116326fef',actual_cwd=str(REPO),
        publication_base=subprocess.check_output(['git','rev-parse','HEAD'],cwd=WT,text=True).strip(),
        categories=categories,tensor_categories=tc,totals=totals(unique),tensor_totals=totals(tensor),
        deduplicated_path_aliases=aliases,top20=top,directory_aggregates=sorted(aggregates,key=lambda r:r['allocated_bytes'],reverse=True),
        PRICE_reference=price,local_PRICE_paths=[r['absolute_path'] for r in unique if r['classification']=='PRICE_REQUIRED'][:30],
        retained_183=retained,confirmed_reclaimable_bytes=None,deleted_bytes=0,delete_authorized=False,
        filesystem=dict(block_size=vfs.f_frsize,available_bytes=vfs.f_bavail*vfs.f_frsize,free_inodes=vfs.f_favail),
        duration_seconds=time.monotonic()-started,GPU=0,model_load=0,tensor_load=0,large_SHA_reads=0,Slurm=0,WandB_new_run=0,
        new_source_or_raw_mutation=0,raw_inventory=str(LOCAL/'full-inventory.csv'))
    coverage=dict(roots=[str(p) for p in roots],registered_worktrees=len(wts),regular_files_observed=scanned,
        directory_inodes=len(seen_dirs),skips=dict(skips),omissions=omissions,errors=errors,metadata_read=meta_reads,
        metadata_content_bytes=metadata_budget,symlink_follow=False,max_files=400000,max_seconds=180,
        full_inventory_scope='regular+symlink metadata under listed roots; not whole filesystem',
        excluded='secrets/private settings/env/package caches; unregistered roots; symlink targets not traversed',
        limitations=['Not a snapshot: concurrent growth possible','Distinct inode byte duplication NOT_VERIFIED',
            'No complete use graph; no UNREFERENCED_CANDIDATE assigned from filename or size alone',
            'S4 published source/input lock metadata only; no S4 filesystem or scheduler access',
            'External assets outside the explicit registered FE model/stat and projector list not swept'])
    dump(AUDIT/'summary.json',summary);dump(AUDIT/'coverage.json',coverage)
    dump(LOCAL/'coverage-full.json',coverage)
    G=1024**3
    lines=['# Server2 PRICE 기준 읽기전용 저장공간 점검','',
        '삭제 권한 **없음** (`delete_authorized=false`). 파일 삭제/이동/압축/전송/덮어쓰기/권한변경0, GPU/Slurm/W&B 신규 실행0. 원 dirty1391항목 보존.','',
        '## 기준 및 경계','',
        '실제 server2 / SH2 session01a0493a-074c-7f91-9a13-769116326fef / `/mnt/raid5/janghj/ODE-edit`. 현재 registry/ignored boundary와 일치한다. 별도 non-main worktree 사용.','',
        'PRICE MEMIT source87a5a736 및 Alpha source019922b1의 현재 게시된 실행 lock은 Server4 경로다. S4 실물/결과/queue를 읽지 않았다. Server2에 같은 모델/통계/프로젝터가 있어도 S4 실행의 직접 입력이라고 추정하지 않는다.','',
        'FE frozen config의 정확 asset 경로와 기존 migration-map을 우선 사용했다. FE 및 W&B spool은 보호하고 실험 진행률/완료 여부는 조회하지 않았다.','',
        '## 분류별 중복 제외 용량','',
        '|분류|파일 수|논리 GiB|할당 GiB|tensor 논리 GiB|','|---|---:|---:|---:|---:|']
    for cls,c in categories.items():lines.append(f"|{cls}|{c['unique_files']}|{c['logical_bytes']/G:.3f}|{c['allocated_bytes']/G:.3f}|{tc[cls]['logical_bytes']/G:.3f}|")
    lines += ['',f"dev/inode 중복 경로 {aliases}개 제외. 파일명/크기 일치는 동일 bytes 증거가 아니다. 확정 회수가능 용량은 **미확정**이며 실제 삭제0B.",'',
        f"기존 이관183CP: 현재 metadata present {retained['present']}/{retained['expected']}, size 일치 {retained['size_matches']}, 역사 논리합 {retained['historical_bytes']}B. 기존 SHA는 재사용 표시만, 전수 재해시0. 원 S4 제거 후 보존 계약이므로 REPRODUCTION_KEEP.",'',
        '## 큰 파일 상위20 (논리 크기 순)','', '|경로|논리 GiB|할당 GiB|분류|','|---|---:|---:|---|']
    for r in top:lines.append(f"|`{r['absolute_path']}`|{r['logical_bytes']/G:.3f}|{r['allocated_bytes']/G:.3f}|{r['classification']}|")
    lines += ['','## Coverage와 불확실성','',
        f"등록 WT {len(wts)}개, regular metadata {scanned}개, 관측 제한/누락 {len(omissions)}건, 오류 {len(errors)}건. 상한180초/400000files/metadata24MiB. 상세는 coverage.json. symlink 미추적, 외부 HF는 등록 모델 cache root만, EasyEdit는 등록 통계와 정확 P 파일만 metadata 식별.",'',
        'UNKNOWN은 PRICE 비관련/안전삭제 판정이 아니다. 기존 task/재현 raw/유일 사본 가능성을 배제하지 못했으며 전부 KEEP. 실제 live open-FD/전체 참조 graph/다른 host 의존성은 미점검. 모델/C0/P/고정data/context/코드/Git/W&B spool은 보호한다. NoCP 정책은 과거 checkpoint 삭제 권한이 아니다.','',
        f"파일별 전체 metadata: `{LOCAL/'full-inventory.csv'}` (local-only). Git inventory.csv는 top20+분류별 큰 tensor20개씩의 소형 발췌이고 summary의 집계는 전체 scan 결과다. root별 aggregate는 summary.json. 현재 FS available {vfs.f_bavail*vfs.f_frsize}B는 공유 FS 순간값이며 본 작업이 확보한 공간이 아니다.",'',
        '재현: `nice -n 5 ionice -c 3 python3 audits/servers/server2/price-storage-inventory-20261007/inventory_scan.py` (create-once, 자동 재실행 없음). metadata 외 tensor 내용읽기 없음. 후속 삭제는 별도 사용자 승인과 정확 대상 검증이 필요하다.','',
        'NO_BROADCAST_NOT_REQUIRED. 소형 보고만 게시 후 STOP; 다른 실험의 monitoring pause 유지.']
    (REPORT/'report-ko.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(totals=summary['totals'],tensor=summary['tensor_totals'],categories=categories,retained=retained,
        omissions=len(omissions),errors=len(errors),seconds=summary['duration_seconds']),ensure_ascii=False))

if __name__=='__main__':main()
