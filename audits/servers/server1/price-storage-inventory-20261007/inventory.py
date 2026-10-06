"""읽기전용 metadata 점검. payload load/hash, symlink traversal, 원자료 쓰기 없음."""
import collections
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import time

ROOT = Path('/mnt/raid5/janghj/ODE-edit')
WT = Path(__file__).resolve().parents[4]
AUDIT = WT / 'audits/servers/server1/price-storage-inventory-20261007'
OUT = WT / 'local/price-storage-inventory-20261007/scan-r1'
NONCE = 'USER-GH-ALL-SH-PRICE-STORAGE-INVENTORY-20261007-SERVER1'
TENSOR = {'.pt', '.pth', '.ckpt', '.bin', '.safetensors', '.npy', '.npz', '.mmap', '.memmap'}
SKIP_DIR = {'.git', '.venv', 'venv', 'node_modules', '__pycache__', 'site-packages', '.ssh'}
DOC_PREFIX = ('audits/', 'experiment-reports/', 'transfers/', 'plans/', 'messages/head/', 'runs/')
CLASSES = ['PRICE_REQUIRED', 'OTHER_TASK_REQUIRED', 'REPRODUCTION_KEEP', 'UNREFERENCED_CANDIDATE', 'UNKNOWN']
PATH_RE = re.compile(r'/mnt/raid5/janghj/[^\s\"\'<>`{}\[\],;|]+')
PRIV = re.compile(r'(^|/)(\.ssh|\.netrc|credentials?[^/]*|secrets?[^/]*|private[^/]*|settings|[^/]*\.env)(/|$)', re.I)

def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')

def csvwrite(path, rows, fields=None):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        w.writeheader(); w.writerows(rows)

def no_symlink(path):
    p = Path(path)
    for parent in [p, *p.parents]:
        if parent.is_symlink():
            return False
    return True

def tensor_kind(p, size):
    if p.suffix.lower() in TENSOR:
        return 'tensor_or_checkpoint_extension_not_content_verified'
    if not p.suffix and size >= 1024**2:
        return 'extensionless_large_unknown'
    if size >= 100 * 1024**2:
        return 'large_other_not_content_verified'
    return 'other_output_metadata_or_raw'

def main():
    if OUT.exists():
        raise SystemExit('create-once scan output already exists')
    OUT.mkdir(parents=True)
    start = time.monotonic()
    begin = dt.datetime.now(dt.timezone.utc).isoformat()
    refs = collections.defaultdict(set)
    evidence = []
    skipped_docs = collections.Counter()
    total_doc_bytes = 0
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=WT).decode().split('\0')
    # 128MiB / file2MiB, tracked compact evidence only. No local settings/env content.
    for rel in tracked:
        if not rel.startswith(DOC_PREFIX) or 'price-storage-inventory-20261007' in rel:
            continue
        p = WT / rel
        if p.suffix not in {'.json', '.md', '.csv'} or PRIV.search(rel) or not no_symlink(p):
            continue
        size = p.stat().st_size
        if size > 2 * 1024**2 or total_doc_bytes + size > 128 * 1024**2:
            skipped_docs['size_or_total_limit'] += 1; continue
        text = p.read_text(errors='replace'); total_doc_bytes += size
        paths = set(m.rstrip(').:') for m in PATH_RE.findall(text))
        if paths:
            evidence.append(dict(path=rel, bytes=size, sha256=hashlib.sha256(p.read_bytes()).hexdigest(), path_references=len(paths)))
        for path in paths:
            if not PRIV.search(path): refs[path].add(rel)
    old_index = ROOT / 'local/checkpoint-location-index/20260918-v1/scans/server1-scan.json'
    # Existing server1 inventory is metadata, not checkpoint content; do not load remote-server inventories.
    if old_index.is_file() and old_index.stat().st_size < 32 * 1024**2:
        text = old_index.read_text()
        for path in set(PATH_RE.findall(text)):
            if not PRIV.search(path): refs[path.rstrip(').:')].add(str(old_index))
        evidence.append(dict(path=str(old_index), bytes=old_index.stat().st_size, sha256=hashlib.sha256(text.encode()).hexdigest(), role='historical_index_only_not_current_existence'))
    wts = [x[9:] for x in subprocess.check_output(['git', 'worktree', 'list', '--porcelain'], cwd=WT, text=True).splitlines() if x.startswith('worktree ')]
    roots = []; skipped_roots = []; external = []
    for w in wts:
        for suffix in ('local', 'outputs', 'results', 'checkpoints'):
            p = Path(w) / suffix
            if not p.exists() and not p.is_symlink(): continue
            if not no_symlink(p): skipped_roots.append(dict(path=str(p), reason='symlink_not_followed')); continue
            if p.is_dir() and p != WT / 'local': roots.append(str(p))
    # Exact already-recorded external model/C0/projector files only. No new external tree walk.
    for path in sorted(refs):
        if ('/EasyEdit/' in path or '/.cache/huggingface/' in path) and Path(path).suffix.lower() in TENSOR:
            if any(c in path for c in ('*', '?', '{', '}')): continue
            external.append(path)
    roots = sorted(set(roots), key=lambda p: (len(p), p))
    minimal_roots = []
    for p in roots:
        if not any(p == r or p.startswith(r + '/') for r in minimal_roots): minimal_roots.append(p)
    rows = []; errors = []; omissions = []; symlinks = []; visited_dirs = set(); visited_paths = set()
    count_entries = 0; deadline = time.monotonic() + 180
    directory_metadata = []; root_records = []
    uid = os.getuid()

    def add_file(p, s, root):
        name = str(p)
        if name in visited_paths: return
        visited_paths.add(name)
        rows.append(dict(server='server1', absolute_path=name, scan_root=root,
            artifact_kind=tensor_kind(p, s.st_size), logical_bytes=s.st_size,
            allocated_bytes=s.st_blocks * 512, count=1, device_inode_or_hardlink_group=f'{s.st_dev}:{s.st_ino}',
            nlink=s.st_nlink, mtime_ns=s.st_mtime_ns, mtime=dt.datetime.fromtimestamp(s.st_mtime, dt.timezone.utc).isoformat(),
            referenced_task_or_lock='', classification='UNKNOWN', classification_evidence='',
            uniqueness_or_reproducibility='NO_CONTENT_HASH_VERIFICATION', uncertainty='point-in-time metadata; no open-file check', delete_authorized=False))

    for root in minimal_roots:
        before = len(rows); stack = [Path(root)]; reason = 'COMPLETE_WITH_EXPLICIT_EXCLUSIONS'
        while stack:
            if time.monotonic() > deadline or count_entries >= 500000:
                reason = 'BOUND_REACHED'; omissions.append(dict(path=str(stack[-1]), reason=reason, unvisited_directories=len(stack))); break
            p = stack.pop()
            try:
                st = p.lstat(); key = (st.st_dev, st.st_ino)
                if key in visited_dirs: continue
                visited_dirs.add(key)
                directory_metadata.append(dict(path=str(p), logical_bytes=st.st_size, allocated_bytes=st.st_blocks*512, device_inode=f'{st.st_dev}:{st.st_ino}'))
                with os.scandir(p) as it:
                    entries = sorted(it, key=lambda e: e.name)
                for e in entries:
                    count_entries += 1
                    q = Path(e.path)
                    if PRIV.search(str(q)):
                        skipped_docs['private_names_excluded_without_read'] += 1; continue
                    s = e.stat(follow_symlinks=False)
                    if s.st_uid != uid:
                        omissions.append(dict(path=str(q), reason='other_uid_metadata_only')); continue
                    if stat.S_ISLNK(s.st_mode):
                        symlinks.append(dict(path=str(q), allocated_bytes=s.st_blocks*512, logical_bytes=s.st_size, reason='not_followed')); continue
                    if stat.S_ISDIR(s.st_mode):
                        if e.name in SKIP_DIR:
                            omissions.append(dict(path=str(q), reason='runtime_git_or_private_subtree_excluded')); continue
                        if s.st_dev != st.st_dev:
                            omissions.append(dict(path=str(q), reason='filesystem_boundary')); continue
                        stack.append(q)
                    elif stat.S_ISREG(s.st_mode): add_file(q, s, root)
            except OSError as exc:
                errors.append(dict(path=str(p), errno=exc.errno))
        root_records.append(dict(path=root, regular_paths=len(rows)-before, status=reason))
    missing_external = []
    for name in external:
        p = Path(name)
        try:
            if not no_symlink(p):
                symlinks.append(dict(path=name, reason='external_symlink_or_ancestor_not_followed')); continue
            s = p.lstat()
            if stat.S_ISREG(s.st_mode) and s.st_uid == uid: add_file(p, s, 'EXACT_REFERENCED_PROTECTED_EXTERNAL')
        except OSError as exc: missing_external.append(dict(path=name, errno=exc.errno))
    # Local sealed identity metadata can improve references; no general config/settings/raw reads.
    local_meta_bytes = 0
    for r in rows:
        p = Path(r['absolute_path'])
        if p.name not in {'execution.lock.json', 'input-manifest.json', 'asset-manifest.json', 'READY.json', 'receiver-ready.json'}: continue
        if r['logical_bytes'] > 2*1024**2 or local_meta_bytes+r['logical_bytes']>32*1024**2: continue
        try:
            text=p.read_text(errors='replace'); local_meta_bytes+=r['logical_bytes']
            for name in set(PATH_RE.findall(text)):
                if not PRIV.search(name): refs[name.rstrip(').:')].add(str(p))
            evidence.append(dict(path=str(p), bytes=r['logical_bytes'], sha256=hashlib.sha256(text.encode()).hexdigest(), role='local_sealed_path_reference'))
        except OSError as exc: errors.append(dict(path=str(p), errno=exc.errno))

    # A specific historical task directory reference is keep evidence, never proof of active use.
    prefix_refs = {}
    for path, docs in refs.items():
        if '/local/' not in path or not any('/experiment-reports/' in '/' + d or '/transfers/' in '/' + d or '/audits/' in '/' + d for d in docs): continue
        tail = path.split('/local/', 1)[1].split('/')
        if len(tail) >= 2 and path in visited_paths: continue  # exact files handled separately
        if len(tail) >= 2 and not Path(path).suffix: prefix_refs[path] = docs

    for r in rows:
        path = r['absolute_path']; p = Path(path); docs = refs.get(path, set())
        parents = [str(x) for x in p.parents]
        parent_docs = set()
        for parent in parents:
            if parent in prefix_refs: parent_docs.update(prefix_refs[parent])
        relevant = docs | parent_docs
        price = {d for d in relevant if ('jlz-price-cap-base-repair-2k' in d or 'jlz-price-alpha-writer-2k' in d) and not d.endswith('inventory.py')}
        is_protected = ('/.cache/huggingface/' in path or '/EasyEdit/' in path or '/datasets/' in path or '/wandb' in path.lower() or p.name == 'contexts.json')
        if price:
            cls='PRICE_REQUIRED'; why='PRICE 정본/lock의 exact local path 또는 명시 input subtree 참조'
        elif is_protected:
            cls='OTHER_TASK_REQUIRED'; why='현재 점검 envelope protected 자산; active use는 별도 미확인'
        elif relevant:
            cls='REPRODUCTION_KEEP'; why='기존 report/audit/manifest exact path 또는 구체적 보존 subtree 참조; 실행중 판정 아님'
        else:
            cls='UNKNOWN'; why='한정 참조검색에서 사용처/보존범위 확정 못함; 미참조 증명 아님'
        r.update(classification=cls, classification_evidence=why,
                 referenced_task_or_lock=';'.join(sorted(relevant)[:5]) if relevant else ('messages/head/2026-10-07-price-storage-inventory-all-sh.json' if is_protected else ''))
    # One inode charged once globally; strongest keep classification wins across aliases.
    groups = collections.defaultdict(list)
    for r in rows: groups[r['device_inode_or_hardlink_group']].append(r)
    totals={c:dict(unique_files=0, logical_bytes=0, allocated_bytes=0, tensor_unique_files=0, tensor_allocated_bytes=0) for c in CLASSES}
    representatives=[]; crosslinks=[]
    for inode, aliases in groups.items():
        chosen=min(aliases,key=lambda r:(CLASSES.index(r['classification']), r['absolute_path']))
        cls=chosen['classification']; t=totals[cls]
        t['unique_files']+=1; t['logical_bytes']+=chosen['logical_bytes']; t['allocated_bytes']+=chosen['allocated_bytes']
        if chosen['artifact_kind'].startswith(('tensor_', 'extensionless_')):
            t['tensor_unique_files']+=1;t['tensor_allocated_bytes']+=chosen['allocated_bytes']
        for r in aliases:
            r['dedup_charged']=r is chosen;r['inode_classification']=cls
        if len(aliases)>1: crosslinks.append(dict(inode=inode, aliases=len(aliases), logical_bytes=chosen['logical_bytes']))
        representatives.append(chosen)
    top=sorted(representatives,key=lambda r:r['allocated_bytes'],reverse=True)[:20]
    aggregates=collections.defaultdict(lambda:dict(unique_files=0,logical_bytes=0,allocated_bytes=0,tensor_allocated_bytes=0))
    for r in representatives:
        root=r['scan_root']
        rel=r['absolute_path'][len(root):].strip('/').split('/')
        group=root+'/'+rel[0] if root != 'EXACT_REFERENCED_PROTECTED_EXTERNAL' else str(Path(r['absolute_path']).parent)
        a=aggregates[(group,r['inode_classification'])];a['unique_files']+=1;a['logical_bytes']+=r['logical_bytes'];a['allocated_bytes']+=r['allocated_bytes']
        if r['artifact_kind'].startswith(('tensor_', 'extensionless_')):a['tensor_allocated_bytes']+=r['allocated_bytes']
    dirs=[dict(path=p,classification=c,**a) for (p,c),a in aggregates.items()]
    dirs.sort(key=lambda r:r['allocated_bytes'],reverse=True)
    v=os.statvfs(ROOT)
    coverage=dict(start_utc=begin,end_utc=dt.datetime.now(dt.timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-start,
        registered_worktrees=len(wts),registered_roots=len(roots),minimal_scanned_roots=len(minimal_roots),
        limits=dict(scan_seconds=180,entries=500000,tracked_reference_bytes=128*1024**2,local_manifest_bytes=32*1024**2),
        regular_paths=len(rows),entries=count_entries,root_records=root_records,skipped_roots=skipped_roots,
        omitted_subtrees=omissions,errors=errors,symlink_count=len(symlinks),external_exact_paths=len(external),
        missing_external_count=len(missing_external),reference_documents=len(evidence),reference_bytes=total_doc_bytes,
        local_metadata_bytes=local_meta_bytes,skipped_document_counts=dict(skipped_docs),
        no_symlink_follow=True,no_tensor_load=True,no_large_tensor_hash=True,no_scheduler_query=True,
        no_remote_reads=True,no_content_duplicate_claim=True,not_atomic_snapshot=True,
        protected_WB_spool='metadata only; synchronization state not read',
        missing='skipped symlinks/runtime/private subtrees; unregistered worktrees/outside explicit output roots; external paths only already-referenced exact files; open handles/untracked references not exhaustively checked')
    summary=dict(nonce=NONCE,host=os.uname().nodename,session='01a04939-f93a-7b50-bca0-65438eab2062',root=str(ROOT),worktree=str(WT),
        analysis_base=subprocess.check_output(['git','rev-parse','HEAD'],cwd=WT,text=True).strip(),
        delete_authorized=False,files_deleted=0,files_moved=0,source_artifacts_modified=0,
        classifications=totals,regular_paths=len(rows),unique_inodes=len(groups),hardlink_alias_paths_removed=len(rows)-len(groups),
        observed_hardlink_groups=len(crosslinks),top20=top,
        total_logical_bytes=sum(x['logical_bytes'] for x in totals.values()),total_allocated_bytes=sum(x['allocated_bytes'] for x in totals.values()),
        tensor_allocated_bytes=sum(x['tensor_allocated_bytes'] for x in totals.values()),
        directory_inode_allocated_bytes=sum(x['allocated_bytes'] for x in directory_metadata),
        candidate_allocated_bytes=totals['UNREFERENCED_CANDIDATE']['allocated_bytes'],confirmed_reclaimable_bytes=None,
        df=dict(total_bytes=v.f_blocks*v.f_frsize,available_bytes=v.f_bavail*v.f_frsize,free_inodes=v.f_favail,scope='shared /mnt/raid5 filesystem, not project-only'),
        no_content_equality_verification=True,classification_basis='exact refs + explicit protected + conservative UNKNOWN; no candidate solely from name/age/size',
        full_inventory=str(OUT/'inventory-full.csv'),no_broadcast='NO_BROADCAST_NOT_REQUIRED',review='owner CPU metadata only; independent reviewer not used')
    csvwrite(OUT/'inventory-full.csv',rows)
    csvwrite(OUT/'directory-aggregates.csv',dirs)
    dump(OUT/'symlinks.json',symlinks);dump(OUT/'missing-external.json',missing_external)
    dump(OUT/'reference-evidence.json',evidence);dump(OUT/'hardlinks.json',crosslinks)
    dump(OUT/'directory-metadata.json',directory_metadata)
    dump(AUDIT/'summary.json',summary);dump(AUDIT/'coverage.json',coverage)
    csvwrite(AUDIT/'inventory.csv',top)
    csvwrite(AUDIT/'directory-top40.csv',dirs[:40])
    dump(AUDIT/'inventory-seal.json',{p.name:dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in OUT.iterdir() if p.is_file()})
    print(json.dumps({k:summary[k] for k in ('regular_paths','unique_inodes','total_allocated_bytes','tensor_allocated_bytes','classifications','df')},ensure_ascii=False,indent=2))
    print('coverage',coverage['elapsed_seconds'],len(errors),len(omissions),len(symlinks))

if __name__=='__main__': main()
