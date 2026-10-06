"""기존 scan 재사용: 등록 Qwen closure metadata와 누락 exact 경로만 보완."""
import collections,csv,datetime,json,os,stat,time
from pathlib import Path
from inventory_scan import REPO,WT,LOCAL,AUDIT,REPORT,TENSOR,CLASSES,dump

def main():
    os.nice(15);os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
    summary=json.loads((AUDIT/'summary.json').read_text());coverage=json.loads((AUDIT/'coverage.json').read_text())
    dump(LOCAL/'initial-summary.json',summary)
    with (LOCAL/'full-inventory.csv').open() as f:rows=list(csv.DictReader(f))
    fields=list(rows[0])
    for r in rows:
        for k in ('logical_bytes','allocated_bytes','count','nlink','observed_alias_count'):r[k]=int(r[k])
        r['delete_authorized']=False
    by_inode={r['device_inode_or_hardlink_group']:r for r in rows};by_path={r['absolute_path']:r for r in rows}
    closure=REPO/'local/checkpoint-migration-server4/20260911-v1/jvp1k-v1-reference-closure.json'
    d=json.loads(closure.read_text());added=[];stat_errors=[]
    # Only exact external file/realpath already named in the historical receipt.
    for entry in d['verified']:
        v=entry.get('destination',{});s=v.get('realpath',v.get('path',''))
        if not s.startswith('/mnt/raid5/janghj/'):continue
        if not ('/EasyEdit/examples/' in s or '/.cache/huggingface/hub/models--' in s):continue
        if s in by_path:continue
        p=Path(s)
        try:st=p.lstat()
        except OSError as e:stat_errors.append(dict(path=s,error=type(e).__name__));continue
        if not stat.S_ISREG(st.st_mode) or st.st_uid!=os.getuid():continue
        # Reject symlink ancestry rather than following it as a traversal shortcut.
        if any(a.is_symlink() for a in p.parents):stat_errors.append(dict(path=s,error='SYMLINK_ANCESTRY_NOT_FOLLOWED'));continue
        key=f'{st.st_dev}:{st.st_ino}'
        if key in by_inode:continue
        row=dict(server='server2',absolute_path=s,artifact_kind='tensor_extension' if p.suffix in TENSOR else 'protected_model_blob_or_metadata',
            logical_bytes=st.st_size,allocated_bytes=512*st.st_blocks,count=1,device_inode_or_hardlink_group=key,nlink=st.st_nlink,
            observed_alias_count=1,mtime=datetime.datetime.fromtimestamp(st.st_mtime,datetime.timezone.utc).isoformat(),root='EXACT_REGISTERED_CLOSURE',
            referenced_task_or_lock=str(closure),classification='REPRODUCTION_KEEP',classification_evidence='Registered Llama/Qwen base/C0/P reference closure; PRICE S2 analogue not direct S4 path',
            historical_sha256=v.get('sha256',''),historical_size_matches=st.st_size==v.get('bytes'),
            uniqueness_or_reproducibility='Exact closure realpath metadata; no payload read/hash',
            uncertainty='Historical hash not revalidated; S4/current-use parity not established',delete_authorized=False)
        rows.append(row);by_inode[key]=row;by_path[s]=row;added.append(row)
    mapping=WT/'transfers/verifications/2026-09-11-checkpoint-migration-server2-destination/migration-map.csv'
    with mapping.open() as f:mp=list(csv.DictReader(f))
    absent=[]
    for m in mp:
        if m['destination'] in by_path:continue
        p=Path(m['destination'])
        try:
            s=p.lstat();result='PRESENT_AFTER_SCAN';size=s.st_size
        except FileNotFoundError:result='ENOENT';size=None
        except OSError as e:result=type(e).__name__;size=None
        absent.append(dict(path=str(p),bundle=m['bundle'],arm=m['arm'],batch=m['batch'],historical_bytes=int(m['bytes']),
            historical_sha256=m['sha256'],current_status=result,current_bytes=size,cause='NOT_IDENTIFIED',deleted_by_this_task=False))
    dump(LOCAL/'missing-migration-exact-stat.json',absent)
    with (AUDIT/'missing-migration-paths.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(absent[0]));w.writeheader();w.writerows(absent)
    def totals(rr):return dict(unique_files=len(rr),logical_bytes=sum(r['logical_bytes'] for r in rr),allocated_bytes=sum(r['allocated_bytes'] for r in rr))
    cats={c:totals([r for r in rows if r['classification']==c]) for c in CLASSES}
    tensors=[r for r in rows if r['artifact_kind']=='tensor_extension']
    tc={c:totals([r for r in tensors if r['classification']==c]) for c in CLASSES}
    top=sorted(rows,key=lambda r:r['logical_bytes'],reverse=True)[:20]
    summary.update(categories=cats,tensor_categories=tc,totals=totals(rows),tensor_totals=totals(tensors),top20=top,
        supplement=dict(source=str(closure),added_files=len(added),added=totals(added),stat_errors=stat_errors),
        missing_migration=dict(count=len(absent),ENOENT=sum(r['current_status']=='ENOENT' for r in absent),
            historical_bytes=sum(r['historical_bytes'] for r in absent),by_arm=dict(collections.Counter(r['arm'] for r in absent)),
            current_cause='NOT_IDENTIFIED',restore_or_transfer_attempted=False),
        raw_inventory=str(LOCAL/'full-inventory-final.csv'))
    summary['directory_aggregates'].append(dict(path='EXACT_REGISTERED_CLOSURE_SUPPLEMENT',**totals(added)))
    coverage['supplement']=dict(exact_reference=str(closure),new_files=len(added),stat_errors=stat_errors,
        missing_migration_paths_checked=len(absent),whole_rescan=False,large_hash=0)
    coverage['metadata_read'].append(str(closure))
    def csv_write(p,rr,mode):
        with p.open(mode,newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rr)
    csv_write(LOCAL/'full-inventory-final.csv',rows,'x')
    compact={r['absolute_path']:r for r in top}
    for c in CLASSES:
        for r in sorted((r for r in tensors if r['classification']==c),key=lambda x:x['logical_bytes'],reverse=True)[:20]:compact[r['absolute_path']]=r
    csv_write(AUDIT/'inventory.csv',list(compact.values()),'w')
    (AUDIT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (AUDIT/'coverage.json').write_text(json.dumps(coverage,ensure_ascii=False,indent=2)+'\n')
    report=REPORT/'report-ko.md';text=report.read_text();G=1024**3
    for c,v in cats.items():
        import re
        text=re.sub(r'^\|'+c+r'\|.*$',f"|{c}|{v['unique_files']}|{v['logical_bytes']/G:.3f}|{v['allocated_bytes']/G:.3f}|{tc[c]['logical_bytes']/G:.3f}|",text,flags=re.M)
    text=text.replace('full-inventory.csv','full-inventory-final.csv')
    text+='\n## Exact 경로 후속 metadata 확인\n\n'
    text+=f"이관 목록 중 미관측 {len(absent)}개를 exact lstat한 결과 ENOENT {summary['missing_migration']['ENOENT']}개. 역사 합 {summary['missing_migration']['historical_bytes']}B. arm별 {summary['missing_migration']['by_arm']}. 이는 현재 목록 경로의 부재 사실이며 삭제 주체/시점/원인이나 다른 경로의 복사본 존재 여부는 확인하지 않았다. 본 점검 삭제0. 초기72 및 나머지관측111의 크기일치와 혼동하지 않는다. 정확 경로는 missing-migration-paths.csv.\n\n"
    text+=f"기존 jvp reference closure에 이미 등록된 Qwen/공용자산 exact metadata {len(added)}파일을 보완했다({totals(added)['logical_bytes']/G:.3f}GiB). symlink를 따라 탐색하지 않고 receipt의 명시 regular realpath만 사용했다. 본표는 이 보완을 포함하며 원 scan/새 fullinventory를 모두 local 보존했다.\n"
    report.write_text(text)
    checks=dict(unique_inodes=len(by_inode)==len(rows),class_count=sum(v['unique_files'] for v in cats.values())==len(rows),
        class_logical_sum=sum(v['logical_bytes'] for v in cats.values())==totals(rows)['logical_bytes'],
        class_allocated_sum=sum(v['allocated_bytes'] for v in cats.values())==totals(rows)['allocated_bytes'],
        retained_plus_absent=summary['retained_183']['present']+len(absent)==183,delete_authorized=False,model_or_GPU=0)
    assert all(checks[k] for k in ('unique_inodes','class_count','class_logical_sum','class_allocated_sum','retained_plus_absent'))
    dump(AUDIT/'arithmetic-check.json',checks)
    print(json.dumps(dict(totals=summary['totals'],categories=cats,missing=summary['missing_migration'],added=len(added),checks=checks)))

if __name__=='__main__':main()
