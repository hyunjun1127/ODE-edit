"""Enrich an existing metadata snapshot; no new filesystem or experiment scan."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
import inventory_readonly as inv

def main():
    audit=inv.AUDIT; out=inv.OUT
    old=json.loads((audit/'summary.json').read_text()); coverage=json.loads((audit/'coverage.json').read_text())
    if not (out/'initial-summary.json').exists(): inv.save(out/'initial-summary.json',old)
    if not (out/'initial-coverage.json').exists(): inv.save(out/'initial-coverage.json',coverage)
    coverage=json.loads((out/'initial-coverage.json').read_text())
    families=defaultdict(set); exact=defaultdict(set); docs=[]; read_bytes=0
    candidates=[]
    for root,pattern in [('runs','**/*submission*.json'),('tasks/status','**/server4.json'),('audits/servers/server4','**/*inventory*.json')]:
        candidates.extend((inv.WT/root).glob(pattern))
    candidates.extend(inv.WT/p for p in [
        'audits/servers/server4/2026-09-09-blue-downstream-transfer/checkpoint-manifest.json',
        'audits/servers/server4/jlz-v13-mdcd-sequential-2k-20261005-v1/review-20261005-r1/binding-verification.json',
        'experiment-reports/servers/server4/blue-l4-l8-lifelong-b100x100-review-2026-09-09-v1/first-table-inputs.json',
        'experiment-reports/servers/server4/jlz-realization-v9-ridge-bs100x20-s4-20261004-v1/results-publication-20261004/baselines.json'])
    for p in sorted(set(candidates)):
        rel=str(p.relative_to(inv.WT))
        if inv.TASK in rel or inv.sensitive(rel) or p.is_symlink() or not p.exists():continue
        size=p.stat().st_size
        if size>8*1024**2 or read_bytes+size>32*1024**2:continue
        try: data=json.loads(p.read_bytes())
        except (OSError,ValueError):continue
        read_bytes+=size; docs.append({'path':rel,'bytes':size})
        for q in inv.paths(data):
            if q.startswith(str(inv.ROOT)+'/local/'):
                family=q[len(str(inv.ROOT)+'/local/'):].split('/')[0]
                exact[q].add(rel)
                if family not in ['worktrees','state','logs','cache','analysis','datasets']:families[family].add(rel)
    (out/'classification-reference-map.json').write_text(json.dumps({'families':{k:sorted(v) for k,v in families.items()},'exact':{k:sorted(v) for k,v in exact.items()}},ensure_ascii=False,indent=2)+'\n')
    totals={c:Counter() for c in inv.CLASSES}; directories=defaultdict(Counter); kinds=defaultdict(Counter)
    seen=set(); top=[]; tensors=[]; hardlinks=[]; excluded=Counter(); paths=0; inode_classes={}; conflicts=[]
    with open(old['full_inventory_path'],newline='') as f:
        for r in csv.DictReader(f):
            p=r['absolute_path']
            if p.startswith(str(inv.ROOT/'local'/inv.TASK)+'/'):
                excluded['own_worktree_local_rows']+=1;continue
            paths+=1
            local_prefix=str(inv.ROOT)+'/local/'
            family=p[len(local_prefix):].split('/')[0] if p.startswith(local_prefix) else ''
            if r['classification']=='UNKNOWN' and (p in exact or family in families):
                refs=exact.get(p) or families[family]
                r['classification']='REPRODUCTION_KEEP'
                r['referenced_task_or_lock']=sorted(refs)[0]
                r['classification_evidence']='기존 tracked receipt exact path' if p in exact else '기존 receipt가 참조하는 task family; 보수적 재현 보존, 파일별 live use 증명 아님'
            c=r['classification']; totals[c]['path_count']+=1
            inode=r['device_inode_or_hardlink_group']
            if inode in seen:
                totals[c]['duplicate_hardlink_paths']+=1
                if inode_classes[inode]!=c:conflicts.append({'inode':inode,'first_class':inode_classes[inode],'later_class':c})
                continue
            seen.add(inode);inode_classes[inode]=c
            for k in ['logical_bytes','allocated_bytes','count','nlink']:r[k]=int(r[k])
            r['delete_authorized']=False
            totals[c]['unique_files']+=1;totals[c]['logical_bytes']+=r['logical_bytes'];totals[c]['allocated_bytes']+=r['allocated_bytes']
            group=local_prefix+family if family else (str(inv.ROOT)+'/[non-local]' if p.startswith(str(inv.ROOT)+'/') else 'EXTERNAL_EXACT_REFERENCES')
            for agg in [directories[group],kinds[r['artifact_kind']]]:
                agg['files']+=1;agg['logical_bytes']+=r['logical_bytes'];agg['allocated_bytes']+=r['allocated_bytes']
            inv.heapq.heappush(top,(r['logical_bytes'],p,r))
            if len(top)>20:inv.heapq.heappop(top)
            if r['artifact_kind']!='ordinary_file':
                inv.heapq.heappush(tensors,(r['logical_bytes'],p,r))
                if len(tensors)>100:inv.heapq.heappop(tensors)
            if r['nlink']>1 and len(hardlinks)<100:hardlinks.append(r)
    old.update(classification_totals=totals,total_paths=paths,total_unique_files=len(seen),kind_totals=kinds,
        directories=dict(sorted(directories.items(),key=lambda x:-x[1]['allocated_bytes'])),top20=[x[2] for x in sorted(top,reverse=True)],observed_hardlinks=hardlinks)
    old['classification_pass']='metadata snapshot reused; tracked receipt enrichment; no second stat/scan'
    old['raw_inventory_note']='full CSV의 초기 UNKNOWN은 classification-reference-map과 classify_snapshot.py로 재분류; own inventory WT rows 제외. 원 metadata CSV는 변경하지 않음.'
    coverage['reference_documents']+=docs
    coverage['historical_reference_families']={k:sorted(v)[:3] for k,v in families.items()}
    coverage['classification_correction']={'reason':'절대 WT 경로에 task명이 포함되어 historical receipt 후보가 모두 제외된 필터를 상대경로로 수정. metadata 재스캔 없이 재분류.', 'excluded_rows':excluded,'reference_bytes_read':read_bytes,'cross_class_inode_conflicts':conflicts}
    # Only own generated compact deliverables replaced; original scan receipts retained ignored-local.
    for name,data in [('summary.json',old),('coverage.json',coverage)]:
        (audit/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    selected={x[1]:x[2] for x in top+tensors}
    with (audit/'inventory.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,list(next(iter(selected.values())).keys()),lineterminator='\n');w.writeheader();w.writerows(sorted(selected.values(),key=lambda r:-r['logical_bytes']))
    # Independent arithmetic check from the metadata stream, not from stored aggregates.
    check={'paths':paths,'unique_files':len(seen),'class_count_sum':sum(x['unique_files'] for x in totals.values()),
        'allocated_sum':sum(x['allocated_bytes'] for x in totals.values()),'directory_allocated_sum':sum(x['allocated_bytes'] for x in directories.values()),
        'cross_class_inode_conflicts':conflicts,'delete_authorized':False,'level':'same-owner separate reduction, not independent reviewer'}
    assert check['class_count_sum']==len(seen)
    assert check['allocated_sum']==check['directory_allocated_sum']
    assert not conflicts
    (audit/'reduction-check.json').write_text(json.dumps(check,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'classes':totals,'historical_families':len(families),'documents':len(docs),'check':check},ensure_ascii=False))

if __name__=='__main__':main()
