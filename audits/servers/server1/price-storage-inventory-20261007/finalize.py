"""기존 stat inventory 재사용: 보존근거 보완/독립 inode reducer/소형 보고 생성."""
import collections
import csv
import hashlib
import json
from pathlib import Path
from inventory import AUDIT, OUT, WT, CLASSES, NONCE, dump, csvwrite

REPORT = WT / 'experiment-reports/servers/server1/price-storage-inventory-20261007/report-ko.md'
ENFC = '/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-enfc-single-batch-m-v1/local/enfc-single-batch-m/20260918-v1'
KEEP_DOCS = [
    'messages/head/2026-09-18-sh1-enfc-single-batch-resume.md',
    'experiment-reports/servers/server1/enfc-single-batch-m-2026-09-18-v1/completed-review-r1/diagnostic-report-ko.md',
    'experiment-reports/servers/server1/enfc-single-batch-m-2026-09-18-v1/completed-review-r1/artifact-manifest.json',
    'messages/head/2026-10-06-fe-sequential-2k-sh2.json',
    'plans/global/jlz-price-cap-base-repair-2k/execution.json',
    'plans/global/jlz-price-alpha-writer-2k/execution.json',
    'runs/jlz-price-cap-base-repair-2k/submission.json',
    'runs/jlz-price-alpha-writer-2k/submission.json',
    'project/run_scripts/jlz_price_alpha_writer/prepare.py',
    'project/run_scripts/jlz_interference_l1/cap_prepare.py',
]

def gi(x): return f'{x/2**30:.4f}'

def main():
    final=OUT.parent/'final-r1'
    if final.exists(): raise SystemExit('create-once final output already exists')
    final.mkdir()
    s=json.loads((AUDIT/'summary.json').read_text())
    coverage=json.loads((AUDIT/'coverage.json').read_text())
    rows=list(csv.DictReader((OUT/'inventory-full.csv').open()))
    for r in rows:
        for k in ('logical_bytes','allocated_bytes','count','nlink','mtime_ns'): r[k]=int(r[k])
        r['delete_authorized']=False
        if r['absolute_path'].startswith(ENFC+'/') and r['classification'] != 'OTHER_TASK_REQUIRED':
            r.update(classification='REPRODUCTION_KEEP',classification_evidence='ENFC 완료보고 raw/teacher/imports 보존 명시 + exact 실행WT/task path',
                     referenced_task_or_lock=';'.join(KEEP_DOCS[:2]))
    groups=collections.defaultdict(list)
    for r in rows:groups[r['device_inode_or_hardlink_group']].append(r)
    reps=[]
    totals={c:dict(unique_files=0,logical_bytes=0,allocated_bytes=0,tensor_unique_files=0,tensor_allocated_bytes=0) for c in CLASSES}
    for aliases in groups.values():
        assert len({(r['logical_bytes'],r['allocated_bytes']) for r in aliases})==1, 'inode changed during scan'
        rep=min(aliases,key=lambda r:(CLASSES.index(r['classification']),r['absolute_path']))
        reps.append(rep); cls=rep['classification']
        for r in aliases:r.update(dedup_charged=r is rep,inode_classification=cls)
        t=totals[cls];t['unique_files']+=1;t['logical_bytes']+=rep['logical_bytes'];t['allocated_bytes']+=rep['allocated_bytes']
        if rep['artifact_kind'].startswith(('tensor_','extensionless_')):
            t['tensor_unique_files']+=1;t['tensor_allocated_bytes']+=rep['allocated_bytes']
    # Independent grouping arithmetic, conservation across reference-only reclassification.
    assert len(groups)==s['unique_inodes']
    assert sum(t['allocated_bytes'] for t in totals.values())==s['total_allocated_bytes']
    assert sum(t['logical_bytes'] for t in totals.values())==s['total_logical_bytes']
    assert sum(r['allocated_bytes'] for r in reps)==sum(t['allocated_bytes'] for t in totals.values())
    assert sum(r['dedup_charged'] for r in rows)==len(groups)
    assert all(r['classification'] in CLASSES and not r['delete_authorized'] for r in rows)
    top=sorted(reps,key=lambda r:r['allocated_bytes'],reverse=True)[:20]
    ag=collections.defaultdict(lambda:dict(unique_files=0,logical_bytes=0,allocated_bytes=0,tensor_allocated_bytes=0))
    for r in reps:
        root=r['scan_root']
        group=(root+'/'+r['absolute_path'][len(root):].strip('/').split('/')[0]) if root!='EXACT_REFERENCED_PROTECTED_EXTERNAL' else str(Path(r['absolute_path']).parent)
        a=ag[(group,r['classification'])];a['unique_files']+=1;a['logical_bytes']+=r['logical_bytes'];a['allocated_bytes']+=r['allocated_bytes']
        if r['artifact_kind'].startswith(('tensor_','extensionless_')):a['tensor_allocated_bytes']+=r['allocated_bytes']
    dirs=sorted([dict(path=p,classification=c,**v) for (p,c),v in ag.items()],key=lambda r:r['allocated_bytes'],reverse=True)
    possible=collections.defaultdict(list)
    for r in reps:
        if r['artifact_kind'].startswith('tensor_') and r['logical_bytes']>=100*1024**2:possible[(Path(r['absolute_path']).name,r['logical_bytes'])].append(r)
    pairs=[]
    for (name,size),rr in possible.items():
        if len(rr)<2:continue
        for r in rr:pairs.append(dict(group=f'{name}:{size}',path=r['absolute_path'],logical_bytes=size,allocated_bytes=r['allocated_bytes'],device_inode=r['device_inode_or_hardlink_group'],classification=r['classification'],byte_equality='NOT_VERIFIED_NAME_SIZE_ONLY',delete_authorized=False))
    unknown=[r for r in reps if r['classification']=='UNKNOWN' and r['artifact_kind'].startswith(('tensor_','extensionless_'))]
    csvwrite(final/'inventory-full.csv',rows);csvwrite(final/'directory-aggregates.csv',dirs)
    csvwrite(AUDIT/'inventory.csv',top);csvwrite(AUDIT/'directory-top40.csv',dirs[:40])
    if pairs:csvwrite(AUDIT/'possible-same-name-size.csv',pairs)
    if unknown:csvwrite(AUDIT/'unknown-tensors.csv',unknown)
    s.update(classifications=totals,top20=top,full_inventory=str(final/'inventory-full.csv'),
        reference_refinement='ENFC explicit preservation; original scan-r1 retained; no second metadata scan',
        unknown_tensor_files=len(unknown),unknown_tensor_allocated_bytes=sum(r['allocated_bytes'] for r in unknown),
        possible_name_size_groups=len({r['group'] for r in pairs}),duplicate_byte_equality='NOT_VERIFIED',
        current_price_local_execution='NOT_FOUND_IN_SCANNED_ROOTS; canonical execution locks are server4 /data paths; no live scheduler/result inspection',
        confirmed_reclaimable_bytes=None)
    dump(AUDIT/'summary.json',s)
    coverage['focused_reference_supplement']=[dict(path=p,bytes=(WT/p).stat().st_size,sha256=hashlib.sha256((WT/p).read_bytes()).hexdigest()) for p in KEEP_DOCS]
    coverage['reference_search_limit']='Broad compact-document read bound128MiB reached; focused PRICE/FE/ENFC documents supplemented. No global absence-of-reference proof.'
    coverage['accounting']='Regular-file st_blocks*512 only, globally dedup(dev,inode). Directory inode blocks separate, symlink targets excluded. Reflink/compression/shared-extents not measured.'
    dump(AUDIT/'coverage.json',coverage)
    bindings=[]
    for task in ('jlz-price-cap-base-repair-2k','jlz-price-alpha-writer-2k'):
        p=WT/'runs'/task/'submission.json';x=json.loads(p.read_text())
        bindings.append(dict(task=task,source=x['source'],lock=x['lock'],evidence=str(p.relative_to(WT)),live_read=False))
    dump(AUDIT/'price-bindings.json',dict(bindings=bindings,local_path_rewrite=False,remote_access=False,asset_notes='서버4 모델/C0/P와 서버1 같은 이름/size 사본의 현재 byte equality는 미검증. 로컬 공통자산은 보호하되 server4 실사용으로 승격하지 않음.'))
    dump(AUDIT/'postcheck.json',dict(independent_reviewer=False,owner_separate_reducer=True,
        checks=['unique inode totals independently regrouped','apparent/allocated conservation','classification-only refinement preserves byte totals','one charged alias per inode','all rows delete_authorized=false','top20 sorted allocated bytes'],
        passed=True,GPU=0,scheduler_queries=0,source_payload_reads=0,large_tensor_hashes=0,
        qualification='CPU metadata arithmetic only; no tensor/model/scientific validation'))
    table='\n'.join(f"| {c} | {v['unique_files']:,} | {gi(v['logical_bytes'])} | {gi(v['allocated_bytes'])} | {gi(v['tensor_allocated_bytes'])} |" for c,v in totals.items())
    toptext='\n'.join(f"| {i} | `{r['absolute_path']}` | {gi(r['logical_bytes'])} | {gi(r['allocated_bytes'])} | {r['classification']} |" for i,r in enumerate(top,1))
    doc=f'''# server1 PRICE 기준 저장공간 점검

## 결과와 권한

Nonce `{NONCE}`. **delete_authorized=false**. 삭제·이동·압축·전송·원자료 overwrite·권한변경 0. GPU/model/pickle/torch load·새 Slurm/W&B run·scheduler 조회 0. 기존 jobs/pending과 source/raw는 변경하지 않았다.

이번 범위에서 regular path {s['regular_paths']:,}개, device/inode 중복 제외 {s['unique_inodes']:,}개를 계수했다. apparent **{gi(s['total_logical_bytes'])}GiB**, allocated **{gi(s['total_allocated_bytes'])}GiB**이며, tensor/CP 확장자 및 큰 무확장자 분류는 allocated **{gi(s['tensor_allocated_bytes'])}GiB**다. 이름/확장자는 내용 검증이 아니다. 별도 directory inode allocated {gi(s['directory_inode_allocated_bytes'])}GiB는 위 합계에 포함하지 않는다.

현재 PRICE 두 실행 lock은 server4 `/data/janghj/ODE-edit/local/`를 참조한다. 이 server1 점검에서 직접 참조되는 PRICE 실행 payload는 식별하지 못했다. **PRICE_REQUIRED=0은 공통 모델·C0·P가 불필요하다는 뜻이 아니다.** 다른 서버에 있는 파일의 존재/실행 상태는 재조회하지 않았다. SHA/size/path의 과거 정본과 로컬 현재 metadata를 구분했다.

## 분류별 용량

단위 GiB=2^30 bytes. hardlink 144개 alias를 중복 제외했다. `OTHER_TASK_REQUIRED`에는 명시 protected 자산을 포함하며 현재 사용중이라는 runtime 증명은 아니다.

| 분류 | unique files | apparent GiB | allocated GiB | tensor 계열 allocated GiB |
| --- | ---: | ---: | ---: | ---: |
{table}

`UNREFERENCED_CANDIDATE`로 확정한 항목은 0이다. 확인된 회수 가능 공간은 **NOT_ESTABLISHED**이며 0 bytes라고 측정한 것이 아니다. UNKNOWN은 보존한다. 이 점검 결과만으로 어떤 파일도 삭제하도록 승인하지 않는다.

## 보존 및 불명확 항목

- ENFC B1 imports의 S64/Dev128 teacher `.npy` 24개와 reference token은 기존 ENFC 보존 envelope/완료 보고에 결속하여 REPRODUCTION_KEEP로 보완했다. 첫 broad 검색의 UNKNOWN 결과와 원 metadata는 `scan-r1`에 보존하고 재분류만 수행했다. teacher payload는 읽지 않았다.
- 모델 shards/tokenizer, Llama/Qwen/GPT-J 등 native C0/projector, fixed dataset/context, W&B spool은 보호했다. W&B 동기화/인증 상태는 읽지 않았다. 과거 noCP 정책은 이들 삭제 권한이 아니다.
- `{unknown[0]['absolute_path'] if unknown else '없음'}`: {gi(sum(r['allocated_bytes'] for r in unknown))}GiB, tensor 계열 UNKNOWN {len(unknown)}개. 사용처/독립 보존 참조를 이번 한정 검색에서 확정하지 못해 보존한다.
- 동일 이름+size의 별도 inode 그룹 {s['possible_name_size_groups']}개를 [목록](../../../../audits/servers/server1/price-storage-inventory-20261007/possible-same-name-size.csv)에 기록했다. 예: EasyEdit와 `00.KE/EasyEdit`의 C0. **byte 동일성/중복 삭제 가능성 미검증**이며 모두 별도 allocation으로 계수했다. 현재 large-file SHA 재계산은 하지 않았다.
- FE 등 타 task 입력 보존이 우선이다. FE envelope는 server2 배정이며 그 서버의 상태/파일은 조회하지 않았다. 활성 사용처 전수 조사는 하지 않았으므로 UNKNOWN에 무사용 판정을 부여하지 않는다.

## 상위20 regular 파일

allocated 기준, inode당 한 행. 모델 snapshot symlink의 target은 아래 표에서 제외되어 있다.

| 순위 | 경로 | apparent GiB | allocated GiB | 분류 |
| --- | --- | ---: | ---: | --- |
{toptext}

디렉터리별 상위40 및 각 inode/nlink/mtime/reference는 audit CSV에 있다. 디렉터리 합계는 globally charged 대표 inode 기준이므로 별도 `du` subtree 합계와 다를 수 있다.

## 범위와 누락

- 실제 host `devbox`, root `/mnt/raid5/janghj/ODE-edit`, SH1 session `01a04939-f93a-7b50-bca0-65438eab2062`. 역사 registry29e4와 구분. 전용 non-main WT `{WT}`; analysis base `{s['analysis_base']}`. Root dirty에는 손대지 않았다.
- 등록 worktree {coverage['registered_worktrees']}개에서 local/outputs/results/checkpoints를 발견했고, nested root 중복 제외 {coverage['minimal_scanned_roots']}개를 순회했다. 실제 {coverage['entries']:,} entries, {coverage['elapsed_seconds']:.2f}초, stat errors {len(coverage['errors'])}개. max180초/500,000 entries 한도에는 도달하지 않았다.
- 외부는 기존 문서/index에 명시된 exact tensor 파일 {coverage['external_exact_paths']}경로만 metadata 확인했다. `00.KE/EasyEdit`도 기존 receipt의 exact 파일만이며 해당 repo 트리를 순회하지 않았다.
- symlink {coverage['symlink_count']}경로는 따라가지 않았다. Llama/Qwen HF snapshot shards 등이 포함된다. 환경/Git/private subtree {len(coverage['omitted_subtrees'])}개 제외; full model blob allocation/SDK env 전체는 포함하지 않는다. 이 합계는 서버 전체/project 전체의 완전한 disk usage가 아니다.
- 문서 broad 검색은128MiB 상한에 도달했다. PRICE/FE/ENFC 지정 문서는 별도 보완했다. 원 체크포인트 index(2026-09-18)는 경로 발견 근거일 뿐 현재 존재 증명이 아니며 현재 stat만 용량에 포함했다. 기존2026-09-30 cleanup 기록을 읽었으나 그 과거 권한/스크립트를 실행·상속하지 않았다.
- 정적 metadata의 시점은 `{coverage['start_utc']}`~`{coverage['end_utc']}`. 기존 실행과 동시 접근할 수 있어 원자적 snapshot이 아니다. open-file, reflink/shared extents, 전체 hidden reference, live job dependency는 검증하지 않았다. 23 symlink target과 제외 subtree는 coverage에 명시한다.
- `/mnt/raid5` 공유 volume 가용 {gi(s['df']['available_bytes'])}GiB, total {gi(s['df']['total_bytes'])}GiB는 조회 시점 filesystem 수치일 뿐 ODE-edit 단독 수치/예약/회수가능량이 아니다.

## 근거·재현·검산

Audit: `audits/servers/server1/price-storage-inventory-20261007/`. `summary.json`, `inventory.csv`(top20), `coverage.json`, `directory-top40.csv`, `unknown-tensors.csv`, `possible-same-name-size.csv`, `price-bindings.json`, `postcheck.json`.
대규모 full inventory: `{final/'inventory-full.csv'}`. 초기 metadata/refs: `{OUT}`. Git에는 compact metadata와 점검 코드만, raw/tensor/prompt/full logs는 넣지 않았다. `NO_BROADCAST_NOT_REQUIRED`.

```bash
nice -n 19 ionice -c 3 python3 audits/servers/server1/price-storage-inventory-20261007/inventory.py
python3 audits/servers/server1/price-storage-inventory-20261007/finalize.py
```

create-once 출력으로 동일 경로 재실행은 거부한다. 현재 결과는 기존 CSV를 읽어 재집계할 수 있다. 독립 agent는 사용하지 않았고 owner의 별도 CSV reducer가 inode/용량 보존·분류·중복 제외를 검사했다. 이는 tensor byte/모델/과학적 유효성 검증이 아니다.

**TASK_COMPLETE_STOP** — 후속 삭제 지시 전 모든 기존 자료 보존. 자동 점검/청소/monitor/실험 재개 없음.
'''
    REPORT.parent.mkdir(parents=True,exist_ok=True);REPORT.write_text(doc)
    receipt=dict(nonce=NONCE,status='TASK_COMPLETE_STOP',delete_authorized=False,
        actual_host='devbox',session=s['session'],cwd=str(WT),root=s['root'],legacy_cwd='historical29e4_not_used',
        nonce_duplicate_before='NO_PRIOR_SH1_RECEIPT_OR_BRANCH',report=str(REPORT.relative_to(WT)),
        report_sha256=hashlib.sha256(REPORT.read_bytes()).hexdigest(),total_allocated_bytes=s['total_allocated_bytes'],
        tensor_allocated_bytes=s['tensor_allocated_bytes'],confirmed_reclaimable_bytes=None,
        scheduler_queries=0,model_loads=0,GPU=0,existing_jobs_changed=0,
        next_action='STOP; no deletion authorized',analysis_base=s['analysis_base'],publication_commit='Git containing this receipt; final HEAD separately reported')
    for rel in ('messages/acks/server1/price-storage-inventory-20261007.json','messages/server-heads/server1/price-storage-inventory-20261007.json','tasks/status/price-storage-inventory-20261007/server1.json'):dump(WT/rel,receipt)
    dump(AUDIT/'final-inventory-seal.json',{p.name:dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in final.iterdir()})
    print(json.dumps(dict(classifications=totals,unknown_tensors=len(unknown),report_sha256=receipt['report_sha256']),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
