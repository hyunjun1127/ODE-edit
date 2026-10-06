# server4 PRICE 저장공간 읽기전용 점검

## 결론

점검 범위에서 PRICE 직접 입력 외의 과거 checkpoint/tensor가 존재한다. 다만 재현·비교 보존 근거가 있으며 **삭제 대상으로 확정한 파일은 없다**. `delete_authorized=false`; 원자료 삭제·이동·압축·전송·권한 변경은 수행하지 않았다. 기존 실험/pending job과 W&B는 변경·조회하지 않았다.

2026-10-07 01:49:40 KST부터 18.54초 동안 한정 메타데이터 scan을 수행했다. 최종 집계는 337,334개 경로, dev/inode 중복 제거 335,836개 파일, allocated **99.805 GiB**다. 이는 전체 서버/파일시스템 사용량이 아니라 아래 허용 범위의 관측 합계다.

## 범주별 용량

GiB=2^30 bytes. hardlink는 dev/inode당 한 번, 현재 PRICE exact 입력을 우선 배정했다. apparent는 파일 길이, allocated는 st_blocks×512다.

| 분류 | 고유 파일 수 | apparent GiB | allocated GiB | 판단 |
|---|---:|---:|---:|---|
| PRICE_REQUIRED | 3,458 | 51.196 | 51.203 | 현재 MEMIT/Alpha config·lock 입력 및 해당 실행 namespace 보존 |
| OTHER_TASK_REQUIRED | 4,112 | 0.349 | 0.358 | W&B SDK/spool 등 보호; 업로드 완료 여부 미검증 |
| REPRODUCTION_KEEP | 300,288 | 40.451 | 40.924 | 과거 task/비교·재현 receipt 참조; 보수적 family 보존 포함 |
| UNREFERENCED_CANDIDATE | 0 | 0 | 0 | 미참조를 확정하지 못했음; 전 서버에 후보가 없다는 뜻 아님 |
| UNKNOWN | 27,978 | 7.261 | 7.320 | 참조/용도 검증 불완전, 그대로 보존 |

관측 hardlink 중복 경로는 1,498개다. 동일 이름/크기나 서로 다른 inode의 내용 동일성을 확인하지 않았다. nlink가 관측 링크 수보다 큰 경우 외부 링크 존재 가능성이 있다. 이 수치는 제거 가능한 공간이 아니다. 확정 reclaimable bytes와 후보 reclaimable bytes는 **NOT_DETERMINED**다.

## PRICE 직접 입력 외 tensor/checkpoint

확장자 기반 유형 분류이므로 모든 .pt/.bin 파일의 내용이 실제 tensor임을 증명하지는 않는다. 아래는 dev/inode 중복 제외 allocated 합계다.

| local 하위 경로 | 파일 수 | GiB | 분류·보존 근거 |
|---|---:|---:|---|
| native-delayed-write-e3 | 12 | 13.1255 | BASE_MEMIT의 W-method-state.pt 등 과거 checkpoint/재현 task 참조 → REPRODUCTION_KEEP |
| jlz-realization-v9 | 525 | 2.2610 | 과거 v9 산출물·원실험 보존 → REPRODUCTION_KEEP |
| single-layer-edit-preserving-correction | 1 | 1.5303 | T/attempt-v1/output/P-star-basis.pt, 기존 inventory/재현 family → REPRODUCTION_KEEP |
| jlz-twoarm | 1 | 0.1602 | 과거 task 참조 → REPRODUCTION_KEEP |
| bg1-c4-ours-first | 1 | 0.0004 | 과거 task 참조 → REPRODUCTION_KEEP |
| bpcw512 | 2 | 0.000008 | 과거 task 참조 → REPRODUCTION_KEEP |
| state/phase123-server1-sample-stream-v1 | 1 | 0.0019 | selected-counterfact-records.bin; dataset성 경로지만 현재 용도 미검증 → UNKNOWN/보존 |
| checkpoint-migration-server2/.../synthetic-guard-fixture-v1 | 5 | 0.000019 | 각 40-byte .pt 파일; fixture 이름만으로 삭제/내용 판정 불가 → UNKNOWN/보존 |

과거 NoCP 정책을 기존 checkpoint 삭제 권한으로 해석하지 않았다. 다른 active/pending task 전체의 live open-file 검사는 하지 않았으므로 UNKNOWN의 비사용을 주장하지 않는다. 기존 status/submission/inventory와 baseline 참조 문서 101개를 한정 대조했다. family 단위 REPRODUCTION_KEEP는 보수적 보호이며, 개별 파일의 현재 프로세스 사용 증명이 아니다.

## 큰 파일 상위 20

전체 절대 경로·mtime·dev/inode·nlink·분류 근거는 [inventory.csv](../../../../audits/servers/server4/price-storage-inventory-20261007/inventory.csv)에 있다. 아래 HF blob은 현재 config가 등록한 exact 실제 파일이며 snapshot symlink를 따라 탐색한 것이 아니다. 두 용량은 세 자리 반올림에서 동일하다.

| 순위 | 파일 식별 | apparent / allocated GiB | 분류 |
|---:|---|---:|---|
| 1 | Qwen Alpha projector null_space_project_Qwen2.5-7B-Instruct.pt | 6.685 / 6.685 | PRICE_REQUIRED |
| 2 | Llama HF blob 8d4782b4… | 4.656 / 4.656 | PRICE_REQUIRED |
| 3 | Llama HF blob d8cf9c4d… | 4.635 / 4.635 | PRICE_REQUIRED |
| 4 | Llama HF blob 3acdd690… | 4.578 / 4.578 | PRICE_REQUIRED |
| 5 | Llama Alpha projector null_space_project_Meta-Llama-3-8B-Instruct.pt | 3.828 / 3.828 | PRICE_REQUIRED |
| 6 | Qwen HF blob a1333e62… | 3.674 / 3.674 | PRICE_REQUIRED |
| 7 | Qwen HF blob 8efdec4c… | 3.599 / 3.599 | PRICE_REQUIRED |
| 8 | Qwen HF blob f5d25a27… | 3.599 / 3.599 | PRICE_REQUIRED |
| 9 | Qwen HF blob 1a72d403… | 3.312 / 3.312 | PRICE_REQUIRED |
| 10 | single-layer-edit-preserving-correction/.../P-star-basis.pt | 1.530 / 1.530 | REPRODUCTION_KEEP |
| 11 | Qwen C0 layer8 mom2 | 1.337 / 1.337 | PRICE_REQUIRED |
| 12 | Qwen C0 layer7 mom2 | 1.337 / 1.337 | PRICE_REQUIRED |
| 13 | Qwen C0 layer6 mom2 | 1.337 / 1.337 | PRICE_REQUIRED |
| 14 | Qwen C0 layer5 mom2 | 1.337 / 1.337 | PRICE_REQUIRED |
| 15 | Qwen C0 layer4 mom2 | 1.337 / 1.337 | PRICE_REQUIRED |
| 16 | native-delayed-write-e3/.../BASE_MEMIT/B100/W-method-state.pt | 1.094 / 1.094 | REPRODUCTION_KEEP |
| 17 | 같은 BASE_MEMIT/B090/W-method-state.pt | 1.094 / 1.094 | REPRODUCTION_KEEP |
| 18 | 같은 BASE_MEMIT/B080/W-method-state.pt | 1.094 / 1.094 | REPRODUCTION_KEEP |
| 19 | 같은 BASE_MEMIT/B070/W-method-state.pt | 1.094 / 1.094 | REPRODUCTION_KEEP |
| 20 | 같은 BASE_MEMIT/B060/W-method-state.pt | 1.094 / 1.094 | REPRODUCTION_KEEP |

확인된 프로젝트 외부 exact 자산 33개가 allocated 50.188 GiB를 차지한다. ODE-edit local의 큰 디렉터리는 native-delayed-write-e3 14.031 GiB, jlz-realization-v9 3.568 GiB, single-layer-mechanism-first 3.139 GiB, historical-update-timeaxis 2.859 GiB다. 디렉터리 합계는 tensor만이 아니라 raw/소스/일반 파일도 포함하며 전체 표는 summary.json에 있다.

## 경계·입력 결속

- 실제 session: 01a04939-b5c7-7a03-ba2d-ef3343d62cfd, server4.
- 실제 app CWD: /data/janghj/ODE-edit, origin hyunjun1127/ODE-edit. 현재 registry와 일치.
- own WT: /data/janghj/ODE-edit/local/price-storage-inventory-20261007/worktree.
- branch: codex/server4-price-storage-inventory-20261007.
- authority main: 4ea217b5258ff786e83a5dcf617d311265a4ff94.
- envelope SHA256: b2c34d7e3fa179012c62d5e81bfa1888cdacf1a1e372cb9f89223228eac35408.
- MEMIT 실행 source: 87a5a736c455d5082f5f666ae562f9edc4e5d3b2.
- Alpha 실행 source: 019922b1524efac596bdb8bf7581d44f1721ed59; 결과가 아닌 등록 게시 main 7cad527a.
- 양 task의 local attempt/config.json, execution.lock.json를 읽고 작은 문서 SHA를 기록했다. 기존 asset SHA는 fresh lstat(size/inode/mtime)와 대조했으며, 검사 가능한 기존 binding에서 불일치는 발견되지 않았다. 이것은 내용 SHA 재검산/모델 검증이 아니다.
- 현재 raw/output 경로 6개가 lstat 시 존재하지 않았다. pending/output 미생성 가능성이 있으나 원인을 추정하지 않고 coverage.json에 FileNotFoundError 그대로 기록했다. job 상태/성과를 확인하지 않았다.

## Coverage와 한계

등록 WT 60개를 파악했다. root/local 및 WT의 실제 local/output/outputs/results만 한정 탐색하고, WT source 전체·Git 내부·민감 경로는 제외했다. 15,979개 디렉터리를 방문했고 180초/100만 파일 상한에 도달하지 않았다. symlink는 따라가지 않았다(탐색 중 44개 제외). 다른 사용자/home 전체/다른 repo/credential은 탐색하지 않았다. 외부 자산은 원 receipt에 등록된 exact path만 lstat했다.

초기 scan의 historical-reference 필터가 own WT 절대경로의 task명에 걸려 과거 참조를 제외하는 문제가 있었다. 상대경로 필터로 교정하고 **기존 metadata CSV를 재분류**했다. raw scan/초기 요약은 local에 보존했고 tensor/filesystem 재스캔은 하지 않았다. 자체 WT local 행 687개는 최종 집계에서 제외했다. 분류표/디렉터리 총합 및 inode 중복을 별도 CPU reducer로 재집계했으며 owner 검산이다. 별도 독립 reviewer/GPU PASS가 아니다.

동시 write가 가능한 비원자적 filesystem snapshot이다. scan 끝의 사용자 available은 **60.456 GiB**, free inode는 223,092,717이었다. quota 도구/사용자 quota는 확인하지 못했다. reserved 포함 free와 사용자 available은 다르며, 이 available은 실험용 독점 reserve가 아니다. 이 범위만으로 전체 디스크 사용 원인 또는 과거 ENOSPC 소비 주체를 확정하지 않는다.

## 산출물·안전 확인

- [summary.json](../../../../audits/servers/server4/price-storage-inventory-20261007/summary.json): 분류/유형/디렉터리 합계, 상위20.
- [coverage.json](../../../../audits/servers/server4/price-storage-inventory-20261007/coverage.json): 경계·누락·참조·asset stat 결속.
- [reduction-check.json](../../../../audits/servers/server4/price-storage-inventory-20261007/reduction-check.json): 335,836개 고유 파일, allocated 107,164,385,280 bytes 합계 일치.
- full metadata CSV: /data/janghj/ODE-edit/local/price-storage-inventory-20261007/metadata/inventory-full.csv (147,250,431 bytes, Git 제외).
- CSV의 초기 분류는 classification-reference-map.json + classify_snapshot.py로 보강한다. 최종 분류는 tracked summary/compact CSV가 정본이다.
- full metadata receipt 및 최초 요약/coverage는 위 ignored local에 보존.
- tensor/model/raw content read/model load/torch.load/pickle/GPU/새 Slurm/Slurm 변경/새 W&B/대형 tensor SHA 재계산: 모두 0. 허용된 소형 config/lock/기존 inventory 내용은 읽었다.
- NO_BROADCAST_NOT_REQUIRED: 소형 source/metadata/report Git 게시만 필요; local raw/spool/SDK/credential 전송 없음.
- REVIEW_COMPLETE_STOP. 반복 monitor/heartbeat/자동 정리는 만들지 않는다. 후속 사용자 삭제 승인 전 아무것도 제거하지 않는다.
