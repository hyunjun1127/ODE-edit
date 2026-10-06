# Server3 PRICE 기준 저장공간 점검 — 2026-10-07

**INVENTORY_COMPLETE_NO_DELETE / delete_authorized=false.** 파일 삭제·이동·압축·전송·원파일 overwrite·권한변경은 하지 않았다. 기존 실험/job/대기열·W&B 상태도 변경하지 않았다. 본 보고는 후속 판단용 메타데이터 점검이며 삭제 가능 목록 확정이 아니다.

현재 PRICE MEMIT/Alpha의 게시된 submission·config/lock 참조는 **server4** 소유다. Server3에는 해당 두 local task root/lock가 없다. 같은 `/data/janghj/...` 문자열이 있어도 동일 서버 파일로 간주하지 않았다. 따라서 이 스냅샷에서 `PRICE_REQUIRED`로 확인된 server3 실행 payload는 0이다. Git에 있는 PRICE 정본·소스 사본 및 SH3 공통 자산은 보존한다.

## 핵심 용량

| 항목 | apparent GiB | allocated GiB | 판정 |
|---|---:|---:|---|
| MEMIT-HJ edited CP | 4.949 | 4.949 | REPRODUCTION_KEEP |
| EN derived basis | 1.530 | 1.530 | REPRODUCTION_KEEP |
| GSS cold history/maps | 1.780 | 1.784 | REPRODUCTION_KEEP |
| GPT-J 원모델2개 및 projector/metadata | 51.093 | 51.093 | REPRODUCTION_KEEP |
| 등록 Llama HF 모델 cache | 14.966 | 14.966 | OTHER_TASK_REQUIRED |
| Llama/Qwen C0 10개 | 10.513 | 10.513 | OTHER_TASK_REQUIRED |
| Llama/Qwen projector2개 | 10.513 | 10.513 | OTHER_TASK_REQUIRED |
| Git checkout/source/report 사본 — 제외 경로 밖 | 13.519 | 13.790 | OTHER_TASK_REQUIRED |

- **MEMIT-HJ 1k CP가 실제 남아 있다:** `local/memit-hj/20260930-v2/attempt-v1/output/temporary-checkpoints/000-01000-0000.pt`, logical5,313,662,767B. 기존 manifest cursor1000/원 reload_state_output=true/기존 SHA와 크기를 대조했다. 이번에 tensor load·재해시·재로드하지 않았다. Refcount0은 이번 삭제 권한이 아니므로 보존한다. 점검 범위에서 다른 edited CP payload는 확인되지 않았으며, 이를 모든 경로의2k CP 부재 증명으로 확대하지 않는다.
- EN `pstar-derived-v1/basis.npy`는 기존 receipt가 결속한 파생 입력이며, 과거 별도 cleanup receipt에서도 보호했다. GSS cold-history/maps2,400파일도 old task의 teacher/sketch 재현 입력으로 보존한다. 두 arm에서 같은 파일명·size가 보여도 SHA/내용 동일 판정을 하지 않았다.
- GPT-J의 두 대형 blob은 각각24,207,819,307B와24,207,760,308B이고 projector는6,442,452,841B다. Bootstrap에 등록된 과거 원모델·자산이다. 현재 PRICE 실행에 필요하다는 증거는 없지만 원자산 보존 범위에 포함한다. 두 blob을 중복이라고 단정하지 않는다.
- Llama/Qwen projector는 PRICE Alpha source의 기존 SHA/size와 SH3 readiness receipt가 일치하고, current dev/inode/size/mtime도 prior stat와 일치한다. L4..L8→slot0..4 및 threshold.02는 기존 source/manifest 근거이며 이번 tensor 검증이 아니다.
- **Qwen 등록 HF model directory는 FileNotFoundError.** readiness 경로의 현재 부재만 확인했다. 다른 위치 보유 여부나 삭제 원인은 UNKNOWN이며 home/cache 전체를 검색하지 않았다. Qwen C0/P는 현물로 있다.

## 분류별 집계

아래는 선택 범위의 정규 파일을 dev/inode로 한 번만 세었다. `OTHER_TASK_REQUIRED`에는 명시 보존된 공통 자산·runtime·source/worktree도 포함되며, 현재 job에서 사용 중임을 모두 확인했다는 뜻은 아니다.

| 분류 | 고유 파일 | apparent bytes | allocated bytes | allocated GiB |
|---|---:|---:|---:|---:|
| PRICE_REQUIRED | 0 | 0 | 0 | 0.000 |
| OTHER_TASK_REQUIRED | 136,940 | 53,492,504,866 | 53,799,190,528 | 50.104 |
| REPRODUCTION_KEEP | 32,997 | 67,099,115,813 | 67,166,412,800 | 62.554 |
| UNREFERENCED_CANDIDATE | 0 | 0 | 0 | 0.000 |
| UNKNOWN | 1 | 507 | 4,096 | 0.000 |
| 합계 | 169,938 | 120,591,621,186 | 120,965,607,424 | 112.658 |

정규 경로 170,082개에서 hardlink 중복 경로 144개를 제외했다. 여러 경로로 발견한 inode group은 144개다. nlink>1인 고유 파일2,479개 중에는 범위 밖 링크가 있을 수 있다. 한 경로 제거가 실제 공간 회수로 이어진다고 추정하지 않았다.

별도 tensor/model-cache 표(원모델 cache의 작은 tokenizer/config도 포함):

| 분류 | 고유 파일 | logical bytes | allocated bytes |
|---|---:|---:|---:|
| PRICE_REQUIRED | 0 | 0 | 0 |
| OTHER_TASK_REQUIRED | 28 | 38,645,575,530 | 38,645,702,656 |
| REPRODUCTION_KEEP | 2,419 | 63,728,984,693 | 63,732,924,416 |
| UNREFERENCED_CANDIDATE | 0 | 0 | 0 |
| UNKNOWN | 0 | 0 | 0 |

`UNREFERENCED_CANDIDATE`로 승격할 충분한 근거가 있는 항목은0이다. 이는 불필요 자료가 전혀 없다는 증명이 아니다. PRICE 외 재현 자료와 보호 원모델의 용량은 위에 별도로 제시했다. **확정 reclaimable bytes=0**, 실제 삭제0B. UNKNOWN의507B는 local README이며, 제외 범위/현재 사용자 미확인 문제는 coverage에 따로 기록한다.

## 파일 크기 상위20

allocated 기준이며 모든 경로는 현재 server3 로컬이다. 상세 dev/inode/nlink/mtime/근거는 [inventory.csv](../../../../audits/servers/server3/price-storage-inventory-20261007/inventory.csv)에 있다. CSV에는 상위20과 directory aggregate가 함께 있어 행 전체를 합산하면 안 된다.

| 순위 | 절대경로 | logical bytes | allocated GiB | 분류 |
|---:|---|---:|---:|---|
| 1 | `/data/janghj/.cache/huggingface/hub/models--EleutherAI--gpt-j-6b/blobs/0e183edc2025ecfdba4429ba43c960224103b3c3dc26616503cdc2158a3d6c93` | 24,207,819,307 | 22.545 | REPRODUCTION_KEEP |
| 2 | `/data/janghj/.cache/huggingface/hub/models--EleutherAI--gpt-j-6b/blobs/39719bd3194cf2f63e722241a1c1b60afc1ef07de41fe86bf71c934f1bf8836e` | 24,207,760,308 | 22.545 | REPRODUCTION_KEEP |
| 3 | `/data/janghj/EasyEdit/examples/null_space_project_Qwen2.5-7B-Instruct.pt` | 7,177,504,758 | 6.685 | OTHER_TASK_REQUIRED |
| 4 | `/data/janghj/EasyEdit/examples/null_space_project_gpt-j-6b.pt` | 6,442,452,841 | 6.000 | REPRODUCTION_KEEP |
| 5 | `/data/janghj/ODE-edit/local/memit-hj/20260930-v2/attempt-v1/output/temporary-checkpoints/000-01000-0000.pt` | 5,313,662,767 | 4.949 | REPRODUCTION_KEEP |
| 6 | `/data/janghj/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/blobs/8d4782b4a69ef03845159ce1a15e272aadaaf134dc138d68f616098e8531729c` | 4,999,802,720 | 4.656 | OTHER_TASK_REQUIRED |
| 7 | `/data/janghj/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/blobs/d8cf9c4d0dd972e1a2131bfe656235ee98221679711a3beef6d46dadf0f20b5c` | 4,976,698,672 | 4.635 | OTHER_TASK_REQUIRED |
| 8 | `/data/janghj/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/blobs/3acdd690e65c24f42a24581b8467af98bd3ca357444580f8012aacd2bd607921` | 4,915,916,176 | 4.578 | OTHER_TASK_REQUIRED |
| 9 | `/data/janghj/EasyEdit/examples/null_space_project_Meta-Llama-3-8B-Instruct.pt` | 4,110,419,877 | 3.828 | OTHER_TASK_REQUIRED |
| 10 | `/data/janghj/ODE-edit/local/en-adaptive-nullspace/20260920-v1/inputs/pstar-derived-v1/basis.npy` | 1,643,020,416 | 1.530 | REPRODUCTION_KEEP |
| 11 | `/data/janghj/EasyEdit/examples/data/stats/Qwen2.5-7B-Instruct/wikipedia_stats/model.layers.4.mlp.down_proj_float32_mom2_100000.npz` | 1,435,501,774 | 1.337 | OTHER_TASK_REQUIRED |
| 12 | `/data/janghj/EasyEdit/examples/data/stats/Qwen2.5-7B-Instruct/wikipedia_stats/model.layers.5.mlp.down_proj_float32_mom2_100000.npz` | 1,435,501,774 | 1.337 | OTHER_TASK_REQUIRED |
| 13 | `/data/janghj/EasyEdit/examples/data/stats/Qwen2.5-7B-Instruct/wikipedia_stats/model.layers.6.mlp.down_proj_float32_mom2_100000.npz` | 1,435,501,774 | 1.337 | OTHER_TASK_REQUIRED |
| 14 | `/data/janghj/EasyEdit/examples/data/stats/Qwen2.5-7B-Instruct/wikipedia_stats/model.layers.7.mlp.down_proj_float32_mom2_100000.npz` | 1,435,501,774 | 1.337 | OTHER_TASK_REQUIRED |
| 15 | `/data/janghj/EasyEdit/examples/data/stats/Qwen2.5-7B-Instruct/wikipedia_stats/model.layers.8.mlp.down_proj_float32_mom2_100000.npz` | 1,435,501,774 | 1.337 | OTHER_TASK_REQUIRED |
| 16 | `/data/janghj/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/blobs/67e9ad31c8c32abf3a55ee7fc7217b3ecb35fd3c74d98a5bd233e0e4d6964f46` | 1,168,138,808 | 1.088 | OTHER_TASK_REQUIRED |
| 17 | `/data/janghj/EasyEdit/examples/data/stats/Meta-Llama-3-8B-Instruct/wikipedia_stats/model.layers.4.mlp.down_proj_float32_mom2_100000.npz` | 822,084,814 | 0.766 | OTHER_TASK_REQUIRED |
| 18 | `/data/janghj/EasyEdit/examples/data/stats/Meta-Llama-3-8B-Instruct/wikipedia_stats/model.layers.5.mlp.down_proj_float32_mom2_100000.npz` | 822,084,814 | 0.766 | OTHER_TASK_REQUIRED |
| 19 | `/data/janghj/EasyEdit/examples/data/stats/Meta-Llama-3-8B-Instruct/wikipedia_stats/model.layers.6.mlp.down_proj_float32_mom2_100000.npz` | 822,084,814 | 0.766 | OTHER_TASK_REQUIRED |
| 20 | `/data/janghj/EasyEdit/examples/data/stats/Meta-Llama-3-8B-Instruct/wikipedia_stats/model.layers.7.mlp.down_proj_float32_mom2_100000.npz` | 822,084,814 | 0.766 | OTHER_TASK_REQUIRED |

## PRICE 및 다른 작업 참조

| 실행 | source | lock SHA | 실행 소유·이 서버 현물 |
|---|---|---|---|
| jlz-price-cap-base-repair-2k | `87a5a736c455d5082f5f666ae562f9edc4e5d3b2` | `ea45afde90a88f315edb61c35807a9b3419c2c4dd29710870fe269afd4db42b9` | server4; S3 동일문자열 lock 부재 |
| jlz-price-alpha-writer-2k | `019922b1524efac596bdb8bf7581d44f1721ed59` | `cae341a900fc53335f9711fbdfa69c0baa9d80851c9a85c2bf5b3411a1ad2d35` | server4; S3 동일문자열 lock 부재 |

읽은 기준은 현재 main의 PRICE submission/source-input-preflight, Alpha pinned source019922b1, SH3 readiness asset inventory와 과거 CP·basis·실행 lock다. MEMIT87a5a736 원 Git object는 로컬에서 확보되지 않아 frozen source 동일성은 미검증으로 표시했고, tracked preflight/config/lock 참조와 현재 게시 소스만 사용했다. Source Alpha prepare 파일은019922b1 Git object와 현재 main bytes가 일치한다. Remote frozen raw/lock를 가져오지 않았다.

FE는 기존 tracked status상 server2 소유이고 S3 local에 FE task root는 없다. S3 다른 실험 입력·source·raw·실패증거는 task의 KEEP 경계와 본 envelope에 따라 보호했다. 어떤 old job도 resume/cancel/hold하거나 결과 진행률을 조회하지 않았다. Scheduler/open-file 조회0이므로 전체 server3 job이 idle이라고 주장하지 않는다.

W&B ignored local files는 보호했다. 동기화 상태는 조회하지 않았고 SDK/로그인/smoke/new run0이다. 과거 NoCP는 현재 CP 삭제 승인으로 해석하지 않았다.

## 스냅샷·범위·한계

- 실제 host/user: ubuntu/janghj. Session `01a0b9c8-d12c-7133-b336-cc1b5fc2a6b3`, actual CWD `/data/janghj/ODE-edit`, origin `hyunjun1127/ODE-edit`. Registry CWD와 일치하며 전용 non-main WT boundary PASS. 기존 root dirty를 보존했다.
- 1차 관측 `2026-10-06T16:46:17.071399+00:00`, 기존 bootstrap GPT-J 보충 `2026-10-06T16:48:29.235794+00:00`. 순차 lstat/scandir만 사용했고 1차 metadata scan 2.454초 + 보충 0.004초, nice15/ionice idle/CPU affinity1이다.
- `/data` filesystem: total7,619,770,974,208B, 1차 available119,471,538,176B (111.267GiB), 보충 시 available114,191,192,064B (106.349GiB). 비독점 값이며 차이를 본 task 또는 특정 job 사용량으로 귀속하지 않는다. reserved 포함 free와 일반 사용자 available을 구분했다. 1차 available inodes224,054,892.
- Root/local 및 등록 worktree(모두 root 아래), readiness의 정확 Llama/Qwen modelcache·P·C0/hparams, bootstrap의 GPT-J model/P를 점검했다. 등록 worktree 18개 중 본 점검 worktree/self는 제외했다. 경로별 상세 및누락은 coverage.json에 있다.
- 170,082 regular paths, 1,090 symlink entries. Symlink를 따라가지 않았다. Tensor/pickle/model load0 및 전수 payload SHA 재계산0. 근거 metadata JSON40개/9,839,819B만 제한적으로 읽었다.
- `.git`/private/config 경로, venv/site-packages/uv-cache/node_modules 등 package infrastructure는 제외했다. 디렉터리 inode block, 제외 환경, 다른 사용자·다른 repo·미등록 cache는 합계에 없다. 따라서 이112GiB대 합계는 프로젝트 전체 du나 filesystem 사용량이 아니다. SDK 중복/환경 정리 여부도 판단하지 않는다.
- 동일 이름·동일 크기를 byte equality로 처리하지 않았다. 기존 SHA+stat 일치는 보호 근거이며 새로운 암호학적 내용 검증이 아니다. 스냅샷 중 다른 프로세스가 파일을 바꿀 수 있으며 freeze/lock을 걸지 않았다.
- Owner audit만 수행했다. 독립 subagent/red reviewer는 사용하지 않았다. 분류 합계·inode 합계·CSV top20·delete flag·참조 manifest를 별도 CPU 산술 검사했다. 추가 GPU/평가/Slurm/W&B 작업은 없다.

## 재현·산출물

- [summary.json](../../../../audits/servers/server3/price-storage-inventory-20261007/summary.json)
- [coverage.json](../../../../audits/servers/server3/price-storage-inventory-20261007/coverage.json)
- [PRICE bindings](../../../../audits/servers/server3/price-storage-inventory-20261007/price-bindings.json)
- [manifest.json](../../../../audits/servers/server3/price-storage-inventory-20261007/manifest.json)

Full metadata와 기존 소형 evidence 목록은 ignored `/data/janghj/ODE-edit/local/price-storage-inventory-20261007/`에 있다. Git에는 집계·상위20·소형 분류 근거·점검 코드만 게시한다. 원 raw/prompt/tensor/model/CP/private 설정은 Git으로 복사하지 않았다.

Frozen metadata에서 집계 재현(원 payload 접근 없음):

```bash
cd /data/janghj/ODE-edit/local/price-storage-inventory-20261007/worktree
ionice -c 3 python3 audits/servers/server3/price-storage-inventory-20261007/reduce_inventory.py
```

**후속 사용자 삭제 지시 전 어떤 자료도 제거하지 않는다. 점검 완료 후 STOP.**
