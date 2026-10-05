# Causal Allocation Editing — 사용자 취소 및 등록 사실 보고

현재 상태는 `USER_STOPPED`다. 사용자 요청 “server4 실험 취소하자”에 따라 server4의 이 실험만 취소했다.

## 사용자 취소 — 2026-10-06 KST

| 역할 | Job ID | 취소 직전 | 최종 상태 | parent 할당 GPU초 |
|---|---:|---|---|---:|
| qualification | 59163 | COMPLETED | COMPLETED, 취소 호출 없음 | 85 |
| main | 59164 | RUNNING | CANCELLED by 1025 | 2346 |
| CPU collector | 59165 | PENDING Dependency | CANCELLED by 1025 | 0 |

정확 owner/name/node/source/config/lock/script/전체 argv 및 dependency를 대조한 뒤 collector→main 순서로 취소했다. 2026-10-06 05:29:46 KST에 취소, 05:30:22 KST까지 한정 종료 확인을 마쳤다. Exact-job queue는 비었고 main batch/extern도 terminal이다. 현재 task GPU 할당은 0이며 무관 job 변경은 0이다.

Parent 할당 GPU 시간 합계는 2431초(약 0.6753 GPUh)다. Batch/extern 시간을 중복 가산하지 않았으며 active kernel 시간은 미측정이다. 원 source/archive/config/raw/log 및 완료 qualification은 그대로 보존했다. 취소 작업에서는 모델 로드·결과 상세 리뷰·새 제출을 하지 않았다. Qualification job의 `COMPLETED`만으로 실제 LM 정합·양수 λ·B1/B2·W20을 인증하지 않는다. NoCP/exact resume `NOT_AVAILABLE`, 자동 재개·retry·반복 모니터링은 없다.

상세 취소 receipt: [user-stop-20261006.json](../../../../audits/servers/server4/causal-allocation-editing/user-stop-20261006.json). Raw local KEEP, 취소 receipt만 소형 게시하므로 `NO_BROADCAST_NOT_REQUIRED`다.

## 아래 내용은 취소 전 등록·초기 인계의 역사 기록

등록 당시 `SUBMITTED_INITIAL_RESOURCE_PENDING`: 구현·CPU/source 검사 후 전량 held 등록·검사·release했다. GPU qualification과 본선 완료를 뜻하지 않는다. 아래 pending/미관측 상태는 당시 snapshot이며 현재 상태를 뜻하지 않는다.

- 새 production CPU 회귀 63/63 PASS. 별도 소스 리뷰에서 확인된 mechanical blocker는 수리 후 0이다.
- 정본 manifest/envelope SHA, first2000 순서, 20 pack과 14,000 native full-token 행을 결속했다. 원 설계·기존 task/source/raw는 변경하지 않았다.
- 한 경로, cold W0/H0, BS100×20, L4–L8. W5/10/15/20 누적 평가이며 W20 분모는 R2000/P4000/N20000이다.
- 실제 LM qualification·양수 λ·속도·실현률·W20은 아직 미확인이다. CPU PASS를 GPU PASS로 표기하지 않는다.
- 등록 경로: cap1 최소 qualification → exact READY 확인 → fresh 본선 native calibration/20 commits → afterany CPU collector. 신규 baseline/추가 arm/추가 full-B fit는 없다.
- GPU job은 각 1GPU/8CPU/59392MiB, main 요청 상한 48h. 입력 기반 host 실행 추정은 38.0GiB이며 실제 peak/ETA는 미측정이다.

소스·소형 receipt만 Git에 게시한다. 원 raw/tensor/model/fullstdout는 local KEEP (`NO_BROADCAST_NOT_REQUIRED`). noCP, exact resume 불가. 품질·집중·고정예산 미수렴은 관측 결과이며 기술 gate가 아니다. 양수 가격 보정과 정합/비유한/solve/commit 검사는 정본 기술 경계대로 유지한다.

## 등록 receipt

| 역할 | Job ID | 요청 | 한정 release 후 snapshot |
|---|---:|---|---|
| qualification | 59163 | GPU1 / CPU8 / 59392MiB / 4h | PENDING: ReqNodeNotAvail, May be reserved for other job |
| fresh 2k main | 59164 | GPU1 / CPU8 / 59392MiB / 48h | PENDING: afterany 59163 |
| CPU collector | 59165 | GPU0 / CPU8 / 24576MiB / 4h | PENDING: afterany 59163, 59164 |

실행 source는 `a1332fd70f0d4898b74399a349e41a225f0baf04`, config SHA는 `fd15741517cf6600f832384d9fd17ee6e26c4dac21fe16b07053bc256dea2f81`, lock SHA는 `d17b71ee91a5183abad5e2e2818b15609e1a4f190aba7f7762db33979dab59cc`다. 실행 archive는 create-once이며 이후 보고 게시 commit과 분리된다.

동일 nonce 중복 등록은 없었고, fresh own GPU 자원/admitted pending은 등록 전 0이었다. 현재 엄격 cap은 project3/task1로 결속했다. 각 job의 owner/name/node/partition/full argv/source/script/memory/GPU/CPU/wall/exportNONE/Requeue0/dependency를 held 상태에서 확인하고 모두 release했다. 기존 job 취소·변경은 0이다.

현재 B1 commit→B2 own entry, actual LM parity, 양수 λ, 실제 비용/peak 및 W20은 `NOT_OBSERVED`다. 첫2k W0 raw는 동일 모델/runtime/token/evaluator에 결속했으나 실제 cold state 확인을 통과해야 재사용한다. Resource pending 상태에서 초기 인계를 종료하며 `monitoring_active=false`, `automatic_resume=false`, `automatic_retry=false`다. Sealed runner/collector는 승인 경계를 따라 자연 진행하며, 추가 반복 polling/heartbeat는 하지 않는다.
