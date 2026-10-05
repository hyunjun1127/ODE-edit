# Causal Allocation Editing — 등록 및 초기 인계 사실 보고

현재 `SUBMITTED_INITIAL_RESOURCE_PENDING`: 구현·CPU/source 검사 후 전량 held 등록·검사·release했다. GPU qualification과 본선 완료를 뜻하지 않는다.

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
