# Temporal-routing 진단 — 실행 준비 사실

권한: `ODEEDIT-GH-SH4-TEMPORAL-ROUTING-DIAGNOSTIC-20260929-R1`.
[정본 설계](../../../../plans/global/2026-09-29-temporal-routing-diagnostic-v1/design-ko.md),
[실행 envelope](../../../../messages/head/2026-09-29-temporal-routing-diagnostic-sh4.md).

## 입력·경계

전용 branch `codex/server4-temporal-routing-diagnostic-20260929-v1`; server4/session `01a04939-b5c7-7a03-ba2d-ef3343d62cfd`.
원본/이전 task/다른 job은 변경하지 않는다. 정본 14개 size/SHA 일치, 원 CSV CRLF 불변이다.
CSV 전체를 parsing하여 15개 branch 각각 동일 100요청, 1,500 fit 행, 120 panel 행, 30 snapshot 행과 metadata order를 검산했다.

기존 parent CP는 `local/alpha-key-concentration-causal/20260923-r1/inputs/checkpoints/BASE_ALPHAEDIT/{B010,B050,B090}/W-method-state.pt`를 읽기 전용 재사용한다.
과거 historical T0의 full-file SHA를 재사용하고 현재 size/inode/mtime를 대조했다. 이번에는 별도로 모든 W tensor hash·shape·finite, 전체 M shape·finite·tensor hash 및 parent metadata를 CPU에서 확인했다. 이번 full-file SHA 재검산이라고 쓰지 않는다. P는 신규 전체 SHA/schema/finite 검산했다. 새로운 원격 전송 0.

| 근거 | 실제 SHA256 |
| --- | --- |
| CPU 입력 binding (local) | `101bb78ce768aaba986d90150c661c36d913d1161bc50829e56a1f7ec56a1df0` |
| FULL_READ receipt (local) | `eb84fc23ded386b1a36442570fc150754c45935ec43469ad950fedc9e6441697` |
| 토큰/저장 CPU preflight (local) | `0ec0c42e739baab807e77cc65eae33c725f8d0755067f1b7dc8b56fbabf86af3` |
| owner CPU22 audit (local) | `41155bd9b839e27949abb3922016c62846c925579027cb98a717371b568129e8` |

위 local root는 `/data/janghj/ODE-edit/local/temporal-routing-diagnostic/20260929-v1/`이며 각각 `preflight-v1/input-binding.json`, `preflight-v1/full-read.json`, `preflight-v2/cpu-preflight.json`, `cpu-audit-v2/owner-cpu-audit.json`이다. 큰 input binding의 raw parent metadata는 Git에 넣지 않는다.

## 구현·검증

원 native source `a4d24f...`를 별도 module 이름으로 로드하여 호출하고 numeric AST/식은 수정하지 않는다. 원 compute_z/compute_ks/nethook/representation caller와 dependency hash를 결속했다. NativeTrace는 기존 locals만 관측하며 추가 target fitting/forward를 만들지 않는다. 선택층 post-write history를 native 함수가 정확 한 번 append했는지 별도 CPU outer product와 bitwise 대조한다.

CPU22 PASS: signed A/B/C expansion, live-key C, 정규화, tie-failure, TF/preference 구분, pair identity, 덮어쓰기 차단 atomic rename, tensor reload, fixed random norm, 입력 전체 표, tiny CPU Llama의 all-token hook/capture·zero/random patch. 작은 CPU 모델은 실제 Llama-3-8B 검증이 아니다. 별도 subagent/red 미사용: owner source 감사 + CPU 회귀검사다.

Actual 검사는 각 branch 안에 통합한다: parent5W/5M restore, L2=10/physical slot, selected-key invariance, nonselected W/M 불변, observer 비변이, history1→다음 entry, finite 및 snapshot 실제 IO/reconstruction. 새 numerical waiver는 없다.
각 비교의 evidence를 failure 전에 남긴다. 정상 음성 결과/망각은 commit을 취소하거나 retry하는 조건이 아니다.

정해진 TF prefix/모든 valid input token에서 capture한다. Base sensor/observer true-answer prediction token에서 full-vocabulary W0 teacher KL을 token mean→question mean으로 집계한다. History는 유효 새 목표 NLL이다. Current R+2P, milestone fixed10N, pre/post observer와 at-write retention을 따로 저장한다. `A/B/C` FP64 분석 산술은 native FP32 solve를 바꾸지 않는다. 미명시 상수 epsilon=1e-12/random seed=20260929를 결과 전에 고정했다.

Snapshot n50/n100은 실제 FP32 [4096,14336] full selected tensor다. create-once `renameat2(RENAME_NOREPLACE)`, fsync, 재로드 bitwise, full parent 복원+overlay 출력 검사를 수행하도록 구현했다. `exact_editor_resume=NOT_AVAILABLE`.

## 자원·비용 계획

| 항목 | 계획/관측 |
| --- | --- |
| project/task cap | 2 / 2, array 0–14%2 |
| branch job | 1 GPU, 8 CPU, 60,416 MiB host, exportNONE, Requeue0 |
| CPU collector | 0 GPU, 8 CPU, 24,576 MiB, afterany |
| GPU 장치 | RTX PRO 6000 Blackwell, 장치별 97,887 MiB 관측 |
| host/GPU 예상 peak | 48 / 64 GiB, 실측 아님 |
| token 계획 | 전체 catalog 940 TF rows, 각 branch 844, 최장 input36 |
| capture/teacher/snapshot payload | 86,787,686,400 B (계산값) |
| 총 admission reserve | 155,507,163,136 B (payload + 64 GiB 여유) |
| preflight free | 807,852,158,976 B; 공유 FS 독점 예약 아님 |
| snapshot payload만 | 7,046,430,720 B, 30개 |
| wall | branch당 24h 보수 등록, 실제 ETA/사용량 아님 |
| 신규 native budget | 1,500 scientific fits; duplicate 기술 fit 계획0 |

원본 native solve는 FP32다. 자원 검토에 FP64 diagnostic scratch를 포함하되 FP64 native solve로 바꾸지 않는다. 새 fitting 이후 실제 timing/peak는 프로그램 cost에 남긴다. 독립 branch의 W0/entry observer setup은 반복되며 숨긴 dedup/speedup을 주장하지 않는다.

등록 전 resource snapshot: 본 사용자 S4 queue 없음, 물리 GPU8/8 할당. 제출 직전 admission을 다시 확인한다. 다른 job을 취소/변경하지 않는다. 아직 job/actual 초기 gate 없음.

전량 정상 등록·held 검사·release 후 대표 step1 commit/history1→step2 entry를 확인하거나, 실제 resource pending을 확인하면 agent monitoring을 중지한다. 등록된 15 branch/collector만 자연 진행한다.
`NO_BROADCAST_NOT_REQUIRED`: 같은 host 입력/진단이며 별도 원격 raw 소비자가 없다.
