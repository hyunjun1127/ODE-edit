# L4-preserving repair 두 arm — 준비·제출 사실 보고

상태: **공통 기술 pilot 49238 PENDING / 본실험 두 arm 미등록 / MONITORING_PAUSED_AWAITING_USER**.
마지막 scheduler 관측은 2026-09-17T04:43:51.554816+00:00이며 이후 결과·로그·scheduler를 조회하지 않았다. 현재 상태나 완료를 추정하지 않는다.

## 최신 사용자 checkpoint 미저장 적용

“checkpoint는 저장하지 말고 진행하라”가 기존 저장 요구보다 우선한다. W4/W8/M4/RNG disk checkpoint는 생성하지 않는다. RAM의 batch간 상태 전달·rollback, 평가 NLL, commit/hash/received ledger, Q/response/기술 probe 증거는 유지한다. 이들은 complete continuation checkpoint가 아니며 exact restart는 NOT_AVAILABLE이다. 기존 파일 삭제·덮어쓰기0.

## 승인 범위와 실제 실행 수준

| 대상 | 승인 범위 | 이번 등록 | 실제 검증 수준 |
| --- | --- | --- | --- |
| 공통 pilot | 첫100 native L4 fit/solve1 공유, R-GD/R-QP 기술 연결 | 49238, held 점검 후 release | 마지막 PENDING, 실제 모델 NOT_RUN |
| R-GD | fresh W0/zeroM4, B100×10 | 미등록 | 기술 READY 전, 과학 결과 없음 |
| R-QP | fresh W0/zeroM4, B100×10 | 미등록 | 기술 READY 전, 과학 결과 없음 |

두 main 계획은 unique1000/arm-request2000/20batch, native target2000·solve20·M4append20·repair endpoint 최대120이다. 실측이 아니다. 신규 N4/REFIT4/LD, L8 native target/P8/M8, 별도 paraphrase training/guard, 다른 task 실행0. Pilot native100은 science 분모에 합산하지 않고 main에 warm carry하지 않는다.

## 구현과 기술 판정 경계

자기 entry의 native full L4 write WN 후 W4를 고정하고 physical L8 full-weight repair만 적용한다. R-GD는 Base 1방향 scalar proposal, R-QP는 Base/Current/존재시Past gradient span의 최대3방향이다. FP64 Gram/QR·pN Fisher JVP·최소 condition shift·whitening·많은 guard를 처리하는 작은 active-set QP를 구현했다. Toy2D solver를 production으로 복사하지 않았다.

Actual Current/Past mean≤각 WN mean+1e-4, strict/preference exact ID subset, Base predicted/actual gain>1e-6, agreement≥.1을 유지한다. 최대6 endpoint 중 첫 acceptable만 선택하고 r/2 중 Q/b/A/H는 고정한다. 정상 off는 이번 repairΔ8=0이며 이전 누적 L8은 유지한다. M4 finalizer1/inner0/M8append0, received ledger 전량 갱신이다. Official P/N은 selection.json 봉인 뒤 observer에서만 사용한다.

CPU 51 tests PASS(geometry/QP28, tiny-model response7, engine mock16), import/syntax/shell 검사 PASS. Independent source 검토의 누락을 수정했다. 이는 실제 Llama FD/JVP/GN/materialization/teacher/history 검증 PASS가 아니다. 해당 필수 pilot checks가 없거나 실패하면 READY를 생성하지 않는다. 기술 규칙은 GPU 관측 전에 numerical.py/geometry.py/qp.py와 execution lock으로 고정했다.

## 자산·source·환경

- 실행 HEAD `a89350e2c2b2b0652f0a085b2b6c2e9c3accc89b`, tree `645174a7cbe01b611449ad3cf32abd6c035e671c`. 출판 commit과 구분한다.
- Archive SHA `9dc7f87d50c32fe3c19497a0f44975a0b3bea2626c82f07199c159b09465cbe0`.
- Execution lock SHA `196e9fb464eb4836079c093a05be502fb4c38e78fce60b9c4767ef56b71614c5`.
- Cold capsule SHA `2d5d5c45bbbdf36ed859451242d7d84874d08cda4008d585945e565d9b3b1cb7`. W0 revision8afb486c1db24fe5011ec46dfbe5b5dccdb575c2, method seed20260916, FP32/eager, 양 TF32off, P4 physical4→asset0→local0, M4zero.
- Fixed10k first1000 root `40fe1afb318a4a5da9630425213644a729a1e85fb4e78c18e9a4dcd2870eefdd`, 같은 순서/100씩10batch. 재추출0.
- 기존 C4 reference/S64/Dev128 teacher 재사용 우선; 새 teacher 생성0. 실제 W0 연결 검사는 pilot에 예정되어 있고 아직 PASS가 아니다. C4 sampling seed20260915와 method seed를 분리한다.
- 실제 host/session/repo/환경변수·tracked registry는 일치. 공용 session helper는 root의 오래된 session 설정 및 새 worktree 설정 부재로 BLOCK이다. helper/shared config를 수정하지 않았으며 helper PASS로 기록하지 않는다.

## 자원·비용·저장

제출 직전 bounded resource-only admission에서 기존 Server4 project 예약용량0, cap2를 확인했다. 공통 pilot1GPU/8CPU/60416M/exportNONE/Requeue0/wall12h를 held 상태에서 점검하고 release했다. 기존 job 변경0. 두 main은 READY 이후 실제 pilot 측정으로 wall/storage를 lock하고 가능한 두 slot에 독립 등록해야 한다.

제출 시 available disk 136,619,634,688 bytes, 최소 reserve 51,539,607,552 bytes(48GiB). Q 최대14.1GB 등 진단/평가와 여유를 고려한 계획이며 실제 사용량이 아니다. No-checkpoint가 반영되었다. Allocation/compute/peak/science 비용은 아직 미측정이다. 이전 cold7/teacher 비용은 새 비용에 재합산하지 않는다. Hour hardcap=null, 신규 GPU 실측을 CPU PASS에서 추정하지 않는다.

## 재현·인계

실행 lock: `/data/janghj/ODE-edit/local/l4-preserving-repair/20260917-v1/execution.lock.json`. Frozen source: `/data/janghj/ODE-edit/local/l4-preserving-repair/20260917-v1/source-v1`. 예상 pilot: `/data/janghj/ODE-edit/local/l4-preserving-repair/20260917-v1/technical/attempt-v1`. READY 파일 존재는 인계 이후 재조회하지 않았다. Local resume: `/data/janghj/ODE-edit/local/l4-preserving-repair/20260917-v1/resume-manifest.json`, SHA `447dc9e270e62d410d0b78c67ee5335b22f3d7b0a0657c224ea6d14febf71038`.

CPU 재현은 package README/checks.py의 51개 unittest를 CUDA_VISIBLE_DEVICES=''로 실행한다. Published checks source SHA와 frozen lineage를 비교해야 하며 GPU 작업을 자동 실행하는 재현 명령은 제공하지 않는다.

monitoring_active=false, automatic_resume=false, resume_trigger=explicit_user_call. PENDING에서 시작을 기다리지 않는다. Callback/자동 추가submit/완료 대기/다른 task 재개0. 다음 명시 recall에서 exact pilot 증거를 확인한 뒤 READY 성립 시 승인된 두 main을 제출할 수 있다. 본 보고는 제출 인계이지 본실험 완료 보고가 아니다.
