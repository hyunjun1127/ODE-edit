# GH → SH3: 53996 실패 확인·수리 재제출, 실제 초기 gate까지 관찰

동일 task: `ODEEDIT-GH-SH3-MEMIT-HISTORY-FIXED10K-20260928-R1`.
Override nonce: `ODEEDIT-GH-SH3-MEMIT-HISTORY-REPAIR-INITIAL-GATE-20260928-R1`.
Target: server3 / ubuntu / session `01a0b9c8-d12c-7133-b336-cc1b5fc2a6b3`,
CWD `/data/janghj/ODE-edit`, repo `hyunjun1127/ODE-edit`.

사용자: “fail되엇으니 확인하고 rerun올리라고 전달해. 그리고 초기 gate 성공까지는
모니터링 사용하라고 해”. 이는 동일 task의 명시적 recall 및 모니터링 경계 변경이다.

## 1. 실패 확인과 최소 수리

- 지정 job53996의 owner/name/source/fullargv, 한정 accounting/로그/failure/raw만
  확인해 최초 오류와 도달 단계·완료 batch·실제 allocation 비용을 기록한다.
  사용자 실패 통지와 actual 관측을 구분한다. 다른 task 결과 조회/변경0.
- 원 execution `f2c1d6f17774d6c591d2a0d71e8165a615b26cba`,
  lock `e9d2a322dee39ae67f5d0258279d2df3a8bc7576b726de4a098c39f12b68dfa5`,
  attempt-v1/source/raw/실패/비용은 불변 보존한다.
- 원인과 최소 재현을 선보고하고, 승인 namespace의 adapter/runtime/직렬화/
  자원/호환성 기술 오류를 직접 최소수리한다. 좁은 CPU regression 후 새 immutable
  attempt/source/archive/lock으로 실제 rerun을 등록한다. 단계마다 GH 재승인 불필요.
- Pinned BLUE311b076 `memit.memit_seq_main.apply_memit_seq_to_model` 직접 사용 유지.
  history writer 별도 재구현/BLUE=true/새 arm/z-hook/수식·hparams·history 시점 변경0.
  원 writer 호환성 patch가 불가피하면 pinned 원본 보존 및 exact 최소 diff를 명시하고
  의미 변경이 없음을 검토한다. 원인 미확인 상태에서 같은 실패를 반복-to-PASS하지 않는다.
- noCP로 exact resume 불가: 새 chain은 fresh W0/H0, 동일 fixed10k B100×100이다.
  old metadata/hash만으로 실패 후 W/H 상태를 복원했다고 하지 않는다. 입력/model/C0/
  context와 valid CPU 증거는 identity가 맞으면 재사용하되 old edited state는 재사용0.

## 2. 실제 초기 gate와 모니터링 종료 조건

이전 **release 즉시 monitoring 중단** 지시는 이번 task에 한해 대체한다.
SH3가 수리→held inspection/release→actual 첫 본실험 경계까지 bounded 관찰한다.
CPU PASS/모델 load/제출 성공/RUNNING만으로 initial PASS를 쓰지 않는다.

실제 초기 gate는 동일 rerun 프로그램의 다음 저장 증거다:

1. B1의 exact first100 요청 및 pinned 실제 entrypoint/config/source 결속.
2. B1 native write가 완료되고 유한한 실제 W와 5층 H가 commit됨.
   `cache_c` 전달·반환, prior-H solve, post-all-layer key의 각층1회 append를 확인.
3. B1 Current R/P/N 및 TF/NLL 관측 완료, 분모100/200/1000과 identity 확인,
   observer가 W/H/context/RNG를 임의 변경하지 않았다는 runtime 근거.
4. B1 committed W/H/context/RNG/ledger와 B2 entry의 연속성 확인.
   H reset/중복 append 없이 B2 진입. B2 완료나 100batch 완료를 기다리지 않는다.

RAM state/hash/counter/작은 receipt로 확인하며 이를 위해 checkpoint를 저장하지 않는다.
이는 구현·상태 초기 gate이지 성능 개선·bitwise cross-host numerical certification
gate가 아니다. 근접 수치 차이는 기존 record-only 방침, identity/finite/history/IO
오류는 실제 기술 문제로 분리한다. 별도 긴 GPU smoke/FD/repeat 캠페인 추가0.

Pending 또는 CPU 준비에서 release-only STOP을 재사용하지 않는다. cap-safe 등록을
유지하면서 초기 gate까지 관찰한다. 접근 불가/해결 불가능한 자원·과학계약 변경 필요
등 실제 blocker면 정직하게 보고하며 gate PASS를 꾸미지 않는다.
초기 이전 재차 기술 실패 시 같은 scope의 RCA/최소수리 절차를 적용하며 과학조건은
임의 변경하지 않는다. 과거·신규 실패와 비용을 모두 보존한다.

초기 gate actual PASS 시 job ID/receipt/source/lock/첫 수치를 compact 보고하고
own-scope main 게시 후 `MONITORING_PAUSED_AWAITING_USER`로 turn을 종료한다.
그 뒤 scheduler/log/result/terminal polling·heartbeat·callback·자동 recall0.
같은 runner의 B2–B100과 reducer는 자연 진행한다. 완료 상세리뷰는 별도 user recall.
GH는 수신 ACK만 회수하며 SH3와 중복으로 scheduler/GPU/raw를 감시하지 않는다.

## 3. 변경 없는 권한과 자원

기존 두 envelope의 허용 write/source/report/audit/local/runs/status scope와
nonforce own-scope branch/main 게시 승인을 상속한다. 새 attempt는
`/data/janghj/ODE-edit/local/memit-history-fixed10k/20260928-v1/` 아래 분리한다.
기존 다른 job cancel/hold/throttle0. 실패53996을 재취소할 필요 없다.
현재 예기치 않은 동일task 활성 job이 있으면 identity 확인 후 중복 제출을 막는다.

Server3 project/taskcap1, 1GPU/8CPU/host≤121856MiB, ubuntu/gpu,
exportNONE/Requeue0. 새 source/resource admission·held 검사를 다시 수행한다.
noCP/exact_resume NOT_AVAILABLE, raw local/공유 native 원본 KEEP,
C4 복구/GSS·EN 재개/대형 전송0, NO_BROADCAST_NOT_REQUIRED.

첫 ACK에 이 nonce와 “RCA/repair 진행, 새 제출 전/후”를 구분해 회신한다.
첫 기술 오류와 rerun job mapping 및 최종 initial gate 인계를 각각 보고한다.
