# 50983 교체 및 51055 정리 오류 수리

사용자 “50983 도 그대로 돌리면 오류가 있을 것 같으니 수정해서 다시 올려야한다”, 실패 재확인 요청, 최종 “repair 해서 올려”를 적용한다. 이번은 수정 프로그램 등록 인계이며 전체 과학 완료 보고가 아니다.

## 확인된 실패와 취소

| Job | 확인 상태 | 원인/실행 범위 | Parent GPU-sec |
| --- | --- | --- | --- |
| 50974 | FAILED / 1:0 | 원 hook 궤적 gradient 상대차 gate; 과학 B1 미진입 | 136 |
| 51055 | FAILED / 1:0 | hook 비교 완료 뒤 `list.remove(x): x not in list`; 남은 T0/B1 미진입 | 104 |
| 50983 | CANCELLED by 1025 | 실패한 50974 dependency의 미실행 구버전을 사용자 교체 지시로 취소 | 0 |

50983은 owner janghj/server4/name/source/argv/afterok:50974(failed)를 확인한 뒤 정확 ID 하나만 `scancel`했고, 2026-09-19T17:20:07Z에 terminal cancellation을 확인했다. 다른 job 변경·파일 삭제0. 원 source/raw/lock/실패 자료는 보존한다.

51055의 program failure wall은 99.6631초이며 allocation 104초와 합산하지 않는다. 이전 두 실패 job allocation 합은 240 GPU-sec다. 그 중 재사용 가능한 검증 계산은 남아 있으므로 전부 무효 과학 계산으로 해석하지 않는다.

## 직접 원인과 최소 수정

`Runtime.reset()`은 `self.oracles=[]`를 먼저 수행한다. 이전 수리의 `finally`가 `rt.reset(); rt.oracles.remove(oracle)`을 호출하여 이미 비운 목록에서 제거하다 실패했다. 마지막 `remove`만 제거한다. 과학식/임계값/optimizer/hook 수치 경로의 추가 변경은 없다.

CPU 회귀검사는 성공 return에서의 이중 제거와 기존 method 실패를 cleanup 오류가 덮는 경우를 모두 실행한다. 51055와 수정 source의 함수 AST를 대조하여 **그 finally 차이만 허용**한다. 나머지 6개 hook/model/config 관련 파일은 exact SHA가 같아야 재사용한다. 코드/입력/runtime이 다르면 재사용을 차단한다.

## 재사용하는 실제 검증과 남은 범위

51055의 저장 `hook-summary.json`은 현재 사용자 lenient 정책상 `pass_=true`다. Production은 사전 고정 batch1이며 batch4 진단을 production PASS로 대체하지 않는다.

- Batch1 actual write Current 최대 NLL 차이: **8.96453857421875e-5**, 기준 1e-4 이내.
- Strict/pair 성공 ID exact. 25 loss/24 Adam 종료 일치.
- 궤적 gradient 최대 상대차 **1.3410962613416946e-4**는 기존 1e-4 기준을 초과해 계속 WARN으로 기록한다. 원 strict gate 전체 PASS나 same-point derivative PASS는 아니다.
- Batch4 actual write NLL 차이 **1.220703125e-4**, 원 기준 초과; batch16 미검증.
- Hook 검증만 완료됐으며 `T0_READY`, 남은 decision T0, B1 과학 실행은 없었다.

새 process에서 W0·zero M·model/context/token IDs·입력과 source를 결속하고, 완료된 fixed4 hook 증거 및 기존 native 기준만 재사용한다. Hook z/solve/forward 중복0; **남은 필수 T0는 계속 수행**한다. 이전 실패 process의 복구 성공을 소급 주장하지 않는다.

## 등록 및 저장 경계

새 단일 프로그램은 완료 hook 재사용 → 남은 full T0 → B1 → 기존 CUM gate 조건부 S3/S10이다. 구 50983이나 실패 T0에 의존하지 않는다. 메서드·과학 확대 gate·cap2는 유지한다. 각 job GPU1/CPU8/mem60416MiB/exportNONE/Requeue0, 계획 wall 7일이다. Resource-only admission과 held owner/source/fullargv/resource 검사를 거쳐 release한다.

새 W/M/RNG/전체 weight checkpoint0, exact_resume=NOT_AVAILABLE. RAM state와 작은 target/key·평가/ledger/hash는 유지한다. 기존 checkpoint/teacher/source/raw 삭제·덮어쓰기0. 초기 24GiB·후속 stage 저장 조건 유지, 과거 waiver 상속0.

CPU 213 tests PASS (8.484초; 포함된 신규 cleanup 6 tests는 별도 합산하지 않음). CPU fixture는 actual 새 full T0/science PASS가 아니다. 별도 독립 red agent는 이번 수리에 사용하지 않았다. 전용 worktree의 session helper 설정 부재는 이전과 같은 NOT_PASS이며 실제 registry/host/repo/session 결속을 재사용하고 공용 설정을 수정하지 않았다.

등록 이후 scheduler/result/log 모니터링은 중지한다. `monitoring_active=false`, `automatic_resume=false`, 사용자 recall 대기. `NO_BROADCAST_NOT_REQUIRED`.

## 재현·출처

- Source namespace: `project/run_scripts/single_layer_mechanism_first/`의 `technical_repair.py`, `reuse_completed_hook.py`, `repair_r2_plan.py`, `program.py`, `submit_repair.py`, `test_cleanup_repair.py`.
- CPU: `python -B -m unittest discover -s project/run_scripts/single_layer_mechanism_first -t . -p 'test_*.py'`.
- 이전 실행: `5ea4e4efad5a9420674641dd13a04d4651701a08`; 원 frozen source는 불변이다.
- 새 실행: create-once plan → clean commit → `freeze --phase GATED_PROGRAM --attempt cleanup-repair-r2` → `submit_repair --lock <execution.lock.json>`.
- Exact 새 job/source/archive/lock 및 release 상태는 후속 compact 등록 receipt에 결속한다.

## 실제 등록 결과

**51056 / odeedit_slmf_S10r2_s4**: held 검사13/13 PASS 후 release. 2026-09-19T17:37:05.703395Z admission 시 전체 own resource queue 및 제출 시 server4 목록은 비어 있었다. GPU1/CPU8/mem60416MiB, cap2 내 등록이다. 초기 free 49,219,559,424B/inode225,306,003은 당시 관측이지 전 stage 독점 reserve가 아니다.

Release 직후 PENDING / Reason=None / Dependency=(null). GPU 부족 또는 actual 새 full T0 성공으로 해석하지 않는다. 이후 결과 모니터링0. 최종 frozen source의 CPU213 회귀검사도 8.892초로 PASS했다.

- execution `bdaed735f28cb2d0a24e56cc00723373ef857d9d`, tree `a757ee7b9251a4e21a9b19897d17a56054bb5a4d`.
- archive SHA `44c78f6f8929b4a5fbbef09b165f29ca092ed4aa69843ace1c8b0f9cbf07b16b`, 464,860,994B.
- lock SHA `09c31436af189650125509abb1e7c8dfa5ad0b9535ba5cd7f8c7cdb12050d9ad`.
- output `/data/janghj/ODE-edit/local/single-layer-mechanism-first/20260919-v1/PROGRAM/cleanup-repair-r2/output`.

새 full T0/B1/S3/S10 및 새 allocation은 아직 NOT_OBSERVED/NOT_MEASURED다. 표/JSON/링크·raw-free 검사와 manifest를 남기며, 이전에 확인한 미설치 HTML renderer는 NOT_RUN으로 유지한다. 이번 compact 인계에는 그림이 불필요하다.

[등록 evidence](../../../../../audits/servers/server4/single-layer-mechanism-first-20260919-v1/repair-r2/registration.json). 기존 r1 report/manifest는 해당 publication commit의 역사 기록이며 mutable task status의 현재 bytes와 혼동하지 않는다.
