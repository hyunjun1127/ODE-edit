# ENFC M-only — T 오류 연동 M 중단 인계

상태: **PAIRED_STOP / WAITING_USER_RESUME**. M은 `PROVISIONAL_T_UNRESOLVED`이며
유효한 완료 결과나 성능 PASS가 아니다. S/R/L 제출0, 자동 T수리/M재제출0.

## 실행 및 정확한 중단

| 대상 | 실제 상태 | Exit / 신호 | 새 할당 GPU초 |
|---|---|---|---:|
| T49928 | FAILED | 1:0 | 2921 |
| M49973_0 / b001 | CANCELLED | parent0:0; batch0:15(SIGTERM) | 310 |
| M49973_1–9 / b002–b010 | 미시작 CANCELLED | 0:0 | 0 |

T1+M array%1로 동시에 최대2GPU였고 overlap235초다. 10개 M을 모두 held inspection
후 release했으나 완료10개라는 뜻은 아니다. T 실패를 관측한 뒤 exact linked
M 0–9만 취소했다. 취소 요청은 2026-09-18T03:10:29Z이며 terminal과 빈 exact
queue를 확인했다. 다른 job 변경·파일 삭제0. SIGTERM을 runtime rollback
성공으로 해석하지 않는다. parent allocation 합계 **3231초 = 0.8975GPUh**,
step/extern 중복합산0, 기존 teacher/native 비용은 재청구하지 않는다.

## 직접 원인 / 수치 검증 범위

Frozen `technical.py:140`이 projector evidence를 `require`로 전달했고,
line36의 `dict(status=..., **evidence)`가 evidence 안의 같은 `status`와 충돌했다.
오류는 `TypeError: dict() got multiple values for keyword argument 'status'`다.
이는 receipt 생성 연결 오류다. OOM/시간제한/FD 불일치/효능 실패가 아니다.
실제 frozen 함수의 작은 CPU 재현에서 같은 TypeError를 확인했고 최소 helper
수리는 CPU regression을 통과했다. 실행 중 T 원본은 변경하지 않았다.

Native cold8 반복2, teacher KL0, noop NLL/logit0, direct/cached gradient 상대0,
보호입력48개 physical/cached logits0 및 key exact는 해당 endpoint에서 확인됐다.
Pstar allowed14326/14336, K443/rank443/free13883은 저장 geometry다.
Projector 함수 return은 있었으나 receipt가 실패했으므로 **그 잔차 값/판정을
PASS로 복원하지 않는다**. FD12-scale, 실제 nonzero invariant, 최종 restore,
T_READY는 NOT_RUN. 자세한 단계는 [technical-coverage.csv](technical-coverage.csv).

## M 실제 완료 범위와 재사용

B1 native capsule은 retained cold7 WN/target/key/zeroM/context identity로 재사용했다.
이번 M 신규 native fit0/target0. b001은 input/cache/geometry까지만 갔으며
controller selection0, final L4 endpoint0, official evaluation0, M initial0다.
9개 나머지 episode는 미시작이다. 최종 endpoint80개 보존 계약은 유지됐으나
실제 만들어진 final endpoint는 없으며 없는 checkpoint를 있다고 쓰지 않는다.
Partial geometry/tensor는 보존하되 SIGTERM 시 serialization 완결성까지
검산하지 않았고 terminal-valid endpoint로 사용하지 않는다.

330행 item-level [재사용 판정](m-reuse-decisions.csv)과
[10episode 계획](m-execution-plan.csv)을 보존한다. B1 fit 재사용/나머지 최대9fit,
W0 및 B1 native canonical pair 재사용, 없는 greedy32/Dev만 신규라는 계획이었다.
같은 episode의 byte-identical endpoint 관측은 공유하며 분모/비용을 중복계상하지
않는다. M은 독립 W0/M0 cold100×10이며 이전 sequential B2+를 cold로 바꾸지 않는다.

## Source·검증·저장 경계

- T frozen HEAD `729d4544e860207af44119b6b283cf00e851df21`, tree `4ce68fc9c6a70a40b28fa3f68fabcd88f8f73ef0`.
- M frozen HEAD `76bb90372b6ddc05f53374812bfc2df90153601e`, tree `4c5dfdb12615f2c7b391f8ecf7b1788b59919158`.
- T lock `eb103fd12c9584e4989be2e892e6de2d3ed6075b0cf150261e9a00a65344bf2b`.
- M lock `cc7510f4cd6fb41bf24195205b6c12549041f01b149e99aabfb01907eb88f6ea`.
- M source는 receipt helper 수리와 M/reuse/observer/병행 연결을 포함한다.
  Core alltoken/geometry/binding bytes 및 model 수치 메서드 AST를 별도 대조했다.
  다른 orchestration 전체 GPU 동일성을 주장하지 않는다.
- CPU focused tests와 actual T 수치는 분리한다. 원11 toy/1430 design receipt는
  이전 근거이며 real-model PASS가 아니다. M final L4 저장과 cold independent
  history0(명시 M cells)를 구현했고, 과거 noCP/FDskip/mean plateau는 상속하지 않았다.
- Raw/prompt/gradient/teacher/full stdout은 local-only. 본 package는 compact
  수치·경로·SHA 및 source만 포함한다. Reference-only 과거 task 재개0.

## 최소 복구 제안 — 실행하지 않음

이미 저장된 유효 cold8 native/teacher-repeat/G/Pstar를 identity로 재사용하고,
receipt helper 수리 source에서 projector receipt와 미실행 FD/invariant/restore만
새 immutable 기술 attempt로 확인하는 범위가 최소다. 현재 준비된 continuation
코드는 제안일 뿐 실행하지 않았다. M partial의 geometry는 보존되지만 full
continuation checkpoint가 아니다. T수리 또는 M 재제출은 사용자 recall이 필요하다.
수치 threshold/grid/과학식 변경과 새 S/R/L은 제안/실행하지 않는다.

재현: 같은 pinned environment에서 `python -B -m
project.run_scripts.single_layer_edit_preserving_correction.preflight --receipt
NEW_CREATE_ONCE_PATH`는 CPU-only 검사다. 이 package의 생성은
`python -B -m project.run_scripts.single_layer_edit_preserving_correction.handoff
--worktree WORKTREE`였으며 기존 package가 있으면 덮지 않는다. GPU 재평가는 하지 않는다.

최종 manifest/rooted receipt는 아래 파일과 local evidence inventory를 결속한다.
완료/효능 claim 또는 후속 실험 선택은 하지 않는다. 자동 monitoring/callback0.
