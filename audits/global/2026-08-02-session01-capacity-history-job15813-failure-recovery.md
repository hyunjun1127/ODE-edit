# Session 01 capacity/history — Job 15813 failure 및 recovery 판정

- 날짜: 2026-08-02
- 범위: `odeedit_capacity_history_pair_v1`, Slurm job `15813`
- 판정: **technical failure; scientific result 없음; full-pair v2 recovery 1회 허용**

## Repo/protocol에서 확인한 사실

- Parent job은 2026-08-02 13:25:36--13:35:20 KST에 실행됐고 `FAILED 1:0`이다.
- `15813.2`가 `FAILED 2:0`으로 최초 종료했다. Parent launch order에 따라 이는
  Qwen MEMIT worker다. `15813.0`, `.1`, `.3`은 parent fail-fast에 의해
  `CANCELLED`, exit `0:9`가 됐다.
- Qwen MEMIT native에는 edit 1--3의 actions/receipts와 edit 4 direct-z artifact까지만
  존재하고 summary가 없다. Llama MEMIT native, Llama Alpha native-history,
  Qwen Alpha native-history summary는 pass지만 sibling lifecycle의 partial output이다.
- Llama의 QP directory 두 개는 첫 edit direct-z까지만 생성됐고 controller summary가
  없다. QP terminal result와 evaluator artifact는 전혀 없다.
- Failed worker stdout에는 sanitized `MV0Error` type만 남았고 source line이 없어 기존
  artifact만으로 exact exception location을 확정할 수 없다.
- `sacct`상 Qwen MEMIT step MaxRSS는 `8,869,620 KiB`, request는 `65,000M`이며 log에
  OOM 표시는 없다. 따라서 host-memory OOM을 원인으로 단정할 근거가 없다.

## GH 추정

- Failure 시점은 Qwen MEMIT의 네 번째 native edit 내부 또는 그 결과 기록 경계다.
  현재 log만으로 proposal numeric failure와 artifact serialization failure를 구분할
  수 없다.
- Llama QP partial directory는 Llama native worker가 먼저 다음 branch에 진입했기
  때문에 생긴 것이며 QP scientific failure를 뜻하지 않는다.

## Recovery 변경 경계

- Scientific contract는 변경하지 않는다: 동일 case/order, 두 모델, MEMIT/Alpha
  history, QP objective, trust/retry/first-hit, precomputed covariance/projector,
  evaluator firewall, metric과 lenient gate를 유지한다.
- Controller는 exception text나 runtime values 대신 `source/function/line`만 남기는
  sanitized traceback을 기록한다.
- Parent는 model×family worker stdout/stderr를 job별 ignored local path로 분리한다.
- Failed `c0_v1` artifact는 삭제하거나 resume하지 않는다. Recovery는 전부
  `c0_v2` path와 marker를 사용한다.
- Recovery도 Llama/Qwen과 두 family를 동시에 시작하는 full 4-GPU pair다. Partial
  model/family replay는 금지한다.

## Agent 경계

- Terra Ultra RCA/QP/Slurm reviewer 세 개를 시도했으나 모두 runtime metadata를
  명시적으로 검증하지 못해 repo를 읽지 않고 종료했다. 독립 agent review로 세지
  않으며 GH가 최소 incident 판정만 수행했다.

## 제출 gate

- Targeted regression, shell syntax, `git diff --check`, session/resource/output collision,
  clean pushed main을 모두 통과해야 한다.
- 동일 failure가 반복되면 v2 artifact와 worker-isolated traceback으로 exact RCA를
  먼저 닫고 scientific rescue나 partial retry를 하지 않는다.

- 최종 판정: `PASS_TECHNICAL_RECOVERY_ONLY`
