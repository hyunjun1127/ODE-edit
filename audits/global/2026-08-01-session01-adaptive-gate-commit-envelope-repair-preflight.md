# Adaptive-gate zero-outcome commit-envelope repair audit

대상: v2 job `15731` → MEMIT/Alpha v3 pair

## 확인 사실

- repo/protocol에서 확인한 사실:
  v2 job `15731`은 `FAILED 1:0`, elapsed `00:37:49`다. v1의 lineage label
  지점은 통과했고, 첫 case action commit에서 generic outcome firewall이
  action key `outcomes_unseen_at_commit` 자체의 문자열을 거부했다. 양 model
  `outcomes.jsonl`은 다시 0 byte이며 feature/action/receipt/analysis도 0건이다.
  실패 line 뒤 코드를 결과와 무관하게 검토하니 reusable qstep commit
  helper가 branch order와 receipt field를 qstep에 hardcode하고 있어 adaptive
  7-arm envelope와 맞지 않는 latent plumbing mismatch도 있었다.
- proposal에서 온 내용:
  feature/action/path를 outcome 전에 durability commit해야 하며, 어떤
  evaluation outcome도 controller에 들어가면 안 된다.
- GH 추정:
  forbidden key를 `evaluation_unseen_at_commit`으로 바꾸고 commit helper의
  expected branch/event/receipt schema를 명시 인자로 받되 qstep default를
  유지하면 firewall을 완화하지 않고 adaptive envelope만 정확히 표현한다.
- 사용자 확인 필요:
  없음. model별 method 차이는 없고 두 model exact 동일 commit/controller를
  사용한다.

## 변경과 불변

- 변경:
  - forbidden action key를 outcome 문자열이 없는 이름으로 교체
  - action에 이미 고정된 per-hop C energy를 명시
  - reusable `_commit_action`이 expected branch order, stream event 이름,
    receipt schema를 인자로 받도록 확장; legacy qstep default는 그대로
  - adaptive call은 exact 7-arm order와 별도 receipt schema를 전달
- 불변:
  cases/model/target/direct-z/layer/K/hop/probe/score/margin/G-A-B-C/seed/NFE,
  EasyEdit/cache/projector, lenient common-model gate 모두 변경 없음
- 보존:
  v1/v2 raw/log/marker를 read-only 보존하고 v3 identity만 새로 사용

## 최소 regression/resource gate

- synthetic adaptive feature/action/receipt commit test가 exact 7-arm order,
  outcome-free key, adaptive receipt schema와 file SHA-256을 검증
- targeted adaptive/legacy qstep regression: `13 tests`, `OK`
- 전체 Motivation regression: `192 tests`, `OK`, `10.956s`
- Python compile, shell syntax, `git diff --check`: 통과
- v1/v2 Slurm state는 모두 exact `FAILED`, 네 outcome file은 모두 regular
  non-symlink 0 byte
- v3 helper가 위 조건, clean pushed main, session/cap, v3 output/marker absent를
  제출 직전에 재검사
- Alpha v3는 MEMIT v3 exact `COMPLETED`와 양 summary `all_pass=true` 전 차단

Terra Ultra 제한 reviewer는 runtime metadata를 확인할 수 없어 파일을 읽지
않고 즉시 종료했다. 미확인 agent 결과는 audit 근거로 쓰지 않았다.

- 최종 판정: `PASS` — zero-outcome adaptive commit-envelope repair v3 pair에만 유효
