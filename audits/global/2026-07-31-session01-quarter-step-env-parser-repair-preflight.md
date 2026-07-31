# Quarter-step pre-run env-parser repair audit

범위: job `15700`의 scientific runner 진입 전 wrapper failure와 단일 retry

## 확인된 사실

- repo/protocol:
  - job `15700`은 `2 GPU / 130000M`을 할당받았으나 `00:00:03` 후
    `FAILED 1:0`으로 종료됐다.
  - child step `15700.1`은 model load, output directory 생성,
    feature/action/outcome/receipt 이전에 `environment file contains an
    unsafe assignment`로 exit `2`했다.
  - sibling은 pair fail-fast에 따라 종료됐다.
  - 두 canonical qstep run directory는 생성되지 않았다.
- 원인:
  `servers/local/session-boundary.env`의 canonical Sol Ultra 경계 두 줄은
  안전한 single-quoted 값 `'Sol Ultra'`를 사용하지만 새 child wrapper가
  unquoted token regex만 허용했다.
- GH 추정:
  없음. local stderr와 env line/parser 조건이 exact하게 일치한다.
- 사용자 확인 필요:
  없음. scientific commitment 전의 bounded wrapper repair이며 사용자의
  빠른 동시 제출 지시 범위 안이다.

## 최소 repair

- 일반 env 값은 기존 strict unquoted regex를 유지한다.
- 예외는 exact 두 assignment만 허용한다.
  - `ODEEDIT_REQUIRED_CODEX_MODEL_PROFILE='Sol Ultra'`
  - `ODEEDIT_CONFIRMED_CODEX_MODEL_PROFILE='Sol Ultra'`
- 두 값을 allowlist에 포함하고 load 후 각각 exact `Sol Ultra`인지 다시
  검사한다.
- arbitrary quote, command substitution, shell expansion, 다른 profile은
  여전히 거부한다.
- scientific runner, case, model, K, 거리, probe, metric, seed, threshold,
  outcome/analysis contract는 변경하지 않았다.

## 재검증

- `bash -n`: child/submit helper 통과
- dirty-tree bounded parser preflight:
  exact fake Slurm identity로 child를 실행했을 때 env parser를 통과하고
  의도된 clean-Git gate에서 silent exit `2`; unsafe-assignment stderr 없음
- original job state: exact `FAILED`, exit `1:0`, elapsed `00:00:03`
- outputs: Llama/Qwen canonical run directory 모두 없음
- retry helper:
  original marker의 numeric job ID와 Slurm `FAILED` 상태를 확인하고,
  repair audit exact verdict, clean pushed main, no output, no active job,
  session/resource cap을 다시 통과해야 새 `retry1` marker를 만든다.

residual blocker 없음. 실패 raw는 log/state만 있고 scientific artifact는
없으므로 archive 이동이나 삭제를 하지 않는다.

- 최종 판정: `PASS` — quarter-step pre-run env-parser repair retry 한 건에만 유효
