# Adaptive-gate zero-outcome lineage-label repair audit

대상: original job `15730` → MEMIT/Alpha v2 pair

## 확인 사실

- repo/protocol에서 확인한 사실:
  original job `15730`은 `FAILED 1:0`, elapsed `00:26:53`이다. Llama 첫
  case에서 `FrozenTargetLineage.derive_quarter_step`의 label allowlist가
  `gated_fixed_step_2` 또는 `gated_refreshed_step_2`를 거부했다. 두 model의
  `outcomes.jsonl`은 모두 0 byte이고 feature/action/receipt/analysis도 0건이다.
  Qwen child는 pair fail-fast로 종료됐다. 따라서 scientific outcome이나
  gate 선택을 이용한 변경은 없다.
- proposal에서 온 내용:
  frozen target lineage는 trajectory ancestry와 action identity를 잠그기
  위한 technical 장치이며 path 이름 자체가 별도 scientific treatment는
  아니다.
- GH 추정:
  gate 선택 `fixed/refreshed`를 이미 허용된 quarter-step label
  `fixed_step_i/refreshed_step_i`에 매핑하면 ancestry/action hash는 그대로
  유지되며 새 treatment를 추가하지 않는다.
- 사용자 확인 필요:
  없음. 사용자는 빠른 제출과 완료까지 명시했고, repair는 outcome 전
  plumbing failure의 최소 범위다.

## 변경과 불변

- 변경:
  `gated_{selection}_step_i`를 existing allowlist의
  `{selection}_step_i`로 매핑하는 pure helper 1개
- 불변:
  model, cases, target, direct-z, layer, K, `D/4`, `D/64`, score 식, 2%
  margin, G/A/B/C path, NFE, seed, cache/projector, branch order, lenient gate
- model 공통성:
  Llama/Qwen exact 동일 helper/controller를 사용하며 alias branch 없음
- 보존:
  original v1 raw/log/marker는 삭제·수정하지 않고 v2 output/marker/job
  identity를 새로 사용

## 최소 regression/resource gate

- targeted label/gate/lineage/legacy regression: `19 tests`, `OK`
- 전체 Motivation regression: `191 tests`, `OK`, `10.796s`
- Python compile: runner/path 통과
- shell syntax: child/pair/helper 통과
- original outcome files: Llama/Qwen 모두 regular non-symlink, 0 byte
- original marker: numeric job `15730`; exact Slurm `FAILED`
- v2 helper는 위 조건, clean pushed main, session/cap, v2 output/marker absent를
  제출 직전에 다시 검사한다.
- Alpha v2는 MEMIT v2 exact `COMPLETED`와 양 summary `all_pass=true` 전에는
  계속 차단한다.

Terra Ultra 제한 reviewer는 runtime metadata를 확인할 수 없어 어떤 파일도
읽지 않고 즉시 종료했다. 해당 미확인 결과를 audit 근거로 사용하지 않았다.

- 최종 판정: `PASS` — zero-outcome lineage-label repair v2 pair에만 유효
