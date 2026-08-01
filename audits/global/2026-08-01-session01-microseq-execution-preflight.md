# Session 01 Motivation — microseq execution preflight

- 날짜: 2026-08-01
- 작성: GH 최소 red preflight
- 대상: `odeedit_microseq_pair_v1` 1회
- claim boundary: 4-edit Motivation proxy; lifelong·method superiority claim 금지

## 구분

- proposal에서 온 내용: cumulative C-load와 current-state direction refresh가
  short sequential preservation을 바꿀 가능성
- repo/protocol에서 확인한 사실: 양 model에 동일한
  `always-refresh / K=4 / D/4 / layers 4--8 / seed 37` contract를 적용하며,
  Alpha pair technical-valid 후에만 제출한다.
- GH 추정: exact 4-edit chain은 장기 보존 증거가 아니지만 공통 방향의 작은
  retention/KL/capacity signal을 선별하는 데 충분하다.
- 사용자 확인 필요: 없음

## 필수 kill-test

- action leakage: native와 ODE controller가 각각 네 action을 모두 commit하고
  종료한 뒤, evaluator가 양 branch의 summary·8 receipt·proposal hash를 먼저
  검증해야 evaluation row를 decode한다. 코드 순서 단위검사 통과.
- state/replay: fresh W0, proposal entry/descendant state ID, tensor/file/direction
  hash, receipt durability와 controller Git identity를 fail-closed로 확인한다.
- same method: model alias별 case/policy/threshold 분기 없이 exact constants와
  evaluator hash가 pair fixed contract에 포함된다.
- metric: `KL(W0||Wt)`, target-true NLL delta, prior utility, exact normalized
  `tr(Delta C Delta.T)/tr(W0 C W0.T)`, Frobenius를 scalar로만 기록한다. update가
  없는 layer는 capacity 0으로 처리한다.
- cache/artifact: pinned Wikipedia covariance만 read-only load한다. raw prompt,
  logits, token IDs, factors와 weights는 Git에 기록하지 않는다.
- resource/failure: 두 model을 동시에 1 GPU/65000M씩 실행하고 parent는
  2 GPU/130000M/12h다. 한 child가 nonzero면 sibling을 fail-fast 종료한다.
- boundary: 이 GH 범위 밖의 direct-z task job·파일·결과는 조회·수정·조정하지
  않는다. launcher는 제출 직전 aggregate GPU cap만 검사한다.

## 검증 증거

- full CPU regression: `279 tests OK`
- microseq evaluator/controller/artifact/analysis: `py_compile PASS`
- child/pair/submission wrappers: `bash -n PASS`
- Terra Ultra 독립 review: runtime metadata에서 `gpt-5.6-terra`와 `ultra`를
  검증할 수 없어 agent가 파일을 열지 않고 즉시 종료함. 이 제한 때문에 GH가
  필수 kill-test만 직접 확인했으며, scientific 결과 report에서는 동일 조건의
  Terra Ultra agent를 다시 시도한다.
- GPU smoke: 아직 미실행. Alpha technical-valid 전 제출 금지와 controller/
  evaluator fail-closed contract로 첫 이상에서 scientific interpretation 없이
  job을 실패시킨다.

- 최종 판정: `PASS` — Alpha technical-valid 이후 locked four-edit pair 1회에만 유효
