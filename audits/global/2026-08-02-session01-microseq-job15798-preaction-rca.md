# Session 01 Motivation — microseq job 15798 pre-action RCA

- 시각: 2026-08-02 03:36:51--03:39:03 KST
- job: `15798` / `odeedit_microseq_pair_v1`
- 판정: **technical failure; scientific evidence 없음; exact repair rerun PASS**

## 네 범주

- proposal에서 온 내용: controller는 evaluation prompt/outcome을 보지 않은 채
  네 edit action을 모두 commit하고, 별도 evaluator만 terminal 이후 평가해야 한다.
- repo/protocol에서 확인한 사실: Llama native controller는 첫 edit의 action
  commitment 전 `_assert_outcome_free`에서 `ContractError`로 종료했다. actions 0,
  receipts 0, events 1, `outcome_fields_loaded=false`다. Qwen child는 pair fail-fast로
  종료했고 parent exit는 `1:0`이다.
- GH 추정: generic substring guard와 negative firewall attestation의 naming 충돌인
  단일 구현 오류이며 model·GPU·data·projector·scientific policy 문제는 아니다.
- 사용자 확인 필요: 없음. 사용자는 초기 pipeline의 즉시 실패 예방, 빠른 실험
  제출, 동일 model policy를 지시했고 이 repair는 그 범위 안의 fail-closed 수정이다.

## Root cause

`microseq_controller`가 outcome 부재를 증명하려고 feature/action에
`outcome_fields_loaded=False`와 `all_actions_before_outcomes=True`를 넣었다.
shared `_assert_outcome_free`는 exact payload field뿐 아니라 key 문자열에
`outcome`이 들어간 모든 항목을 거부했으므로, 실제 outcome 값이 없음에도 첫
attestation에서 실패했다.

## 최소 repair와 red gate

- 두 exact key만 올바른 builtin boolean일 때 허용한다.
- `outcome_fields_loaded=True/0`, `all_actions_before_outcomes=False/1`, 임의
  `outcomes_unseen_at_commit`, nested `progress`는 계속 `ContractError`다.
- model, case order, seed, layers, `K=4`, `D/4`, proposal/evaluator/metric,
  EasyEdit/cache와 scientific gate는 변경하지 않았다.
- targeted test: PASS
- `py_compile`: PASS
- full CPU regression: `Ran 290 tests`, `OK`

## 실패 artifact와 영향 범위

- failed raw/marker는 삭제하지 않고 ignored local failure archive로 이동해 보존한다.
- Slurm stdout/stderr는 job ID 파일명 그대로 보존한다.
- proposal artifact와 target artifact 각 1개는 incomplete technical provenance일
  뿐 결과 분석·pair gate·Motivation claim에 사용하지 않는다.
- Qwen은 scientific action을 commit하지 못했고 partial rescue하지 않는다.

## GH emergency 직접 명령 기록

- 사유: startup 즉시 실패를 확인해 GPU 낭비와 invalid partial continuation을 막기
  위한 running-job triage 및 exact repair.
- 명령 범위: `sacct`, 해당 job log/failed summary 읽기, relevant 세 함수/테스트
  점검, value-locked guard patch, targeted/compile/full CPU test, failed local
  artifact archive, clean main commit/push 후 동일 pair 1회 재제출.
- 영향 범위: ODE-Edit server1 job `15798`와 microseq local/artifact/code path만.
  EasyEdit 및 별도 task/job/repository는 변경·모니터링하지 않는다.
- 후속 보고: 본 audit, `messages/inbox/server1.md`, 재제출 job의 30분 monitor,
  model별 report와 pair postrun audit.

## 재제출 중단 조건

- failed run이 action/receipt 0 또는 evaluator-load false가 아니면 중단
- repair 외 scientific diff, dirty/unpushed main, session/resource mismatch면 중단
- 동일 contract/firewall failure 재발 시 추가 즉석 retry 금지

- 최종 판정: `PASS` — failed evidence 보존 및 clean pushed repair 뒤 exact same
  two-model pair의 1회 재제출만 허용
