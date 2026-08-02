# Job 15842 adaptive-lineage recovery

- 날짜: 2026-08-02
- job: `15842` / `odeedit_capacity_history_pair_c1_v1`
- 상태: `FAILED 1:0`; scientific output 없음
- 후속 namespace: `odeedit_capacity_history_pair_c1_v2`

## 확인된 사실

- resource는 GPU `4`, CPU `32`, memory `260000M`으로 cap 안이었다.
- Llama/Qwen × MEMIT/Alpha-history native controller 네 개는 모두 4/4 terminal/pass였다.
- Llama MEMIT QP는 edit 1 feature/action/receipt를 commit하기 전에
  `FrozenTargetLineage.derive_adaptive_step` line 668에서 `ContractError`로 종료했다.
- 해당 QP run은 feature `0`, action `0`, receipt `0`, evaluator `0`이다.
- Pair wrapper는 첫 failure를 감지하고 다른 세 QP worker를 종료했다.
- Raw failed directories/logs/marker는 ignored `local/`에 보존하며 Git에 넣지 않는다.

## RCA

c1 controller는 최대 네 accepted round와 `capacity_round_4`를 사용하도록 바뀌었지만,
shared lineage module의 `_ADAPTIVE_STEP_LABELS`는 기존 c0의 세 round만 잠그고 있었다.
따라서 네 번째 accepted transition이 controller 계산을 끝내기 전에 lineage identity
gate에서 거부됐다. 이는 QP 수치나 모델 결과가 아니라 integration envelope mismatch다.

## 최소 수정

- `_ADAPTIVE_STEP_LABELS`: `range(1, 4)` → `range(1, 5)`
- adaptive envelope error text: three → four hops
- toy lineage unit test scale count: 3 → 4
- fresh c1_v2 run/job/output/marker identity

EasyEdit, precomputed covariance/projector/Wikipedia artifact, controller QP algebra, gate,
model별 policy는 변경하지 않는다.

## GH emergency/direct action 기록

사용자 지시의 빠른 Motivation 실험과 server1 SH 부재 때문에 GH가 기존 승인 범위에서
job `15842`를 직접 제출·초기 모니터링했다. 실행 명령은 tracked
`project/run_scripts/submit_session01_capacity_history_pair_server1.sh` 한 번이었다.
Failure 후 추가 Slurm 명령으로 state를 변경하지 않았고 wrapper의 fail-fast cancellation만
작동했다. 후속도 같은 cap과 tracked helper 한 번으로 제한한다.
