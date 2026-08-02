# Job 15843 adaptive-scale tolerance recovery

- 날짜: 2026-08-02
- job: `15843` / `odeedit_capacity_history_pair_c1_v2`
- 상태: `FAILED 1:0`; scientific output 없음
- 후속 namespace: `odeedit_capacity_history_pair_c1_v3`

## 확인된 사실

- 네 native controller는 모두 4/4 terminal/pass였다.
- Llama MEMIT QP는 edit 1 feature/action/receipt commit 전 동일한
  `FrozenTargetLineage.derive_adaptive_step` gate에서 종료했다.
- c1_v2 source에는 `capacity_round_1..4` label envelope가 이미 반영돼 있었다.
- QP feature/action/receipt는 0개이고 evaluator는 시작하지 않았다. 다른 세 worker는
  pair fail-fast로 취소됐다.

## RCA

`derive_adaptive_step`의 남은 거부 조건은 label membership 또는
`0 < step_scale <= 0.25`다. c1_v2는 four-label membership을 unit/runtime source에
확인했으므로 exact upper-bound 비교가 원인이다. QP solution은 trust distance에
`1e-7` absolute numerical residual을 허용한다. Llama MEMIT edit 1의 native
C-distance `0.014959074...`에서는 이것이 scale ratio 약 `6.7e-6`까지 될 수 있는데,
lineage는 이를 zero-tolerance로 비교했다.

## 최소 수정

- adaptive lineage scale upper bound에 `1e-5` numeric tolerance 추가
- applied proposal/coefficients/descendant는 반올림·rescale하지 않고 exact hash로 bind
- unit: `capacity_round_4, scale=0.250005` 허용
- unit: `scale=0.25002`, `capacity_round_5` 거부
- fresh c1_v3 job/output/marker identity

이 tolerance는 QP solver가 이미 허용한 numerical envelope만 정렬한다. Progress target,
trust fraction, capacity term, model/family policy, EasyEdit/precomputed artifact는 바꾸지
않는다.
