# Capacity/history matched-quarter c1 preflight

- 날짜: 2026-08-02
- 방법명: **ODE-Edit**
- 대상 job: `odeedit_capacity_history_pair_c1_v2`
- 상태: pre-push 필수 검증 통과

## 감사 범위

필수 항목만 검사한다: c0 under-edit RCA, c1 terminal/progress semantics, model-common
identity, evaluation firewall, EasyEdit/precomputed artifact read-only boundary, Slurm cap,
clean pushed Git, fresh output namespace.

## c0 RCA

- code: `MAX_ACCEPTED_ROUNDS=3`, `INITIAL_TRUST_FRACTION=0.25`,
  `REQUESTED_PROGRESS_FRACTION=0.75`, exact-top1 terminal.
- raw accepted path/native-distance mean:
  - Llama MEMIT `0.454052`
  - Llama Alpha-history `0.457809`
  - Qwen MEMIT `0.399978`
  - Qwen Alpha-history `0.409667`
- 판정: c0 cost 감소와 efficacy harm은 matched-strength comparison이 아니므로 method
  kill evidence로 사용하지 않는다. Raw artifact와 수치는 보존한다.

## c1 확인 결과

- [x] `K=4`, initial cap `D/4`, fixed 75% request 제거
- [x] ordered-native authorized rewrite utility를 controller reference로 사용
- [x] exact-top1 diagnostic-only, native-reference terminal
- [x] retry failure fail-closed; unmatched early stop 금지
- [x] Llama/Qwen 및 MEMIT/Alpha-history 동일 policy/run contract
- [x] capacity + frozen-target-lineage unit suite `34/34` 통과
- [x] shell syntax, session boundary, resource-cap, diff whitespace 검사 통과
- [x] fresh c1 output/marker와 no active same-name job 확인
- [x] server1 cap GPU `4`, memory `260000M` 허용 확인
- [x] secret pattern 및 5 MiB 초과 tracked-candidate 없음

Commit 전 검사에서 `check-session-boundary.sh`는 repository/session/model을
`hyunjun1127/ODE-edit` / `019fb1ea-03cb-7c20-bb3b-eba5f8d6f5f2` / `Sol Ultra`로
확인했다. Submission helper는 clean `main == origin/main`, staged access, exact source
identity를 다시 요구하므로 commit/push 뒤 한 번 더 실행한다.

## Agent boundary

Terra Ultra runtime metadata를 확인할 수 없었던 세 격리 agent는 지정 파일을 읽지 않고
`BLOCK` 종료했다. 독립 review로 세지 않으며 GH가 필수 검증을 직접 수행한다.

## Job 15842 startup recovery

최초 c1_v1 job `15842`는 네 native controller를 모두 `4/4 all_pass`로 끝낸 뒤,
Llama MEMIT QP 첫 edit의 네 번째 adaptive hop을 lineage에 bind하는 과정에서
`ContractError`로 fail-fast했다. Controller는 `K=4`로 수정됐지만 shared
`_ADAPTIVE_STEP_LABELS`가 여전히 `capacity_round_1..3`만 허용한 것이 원인이다.
QP feature/action/receipt는 0개였고 evaluator는 시작하지 않았으므로 scientific
evidence로 사용하지 않는다. 다른 세 worker는 pair wrapper가 취소했다.

수정은 lineage label envelope를 `capacity_round_1..4`로 확장하고 toy lineage test도
실제 네 scale을 bind하도록 바꾼 것뿐이다. Fresh run/job namespace를 c1_v2로 올려
failed c1_v1 artifact를 덮어쓰지 않는다.

- 최종 판정: `PASS` — clean pushed main에서 c1 4-GPU pair 1회 제출에만 유효
