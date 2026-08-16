# P1R23 Progress-Simplex Sequential-NoH 경계 보고서

## 결론

`ODEEDIT-S05-P1R23-PROGRESS-SIMPLEX-SEQUENTIAL-NOH-V1`의 8개 배열 task는 모두 scheduler terminal에 도달했지만, 유효한 T10 과학 terminal은 0/8이다. 모든 cell이 동일한 exact Progress-Simplex Soft Stage1 certificate 경계(`progress_simplex_routing._solve_soft`, line 504)에서 `FAIL_CLOSED_NO_RETRY`로 종료됐다. 따라서 partial round endpoint는 최종 결과로 수용하지 않고, Neutral–Soft, Atomic, Full-6 Structural Historical, AlphaEdit 비교도 수행하지 않는다.

상태는 `HOLD_NUMERICAL_METHOD_DIRECTION`이다. tolerance, objective, solver, active set을 변경하거나 partial prefix를 재사용·재개하지 않았다.

## 고정 identity와 제출

- scientific basis: `a343d1f6967ef37763009b92d227ade85cd93de0`
- execution checkpoint: `7c3e1b4b5c6049d73ed21c61f3428e486c355244`
- tree: `3728cb98ffc53084cb94c60f487eefc6b2608c6d`
- source manifest SHA/root: `645bce5a21d3458661b14924e9ce3ff1ca32e3aa59fe77904a9b2392ef555081` / `1a2577399437fcfbc22c5c2d0d688fbc9fa26488ccdad248dd401faa6528c664`
- numerical lock root: `34aab65e8574f85dec3caa4d7a99fddacd4149026f1237477f74d48a8ad67a42`
- dry-plan SHA: `98a2cb40a431fe855c0bbfd46ad9cd62d9436c86702fe5eb61808123484578b7`
- Slurm array: `18822`, `0-7%4`, devbox, task당 1 GPU / 8 CPU / 65000 MiB / 23:59
- intent SHA: `6b834e278480cd93ec906e0d259e8ee90d6e34562e56839060d94ed1cc02224c`
- submission receipt SHA: `e421fa6d113ab5744d6342fb41a4274ad5899d4f5b8d3bb2a0a49f2e877445c1`
- essential gates: focused 6/6, compile, bash, session, artifact, namespace, held inspection PASS

## 8-cell terminal 경계

`완료 round`는 성공적으로 K8을 마치고 정확히 한 번 commit 및 cumulative evaluation까지 끝낸 outer round 수다. `accepted k`는 실패 전까지 직렬화된 inner Euler transition 수다.

| task | model | allocation | arm | 완료 round | accepted k | 마지막 stage | exception message SHA prefix |
|---:|---|---|---|---:|---:|---|---|
| 0 | Llama | BG | Neutral | 6 | 48 | post_sequential_round_6 | `c44c806c` |
| 1 | Llama | BG | Soft | 0 | 5 | post_model_context_teacher | `642215ac` |
| 2 | Llama | RS | Neutral | 0 | 4 | post_model_context_teacher | `a3933d61` |
| 3 | Llama | RS | Soft | 7 | 58 | post_sequential_round_7 | `e4885474` |
| 4 | Qwen | BG | Neutral | 6 | 51 | post_sequential_round_6 | `b60c2e1b` |
| 5 | Qwen | BG | Soft | 3 | 30 | post_sequential_round_3 | `c89c1560` |
| 6 | Qwen | RS | Neutral | 0 | 3 | post_model_context_teacher | `13db0b8f` |
| 7 | Qwen | RS | Soft | 9 | 72 | post_sequential_round_9 | `0006f238` |

공통 stack의 마지막 frame은 8/8 모두 `progress_simplex_routing.py::_solve_soft`, line 504다. 이 경계는 launcher, path, serializer, result persistence 결함이 아니며, exact full-six live routing problem에 대한 Stage1 numerical certificate fail-close다.

## History-OFF 및 완료 prefix 감사

완료된 prefix 전체를 raw-free receipt만으로 검사했다.

- accepted transition: 271개
- H-OFF receipt PASS: 271/271
- router-visible history item: 항상 0
- raw historical replay: 항상 0
- projected-key/fixed-rank sketch construction: 항상 0
- functional-H/structural-H status: 항상 `INACTIVE_BY_HISTORY_MODE_OFF`
- functional-H/structural-H decision influence: 항상 0
- 성공 round commit: 31개, 각 `commit_count=1`, `post_commit_verified=true`, rollback 0
- history append/count: 항상 0
- cumulative B10 evaluations: 121개; 각 cell에서 `R(R+1)/2`와 exact 일치
- future-batch/controller evaluator access: 기록상 0

단, inherited outer receipt의 `current_batch_enters_history_after_endpoint_commit=true` 표기는 History-OFF ledger(`append_call_count=0`, history count 0)와 의미상 충돌하는 legacy label이다. 실제 method input과 state는 위의 zero-count receipts로 판정했으며, 이 label은 과학적 H 활동 증거로 사용하지 않는다.

## Terminal 무결성

- scheduler: 8/8 `FAILED 1:0`
- full T10 action freeze: 0/8
- terminal manifest: 0/8
- exact terminal W0 pointer/byte restore receipt: 0/8
- 최종 10 commits / 55 evaluations: 0/8
- partial prefix 재사용·resume·retry·resubmit: 0
- 새로운 모델/GPU/Slurm job: 없음

프로세스 종료만으로 W0 복원을 추론하지 않는다. T10 terminal receipt가 없으므로 모든 최종 Eff/Gen/Loc, continuous NLL/margin, P/H/capacity, Atomic/Structural-Historical/AlphaEdit 대비는 `NOT_RECORDED_AS_VALID_TERMINAL`이다.

## FACT / INFERENCE / NOT_RECORDED

### FACT

1. H-OFF firewall은 실패 전 모든 271개 accepted step과 31개 committed round에서 작동했다.
2. exact full-six Sequential-NoH 경로는 8/8에서 동일 Stage1 certificate code boundary를 만났다.
3. 실패 시점은 cell별로 round0부터 round10 진입 전까지 달랐으나, 실패 source와 certificate phase는 동일하다.
4. 어떤 cell도 T10 terminal contract를 충족하지 않았다.

### INFERENCE

이 실행은 “Historical-H가 없는 sequential 편집 성능”을 평가하지 못했다. 관측된 것은 exact full-six Progress-Simplex Soft Stage1 solver/certificate 경로가 누적 또는 일부 초기 state에서 total하지 않다는 구현/수치 경계다. 이를 H의 효과나 Neutral/Soft 성능 차이로 해석할 수 없다.

### NOT_RECORDED

- 유효한 T10 cumulative Eff/Gen/Loc 및 NLL/margin
- 55-evaluation 완주 trajectory
- terminal structural/functional P, capacity, layer allocation
- final action-freeze/manifest/W0 restore
- Sequential-NoH 대 Atomic/Full-6 Structural Historical/AlphaEdit의 과학적 대비

## Claim boundary

재사용된 outcome-selected stream의 mechanistic/debug execution이며 `scientific_promotion=false`다. 후속 실행에는 explicit numerical-method authority가 필요하다. retention이나 preservation, Historical-H 유무, 보편적 router 성능에 관한 과학적 결론을 내리지 않는다.
