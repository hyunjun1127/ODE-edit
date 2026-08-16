# P1R28 Stage-A B1 기구 점검 보고서

작성 시각: 2026-08-13 Asia/Seoul  
범위: P1R28 RS/Neutral B1 Stage-A의 raw-free receipt, 실패 receipt, 허용된 코드·수치 lock 및 P1R24/P1R27 immutable reference만 읽었다. 모델·평가·Slurm·과학 소스·결과 root에는 쓰지 않았다.

## 판정

**TECHNICAL_FAIL — Stage-A strict HOLD.** TECH-R1의 두 alias 모두 C1의 k1–k4 수락 뒤 다음 dynamic-field capacity 검증에서 동일하게 fail-closed 되었다. 따라서 C2, Stage-A 종단, Stage-B B10 및 어떤 효능/보존 과학 판정도 성립하지 않는다.

## 계보와 범위

| 항목 | 값 |
| --- | --- |
| 지시 | `ODEEDIT-S05-P1R28-P1R24-ANCHORED-CORRECTED-COUPLING-KILL-TEST-V1` |
| P1R24 과학 base | `ce8c6c36348752f1407f7d713d30e6b5c727379b` |
| TECH-R1 source head | `6d4e8f46334822d832877bd317a0f849a384ea98` |
| TECH-R1 Slurm | `19164`, array `0-1%2`, 2 GPU 이하 |
| 수치 lock SHA-256 | `93dcbe85e5736f8116bd3c6b7d58f11e05afecf55c6c45a211a91223e5b75476` |
| sealed B10 identity | seal `3d38b76…ed92f628`, order `984fe6…353b015b` |
| Stage-B | **실행 안 됨** |

초기 attempt `19162`는 두 array task 모두 local session-boundary config 부재로 모델 load 전에 중단됐다. stderr의 동일 문구는 `missing local session boundary config`; 모델·과학 action 및 result root는 이 attempt에서 수립되지 않았다. TECH-R1은 그 순수 세션 바인딩 실패를 고친 별도 namespace다.

## FACT — 유효한 C1 수락 receipt

두 alias 모두 C1(`d = Δz`, `lambda=0`)에서 k1–k4까지 4개씩, 총 8개의 accepted receipt가 있다. 각 receipt는 다음을 동시에 기록한다.

- `theta=h*v`, `h=0.125`, `h_application_count=1`, `second_h_division_count=0`.
- 좌표 contraction ratio는 **8/8 정확히 1**, identity residual은 **8/8 0**, 금지된 `(h*bar_a)^T theta` count는 0이다.
- 현재 velocity trust diagonal과 RoutingProblem diagonal의 최대 차이는 0, radius-squared residual 최대는 `2.22e-16` 이하이며 per-layer upper-cap influence는 0이다.
- 각 수락 step은 materialization 1회·5 weight이고, retry/backtracking, inner heldout, persistent history, overlay F/B는 모두 0이다.
- 각 C1 coupling은 shared full-six VJP 1회이고, lambda 탐색 때문에 추가된 model forward/backward/materialization은 모두 0이다.

| Alias | k | `rho` | `r_max` | BF16 step energy | activation gain | predicted equality residual |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Llama | 1 | 4.081125 | 21.195277 | 0.00855761 | 0.910623 | 0 |
| Llama | 2 | 3.404358 | 17.096500 | 0.01813867 | 1.291171 | 0 |
| Llama | 3 | 4.290337 | 25.408050 | 0.01681355 | 1.031854 | 0 |
| Llama | 4 | 0.386701 | 2.558865 | 0.01498517 | 0.848343 | 0 |
| Qwen | 1 | 4.704941 | 21.228891 | 0.05469310 | 1.185555 | 0 |
| Qwen | 2 | 4.283864 | 18.526611 | 0.08525150 | 1.738598 | 0 |
| Qwen | 3 | 7.083539 | 46.174912 | 0.05492537 | 0.996632 | 8.88e-16 |
| Qwen | 4 | 0.597934 | 2.088528 | 0.15058754 | 1.601029 | 0 |

P1R24 같은 B1 anchor와의 k1 비교도 수립됐다. Llama는 같은 job에서 A0를 재실행했고, target loss, `rho`, velocity 및 materialization identity가 C1과 정확히 같다. Qwen은 exact immutable `ce8c6c3` A0를 사용했고, target loss·`rho`·materialization identity 및 energy가 같으며 velocity의 최대 표기 차이는 FP64 `5.55e-17`이다. 두 alias에서 k1 C1/P1R24 energy ratio와 activation-gain ratio는 정확히 **1.0**이다. 따라서 이 k1 receipt 범위에서는 P1R27의 약 9배 gain failure가 재현되지 않았다.

## TECHNICAL_FAIL — 공통 종료

두 TECH-R1 결과 root의 `failure.json`은 완전히 같은 SHA-256 `dfd08a3bc4604f449cf26ab063b2fb815af8f1f47c0763c34630ecb44507b3c4`이고, 예외 메시지 hash는 `a9a327b881649c37aceba8d46ad75fa74de44498dc73f57710c9f70090373394`다. 마지막 완료 단계는 `post_model_context_teacher`; 분류는 `ODEBFContractError`, status는 `FAIL_CLOSED_NO_RETRY`이다.

허용된 stack은 C1 k4 이후 `build_scalable_dynamic_field → build_p1_dynamic_field → _validate_dynamic_factor_capacity`의 `P1 dynamic arm has nonpositive capacity`에서 끝난다. raw-free failure receipt는 capacity scalar가 정확히 0인지, 음수인지, 비유한인지 직렬화하지 않는다. 따라서 값의 원인 분류는 **NOT_RECORDED**이며, 이를 바탕으로 수치/과학적 해석을 만들 수 없다.

## INFERENCE

수락된 8개 C1 step의 좌표·h·trust·strength certificate는 일관되고 k1 anchor scale도 회복했지만, 공통 capacity totality가 다음 field를 구성하지 못하게 했으므로 Stage-A의 필수 `K8/action-freeze/W0 restore` gate는 통과하지 않았다. 이는 C1/C2의 효능 또는 lag tracking의 과학적 반증이 아니라, fail-closed 기술 경계다.

## SCIENTIFIC_FAIL

기록된 과학적 실패 판정은 없다. C2는 시작되지 않았고, Stage-B B10과 terminal external evaluator도 실행되지 않았다.

## NOT_RECORDED

- capacity validator에 전달된 정확 scalar와 0/비유한 구분;
- C1 k5–k8, C2 전 step, lambda trajectory와 tracking-conflict rectification;
- terminal action-freeze, terminal W0 pointer/byte restore, external Eff/Gen/Loc 및 NLL;
- P1R28 Stage-B B10 결과.

## HOLD 경계

이 보고서는 repair, 재제출 또는 과학적 결정을 권한하지 않는다. strict Stage-A 경계에 따라 P1R28은 **HOLD**이며, 이 receipt 집합만으로 Stage-B를 열 수 없다. P1R27은 immutable C0 negative reference로만 남고, 그 보고서는 변경하지 않았다.

## 근거 artifact

- TECH-R1 submission receipt: `local/odebf/state/p1r28-corrected-coupling/s05-p1r28-smoke-6d4e8f463348-tech-r1.submission-receipt.json` (`fda4c8ea…b290f90f`)
- attempt 19162 receipt: `local/odebf/state/p1r28-corrected-coupling/s05-p1r28-smoke-5457e3a7d0db-v1.submission-receipt.json` (`a4544c57…f849e9c4`)
- failure receipt (both aliases): `local/odebf/results/s05-p1r28-*-b1-rs-c1-c2-pair-v1/failure.json` (`dfd08a3…4507b3c4`)
- representative C1 k1 receipts: Llama `66c74d56…595ff00d`, Qwen `70f4ca37…04b1e3b`

