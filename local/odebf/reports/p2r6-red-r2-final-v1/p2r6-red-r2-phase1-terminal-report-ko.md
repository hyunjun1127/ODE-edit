# P2R6 RED-R2 Phase 1 종료 보고서

## 1. 판정

- 상태: `PHASE1_TECHNICAL_INVALID_STRICT_STOP`
- 분류: `TECHNICAL_INTEGRATION_INCOMPLETE`
- 과학 실패: `false`
- 완전한 두 모델 Phase 1 행렬: `false`
- Phase 1 과학 게이트: `NOT_RUN_INCOMPLETE_TECHNICAL_INVALID_MATRIX`
- 보편 controller 선택: `NOT_RUN`
- Phase 1→2 action freeze: `NOT_CREATED`
- Phase 2: `CLOSED`
- Historical/Sequential/B10x10: `NOT_AUTHORIZED`

이 보고서는 유효한 네 endpoint의 사실과 기술 실패 사실만 기록한다. AR/AS 후보의 완전한 두 모델 행렬이 없으므로 후보 간 과학 비교, controller 선택, 과학적 PASS/FAIL은 수행하지 않았다.

## 2. 실행 및 분모

| 항목 | 값 |
|---|---:|
| Slurm array job | `20107`, `0-1%2` |
| Llama scheduler child | `20108`, `FAILED`, exit `1:0` |
| Qwen scheduler task | `20107`, `CANCELLED_BY_OWNER_AFTER_PAIRED_STRICT_FAILURE` |
| 계획 jobs / endpoint / request endpoint | 2 / 8 / 80 |
| 유효 endpoint / request endpoint | 4 / 40 |
| 기술 무효 endpoint | 1 |
| 취소 불완전 endpoint | 1 |
| 미시작 endpoint | 2 |
| 종료 뒤 재시도·재제출·repair | 0 |
| 종료 뒤 model action | 0 |
| 종료 뒤 P2R6 active/pending GPU | 0 |

Endpoint별 상태:

| 모델 | case | A0-CAP | AETA-CAP | AR-CAP | AS-CAP |
|---|---:|---|---|---|---|
| Llama | 5 | VALID | VALID | TECHNICAL_INVALID | NOT_STARTED |
| Qwen | 1 | VALID | VALID | CANCELLED_INCOMPLETE | NOT_STARTED |

Llama AR prefix에는 target microstep receipt 6개와 writer step receipt 1개가 있으나 terminal·manifest·action-freeze가 없으므로 endpoint로 집계하지 않았다. Qwen AR prefix에는 target microstep receipt 3개, writer step receipt 0개가 있고 endpoint가 없다. 두 prefix의 성능 수치는 `NOT_RECORDED`로 유지한다.

## 3. 첫 번째 거짓 gate

| 필드 | 값 |
|---|---|
| 모델 / arm | Llama / AR-CAP |
| 단계 | `AR_CAPACITY_IN_SEMANTIC_REGION` |
| 예외 | `P2R6RoutingTechnicalError` |
| exception message SHA256 | `803c8e321259229f3ea3a0aa757cadd0ee485f03b2181ab55ae871e9e7a28e22` |
| failure identity | `bdbb49de86b35698d137adbba33073f4680b44c3e6dc4a45d829264d84f4893f` |
| backend | `DENSE_FP64_PRIMAL_DUAL_PREDICTOR_CORRECTOR_ORIGINAL_ALPHA` |
| solver status | `MAX_ITERATIONS` |
| constraint envelope | 0 |
| external tolerance | `1e-8` |
| `r_pri` | `4.10481946452304e-09` |
| `r_dual` | `0` |
| `r_stat` | `2.252975825678405e-06` |
| `r_comp` | `165.7961319173045` |
| objective 최소 eigenvalue / PSD | `0.011158391701233217` / `true` |
| certificate | FAIL |

`r_stat`와 `r_comp`가 고정 tolerance를 초과했다. selected AR-CAP 실행은 endpoint 전에 fail-close되었고 retry는 0이었다. 이후 Llama AS는 시작하지 않았고, Qwen 진행 중 AR은 소유자 취소되어 완전한 paired matrix를 만들지 않았다.

## 4. 유효 endpoint 무결성

네 유효 endpoint 모두 다음을 통과했다.

- action freeze가 evaluator보다 먼저 생성됨: 4/4 PASS
- heldout controller access: 0/4 endpoint에서 모두 0
- target microstep: 각 24
- writer transition/materialization: 각 8/8
- retry: 각 0
- shadow technical invalid: 각 0
- W0 pointer+byte restore: 4/4 PASS
- manifest의 terminal SHA 및 action-freeze SHA와 실제 파일 재해시: 4/4 PASS

실패한 Llama AR의 W0 pointer+byte restore도 PASS다. 취소된 Qwen AR 이후 W0 상태는 `NOT_RECORDED`; 마지막 확인 지점은 AETA 완료 뒤 `POST_AETA_STAGE_006_PASS`다.

## 5. 유효 endpoint 성능 수치

### 5.1 W-only terminal

| 모델 | arm | Eff | Gen | Loc | Eff new-NLL | Eff margin | Gen new-NLL | Gen margin | Loc new-NLL | Loc margin |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama | A0-CAP | 9/10 | 19/20 | 80/100 | 1.212656 | 13.274844 | 1.335829 | 8.839952 | 9.174336 | 3.614893 |
| Llama | AETA-CAP | 10/10 | 19/20 | 80/100 | 0.541419 | 12.386706 | 1.652226 | 7.883711 | 9.236016 | 3.625469 |
| Qwen | A0-CAP | 10/10 | 19/20 | 92/100 | 0.001252 | 18.229998 | 2.505222 | 9.495559 | 12.517813 | 7.027715 |
| Qwen | AETA-CAP | 10/10 | 19/20 | 92/100 | 0.003078 | 17.353172 | 2.539826 | 8.950799 | 12.518750 | 7.037051 |

### 5.2 z-inject terminal 및 full-six W objective

| 모델 | arm | z Eff | z Gen | z Eff new-NLL | z Eff margin | z Gen new-NLL | z Gen margin | W full-six target-new NLL |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Llama | A0-CAP | 9/10 | 19/20 | 1.300152 | 13.156098 | 1.482367 | 8.559039 | 1.161929 |
| Llama | AETA-CAP | 10/10 | 19/20 | 0.650168 | 13.637332 | 1.083964 | 9.321505 | 0.856820 |
| Qwen | A0-CAP | 10/10 | 19/20 | 0.001153 | 18.292597 | 2.546587 | 9.495601 | 0.000692 |
| Qwen | AETA-CAP | 10/10 | 19/20 | 0.001357 | 18.398643 | 2.284327 | 9.831298 | 0.000724 |

AR/AS terminal Eff/Gen/Loc/NLL/margin은 모두 `NOT_RECORDED`다. 위 A0/AETA 수치는 완전하지 않은 AR/AS 후보 행렬의 과학 판정에 사용하지 않았다.

## 6. routing, P, capacity

| 모델 | arm | 최종 route status | mass min / max | active mass / semantic | rank / nullity | semantic-region max violation | predicted capacity | actual BF16 capacity | cumulative Structural-P | final negative actual count |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama | A0-CAP | `P2R6_A0_CAP_CERTIFIED` | 0.009021 / 1.000000 | 9 / 10 | 50 / 0 | 1.1570e-19 | 62.241856 | 62.325160 | 0.079516 | 0 |
| Llama | AETA-CAP | `P2R6_AETA_CAP_CERTIFIED` | 6.2478e-10 / 1.000000 | 1 / 10 | 28 / 22 | 1.5829e-17 | 43.374220 | 43.455239 | 0.054665 | 0 |
| Qwen | A0-CAP | `P2R6_A0_CAP_CERTIFIED` | 1.000000 / 1.000000 | 10 / 10 | 50 / 0 | 1.3553e-20 | 432.869169 | 433.102459 | 4.021854 | 1 |
| Qwen | AETA-CAP | `P2R6_AETA_CAP_CERTIFIED` | 0.280838 / 1.000000 | 2 / 10 | 28 / 22 | 6.9389e-18 | 123.157059 | 123.372480 | 1.301196 | 0 |

네 유효 endpoint의 final route에서 Neutral fallback=0, preservation strength attenuation=0이었다. Allocation SHA는 machine table에 기록했다. AR/AS routing/P/capacity endpoint는 `NOT_RECORDED`다.

## 7. compute ledger

네 유효 endpoint 각각의 집계는 동일했다.

| 항목 | endpoint당 값 |
|---|---:|
| completed K | 8 |
| target F/B | 125 / 120 |
| KL F/B | 125 / 120 |
| physical capture F | 45 |
| physical response F / batched VJP | 40 / 40 |
| post-write objective F | 40 |
| routing QP solve / certificate | 48 / 48 |
| candidate F / materialization | 0 / 0 |
| writer materialization | 8 |

| 모델 | arm | target 초 | writer 초 | z evaluator 초 | W evaluator 초 | total 초 |
|---|---|---:|---:|---:|---:|---:|
| Llama | A0-CAP | 22.486 | 88.301 | 1.224 | 1.209 | 125.743 |
| Llama | AETA-CAP | 22.740 | 91.260 | 1.227 | 1.214 | 129.536 |
| Qwen | A0-CAP | 18.750 | 118.620 | 1.118 | 1.111 | 153.478 |
| Qwen | AETA-CAP | 18.329 | 118.128 | 1.124 | 1.113 | 153.326 |

금지 영향 counter는 네 유효 endpoint 모두 candidate F/materialization, clamp-OFF access, explicit lag, remaining-horizon division, semantic debt, retry, backtracking, functional-P veto, hard-P budget, Historical, Sequential, shadow model F/B/materialization이 0이었다.

## 8. 식별자 및 증거

- 최종 source HEAD/tree: `1246e5047bdac2e59d5cc0642ebd8d7104c6d066` / `ed7c236066afe1c8140ac8b306d5be1785cc08c5`
- source parent: `18a7dedbe6007ad297c9966d20f4facfc380d251`
- source manifest: `project/run_scripts/ode_bf/locks/source_manifest_s05_p2r6_red_r2_final.json`; SHA `9f0cb669310f5a197537bbe0543701d4306ed7b53d5210a48e1d33db97dbf2c0`; root `1a6ae84883ffcfff1578caf6297ad1d8b261ec0f2e75f65b4f0a5c7260c1d958`; entries 28
- numerical lock: `project/run_scripts/ode_bf/locks/numerical_lock_s05_p2r6_red_r2_final.json`; SHA `b0cc41404cbee0fc401b879bd150c1551703fe53332b5a4e22aea9e2bfd8d2b0`; root `2e0389cd3a352ece3f23ff81aa8cc446e6811db762760f332f9073b0d66e439a`
- submission receipt: `local/odebf/state/p2r6-red-r2-final-v1/s05-p2r6-red-r2-final-phase1-none-1246e5047bda-v1.submission-receipt.json`; SHA `d20fc0a7c500d058ca64ddb12f6af6e7df798cd2aa6c19361dd1ccb81de38876`
- terminal technical-invalid receipt: `local/odebf/state/p2r6-red-r2-final-v1/phase1-technical-invalid-terminal-job20107.json`; SHA `6089e7e8cc9f520a75054ad1d79829ced4747372aa6fb00319a9fd262f3308d1`; root `2c8c59105dff56cc64077bc405a3e4f6a2093590d28c18e19a617be6b6468a93`

네 유효 endpoint의 terminal/manifest/action-freeze SHA, endpoint identity, allocation SHA 및 세부 수치는 동봉한 8행 machine table에 기록했다.

## 9. 최종 gate 상태

- Technical validity: `FAIL`
- Phase 1 matrix completeness: `FAIL`
- Scientific gate: `NOT_RUN`
- `SCIENTIFIC_FAIL_SIGNAL`: `NOT_RUN`
- `INCONCLUSIVE_NO_UNIVERSAL_CONTROLLER`: `NOT_RUN`
- `MECHANISM_NO_SIGNAL`: `NOT_RUN`
- controller selection: `NOT_RUN`
- Phase 2 release: `CLOSED`
- scientific promotion: `false`

추가 GPU 실행, evaluator 재실행, retry, tuning, endpoint imputation은 수행하지 않았다.
