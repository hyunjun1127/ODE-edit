# P1R28 Stage-A TECH-R5 B1 기전 검증

## 판정

`STAGE_A_MECHANISM_PASS`.

TECH-R5의 두 B1 모델 작업(19177_0 Llama, 19177_1 Qwen)은 모두 종료 코드 0으로 완료했다. P1R28 계약의 Stage-A 기술·기전 게이트에서 좌표, 상태 전이, K8, W0 복원, 동일-샘플 P1R24 앵커 스케일 회복, zero-action 총체화에 실패한 항목은 없었다. 따라서 상위 계약에 따라 Stage B 단일 sealed-B10 게이트는 **기전상 진행 가능**하다. 이 보고서는 Stage-B 제출이나 효능 결론을 내리지 않는다.

## 범위와 계보

| 항목 | 값 |
| --- | --- |
| 작업 | 19177, `0-1%2`, 두 task `COMPLETED/0:0` |
| 과학 소스 HEAD | `8a8de0e7b4c1666a7bc22d02468410e97f14eeec` |
| P1R28 base | `ce8c6c36348752f1407f7d713d30e6b5c727379b` |
| 수치 잠금 SHA | `93dcbe85e5736f8116bd3c6b7d58f11e05afecf55c6c45a211a91223e5b75476` |
| 소스 매니페스트 SHA | `bc9ad7a79891ba5a03d89385278543fb24b1b023841462803ebdef138b2e9e83` |
| 요청 수 | alias별 B1 1개, 역사/순차 상태 0 |
| P1R24 A0 | Llama는 같은 job에서 exact CE8C6C3 재실행 1회, Qwen은 exact immutable CE8C6C3 영수증 재사용 |

이 검토는 raw-free terminal/manifest/action-freeze/accepted-step 영수증과 관련 P1R28 수치 잠금만 읽었다. 프롬프트·타깃·텐서·사설 로그를 읽거나 모델/평가/Slurm을 실행하지 않았다.

## 무결성 및 실행 완료

| Alias | terminal SHA-256 | manifest SHA-256 | action-freeze SHA-256 | W0 복원 | terminal/manifest 재해시 |
| --- | --- | --- | --- | --- | --- |
| Llama | `eec91ab3a4e8b4b271676a99537bbb29f88a7d7cc0903f72b2ef6331f5a98793` | `20aee3d0ef07518132eb9dc34c69b5f1c108a6a6a724a6c92bc3fbc1129c206e` | `c03a2c8fc36168156d6a83db3f67a18bff314a5d047d7e6a1d62ac29db60a0ad` | PASS | PASS |
| Qwen | `fb7bc875ff7c72b42502cec99d5e06c9a47138f3ce26c2dbd3238ebb9ab48cb3` | `0556a6fd60a454a7a14f02d333c58e97a3e31d2853568e6923ab0b1929036547` | `25cc74d2d69e9cf96c6dd464650e4c008770e8baa4838ac4eb2fe50007679944` | PASS | PASS |

양쪽 모두 action freeze가 held-out 접근보다 앞섰고, inner held-out access, persistent-history append, replay-H decision influence, sequential-controller influence는 모두 0이다. Terminal/manifest가 서로의 SHA를 정확히 참조한다.

## C1/C2 전이 기전

모든 네 corrected rollout(C1/C2 × 2 alias)은 `K=8`, `tau=1`, materialization 8회, 동적 field refresh 8회, static initial field 재사용 0으로 끝났다.

| Alias | Arm | JOINT_WRITE | SEMANTIC_NO_POSITIVE_DIRECTION | C2 λ backend calls | λ=1 write | tracking rectified | technical fail | retry/backtrack/reject |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Llama | C1 | 4 | 4 | 0 | 0 | 0 | 0 | 0 |
| Llama | C2 | 4 | 4 | 4 | 4 | 0 | 0 | 0 |
| Qwen | C1 | 4 | 4 | 0 | 0 | 0 | 0 | 0 |
| Qwen | C2 | 4 | 4 | 4 | 4 | 0 | 0 | 0 |

각 C2의 양의 semantic 구간 k1–k4에서 λ=1은 모두 feasible였다. 따라서 `TRACKING_CONFLICT_RECTIFIED`는 관측되지 않았다. 이는 conflict 분기가 필요 없었던 것이며, 그 분기의 실증은 `NOT_RECORDED`이다. 양의 semantic slope가 있는데 tracking 때문에 stall한 전이는 0이다.

k5–k8에서는 결합 target가 동결되어 semantic slope와 selected displacement가 정확히 0이었다. 이는 `SEMANTIC_NO_POSITIVE_DIRECTION`으로 총체화되었고, 각 step에서 W, controller z, physical terminal z의 before/after hash가 같았다. 이 경로는 technical failure나 tracking-induced stall이 아니다.

## 좌표 및 h 검증

모든 32 corrected accepted transition에서 다음이 동시에 성립했다.

- `(h * bar_a)^T v = bar_a^T (h * v)`의 ratio는 1, identity residual은 0.
- `theta = h v`, `h_application_count=1`, `second_h_division_count=0`.
- 금지 contraction `(h*bar_a)^T theta`의 실행 count는 0.
- per-layer full-residual 복제 count, 추가 model forward/backward/materialization count는 0.
- predicted `a_v^T v=rho`의 equality residual 최대값은 `8.88e-16` 이하이고, coverage는 수치 반올림을 제외하면 1이다.

이는 P1R27의 coordinate/scale 오류를 재도입하지 않았음을 뜻한다.

## P1R24 동일-샘플 앵커 스케일

첫 accepted step의 C1/C2 값은 각 alias의 paired P1R24 A0와 정확히 같다.

| Alias | Arm | first-step realized gain | A0 gain | energy | A0 energy | energy ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Llama | C1 | 0.9106226414 | 0.9106226414 | 0.0085576101 | 0.0085576101 | 1.0 |
| Llama | C2 | 0.9106226414 | 0.9106226414 | 0.0085576101 | 0.0085576101 | 1.0 |
| Qwen | C1 | 1.1855549875 | 1.1855549875 | 0.0546930964 | 0.0546930964 | 1.0 |
| Qwen | C2 | 1.1855549875 | 1.1855549875 | 0.0546930964 | 0.0546930964 | 1.0 |

양의 action k1–k4의 energy-ratio 범위는 Llama C1 `0.8668–1.0000`, Llama C2 `0.9381–1.0204`, Qwen C1 `0.8726–1.0000`, Qwen C2 `0.9159–1.0000`이다. `realized/intended gain > 9`인 transition은 0개다. 따라서 P1R27 음성 대조의 약 9배 first-step gain은 제거되었고, C1은 요구된 same-sample P1R24 strength scale로 복귀했다.

## B1 종료 지표 (관측 전용)

| Alias | Arm | z8 target-new NLL | terminal W-only target-new NLL | edit-core seconds |
| --- | --- | ---: | ---: | ---: |
| Llama | A0 | 0.005004 | 0.016095 | — |
| Llama | C1 | 0.004827 | 0.022450 | 123.80 |
| Llama | C2 | 0.004966 | 0.019487 | 138.99 |
| Qwen | A0 | 0.007715 | 0.009738 | — |
| Qwen | C1 | 0.007584 | 0.020280 | 124.38 |
| Qwen | C2 | 0.006796 | 0.010691 | 174.16 |

B1은 기술/기전 smoke이며 official held-out Eff/Gen/Loc gate가 아니다. 따라서 이 수치로 Stage-B의 효능 판정을 대체하지 않는다.

## FACT / INFERENCE / NOT_RECORDED

### FACT

- 19177의 Llama/Qwen task가 모두 exit 0으로 끝났고 terminal, manifest, action-freeze, W0 복원 연결을 독립 재해시했다.
- C1/C2의 32 transition 모두 coordinate/h gate를 통과했고, materialization 8회 및 retry/rescue/imputation 0을 만족했다.
- 첫 step의 C1/C2 energy와 gain은 같은 샘플의 P1R24 A0와 정확히 일치했다.
- k5 이후의 zero action은 all-zero semantic slope를 근거로 한 `SEMANTIC_NO_POSITIVE_DIRECTION`이고 W와 z를 함께 hold했다.

### INFERENCE

- TECH-R5는 이전 factor-capacity zero 처리 결함을 과학 규칙 변경 없이 totalize했다. 따라서 Stage-A strict mechanism gate는 PASS다.
- P1R28 parent contract에 의해 Stage B sealed-B10 gate를 진행할 조건이 충족되었다.

### NOT_RECORDED

- B1에서는 C2 tracking conflict가 발생하지 않아 `TRACKING_CONFLICT_RECTIFIED`의 실제 경로는 관측되지 않았다.
- B1에는 official terminal held-out efficacy/generalization/locality 평가가 없으므로 P1R24와의 효능 동등성·우월성은 판정하지 않는다.

## 경계

이 결과는 P1R28 Stage-A의 기전/무결성 판정만 제공한다. 과학적 promotion은 false이며, B10 결과 없이 C1/C2의 성능 또는 preservation claim을 하지 않는다. 이전 19162/19164 Stage-A 보고서와 모든 실패 root는 변경하지 않았다.
