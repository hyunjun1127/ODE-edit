# P1R23 Compute-Aware Progress-Simplex Atomic B10 최종 보고서

## 범위와 결론

- 범위: 동일한 outcome-selected P1R23 B10 seal/order/W0/evaluator에서
  `{Llama,Qwen} × {BG,RS} × {Compute-Simplex-Neutral,Compute-Simplex-Soft}` 8개 trajectory.
- 8/8 trajectory가 K8, `tau=1`을 완료했다. early-stop threshold는 소스에 존재하지 않아
  `NOT_DEFINED`로 고정했고, 조기 종료 없이 K8을 실행했다.
- 4개 paired job 모두 scheduler `COMPLETED 0:0`; terminal, manifest, action-freeze,
  W0 byte/pointer restore를 독립 재해시해 PASS했다. scientific-invalid, retry,
  backtracking, reject, persistent commit/history, P1R20 access는 모두 0이다.
- Eff/Gen은 전 셀에서 10/10, 20/20을 유지했다. Locality는 79–82/100이며,
  exact-objective Progress-Simplex 대비 셀별 0–2 count 차이로 catastrophic collapse가 없다.
- Soft는 32/32 field에서 allocation decision influence가 있었고, 동일 상태 Neutral shadow보다
  structural P/H risk를 32/32 낮췄다. terminal functional-P도 4/4 pair에서 낮았다.
- online model compute는 trajectory당 130F/80B에서 26F/16B로 80% 감소했다.
  edit-core는 exact-objective 대비 6.8–27.8% 감소했다. 다만 repaired Optimized Native 대비
  primary `<=2.0×` timing 목표는 Qwen-BG Soft 한 셀만 만족하므로 timing 목표 자체는 미달이다.
- 따라서 Atomic B10 lenient package gate는 **PASS/PARETO_VALID**로 판정한다. 이는
  Historical 검증을 허용하는 후보 판정이며 Main Table 또는 scientific promotion이 아니다.

## 고정 identity

- instruction: `ODEEDIT-S05-P1R23-PROGRESS-SIMPLEX-BF-COMPUTE-A1`
- method: `P1R23_COMPUTE_AWARE_PROGRESS_SIMPLEX_STRUCTURAL_PH_V1`
- request order: `984fe6ecc419fd0f701dae6032f33556f65a94dbb2850b57196be56c353b015b`
- numerical lock SHA: `15d770bf63b428e1d8f95c5848d70b45842676d4f9d5b7b26e153c4f9296eb88`
- initial execution head (6 trajectories): `c9da7ebd0095fcba0a63cdf2a2e702ab16c2515e`
- Llama-BG certificate-only TECH-R3 execution head (2 trajectories):
  `f64d0c1221ac0eecc53a1fbb9685a9a23835af0e`
- TECH-R3 scientific child: `794885a48c09046b454b88eee7880e08c390902a`
- TECH-R3 source manifest root:
  `8ea9bb3e5b71501edb830ee5635ca8659cd020a763631f49b2e8e0f873df230b`

TECH-R3는 SLSQP `success/status/message`를 observation-only telemetry로 분리하고,
명시적 finite/simplex/nonnegative/q-equality/energy certificate만 유지한다. objective, q,
active set, energy trust, tolerance, sample, evaluator는 바뀌지 않았다.

## Endpoint FACT

NLL 표기는 `new NLL (margin=old-new)`이다.

| Model | Alloc | Arm | E/G/L | E NLL (margin) | G NLL (margin) | controller full-6 NLL | edit-core s |
|---|---|---|---:|---:|---:|---:|---:|
| Llama | BG | Neutral | 10/20/81 | 0.027397 (10.494478) | 0.833589 (8.663286) | 0.039194 | 79.697 |
| Llama | BG | Soft | 10/20/81 | 0.033040 (10.295085) | 0.841425 (8.616388) | 0.046465 | 88.437 |
| Llama | RS | Neutral | 10/20/82 | 0.018537 (16.250213) | 0.520277 (12.240660) | 0.051830 | 77.486 |
| Llama | RS | Soft | 10/20/81 | 0.020814 (16.147936) | 0.529063 (12.086562) | 0.054200 | 88.193 |
| Qwen | BG | Neutral | 10/20/79 | 0.071179 (12.722571) | 2.244556 (7.705444) | 0.066702 | 97.540 |
| Qwen | BG | Soft | 10/20/79 | 0.188727 (12.370648) | 2.187427 (7.754761) | 0.141790 | 82.589 |
| Qwen | RS | Neutral | 10/20/80 | 0.042779 (18.710346) | 1.413828 (9.343204) | 0.019084 | 111.653 |
| Qwen | RS | Soft | 10/20/79 | 0.053104 (17.678146) | 1.426690 (9.402998) | 0.029380 | 116.129 |

Locality NLL는 receipt의 preservation 정의(`new-old`)를 따른다. Soft-minus-Neutral
Locality count는 Llama-BG 0, Llama-RS -1, Qwen-BG 0, Qwen-RS -1이다. 따라서 Soft의
P/H 개선을 보편적 locality count 개선이라고 해석하지 않는다.

## Routing, P/H, capacity, realization

| Pair | active/DOF | Soft allocation changed | Soft structural risk Δ range | terminal functional-P N→S | capacity N→S | delayed realization ratio N / S |
|---|---:|---:|---:|---:|---:|---:|
| Llama-BG | 5 / 4 | 8/8 | -0.01998…-0.00368 | 0.006766→0.006144 | 3.0522→3.1267 | 0.135–1.090 / 0.106–1.084 |
| Llama-RS | 5 / 4 | 8/8 | -0.02212…-0.00270 | 0.011140→0.008989 | 3.3190→3.4109 | 0.492–1.082 / 0.481–1.085 |
| Qwen-BG | 5 / 4 (1 field: 4 / 3) | 8/8 | -0.20108…-0.10841 | 0.078103→0.068051 | 17.3733→11.8777 | 0.170–1.126 / -0.032–1.148 |
| Qwen-RS | 5 / 4 | 8/8 | -0.21179…-0.13220 | 0.066694→0.048910 | 19.2858→11.8467 | 0.419–0.983 / 0.318–0.986 |

- Soft q-equality residual 최대 `4.44e-16`; selected energy ratio는 항상 frozen Neutral-relative
  bound 안이었다.
- 64개 field 중 63개는 active layer 5/DOF4였고, Qwen-BG Neutral k5 한 field만
  active layer 4/DOF3였다. 모든 Soft field에는 실제 allocation 자유도가 있었다.
- Soft의 `v>1` 계수는 Llama-BG 18, Llama-RS 17, Qwen-BG 22, Qwen-RS 24개로,
  제거된 per-layer cap이 숨은 clipping으로 재도입되지 않았다.
- Qwen에서 actual progress가 비양수였던 observation은 4건이다: BG-Soft k7
  `-0.00856`, BG-Soft k8 `-0.00376`, RS-Neutral k8 `-0.00629`, RS-Soft k8
  `-0.01349`. Llama의 actual progress는 전 step 양수였다. Qwen의 네 trajectory도
  K8 endpoint E/G가 모두 10/20이고 누적 edit가 유지되어 전체 trajectory의 완전 붕괴는
  아니며, 이 late-step 부호는 observation-only로 보존한다.
- Llama에서는 Soft capacity가 소폭 증가했고, Qwen에서는 크게 감소했다. 반면 functional-P는
  네 pair 모두 개선됐다. 보존 신호는 metric별로 혼합되어 있다.
- Atomic에서는 Historical H가 empty/inactive이며 decision influence 0이다. Historical 효과는
  이 실행에서 `NOT_RECORDED`다.

## Compute FACT

| Cell | Compute edit-core s | Exact-objective s | 감소율 | Native timing-only ratio | field+slope s | target s | materialize s | router s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| L-BG N | 79.697 | 97.430 | 18.2% | 2.140× | 44.302 | 5.964 | 13.263 | 0.051 |
| L-BG S | 88.437 | 101.023 | 12.5% | 2.374× | 49.668 | 6.384 | 13.272 | 0.046 |
| L-RS N | 77.486 | 103.041 | 24.8% | 2.080× | 42.514 | 5.404 | 13.436 | 0.046 |
| L-RS S | 88.193 | 102.685 | 14.1% | 2.368× | 55.299 | 5.204 | 13.259 | 0.043 |
| Q-BG N | 97.540 | 111.782 | 12.7% | 2.113× | 60.844 | 4.742 | 15.346 | 0.095 |
| Q-BG S | 82.589 | 114.369 | 27.8% | 1.789× | 46.145 | 4.676 | 15.675 | 0.099 |
| Q-RS N | 111.653 | 127.135 | 12.2% | 2.419× | 73.419 | 4.527 | 15.308 | 0.104 |
| Q-RS S | 116.129 | 124.647 | 6.8% | 2.516× | 76.325 | 5.068 | 15.578 | 0.115 |

- Compute-A1: 각 trajectory 26 model forwards, 16 backwards/autograd. Llama 20,239,
  Qwen 18,429 processed tokens.
- exact-objective reference: 각 trajectory 130 forwards, 80 backwards; Llama 25,642,
  Qwen 23,302 tokens. functional-preservation basis 35.0–40.7초가 online compute에 포함됐다.
- Compute-A1 functional routing probe는 step당 0F/0B이며 basis bookkeeping은
  0.00023–0.00039초다. terminal objective는 1F, 약 0.21–0.22초; official endpoint
  evaluation은 state당 약 0.99–1.08초다. inner-step heldout/stepwise evaluator는 0이다.
- scheduler total wall/host MaxRSS: L-BG 247초/6.88GiB, L-RS 248초/7.04GiB,
  Q-BG 276초/10.04GiB, Q-RS 325초/10.38GiB.
- artifact/model-load 시간은 독립 phase로 직렬화되지 않아 `NOT_RECORDED`; total wall에서
  역산해 정밀 값으로 주장하지 않는다.
- Native ratio는 timing-only다. repaired Optimized Native의 alias별 edit-core reference
  (Llama 37.249초, Qwen 46.164초)를 사용했으며 formal rho가 아니다.

## Exact-objective Progress-Simplex 대비

- exact-objective SH2 input package:
  - report SHA `c508242eb798fc0db4a3ac7c84e1e94eb7f656c5ab9e2dd4e7b23edbdf329c29`
  - stepwise SHA `64fa5e0205d640d83afd4e448c75dbf24b71fb230a644fe98697950f50aed212`
  - receipt SHA `d6671f72fd3770271d6de7e228de928adc882c71447fc38ebf04aa7c7385437e`
- exact-objective endpoint는 L-BG 10/20/81 양 arm, L-RS 10/20/81 양 arm,
  Q-BG 10/20/80→10/20/81, Q-RS 10/20/81 양 arm이었다.
- Compute-A1은 E/G count를 모두 유지했고 locality가 셀별 0–2 낮았다. strict hash parity가
  아니라 predeclared lenient scientific parity로 해석한다.
- 두 method 모두 32/32 Soft field에서 allocation이 바뀌고 same-state structural risk가
  감소했다. Compute-A1은 functional signal을 router에서 제거했지만 terminal functional-P는
  네 pair 모두 Neutral보다 낮았다.
- 전체 method-package 차이이므로 runtime/quality 변화가 functional probe 제거 하나의
  고립 인과효과라고 주장하지 않는다.

## TECH-R3 certificate receipt

Llama-BG Neutral trajectory의 accepted-k5(0-based refreshed field k4) Soft-shadow Stage1에서:

- primary SLSQP: `success=false`, status 8, finite PASS, simplex residual 0,
  progress equality residual 0, energy violation `1.694422380182914e-12`,
  energy ratio `1.0000000100213646`, first-false `neutral_relative_global_energy`.
- deterministic convex feasibility restoration: L∞ 이동 `6.767084115288924e-11`,
  energy violation 0, ratio `1.0000000100079292`, 모든 explicit certificate PASS.
- 이 복원은 동일 objective/active set/q/energy bound/tolerance 안에서 certified Neutral seed와의
  convex segment를 사용한다. fallback count는 전체 64 accepted fields 중 1이며 scientific
  retry/backtracking이 아니다. 원 primary certificate 전체가 receipt에 중첩 보존됐다.

## Lenient gate 판정

| Gate | 판정 |
|---|---|
| K8/dynamic refresh/q/simplex/energy/transaction/W0 | PASS |
| E/G Native/P1R23 근접 유지 | PASS (8/8 모두 10/20) |
| catastrophic under-edit 없음 | PASS |
| locality systematic collapse 없음 | PASS (79–82, exact 대비 0–2 차이) |
| Soft allocation 실제 변화와 영향 | PASS (32/32) |
| same-q structural P/H 또는 capacity 개선 | PASS (structural 32/32; Qwen capacity 개선) |
| actual/predicted progress 완전 붕괴 없음 | PASS |
| runtime 유의 감소 | PASS vs exact-objective (6.8–27.8%, 5× F/B 절감) |
| repaired Native primary <=2× | PARTIAL/FAIL (1/8만 만족) |

종합: **Atomic B10 lenient package gate PASS / Pareto-valid candidate**. Historical 단계는
별도 namespace에서 predeclared fixed-rank H sketch와 checkpoint evaluator만 사용해 진행할 수
있다. 이 판정은 B100, Main Table, promotion을 허가하지 않는다.

## 해시 및 claim boundary

| Pair | terminal SHA | manifest SHA | action-freeze SHA | W0 endpoint SHA |
|---|---|---|---|---|
| L-BG TECH-R3 | `6b6db420…467b` | `2b18b3dc…4a1` | `d7a3e5f4…13a8` | `160e8380…423` |
| L-RS | `d853cf9f…d308` | `98dd2312…8d53` | `70032081…34c` | `07f6402e…0c9` |
| Q-BG | `1f5c7cb2…9d22` | `b848e6da…42fa` | `b7d8c145…8553` | `c4a64dbd…00e2` |
| Q-RS | `f7d397f6…324c` | `4c5b8268…7982` | `eb846dbf…9441` | `3b58a460…cfd` |

모든 full hash는 해당 immutable JSON에 보존돼 있다. 본 보고서는 raw prompt/target/tensor를
포함하지 않는다. 동일 outcome-selected atomic B10의 mechanistic/descriptive evidence이며,
`scientific_promotion=false`다.

## NOT_RECORDED

- 별도 artifact/model-load phase 시간
- 공식 Native의 비중첩 full phase counters와 formal rho
- Atomic에서 비어 있지 않은 Historical H 효과
- fresh unseen seal replication, uncertainty, Main Table 승격 근거
