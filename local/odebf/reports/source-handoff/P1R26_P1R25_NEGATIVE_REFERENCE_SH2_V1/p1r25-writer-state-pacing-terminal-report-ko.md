# P1R25 Writer-State Preservation Pacing — Atomic B10 terminal 보고서

작성일: 2026-08-12 (Asia/Seoul)  
instruction: `ODEEDIT-S05-P1R25-WRITER-STATE-PRESERVATION-PACING-V1`  
estimand: `ATOMIC`  
claim boundary: matched-first-order-strength writer-state preservation-aware target pacing, RS Paced-Soft, reused sealed B10  

## 결론

P1R25 production 두 셀은 모두 기술적으로 완결됐다. Llama와 Qwen 모두 K=8, tau=1, terminal action-freeze, 정확한 W0 pointer/byte restore를 통과했고 scientific-invalid, retry, backtracking, history, WA-0 production 영향은 0이었다.

Pacing은 실제로 작동했다. Llama는 4/8 step, Qwen은 7/8 step에서 `s<1`을 선택했다. 동시에 모든 16 step에서 `a(s)^T v = rho_full`을 수치 오차 범위 안에서 지켰으므로, preservation을 위해 first-order writer demand를 낮춘 결과는 아니다.

그러나 first-order 예측이 BF16 실제 W-only 진행으로 충분히 실현되지는 않았다. 8-step 누적 realization은 Llama 0.521556, Qwen 0.269493이고 두 모델 모두 k7에서 actual progress가 음수였다. 따라서 주된 terminal 분류는 두 모델 모두 `WRITE_UNDER_REALIZED`이다. Pacing DOF와 matched strength는 입증됐지만, 동일 sealed B10에서 endpoint preservation이 개선됐다는 증거는 없다. 특히 Qwen은 더 강한 pacing에도 cumulative Structural-P, capacity, functional-P가 크고 Official Native보다 Gen/Loc가 각각 2점 낮았다.

## 1. FACT — 실행 및 무결성

- terminal callback: Slurm `afterany:19063`, callback job `19084`, no-GPU/1 CPU/1 GiB, owner SH2.
- callback 이후 scheduler terminal 상태를 한 번만 조회했다.
- logical task `19063_0` (Llama; accounting raw job `19118`): terminal `COMPLETED`, exit `0:0`, elapsed `00:03:59`, server2, 1 GPU/8 CPU/65000 MiB.
- logical task `19063_1` (Qwen; accounting raw job `19063`): terminal `COMPLETED`, exit `0:0`, elapsed `00:04:48`, server2, 1 GPU/8 CPU/65000 MiB.
- callback 이후 model/GPU/Slurm follow-up action: 0.
- source head: `b4c1fed7f9be2d63f1a1cf016ca81156511cfdf5`
- source tree: `a8e53c3f49d18a505bd5f1325ec522d59d4c9d55`
- scientific Phase-A checkpoint: `daf669493dfb3866a33e489f1798fd127b3bfd72`
- numerical lock SHA-256: `e183de1fb0231113f404e30ed974f0933982d4c5980c84dc0442d72b6b601016`
- source manifest SHA-256: `f11bea5fb8a70bd9b1482ac9ed8ad348ebe3ace0267dbfc2cc1c7675efc9883f`
- request order SHA-256: `984fe6ecc419fd0f701dae6032f33556f65a94dbb2850b57196be56c353b015b`

Result integrity:

| 모델 | terminal SHA-256 | manifest SHA-256 | integrity |
|---|---|---|---|
| Llama3-8B-Instruct | `456ad5c186589303e2cd9ca6bc531cc704de9dbebe9e2684283722d0f74d85f1` | `8a312d7c7599031e7aaeedb77e5c18ac470654bbab6fedf0f62d98da638a307e` | terminal/action-freeze/W0/8 accepted-receipt chain PASS |
| Qwen2.5-7B-Instruct | `f01170724cdfd983db2b21bda5cb8ea9582e0b648404199576cf763d78b63abb` | `b3bab612fd034c5235bb04e2af2bec63c952fdcb9a8794ae5fee80a0342173ad` | terminal/action-freeze/W0/8 accepted-receipt chain PASS |

두 manifest의 terminal hash, 두 terminal의 action-freeze/W0 endpoint hash, 그리고 16개 accepted receipt의 identity hash가 실제 파일과 일치했다. 관련 파일은 모두 regular non-symlink, mode 0600이며 JSON parse가 통과했다.

계약 무결성:

- K=8, h=1/8, tau=1, accepted materialization 8회/모델.
- entry-relative cumulative BF16 reconstruction; incremental BF16 accumulation 0.
- field build 8회, frozen-initial-field reuse 0, static split 0.
- 각 모델에서 8개 state/target/key/slope/field identity가 모두 갱신됐다.
- candidate별 model F/B/materialization 0; accepted step당 실제 materialization 1회.
- retry/reject/backtracking, inner heldout evaluator, overlay F/B, functional inner probe, history/H/P1R20 influence 모두 0.
- W0 pointer identity와 parameter bytes restore PASS; persistent commit 0.
- production WA-0 count 0.

## 2. FACT — pacing 및 matched strength

| 모델 | selected s (k1→k8) | s<1 | mean s | max equality residual | coverage 범위 | capacity influence |
|---|---|---:|---:|---:|---:|---:|
| Llama | .125, 1, .875, .96875, 1, .84375, 1, 1 | 4/8 | .851563 | 2.22e-16 | [1−1.1e-16, 1] | 0/8 |
| Qwen | .203125, .625, .4375, .59375, .515625, .28125, .421875, 1 | 7/8 | .509766 | 1.78e-15 | [1, 1+2.2e-16] | 1/8 |

- 두 모델 모두 `NO_PACING_DOF`가 아니었다.
- S64 grid는 `{j/64 | j=0..64}`의 65점이며 exact 0과 1을 포함했다.
- `rho_full`, selected predicted progress, target `rho_write`가 전 16 step에서 일치했다.
- strength relaxation은 전부 0이었다.
- `C_ref`는 전 step finite/positive였고 capacity tie tolerance는 dimensionless `1e-8`이었다.
- capacity가 최종 후보 집합을 좁힌 것은 Qwen k1의 1회뿐이었다(P-tied 6 → capacity-tied 1).
- post-BF16 observed energy의 selection influence는 0이었다.

## 3. FACT — stepwise pacing, P, realization

표의 `actual`은 k1–k7에서는 다음 refreshed field의 W-only NLL delayed reuse, k8에서는 terminal objective로 완성한 값이다. `P_after`는 raw cumulative Structural-P이다.

### Llama

| k | s | predicted rho | actual W-only progress | actual/predicted | P_after | C_bar |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | .125000 | 4.830061 | 3.066269 | .634830 | .000166812 | 1.000000 |
| 2 | 1.000000 | 4.229612 | 1.618700 | .382706 | .001316342 | 1.000600 |
| 3 | .875000 | 2.635253 | 1.710374 | .649036 | .002150868 | 1.059005 |
| 4 | .968750 | 1.871172 | 1.182201 | .631797 | .002903885 | 1.019379 |
| 5 | 1.000000 | 1.018916 | .278213 | .273048 | .004038635 | 1.002635 |
| 6 | .843750 | .436701 | .198389 | .454291 | .004077579 | 1.179891 |
| 7 | 1.000000 | .337580 | −.021353 | −.063252 | .004819199 | 1.003964 |
| 8 | 1.000000 | .508711 | .243259 | .478188 | .005194318 | 1.003047 |

### Qwen

| k | s | predicted rho | actual W-only progress | actual/predicted | P_after | C_bar |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | .203125 | 10.771235 | 3.410022 | .316586 | .008634636 | .752343 |
| 2 | .625000 | 6.266634 | 2.416553 | .385622 | .098604954 | .781103 |
| 3 | .437500 | 3.779935 | 1.438547 | .380574 | .122022581 | 1.747444 |
| 4 | .593750 | 4.326468 | .754579 | .174410 | .190800715 | 1.166689 |
| 5 | .515625 | 2.937072 | .584154 | .198890 | .207361553 | 1.489173 |
| 6 | .281250 | 2.307750 | .382284 | .165652 | .193847656 | 2.804939 |
| 7 | .421875 | 1.299622 | −.057488 | −.044234 | .182354465 | 2.599913 |
| 8 | 1.000000 | 2.141969 | .188486 | .087997 | .191584639 | .833691 |

Aggregate:

| 모델 | Σ predicted | Σ actual | cumulative realization | negative actual steps | initial→terminal six-context W-only NLL |
|---|---:|---:|---:|---:|---:|
| Llama | 15.868006 | 8.276052 | .521556 | 1 | 8.365490 → .089438 |
| Qwen | 33.830686 | 9.117138 | .269493 | 1 | 9.309228 → .192090 |

Qwen의 P가 k6–k7에서 감소한 것은 cumulative quadratic form의 허용된 negative cross term과 일치한다. P algebra identity residual의 최대값은 Llama 4.34e-19, Qwen 2.78e-17이다.

P AUC는 receipt에 authoritative 적분 convention이 이름 붙어 있지 않다. 따라서 다음 두 convention을 명시해 함께 기록한다.

| 모델 | h·Σ P_after (right rule) | P0=0 trapezoidal tau-AUC | terminal P |
|---|---:|---:|---:|
| Llama | .003083455 | .002758810 | .005194318 |
| Qwen | .149401400 | .137427360 | .191584639 |

## 4. FACT — terminal endpoint와 Official Native

아래 Official Native는 같은 sealed B10/order/W0/evaluator를 사용한 pinned Official EasyEdit AlphaEdit reference다. P1R25와 Native의 알고리즘은 다르므로 이는 endpoint reference이며 pacing-only causal control은 아니다.

| 모델/방법 | Eff | Gen | Loc | Eff new NLL | Gen new NLL | Loc new NLL | Eff margin | Gen margin | Loc margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama P1R25 | 10/10 | 20/20 | 81/100 | .074291 | .856474 | 8.281406 | 11.297584 | 8.502901 | 3.962854 |
| Llama Official Native | 10/10 | 20/20 | 79/100 | .002545 | .750505 | 8.082656 | 12.628705 | 9.858870 | 3.757390 |
| Qwen P1R25 | 10/10 | 18/20 | 78/100 | .222406 | 2.752930 | 8.845625 | 11.183844 | 6.072852 | 3.663276 |
| Qwen Official Native | 10/10 | 20/20 | 80/100 | .009536 | 1.287744 | 8.831719 | 15.415464 | 11.134131 | 3.693335 |

Official Native evidence:

- Llama terminal SHA-256 `e5a97c50be31b3864b082ac4d335f3852328c5b9029eb08c708667b7ae6eb053`
- Qwen terminal SHA-256 `9e8bfe49d4eeac33625d9ab3d0fd492962c9095b524980f3aa56e10fa28e1fd8`
- shared evaluation case identity `04012bcc468cea7cf896a09ca29841146e452692e504016b3d3463526c6e4354`
- evaluator source `25c3f49039e7bacb58de6d9ab8139f6f24f21532434a08690e9ec71ea939e145`

동일 sealed B10에서 Llama는 Native보다 Loc count가 +2였지만 Eff/Gen continuous NLL과 margin은 열세였다. Qwen은 Native보다 Gen −2, Loc −2이고 continuous edit strength도 열세였다. 따라서 두 모델에 걸친 endpoint preservation 개선 신호는 성립하지 않는다.

Terminal controller-space realization:

| 모델 | terminal W-only target-new NLL | z-oracle NLL | W−z gap |
|---|---:|---:|---:|
| Llama | .089438 | .083827 | +.005611 |
| Qwen | .192090 | .214692 | −.022602 |

## 5. FACT — Structural/functional preservation과 capacity

| 모델 | terminal Structural-P | functional-P mean | raw max | smooth max | endpoint BF16 cumulative capacity |
|---|---:|---:|---:|---:|---:|
| Llama | .005194 | .008514 | .067314 | .044419 | 3.812283 |
| Qwen | .191585 | .032950 | .314105 | .291079 | 18.238179 |

Functional-P의 legacy `passed=false`는 관찰값이며 hard decision influence는 0이었다. P budget, veto, rollback, retry는 없었다. Historical H sample count와 decision influence는 0이었다.

## 6. FACT — compute 및 overhead

| 모델 | edit_core | scheduler elapsed | F | B/autograd | slope B | target B | capture F | logical groups | tokens processed/padded | materialize |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama | 181.636 s | 239 s | 180 | 125 | 40 | 85 | 45 | 36 | 27207/30460 | 8 |
| Qwen | 221.400 s | 288 s | 180 | 125 | 40 | 85 | 45 | 36 | 24687/30380 | 8 |

Recorded edit-core phase wall:

| 모델 | refresh | field+slope | target grad | materialize/write | terminal objective | initial capture | routing solve |
|---|---:|---:|---:|---:|---:|---:|---:|
| Llama | 1.941 | 41.702 | 7.712 | 10.862 | .286 | .267 | .000025 |
| Qwen | 5.257 | 54.904 | 7.854 | 14.146 | .233 | .132 | .000025 |

위 named phases의 합과 edit-core 사이에 Llama 약 118.865 s, Qwen 약 138.874 s의 미귀속 시간이 있다. receipt만으로 원인을 재구성하지 않는다.

Pacing 자체의 predeclared `<=1.2×` pure-edit overhead는 동일 server2 B1 paired control로 평가했다.

| 모델 | B1 s=1 control | B1 paced | ratio | gate |
|---|---:|---:|---:|---|
| Llama | 45.371 s | 54.412 s | 1.199267× | PASS |
| Qwen | 59.043 s | 51.552 s | .873122× | PASS |

B1은 one-request technical smoke이며 B10 endpoint estimand가 아니다. 반대로 GH가 제공한 broader P1R24 reused-stream B10×10 mean edit-core 93.299/107.501 s와 본 production 181.636/221.400 s의 단순 비는 1.947×/2.060×이지만, server/sample-stream 조건이 달라 pure pacing overhead로 해석할 수 없다.

Official Native edit-core는 Llama 31.908 s, Qwen 27.273 s이고 P1R25/Official 비는 5.692×/8.118×이다. Official Native는 K1 direct-z algorithm이므로 역시 pacing-only denominator가 아니다.

## 7. FACT — WA-0 분리

WA-0는 B1 paced sibling의 k={0,7}에서만 observation-only로 실행됐고 production에는 없었다. trajectory/pace/route/write/endpoint influence는 모두 0이며 별도 F/B/token/time ledger를 가졌다.

| 모델 | k | selected s | cosine(g_int,g_WA) | writer Jacobian gain | predicted/actual progress | WA-0 F/B | wall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Llama | 0 | .140625 | .533510 | 1.025815 | 4.081125 / 2.521991 | 1/1 | .080 s |
| Llama | 7 | 1 | .852456 | .072349 | .001070 / .000917 | 1/1 | .081 s |
| Qwen | 0 | .218750 | .261512 | 1.005137 | 4.704941 / 2.917109 | 1/1 | .076 s |
| Qwen | 7 | 1 | .446727 | .099234 | .011160 / .010099 | 1/1 | .073 s |

이 값은 intervention gradient와 differentiable writer-aware gradient가 같지 않을 수 있음을 보여주는 관찰 신호일 뿐이다. WA-1, explicit z-barrier, write-aware F_z 활성화 근거로 사용하지 않는다.

## 8. INFERENCE

1. **Controller contract는 성공했다.** S64 pacing과 layer realization의 joint selection은 실제로 DOF를 사용했고, first-order writer demand를 낮추지 않은 채 cumulative Structural-P를 1차 objective로 최적화했다.
2. **물리적 realization이 병목이다.** 예측 equality가 정밀함에도 실제 누적 진행은 약 52%/27%에 그쳤다. 이는 target pace 선택 자체보다 local linear prediction→BF16 write의 비선형/상태 의존 오차가 endpoint를 제한했음을 시사한다.
3. **Preservation 개선은 입증되지 않았다.** Llama Loc +2는 작은 단일-model 신호이며 continuous edit metric은 Native보다 약하다. Qwen은 Gen/Loc 모두 Native보다 2점 낮고 P/capacity도 훨씬 크다.
4. **Qwen에서 pacing은 더 강하게 개입했지만 더 안전해지지 않았다.** mean s=.510, 7/8 influence였음에도 terminal P=.192, capacity=18.24, realization=.269였다. 따라서 `s` 감소만으로 writer-state preservation이 보장되지는 않는다.
5. **용어 경계:** 가능한 주장은 “matched-first-order-strength joint target pacing and layer realization under cumulative Structural-P soft geometry”까지다. CBF, locality optimization, global optimum, explicit z-barrier, write-aware target flow, lifelong guarantee는 주장할 수 없다.

## 9. 분류

| 모델 | 기술 완료 | primary failure class | pacing DOF | preservation verdict | compute verdict |
|---|---|---|---|---|---|
| Llama | PASS | `WRITE_UNDER_REALIZED` | ACTIVE 4/8 | `PACING_PRESERVATION_SIGNAL` 미확립 | B1 paired overhead PASS; B10 phase attribution incomplete |
| Qwen | PASS | `WRITE_UNDER_REALIZED` | ACTIVE 7/8 | `PACING_PRESERVATION_SIGNAL` 미확립 | B1 paired overhead PASS; B10 phase attribution incomplete |

`S1_IDENTITY_FAILURE`, `REMAINING_STEP_OR_H_SCALING_FAILURE`, `MATCHED_STRENGTH_FAILURE`, `CUMULATIVE_P_LEDGER_FAILURE`, `NUMERICAL_FAIL`, catastrophic collapse는 발생하지 않았다.

## 10. NOT_RECORDED / 해석 제한

- exact same sealed-B10 P1R24 Fixed-Neutral/Fixed-Soft raw-free endpoint comparator는 현재 SH2 closure에 없다. 별도 trajectory를 재구성하거나 대체하지 않았다.
- broader P1R24 B10×10 결과는 GH가 제공한 read-only descriptive aggregate다: Llama 99/100,177/200,873/1000, realization .549, edit-core 93.299 s; Qwen 100/100,163/200,844/1000, realization .283, edit-core 107.501 s. 이것은 본 original sealed B10 pair와 동일 estimand가 아니며 paired causal comparison에 사용하지 않았다.
- authoritative P AUC convention은 receipt에 기록되지 않았다. right-rule과 P0=0 trapezoidal 값을 모두 명시했다.
- edit-core의 미귀속 118.865/138.874 s 원인은 NOT_RECORDED다.
- scheduler MaxRSS와 authoritative GPU utilization은 terminal receipt에 유효한 값으로 기록되지 않았다.
- 이 재사용된 sealed B10은 scientific promotion/fresh-sample 증거가 아니다.

## 최종 판정

`P1R25_ATOMIC_EXECUTION_PASS_WITH_WRITE_UNDER_REALIZATION; PACING_DOF_AND_MATCHED_STRENGTH_CONFIRMED; PRESERVATION_GAIN_NOT_ESTABLISHED; NO_PROMOTION`

후속 model/GPU/Slurm 작업은 제출하지 않았다.
