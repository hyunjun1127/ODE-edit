# P1R52 Frozen-π Sequential Quota Writer — Atomic B10×10 최종 분석

## 한눈에 보는 결과

이 보고서는 Llama3-8B-Instruct의 동일 sealed 10×B10 스트림에서 `J0`, `SV`, `FPIQ`를 비교한다. J0는 10/10 endpoint, SV는 8/10 endpoint와 case03(k4)·case06(k7) last-valid prefix, FPIQ는 9/10 endpoint와 case07(k2) last-valid prefix를 남겼다. 미완료 endpoint는 어떤 표에도 보간하지 않았다.

| Arm | attempts | endpoints | EFF(W) | GEN(W) | GEN-strict(W) | LOC(W) | W rewrite NLL | W rephrase NLL | full-six W−z gap | coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| J0 | 10 | 10 | 100/100 (1.0000) | 181/200 (0.9050) | 84/100 (0.8400) | 871/1000 (0.8710) | 0.057125 | 2.291020 | 0.037804 | 0.736622 |
| SV | 10 | 8 | 80/80 (1.0000) | 146/160 (0.9125) | 69/80 (0.8625) | 705/800 (0.8812) | 0.092756 | 2.258845 | 0.060584 | 0.647912 |
| FPIQ | 10 | 9 | 90/90 (1.0000) | 162/180 (0.9000) | 75/90 (0.8333) | 781/900 (0.8678) | 0.083025 | 2.486128 | -0.001018 | 0.962554 |

해석의 핵심은 두 가지다. 첫째, endpoint가 존재하는 case에서 FPIQ는 J0보다 writer coverage를 높이고 full-six W−z gap을 줄였지만, absolute W rewrite/rephrase NLL과 terminal full-six W NLL은 평균적으로 개선하지 못했다. 둘째, gap 축소는 accepted-z endpoint 자체의 악화와 마지막 layer velocity 집중·더 큰 update energy를 동반했다. FPIQ 1개와 SV 2개의 typed incomplete도 남아 B100/Historical 승격 안정성은 확보하지 못했다.

## 실험 정의와 단일 writer 변경

- `J0`: 기존 P1R52 joint block-Jacobi writer. Entry Soft router가 정한 5-layer velocity를 한 번에 물리화한다.
- `SV`: layer 4→8 virtual BF16 prefix에서 residual/key/q/current slope를 갱신하지만 entry velocity `v⁰`는 고정한다.
- `FPIQ`: entry `π⁰`만 고정하고 각 prefix에서 `α_rem=clip(L_l−L_z,0,α*)`, `barπ_l=π⁰_l/Σ_{j≥l}π⁰_j`, `s_l=α_rem·barπ_l`, `v_l=s_l/a_l^cur`를 적용한다.
- 공통 좌표는 `R_l=(z*−y_l)/h`, `θ_l=h·v_l`이며 π와 h는 각각 정확히 한 번만 적용한다.
- target/RSA/R42SafeKDC/origin clamp/selection/teacher/K8/h=1/8/finite demand/entry Soft router는 고정했다. Structural-H와 history는 OFF다.

## 기계적·entry identity

- W0에서 세 arm의 k0 target numeric vectors, finite `α*`, entry `π⁰`, entry velocity는 10/10 case에서 max-abs 0으로 일치했다. 정규화된 identity receipt: `067abc9197336e2119aafdec74e885511e215fee8d932ac2d725a959ca4a8ce6`.
- SV/FPIQ layer4 capture=0, added backward=0, FPIQ `v4/v4_entry` residual max=0.000000000000.
- Sequential available prefixes의 full residual identity max=0.000000000000, quota identity max=0.000000000000.
- π application=[1], physical h application=[1], second h=[0].
- virtual-prefix live parameter mutation count sum=0; final virtual BF16=post-commit BF16 PASS 149/149.
- retry/backtracking/P-H hard gate/cap/contraction/veto/debt/request×layer router/heldout decision influence는 모두 0. W0 restore는 attempts 30/30 PASS다.

## endpoint 절대값: accepted-z와 physical W 분리

| Arm | z EFF | W EFF | z GEN | W GEN | z GEN-strict | W GEN-strict | z rewrite NLL | W rewrite NLL | gap | z rephrase NLL | W rephrase NLL | gap | z→W Eff fail | z→W Gen-strict fail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| J0 | 100/100 (1.0000) | 100/100 (1.0000) | 183/200 (0.9150) | 181/200 (0.9050) | 85/100 (0.8500) | 84/100 (0.8400) | 0.026695 | 0.057125 | 0.030430 | 2.136656 | 2.291020 | 0.154365 | 0 | 1 |
| SV | 80/80 (1.0000) | 80/80 (1.0000) | 149/160 (0.9313) | 146/160 (0.9125) | 71/80 (0.8875) | 69/80 (0.8625) | 0.042631 | 0.092756 | 0.050125 | 2.032275 | 2.258845 | 0.226569 | 0 | 2 |
| FPIQ | 90/90 (1.0000) | 90/90 (1.0000) | 163/180 (0.9056) | 162/180 (0.9000) | 76/90 (0.8444) | 75/90 (0.8333) | 0.078542 | 0.083025 | 0.004483 | 2.471746 | 2.486128 | 0.014382 | 0 | 1 |

`terminal_z8_oracle.target_new_nll`과 `terminal_w8_full_six_target_new_nll`은 full-six target-objective panel이다. 위 rewrite/rephrase NLL은 terminal heldout prompt panel이므로 서로 대체하거나 같은 의미로 부르지 않았다.

## exact matched case의 산술 차이

| 비교 | matched cases | ΔW rewrite NLL | ΔW rephrase NLL | Δfull-six W−z gap | Δcoverage | ΔGEN correct | ΔGEN-strict requests | ΔLOC correct | energy ratio(arm/base) | capacity ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SV-J0 | 8 | 0.031551 | -0.000288 | 0.017866 | -0.085337 | 0.125000 | 0.125000 | 0.500000 | 0.861446 | 0.986515 |
| FPIQ-J0 | 9 | 0.025564 | 0.076623 | -0.039433 | 0.226099 | 0.111111 | 0.111111 | 0.333333 | 1.204663 | 0.968362 |
| FPIQ-SV | 7 | 0.006297 | 0.071842 | -0.059687 | 0.313744 | 0.000000 | 0.000000 | -0.285714 | 1.415291 | 0.990013 |

모든 delta는 동일 case endpoint가 양쪽에 존재할 때만 계산했다. 따라서 J0↔SV 8 case, J0↔FPIQ 9 case, SV↔FPIQ 7 case다. 성공 endpoint만으로 전체 10-case를 대표하지 않으며, typed incomplete denominator를 위와 별도로 유지한다.

### case별 endpoint 절대값

| Arm | case | status | W rewrite NLL | W rephrase NLL | z full-six NLL | W full-six NLL | W−z gap | coverage | W GEN | W GEN-strict | LOC |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| J0 | 01 | COMPLETE | 0.023461 | 2.385339 | 0.055925 | 0.060898 | 0.004974 | 0.768878 | 19/20 | 9/10 | 92/100 |
| J0 | 02 | COMPLETE | 0.031012 | 2.575562 | 0.029736 | 0.040365 | 0.010629 | 0.746906 | 16/20 | 6/10 | 86/100 |
| J0 | 03 | COMPLETE | 0.054102 | 3.256244 | 0.065110 | 0.083812 | 0.018701 | 0.773918 | 18/20 | 8/10 | 87/100 |
| J0 | 04 | COMPLETE | 0.023285 | 3.450674 | 0.012988 | 0.029654 | 0.016666 | 0.741026 | 16/20 | 8/10 | 92/100 |
| J0 | 05 | COMPLETE | 0.130017 | 2.330286 | 0.042113 | 0.178788 | 0.136675 | 0.713077 | 16/20 | 7/10 | 86/100 |
| J0 | 06 | COMPLETE | 0.027505 | 1.580902 | 0.026011 | 0.043596 | 0.017585 | 0.726302 | 18/20 | 8/10 | 83/100 |
| J0 | 07 | COMPLETE | 0.054102 | 1.224658 | 0.039011 | 0.071305 | 0.032294 | 0.738123 | 20/20 | 10/10 | 93/100 |
| J0 | 08 | COMPLETE | 0.118298 | 3.173804 | 0.075982 | 0.133506 | 0.057523 | 0.612512 | 19/20 | 9/10 | 80/100 |
| J0 | 09 | COMPLETE | 0.024417 | 1.330144 | 0.047078 | 0.052240 | 0.005162 | 0.797365 | 20/20 | 10/10 | 93/100 |
| J0 | 10 | COMPLETE | 0.085054 | 1.602591 | 0.020815 | 0.098641 | 0.077826 | 0.748111 | 19/20 | 9/10 | 79/100 |
| SV | 01 | COMPLETE | 0.032460 | 2.367493 | 0.049588 | 0.100520 | 0.050932 | 0.638600 | 19/20 | 9/10 | 92/100 |
| SV | 02 | COMPLETE | 0.051715 | 2.647656 | 0.030069 | 0.068422 | 0.038353 | 0.642398 | 17/20 | 7/10 | 86/100 |
| SV | 03 | TYPED_INCOMPLETE(k4) | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED |
| SV | 04 | COMPLETE | 0.047125 | 3.489627 | 0.094001 | 0.103260 | 0.009259 | 0.698082 | 16/20 | 8/10 | 94/100 |
| SV | 05 | COMPLETE | 0.157617 | 2.446753 | 0.082715 | 0.218670 | 0.135955 | 0.636482 | 16/20 | 7/10 | 86/100 |
| SV | 06 | TYPED_INCOMPLETE(k7) | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED |
| SV | 07 | COMPLETE | 0.096216 | 1.233447 | 0.020117 | 0.096028 | 0.075911 | 0.636120 | 20/20 | 10/10 | 94/100 |
| SV | 08 | COMPLETE | 0.103284 | 2.755127 | 0.078586 | 0.111215 | 0.032629 | 0.624621 | 19/20 | 9/10 | 80/100 |
| SV | 09 | COMPLETE | 0.034359 | 1.394734 | 0.048365 | 0.077911 | 0.029546 | 0.654909 | 20/20 | 10/10 | 94/100 |
| SV | 10 | COMPLETE | 0.219275 | 1.735919 | 0.110924 | 0.223012 | 0.112088 | 0.652087 | 19/20 | 9/10 | 79/100 |
| FPIQ | 01 | COMPLETE | 0.026176 | 2.519141 | 0.063439 | 0.060293 | -0.003147 | 0.941592 | 19/20 | 9/10 | 92/100 |
| FPIQ | 02 | COMPLETE | 0.021355 | 2.576953 | 0.026941 | 0.027218 | 0.000277 | 0.954696 | 16/20 | 6/10 | 86/100 |
| FPIQ | 03 | COMPLETE | 0.036853 | 3.427235 | 0.063033 | 0.062883 | -0.000151 | 0.968194 | 18/20 | 8/10 | 87/100 |
| FPIQ | 04 | COMPLETE | 0.039880 | 3.453448 | 0.140406 | 0.130322 | -0.010084 | 1.062902 | 16/20 | 8/10 | 92/100 |
| FPIQ | 05 | COMPLETE | 0.372214 | 2.582727 | 0.481178 | 0.477578 | -0.003600 | 0.952478 | 16/20 | 7/10 | 86/100 |
| FPIQ | 06 | COMPLETE | 0.020462 | 1.607715 | 0.037065 | 0.037106 | 0.000041 | 0.951405 | 18/20 | 8/10 | 85/100 |
| FPIQ | 07 | TYPED_INCOMPLETE(k2) | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED |
| FPIQ | 08 | COMPLETE | 0.078870 | 3.337500 | 0.101073 | 0.100278 | -0.000795 | 0.947136 | 19/20 | 9/10 | 80/100 |
| FPIQ | 09 | COMPLETE | 0.023073 | 1.352837 | 0.054136 | 0.054036 | -0.000099 | 0.960348 | 20/20 | 10/10 | 94/100 |
| FPIQ | 10 | COMPLETE | 0.128344 | 1.517599 | 0.126056 | 0.134454 | 0.008398 | 0.924233 | 20/20 | 10/10 | 79/100 |

## layer quota·velocity·prefix trajectory

| Arm | available K prefixes | mean coverage | negative aggregate prefixes | negative request-prefixes | support exhausted | layer8 quota/remaining mean | layer8 v/v0 median / p90 / max | max v/v0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| J0 | 80/80 | 0.736622 | 0 | 0 | 0 | NOT_RECORDED | NOT_RECORDED / NOT_RECORDED / NOT_RECORDED | 1.000000 |
| SV | 75/80 | 0.647912 | 5 | 31 | 0 | 0.505615 | 1.000000 / 1.000000 / 1.000000 | 1.000000 |
| FPIQ | 74/80 | 0.962554 | 3 | 102 | 0 | 0.283106 | 5.177759 / 7.462564 / 11.981532 | 11.981532 |

| Arm | layer | rows | α_rem mean | entry π mean | suffix π mean | quota mean | applied slope mean | v/v0 median | factor energy mean | prefix progress mean | negative request count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SV | 4 | 75 | 1.974070 | 0.138213 | 1.000000 | 0.335628 | 2.200868 | 1.000000 | 0.030143 | 0.347656 | 14 |
| SV | 5 | 75 | 1.626384 | 0.172859 | 0.861787 | 0.387648 | 2.010965 | 1.000000 | 0.027837 | 0.327138 | 3 |
| SV | 6 | 75 | 1.299275 | 0.184952 | 0.688928 | 0.412791 | 1.750804 | 1.000000 | 0.022437 | 0.262519 | 5 |
| SV | 7 | 75 | 1.035890 | 0.200097 | 0.503976 | 0.466330 | 1.477485 | 1.000000 | 0.021185 | 0.200618 | 8 |
| SV | 8 | 75 | 0.834089 | 0.303879 | 0.303879 | 0.834089 | 1.474571 | 1.000000 | 0.023993 | 0.209185 | 1 |
| FPIQ | 4 | 74 | 1.378262 | 0.147623 | 1.000000 | 0.234414 | 1.685021 | 1.000000 | 0.012076 | 0.225938 | 39 |
| FPIQ | 5 | 74 | 1.152276 | 0.194868 | 0.852377 | 0.294206 | 1.497033 | 1.223382 | 0.019866 | 0.275672 | 23 |
| FPIQ | 6 | 74 | 0.876652 | 0.214154 | 0.657509 | 0.314832 | 1.181607 | 1.647790 | 0.026830 | 0.284542 | 15 |
| FPIQ | 7 | 74 | 0.592062 | 0.208585 | 0.443355 | 0.291064 | 0.814175 | 2.705727 | 0.048785 | 0.252905 | 9 |
| FPIQ | 8 | 74 | 0.339204 | 0.234769 | 0.234769 | 0.339204 | 0.542817 | 5.177759 | 0.109002 | 0.273029 | 16 |

FPIQ는 suffix π mass가 줄수록 late-layer relative share가 커지고, current slope가 낮을 때 `v/v0`가 증가했다. 이는 contract가 예고한 suffix concentration 신호다. aggregate negative prefix는 FPIQ 3개였지만 request-prefix negative count는 102개여서 평균 progress가 가리는 request별 이질성도 존재한다. 세부 1,145-row layer table에는 각 layer의 `α_rem`, π, suffix mass, quota, raw/applied slope, v, v/v0, θ, key/q hash, factor energy와 prefix progress가 있다.

## P/H, energy, capacity, load와 계산량

| Arm | Structural-P mean | BF16 energy mean | squared capacity mean | realization ratio mean | model F | backward | materializations | logical extra slope groups | physical extra slope autograd | edit-core wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| J0 | 0.007367 | 1.833865 | 4.020608 | 0.796527 | 2240 | 1250 | 80 | 0 | 0 | 1017.719 |
| SV | 0.006838 | 1.576439 | 3.898969 | 0.948478 | 4645 | 2280 | 64 | 256 | 1280 | 1677.528 |
| FPIQ | 0.008210 | 2.236050 | 3.886833 | 0.888090 | 5220 | 2565 | 72 | 288 | 1440 | 2081.127 |

Structural-H는 세 arm 모두 0이며 hard P/H gate도 0이다. `SV/FPIQ`의 logical extra slope group은 완료·last-valid prefix당 4개이고, 각 logical group이 B10 microbatch 5개로 실행되어 physical autograd는 5배다. 이 표는 두 수치를 분리해 기록한다. 위 compute 합계는 terminal endpoint가 존재하는 case만 집계하며, typed incomplete의 last-valid logical/physical count는 per-step 표에 유지한다.

## typed failure와 기술 시도 이력

| attempt | arm | case | last-valid K | exception hash | W0 restore | endpoint |
|---|---|---:|---:|---|---|---|
| TECH-R1 | SV | 03 | 4 | `c73c26fd95c01ec736460e60de17e1660d45d5f2742c0642330d8670ba4d63a3` | PASS | NOT_IMPUTED |
| TECH-R1 | SV | 06 | 7 | `c73c26fd95c01ec736460e60de17e1660d45d5f2742c0642330d8670ba4d63a3` | PASS | NOT_IMPUTED |
| TECH-R1 | FPIQ | 07 | 2 | `c73c26fd95c01ec736460e60de17e1660d45d5f2742c0642330d8670ba4d63a3` | PASS | NOT_IMPUTED |

Canonical TECH-R1의 3개 incomplete는 동일 inherited P1R52 origin-clamp contract exception `c73c26…`이다. SV case03은 k4, case06은 k7, FPIQ case07은 k2까지의 last-valid prefix만 보존했다. 각 failure 후 W0 pointer/bytes restore는 PASS였고 다음 독립 case는 계속됐다.

초기 source `626f5de…`의 SV/FPIQ는 각각 10/10 case가 k0 이전 `cc7657…` empty-prefix factor inventory packaging defect로 끝났다. 이는 science endpoint가 없는 technical attempt history다. TECH-R1 `f534614…`는 empty inventory를 packaging에서 제외한 source-backed fix만 포함하며, primary invalid 결과는 어떤 science aggregate에도 포함하지 않았다.

## 성공·실패 원인 분석

- **J0→SV**: matched 8 case에서 Δcoverage=-0.085337, Δfull-six W−z gap=0.017866였다. residual/key/q refresh만으로 생긴 변화는 작거나 혼합되어 stale cross-effect 하나만을 지배 병목으로 단정할 수 없다.
- **SV→FPIQ**: matched 7 case에서 Δcoverage=0.313744, Δfull-six W−z gap=-0.059687로 realization은 개선됐다. 그러나 ΔW rewrite NLL=0.006297, ΔW rephrase NLL=0.071842여서 absolute heldout NLL 개선은 성립하지 않았다. 이는 writer package association이며 단독 요소의 인과 분리는 아니다.
- **Writer realization**: FPIQ matched J0에서 Δfull-six W−z gap=-0.039433, Δcoverage=0.226099로 writer-side gap은 줄었다. 반면 Δz full-six NLL=0.079730, ΔW full-six NLL=0.040296로 둘 다 악화됐다. gap 축소만으로 absolute strength recovery를 주장할 수 없고 coupled trajectory에서 accepted-z endpoint가 약해진 사실을 함께 봐야 한다.
- **Hard case**: FPIQ case05는 J0 대비 W rewrite NLL +0.242197, W rephrase NLL +0.252441, z full-six NLL +0.439065였다. 이 case가 mean rewrite 악화의 대부분을 차지하지만, 제외 후에도 rephrase와 full-six target endpoint의 일관된 개선은 확인되지 않는다. ASSOCIATION_ONLY이며 원인 단독 분리는 불가하다.
- **비용/집중**: FPIQ/J0 matched energy ratio=1.204663, capacity ratio=0.968362; FPIQ layer8 `v/v0` max=11.981532. 따라서 strength 개선은 공짜가 아니며 late-layer concentration이 tail risk다.
- **안정성**: technically valid endpoint 비율은 J0 10/10, SV 8/10, FPIQ 9/10이다. Nonfinite, double-π/h, live-prefix mutation, rollback failure는 관찰되지 않았으나 inherited origin-clamp incomplete가 남았다.

## Promotion 판정

**최종 판정: `HOLD_NO_PROMOTION_TO_B100_OR_HISTORICAL`.**

FPIQ는 technically valid matched endpoint에서 writer coverage와 W/z gap을 개선하는 positive signal을 보였으나, absolute W rewrite/rephrase·full-six NLL과 GEN은 함께 개선되지 않았고 accepted-z endpoint도 약해졌다. 또한 (1) 10개 중 1개 endpoint가 없고, (2) SV도 2개 incomplete이며, (3) FPIQ의 late-layer velocity·energy 집중이 크고, (4) B100/Historical은 현재 계약상 별도 promotion 결정 전 금지되어 있다. 따라서 이 atomic gate는 `WRITER_REALIZATION_SIGNAL_WITH_TARGET_SIDE_REGRESSION_AND_TECHNICAL_INCOMPLETENESS`로 분류하되, 이 결과만으로 sequential/Historical 실행을 승격하지 않는다. 이는 후속 방법 추천이 아니라 본 계약의 promotion 질문에 대한 판정이다.

## 데이터·재현성 경계

- Contract: `/mnt/raid5/janghj/.codex/attachments/76d5dc44-60b9-4c33-baa7-1906233fd24b/pasted-text.txt` SHA `e5c767cdd6cfd498376dacad39155d08d0bc79c89496dbf629725a9b928b4436`, bytes 15657, lines 742, mode 0600.
- Numerical lock: `/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-s05-p1r52-fpiq-atomic-b10x10-v1/project/run_scripts/ode_bf/locks/numerical_lock_s05_p1r52_fpiq_atomic_b10x10.json` SHA `fd901b535140502c3b92eb598a77048d916faee5fac577872bcd16cbfc0b8478`; root `881b8930ded6db48fab1993ee05286be54d8c987cea30c93a7e5b6722a24df16`.
- Source manifest: `/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-s05-p1r52-fpiq-atomic-b10x10-v1/project/run_scripts/ode_bf/locks/source_manifest_s05_p1r52_fpiq_atomic_b10x10.json` SHA `da9270c6681956d3a0290548b4fd6a5b6dea34c2fbf38c2c46ceccbb1a067617`; root `ef84e3575f45a99722003c4200623dc908ff98cc00590acb35278e2e9ec358e1`.
- J0 source/result: `626f5de52394e4f2fce401d56f16a2b532da6501`, `/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-s05-p1r52-fpiq-atomic-b10x10-v1/local/odebf/results/s05-p1r52-fpiq-independent-b10x10-llama3-8b-inst-j0-v1`.
- SV/FPIQ TECH-R1 source/results: `f53461448e123f1caa1aee0bd5d70ab31e5109d6`, `/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-s05-p1r52-fpiq-atomic-b10x10-v1/local/odebf/results/s05-p1r52-fpiq-independent-b10x10-llama3-8b-inst-sv-tech-r1-v1`, `/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-s05-p1r52-fpiq-atomic-b10x10-v1/local/odebf/results/s05-p1r52-fpiq-independent-b10x10-llama3-8b-inst-fpiq-tech-r1-v1`.
- Stepwise heldout evaluation과 writer decision heldout access는 0. Endpoint absent fields are `NOT_RECORDED`; no imputation.
- `scientific_promotion=false`는 실행 namespace의 고정 행정 경계이며, 위 HOLD 판정과 일치한다.

## 산출물

- `p1r52-fpiq-per-case.json`: attempts/endpoints와 terminal z/W panels.
- `p1r52-fpiq-per-step.json`: all complete and last-valid accepted K prefixes.
- `p1r52-fpiq-per-layer.json`: J0 entry layer rows plus SV/FPIQ current-prefix layer rows.
- `p1r52-fpiq-typed-failures.json`: canonical incomplete + primary invalid attempt history.
- `p1r52-fpiq-paired-case-deltas.json`: exact same-case deltas; unmatched rows retain null delta.
- `analysis-manifest.json`, `analysis-receipt.json`, `independent-review.json`: SHA/row/root/review receipts.
