# P1R52 PIR Sequential Writer — Atomic B10×10 독립 최종 분석

## 범위와 입력 무결성

이 분석은 완결된 Llama3-8B-Instruct independent B10×10 영수증만 읽었다. 모델·평가·GPU·Slurm 호출은 0회이며, raw prompt·target·tensor·weight는 산출물에 포함하지 않았다.

- 계약: `/mnt/raid5/janghj/.codex/attachments/b5a5db2f-0b13-4fe7-be17-ea0a9c4cab42/pasted-text.txt` — SHA `60227762fb804d4f52e7501b60c6016014468d112183cc0c2ecf9cc6fbcf730a`, 14,125 bytes, 574 lines, mode 0600.
- authoritative TECH-R1 source: `871d41c668ed46535a08fd4886318f881798886f` / tree `d5858904bd10073cbac5c7bf522a62c536d3c82e`; J0/PIR-G/PIR-U 모두 같은 HEAD에서 실행됐다.
- numerical lock: SHA `0c99fc1c25cb63b9ffd35fe8bbf9ebb614d8c587c738fe7d5b6359c3614b4a08`, root `95e86cfe1cfd6f550a1f0cea492694b2a0165533c50466f10ae7b4f3139e0430`; source manifest: SHA `acb63948385cdbaef7d8ebebb3b549583c7e084f9b00e8e1b204fde334ac964d`, root `20af9387a7674a450fea45c18f7abba9f50d6890a323a19e8c390139a8c824ba`.
- authoritative matrix는 J0, PIR-G, PIR-U 각 10/10 endpoint(총 30/30)다. 이전 pre-repair PIR-G/U의 KeyError 20건은 기술적 배경 증거로만 보존했고 이 표·평균·paired delta에 포함하지 않았다.

## 핵심 절대값

| Arm | attempts/endpoints | W EFF | W GEN | W GEN-strict | LOC | W rewrite NLL | W rephrase NLL | W full-six NLL | coverage | full-six W−z gap | energy | capacity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| J0 | 10/10 | 100/100 (1.0000) | 181/200 (0.9050) | 84/100 (0.8400) | 871/1000 (0.8710) | 0.057125 | 2.291020 | 0.079280 | 0.736622 | 0.037804 | 1.833865 | 4.020608 |
| PIR-G | 10/10 | 100/100 (1.0000) | 179/200 (0.8950) | 82/100 (0.8200) | 871/1000 (0.8710) | 0.068974 | 2.532618 | 0.119305 | 0.665360 | 0.048314 | 1.092923 | 2.652169 |
| PIR-U | 10/10 | 100/100 (1.0000) | 182/200 (0.9100) | 85/100 (0.8500) | 873/1000 (0.8730) | 0.059402 | 2.373478 | 0.068068 | 0.317375 | -0.002565 | 3.147508 | 4.263492 |

`EFF/GEN/LOC`는 terminal weight panel의 success count다. GEN-strict는 request 단위에서 모든 paraphrase prompt가 통과한 수다. rewrite/rephrase NLL와 full-six target-objective NLL는 서로 다른 panel이므로 대체하지 않았다.

## 기술 완결성 및 고정 계약

| Arm | K8 complete | W0 restore | retry/backtracking | history append | heldout inner | additional current-slope backward | materializations | prefix capture | q solve |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| J0 | 80/80 | 10/10 | 0 | 0 | 0 | 0 | 80 | 0 | 0 |
| PIR-G | 80/80 | 10/10 | 0 | 0 | 0 | 0 | 80 | 320 | 320 |
| PIR-U | 80/80 | 10/10 | 0 | 0 | 0 | 0 | 80 | 320 | 320 |

모든 authoritative case에서 action-freeze, K8, W0 pointer/bytes restore, controller reset, cross-case state 0, history OFF, h count 1, retry/backtracking 0, inner heldout access 0이 receipt로 확인됐다. PIR-G/U의 current semantic-slope backward는 0이며, prefix capture와 q solve만 추가됐다.

## target trajectory와 accepted-z / physical-W 분리

| Arm | z EFF | W EFF | z GEN | W GEN | z GEN-strict | W GEN-strict | z rewrite NLL | W rewrite NLL | z rephrase NLL | W rephrase NLL | z full-six NLL | W full-six NLL | z→W EFF fail | z→W GEN-strict fail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| J0 | 100/100 (1.0000) | 100/100 (1.0000) | 183/200 (0.9150) | 181/200 (0.9050) | 85/100 (0.8500) | 84/100 (0.8400) | 0.026695 | 0.057125 | 2.136656 | 2.291020 | 0.041477 | 0.079280 | 0 | 1 |
| PIR-G | 100/100 (1.0000) | 100/100 (1.0000) | 181/200 (0.9050) | 179/200 (0.8950) | 84/100 (0.8400) | 82/100 (0.8200) | 0.039236 | 0.068974 | 2.350186 | 2.532618 | 0.070991 | 0.119305 | 0 | 2 |
| PIR-U | 100/100 (1.0000) | 100/100 (1.0000) | 182/200 (0.9100) | 182/200 (0.9100) | 85/100 (0.8500) | 85/100 (0.8500) | 0.066374 | 0.059402 | 2.378970 | 2.373478 | 0.070633 | 0.068068 | 0 | 0 |

| Arm | PRIMARY | RESCUE | CURRENT | clamp hits | selected target NLL mean | target displacement norm mean | teacher K8 constant |
|---|---:|---:|---:|---:|---:|---:|---:|
| J0 | 791 | 9 | 0 | 20 | 2.020777 | 0.550510 | 10/10 |
| PIR-G | 789 | 5 | 6 | 30 | 2.016110 | 0.550483 | 10/10 |
| PIR-U | 793 | 6 | 1 | 20 | 2.018998 | 0.533781 | 10/10 |

## exact matched J0 대비 산술 차이

| comparison | matched cases | ΔW rewrite NLL | ΔW rephrase NLL | ΔW full-six NLL | ΔW GEN | ΔW GEN-strict | ΔLOC | Δcoverage | Δfull-six W−z gap | Δenergy | Δcapacity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PIR-G-J0 | 10 | 0.011849 | 0.241597 | 0.040024 | -0.200000 | -0.200000 | 0.000000 | -0.071262 | 0.010510 | -0.740942 | -1.368439 |
| PIR-U-J0 | 10 | 0.002277 | 0.082458 | -0.011212 | 0.100000 | 0.100000 | 0.200000 | -0.419247 | -0.040368 | 1.313643 | 0.242884 |

Delta는 `candidate − J0`이며, NLL의 음수는 candidate가 더 낮다는 뜻이다. 10개 exact matched case의 산술 평균만 사용했고 누락·보간은 없다.

## writer realization·P/energy/capacity 경계

| Arm | gamma mean/median/p90/max | Dβ0 mean | entry-strength residual max | predicted/actual realization ratio | negative physical transitions | layer8 energy share mean/p90/max | layer8 coefficient/J0 median/p90/max | intended→realized cosine mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| J0 | NOT_RECORDED/NOT_RECORDED/NOT_RECORDED/NOT_RECORDED | NOT_RECORDED | NOT_APPLICABLE_BY_POLICY | 0.796527 | 0 | NOT_RECORDED/NOT_RECORDED/NOT_RECORDED | 1.000000/1.000000/1.000000 | NOT_RECORDED |
| PIR-G | 0.323446/0.332562/0.351235/0.364171 | 5.620901 | 0.000000000000 | 0.702282 | 0 | 0.622020/0.667421/0.704216 | 2.474557/2.703020/2.781143 | 0.284958 |
| PIR-U | 1.000000/1.000000/1.000000/1.000000 | 4.498163 | NOT_APPLICABLE_BY_POLICY | 0.295348 | 0 | 0.560345/0.636689/0.813734 | 7.002769/15.556521/54.844697 | 0.406750 |

PIR-G/U의 P 값은 `ENTRY_FIELD_MIXED_GEOMETRY_PROXY_NOT_COMPARABLE`로 기록됐다. 따라서 P proxy를 actual endpoint Structural-P나 arm 간 preservation 비교로 사용하지 않았다. 실제 BF16 energy, squared/cumulative capacity, locality, downstream metrics만 primary preservation 표에 두었다.

### layer별 요약

| Arm | layer | rows | beta mean | gamma·beta mean | residual norm mean | residual reduction ratio mean | q norm mean | BF16 energy share mean | coefficient/J0 median | cosine mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PIR-G | 4 | 80 | 0.138223 | 0.045898 | 3.050151 | 0.009766 | 1.017489 | 0.026249 | 0.321718 | 0.147324 |
| PIR-G | 5 | 80 | 0.230105 | 0.075252 | 3.013466 | 0.017501 | 0.881019 | 0.050545 | 0.412296 | 0.204301 |
| PIR-G | 6 | 80 | 0.358407 | 0.116133 | 2.952497 | 0.032405 | 0.814322 | 0.097115 | 0.591232 | 0.302467 |
| PIR-G | 7 | 80 | 0.526581 | 0.169200 | 2.843327 | 0.070627 | 0.830958 | 0.204071 | 1.008431 | 0.485742 |
| PIR-G | 8 | 80 | 1.000000 | 0.323446 | 2.628406 | NOT_RECORDED | 0.829976 | 0.622020 | 2.474557 | NOT_RECORDED |
| PIR-U | 4 | 80 | 0.145059 | 0.145059 | 1.984627 | -0.016474 | 1.017489 | 0.038454 | 1.075283 | 0.266562 |
| PIR-U | 5 | 80 | 0.236302 | 0.236302 | 1.906862 | 0.043026 | 0.883555 | 0.072963 | 1.445337 | 0.341406 |
| PIR-U | 6 | 80 | 0.331747 | 0.331747 | 1.828237 | 0.079745 | 0.821496 | 0.112961 | 2.015346 | 0.418890 |
| PIR-U | 7 | 80 | 0.489368 | 0.489368 | 1.685694 | 0.186361 | 0.836006 | 0.215277 | 3.124689 | 0.596636 |
| PIR-U | 8 | 80 | 1.000000 | 1.000000 | 1.373535 | NOT_RECORDED | 0.822809 | 0.560345 | 7.002769 | NOT_RECORDED |

## receipt 기반 성공·실패 패턴

- PIR-G−J0 (10 matched case): W rewrite NLL Δ=0.011849, W rephrase NLL Δ=0.241597, W full-six NLL Δ=0.040024, GEN Δ=-0.200000, coverage Δ=-0.071262, full-six W−z gap Δ=0.010510. 이 값들은 contract의 absolute-W/GEN 및 coverage-with-strength 조건을 동시에 충족하지 않는 receipt pattern이다.
- PIR-U−J0 (10 matched case): W full-six NLL Δ=-0.011212, GEN Δ=0.100000, W rewrite/rephrase NLL Δ=0.002277/0.082458, coverage Δ=-0.419247, energy Δ=1.313643. PIR-U는 gamma=1 upper-bound control로만 표기하며 primary candidate로 자동 승격하지 않는다.
- layer8 energy share는 PIR-G 0.622020, PIR-U 0.560345; layer8 coefficient/J0 median은 각각 2.474557, 7.002769다. 이 값은 layer-concentration telemetry이며 단독 인과 설명으로 사용하지 않았다.
- case별 hard/tail 값은 per-case 및 per-layer table에 보존했다. 본 분석은 receipt 간 연관만 기술하며 residual refresh·beta·gamma·energy 중 어느 하나의 단독 원인을 추론하지 않는다.

## case별 절대 endpoint

| Arm | case | W rewrite NLL | W rephrase NLL | W full-six NLL | z full-six NLL | full-six W−z gap | coverage | W EFF | W GEN | W GEN-strict | LOC | energy | capacity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| J0 | 01 | 0.023461 | 2.385339 | 0.060898 | 0.055925 | 0.004974 | 0.768878 | 10/10 (1.0000) | 19/20 (0.9500) | 9/10 (0.9000) | 92/100 (0.9200) | 2.146342 | 5.359672 |
| J0 | 02 | 0.031012 | 2.575562 | 0.040365 | 0.029736 | 0.010629 | 0.746906 | 10/10 (1.0000) | 16/20 (0.8000) | 6/10 (0.6000) | 86/100 (0.8600) | 1.727891 | 3.826536 |
| J0 | 03 | 0.054102 | 3.256244 | 0.083812 | 0.065110 | 0.018701 | 0.773918 | 10/10 (1.0000) | 18/20 (0.9000) | 8/10 (0.8000) | 87/100 (0.8700) | 1.748709 | 3.808204 |
| J0 | 04 | 0.023285 | 3.450674 | 0.029654 | 0.012988 | 0.016666 | 0.741026 | 10/10 (1.0000) | 16/20 (0.8000) | 8/10 (0.8000) | 92/100 (0.9200) | 1.658457 | 3.461245 |
| J0 | 05 | 0.130017 | 2.330286 | 0.178788 | 0.042113 | 0.136675 | 0.713077 | 10/10 (1.0000) | 16/20 (0.8000) | 7/10 (0.7000) | 86/100 (0.8600) | 1.755196 | 3.300255 |
| J0 | 06 | 0.027505 | 1.580902 | 0.043596 | 0.026011 | 0.017585 | 0.726302 | 10/10 (1.0000) | 18/20 (0.9000) | 8/10 (0.8000) | 83/100 (0.8300) | 1.959753 | 4.795119 |
| J0 | 07 | 0.054102 | 1.224658 | 0.071305 | 0.039011 | 0.032294 | 0.738123 | 10/10 (1.0000) | 20/20 (1.0000) | 10/10 (1.0000) | 93/100 (0.9300) | 1.702857 | 4.027514 |
| J0 | 08 | 0.118298 | 3.173804 | 0.133506 | 0.075982 | 0.057523 | 0.612512 | 10/10 (1.0000) | 19/20 (0.9500) | 9/10 (0.9000) | 80/100 (0.8000) | 1.774441 | 3.292464 |
| J0 | 09 | 0.024417 | 1.330144 | 0.052240 | 0.047078 | 0.005162 | 0.797365 | 10/10 (1.0000) | 20/20 (1.0000) | 10/10 (1.0000) | 93/100 (0.9300) | 1.576076 | 3.527591 |
| J0 | 10 | 0.085054 | 1.602591 | 0.098641 | 0.020815 | 0.077826 | 0.748111 | 10/10 (1.0000) | 19/20 (0.9500) | 9/10 (0.9000) | 79/100 (0.7900) | 2.288930 | 4.807483 |
| PIR-G | 01 | 0.035600 | 2.598889 | 0.090545 | 0.058448 | 0.032097 | 0.661536 | 10/10 (1.0000) | 19/20 (0.9500) | 9/10 (0.9000) | 92/100 (0.9200) | 1.314990 | 3.638870 |
| PIR-G | 02 | 0.046741 | 2.785840 | 0.056475 | 0.026420 | 0.030056 | 0.667495 | 10/10 (1.0000) | 16/20 (0.8000) | 6/10 (0.6000) | 86/100 (0.8600) | 0.953837 | 2.350515 |
| PIR-G | 03 | 0.065108 | 3.822431 | 0.102817 | 0.059979 | 0.042838 | 0.666066 | 10/10 (1.0000) | 18/20 (0.9000) | 8/10 (0.8000) | 87/100 (0.8700) | 0.895006 | 2.134647 |
| PIR-G | 04 | 0.056665 | 3.552191 | 0.213337 | 0.183671 | 0.029666 | 0.662402 | 10/10 (1.0000) | 16/20 (0.8000) | 8/10 (0.8000) | 92/100 (0.9200) | 1.148765 | 2.535689 |
| PIR-G | 05 | 0.221027 | 2.828198 | 0.286251 | 0.136725 | 0.149527 | 0.648114 | 10/10 (1.0000) | 16/20 (0.8000) | 7/10 (0.7000) | 86/100 (0.8600) | 0.999048 | 1.876163 |
| PIR-G | 06 | 0.037679 | 1.700226 | 0.072666 | 0.037066 | 0.035600 | 0.650932 | 10/10 (1.0000) | 17/20 (0.8500) | 7/10 (0.7000) | 82/100 (0.8200) | 1.193670 | 3.144829 |
| PIR-G | 07 | 0.038385 | 1.494049 | 0.074907 | 0.041287 | 0.033619 | 0.676660 | 10/10 (1.0000) | 20/20 (1.0000) | 10/10 (1.0000) | 94/100 (0.9400) | 1.150575 | 3.003181 |
| PIR-G | 08 | 0.077820 | 3.434045 | 0.107124 | 0.089632 | 0.017491 | 0.669754 | 10/10 (1.0000) | 19/20 (0.9500) | 9/10 (0.9000) | 80/100 (0.8000) | 0.905902 | 2.147244 |
| PIR-G | 09 | 0.033374 | 1.452800 | 0.075126 | 0.049582 | 0.025544 | 0.673705 | 10/10 (1.0000) | 20/20 (1.0000) | 10/10 (1.0000) | 92/100 (0.9200) | 0.923568 | 2.283043 |
| PIR-G | 10 | 0.077341 | 1.657507 | 0.113797 | 0.027099 | 0.086699 | 0.676932 | 10/10 (1.0000) | 18/20 (0.9000) | 8/10 (0.8000) | 80/100 (0.8000) | 1.443875 | 3.407510 |
| PIR-U | 01 | 0.033214 | 2.502454 | 0.064912 | 0.065540 | -0.000628 | 0.317532 | 10/10 (1.0000) | 19/20 (0.9500) | 9/10 (0.9000) | 92/100 (0.9200) | 3.852036 | 5.354010 |
| PIR-U | 02 | 0.022298 | 2.613550 | 0.028200 | 0.028159 | 0.000041 | 0.357278 | 10/10 (1.0000) | 16/20 (0.8000) | 6/10 (0.6000) | 86/100 (0.8600) | 2.055170 | 3.340163 |
| PIR-U | 03 | 0.037473 | 3.436828 | 0.064221 | 0.062974 | 0.001246 | 0.359170 | 10/10 (1.0000) | 18/20 (0.9000) | 8/10 (0.8000) | 87/100 (0.8700) | 2.064634 | 3.414529 |
| PIR-U | 04 | 0.018459 | 3.355505 | 0.020204 | 0.020053 | 0.000151 | 0.271130 | 10/10 (1.0000) | 16/20 (0.8000) | 8/10 (0.8000) | 92/100 (0.9200) | 3.327533 | 4.343968 |
| PIR-U | 05 | 0.052084 | 2.455518 | 0.122657 | 0.121170 | 0.001488 | 0.200646 | 10/10 (1.0000) | 16/20 (0.8000) | 7/10 (0.7000) | 86/100 (0.8600) | 3.954284 | 4.020895 |
| PIR-U | 06 | 0.019476 | 1.540515 | 0.036445 | 0.036219 | 0.000226 | 0.349155 | 10/10 (1.0000) | 18/20 (0.9000) | 8/10 (0.8000) | 83/100 (0.8300) | 3.904063 | 5.676966 |
| PIR-U | 07 | 0.027807 | 1.188510 | 0.052774 | 0.052724 | 0.000051 | 0.289141 | 10/10 (1.0000) | 20/20 (1.0000) | 10/10 (1.0000) | 93/100 (0.9300) | 3.963688 | 5.340894 |
| PIR-U | 08 | 0.077502 | 3.503125 | 0.101794 | 0.102792 | -0.000998 | 0.398236 | 10/10 (1.0000) | 19/20 (0.9500) | 9/10 (0.9000) | 80/100 (0.8000) | 1.877950 | 2.841011 |
| PIR-U | 09 | 0.025024 | 1.375829 | 0.057837 | 0.057852 | -0.000016 | 0.377601 | 10/10 (1.0000) | 20/20 (1.0000) | 10/10 (1.0000) | 94/100 (0.9400) | 2.042709 | 3.176640 |
| PIR-U | 10 | 0.280681 | 1.762949 | 0.131641 | 0.158850 | -0.027209 | 0.253859 | 10/10 (1.0000) | 20/20 (1.0000) | 10/10 (1.0000) | 80/100 (0.8000) | 4.433011 | 5.125845 |

## compute ledger

| Arm | model F | B | target B | slope B | extra current-slope B | tokens | prefix captures | q solves | materializations | edit-core s | terminal evaluator s | job wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| J0 | 2240 | 1250 | 850 | 400 | 0 | 380343 | 0 | 0 | 80 | 1033.433 | 381.434 | 1612.634 |
| PIR-G | 4200 | 1250 | 850 | 400 | 0 | 776238 | 320 | 320 | 80 | 1999.822 | 379.335 | 2575.518 |
| PIR-U | 4185 | 1250 | 850 | 400 | 0 | 773079 | 320 | 320 | 80 | 1952.115 | 378.812 | 2531.017 |

PIR-G/U의 current-slope backward는 계약대로 0이다. `prefix capture=320`, `q solve=320`은 10 case × K8 × later layer 4에 대응한다. terminal evaluator 외 heldout는 0이고, analysis 자체의 added model/backward/generation은 0/0/0이다.

## 이전 FPIQ 배경 증거

읽기 전용 이전 FPIQ 보고서 SHA는 `9a22d60ae2286d2caebe9f74288784a45e325d0bbeb99c9ca40370902fc70496`이다. 그 결과는 J0 10/10, SV 8/10, FPIQ 9/10 및 `HOLD_NO_PROMOTION_TO_B100_OR_HISTORICAL`로 봉인돼 있다. source/head가 달라 이번 authoritative J0/PIR-G/PIR-U 표나 paired average에는 합산하지 않았다.

## 계약상 promotion 판정

- PIR-G: `PIR_G_NO_PROMOTION_CONTRACT_CONDITIONS_NOT_ALL_MET`. absolute-W/GEN condition=False, accepted-z non-regression condition=False, coverage/gap-with-absolute-W condition=False, technical 10/10=True.
- PIR-U: `PIR_U_STRONG_UPPER_BOUND_CONTROL_NOT_AUTOMATIC_PRIMARY_PROMOTION`. 계약상 strong residual-realization control이며, 결과가 더 좋아도 automatic primary promotion 대상이 아니다.
- Historical/sequential follow-up: `NOT_AUTHORIZED_WITHOUT_POSITIVE_FINAL_EVIDENCE_AND_EXPLICIT_FOLLOW_UP_AUTHORITY`. 이 atomic 계약은 positive final evidence와 별도 명시 권한 없이는 후속을 열지 않는다.

이 판정은 contract §15 조건에 대한 receipt 기반 분류다. coverage 또는 W−z gap만의 변화는 promotion 근거로 사용하지 않았다.

## 실패·무결성 이력

authoritative TECH-R1 matrix의 typed failure는 0이다. pre-repair PIR-G/U의 KeyError 20/20은 `PRE_REPAIR_TECHNICAL_BACKGROUND_EXCLUDED`로 typed-failures table에 보존했다. 해당 endpoint는 새 결과에 보간·재사용하지 않았다.

## 산출물

- per-case: terminal z/W panels, absolute counts, NLL, gaps, coverage, energy/capacity.
- per-step/per-layer: alpha, pi/beta/gamma/Dβ0, residual/key/q hashes, energy share, cosine, physical progress and count identities.
- per-request: 2,400 W-only refreshed-field progress rows; no heldout stepwise values.
- paired/typed-failure/compute: exact J0→PIR-G/U arithmetic, excluded historical failures, ledger.
- manifest/receipt/independent-review: hashes, rows, roots and raw-free verification.
