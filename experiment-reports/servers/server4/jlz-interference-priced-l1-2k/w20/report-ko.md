# PRICE W20: 2,000-edit 완료 및 baseline 비교

USER recall에 따라 W15 중간 보고를 W20 완료 결과로 갱신했다. 실행 source `0415aba3c160170d306be8196792f198dad4d122`는 그대로이며, 본 게시 코드는 CPU 리뷰만 추가했다.

## 완료 및 핵심 결과

PRICE GPU59768, collector59769 모두 COMPLETED/exit0. 원 terminal·commit·raw row와 별도 CPU 재집계로 20 fits/20 commits/19 W-H-RNG-context-ledger joins/100 layer history appends 및 W20 R2000/P4000/N20000을 검산했다. B21 없음. 취소된 FLAT/REVERSE 재개 없음.

**Ours: RS 99.900% (1998/2000), PS 91.275% (3651/4000), NS 85.115% (17023/20000), harmonic mean 91.702%.**

## Baseline 비교

W0는 **편집0개/평가first2000**인 Ours의 동일 runtime 초기 관측이다. 나머지는 각 방법의 **W20 시점 first2000 all-seen** 집계다. 마지막 W100 결과를 first2000으로 바꿔 쓴 것이 아니다. 단위 %. Ours/W0 이외는 아래 조건 차이를 유지한 역사 참고이며 동일 runtime의 matched 실험으로 표시하지 않는다.

| 방법 | RS | PS | NS | Harmonic mean | 비교 범위 |
|---|---:|---:|---:|---:|---|
| W0 (ours; 편집0/평가2000) | 8.200 | 10.950 | 88.555 | 13.359 | 동일 runtime W0 재사용·raw 검산 |
| PRICE (ours) | 99.900 | 91.275 | 85.115 | 91.702 | 본 run raw 검산 |
| MEMIT-H | 99.400 | 91.000 | 79.045 | 89.020 | HISTORICAL_REFERENCE |
| AlphaEdit | 99.300 | 93.225 | 68.590 | 84.802 | HISTORICAL_REFERENCE |
| AlphaEdit-BLUE | 99.600 | 97.150 | 76.585 | 89.845 | HISTORICAL_REFERENCE |
| CAKE | 99.150 | 87.750 | 76.405 | 86.781 | HISTORICAL_REFERENCE |
| MEMIT | 64.750 | 61.700 | 51.825 | 58.885 | HISTORICAL_REFERENCE |
| MEMIT-BLUE | 99.400 | 95.575 | 79.715 | 90.722 | HISTORICAL_REFERENCE |
| v12-MEMIT | 98.500 | 90.300 | 76.060 | 87.275 | HISTORICAL_REFERENCE |
| V13 MD | 98.400 | 88.250 | 71.035 | 84.337 | HISTORICAL_REFERENCE |
| V13 CD | 98.600 | 93.600 | 70.045 | 85.465 | HISTORICAL_REFERENCE |

Harmonic mean은 R/P/N 세 family의 **동일 가중 조화평균**이다. 각 원래 정수 분자/분모로 rate r,p,n을 만든 뒤 `100 × 3 / (1/r + 1/p + 1/n)`으로 계산했다(어느 rate든 0이면 0). 표의 반올림된 %를 다시 입력하지 않았고 N의 큰 분모로 가중하지 않았다. [정수 counts 및 source CSV](comparison-W20.csv).

조건 차이:

- Ours: server4 Blackwell, torch2.9.1+cu128/transformers4.57.1, seed20261002, FP32 eager/TF32off, L4–L8. PRICE requested-budget/approximate same-layer response pullback와 native mean-key ridge.
- MEMIT-H/기존 AlphaEdit·BLUE·CAKE 집계는 과거 seed20260907/transformers4.44.2/cuDNN TF32=true 등 다른 runtime이다. MEMIT-H는 native singleton-z/remaining-layer residual divisor, AlphaEdit는 L4–L8/L2=10, BLUE는 L4+L8/L2=1, CAKE는 clamp.5/decay.4/temperature.1 등 설정 차이가 있다. 각 원 보고서 조건을 유지한다.
- `v12-MEMIT`은 S3 H200 NVL에서 JLZ v12 planner + ridge target-tracking writer를 사용한 역사 결과로, native MEMIT-H와 별개다. V13 MD/CD도 이전 독립 trajectory로 이번 Ours와 동일 method 비교가 아니다.
- 과거 BLUE CSV의 내부 arm `*_ORIGINAL`은 실제 display label `*_BLUE (L4+L8)`로 확인해 매핑했다. CAKE의 rewrite-only 중복 RS행은 사용하지 않고 `ACTUAL_FULL_SEEN` R/P/N을 선택했다.
- Baseline의 원래 게시 CSV size/SHA·W20/분모·정수 count·이전 비교표의 일치를 새로 확인했다. 이번에는 baseline 원 raw/tokenizer를 재실행하거나 cross-run paired 검산하지 않았다. 성능 유의성/동일조건 우월성/과학적 promotion을 판정하지 않는다. 신규 baseline fit/모델 평가 0.
- 직전 v12-R은 W15까지만 있고 W20이 없는 자료이므로 이 W20 표의 `v12-MEMIT`으로 대체하거나 혼합하지 않았다.

## Ours 누적 trajectory

R/P는 new NLL < true NLL, N은 true NLL < new NLL이며 ties는 실패다.

| Endpoint | RS (분자/분모) | PS (분자/분모) | NS (분자/분모) | Harmonic mean |
|---|---:|---:|---:|---:|
| W0 (편집0/평가2000) | 8.200 (164/2000) | 10.950 (438/4000) | 88.555 (17711/20000) | 13.359 |
| W5_ALL_SEEN | 99.800 (499/500) | 92.800 (928/1000) | 87.320 (4366/5000) | 93.029 |
| W10_ALL_SEEN | 99.900 (999/1000) | 92.400 (1848/2000) | 86.370 (8637/10000) | 92.562 |
| W15_ALL_SEEN | 99.867 (1498/1500) | 91.467 (2744/3000) | 85.667 (12850/15000) | 91.970 |
| W20_ALL_SEEN | 99.900 (1998/2000) | 91.275 (3651/4000) | 85.115 (17023/20000) | 91.702 |

W20 teacher-forced 지표는 다음과 같다. R/P desired=new, N desired=true. Preference와 TF 정확도는 별개이며 free-generation 정확도가 아니다.

| 종류 | TF strict% | Token micro% | Prompt macro% | true NLL 평균 | new NLL 평균 |
|---|---:|---:|---:|---:|---:|
| R | 99.250 (1985/2000) | 99.259 | 99.250 | 14.123 | 0.052 |
| P | 61.875 (2475/4000) | 62.204 | 62.213 | 9.724 | 2.006 |
| N | 19.795 (3959/20000) | 20.863 | 20.442 | 5.040 | 9.875 |

## Retention

- 같은 first2000 neighborhood W0→W20: 17711→17023 / 20000, 88.555→85.115%, −3.440pp. lost1072 / gained384 / retained16639, W0-correct retention93.9473%.
- 각 birth cohort 편집 직후→W20 neighborhood: 17369→17023 / 20000, lost542 / gained196 / retained16827.
- 편집 직후→W20 rewrite: 2000→1998, lost2 / gained0. Rephrase: 3682→3651, lost90 / gained59.
- current pre/post, first100/first500, birth cohort, preference/TF strict의 같은 row identity 비교는 [metrics.csv](metrics.csv)와 [paired-cohorts.csv](paired-cohorts.csv)에 있다. 미측정 값 0 대체 없음.

## 검산·비용

- 500 BUILD/500 subject forward/480 subject backward/48000 request-update 참여. 후보별 frozen entry price, active/update/expansion, FP64 projector KKT/FP32 local/shared feasibility, B1 cached LOO, terminal last-evaluated exact payload/no-resolve/no-double-add와 rewrite-only H once를 원 scalar receipt에서 재검산했다.
- R/P/N raw identity/order/token/finite/분모와 저장 summary, paired/counters/cost를 기존 collector와 비교해 일치했다. source-bound stdlib reducer 재사용, 모델/activation/P/K tensor replay 없음. 641개 read-set 파일의 SHA를 다시 확인했다. Owner audit이며 별도 독립 reviewer를 사용하지 않았다.
- parent GPU allocation18,471초 = 5.1308 GPUh. CPU collector17초/GPU0. Slurm step/extern를 부모에 중복 가산하지 않았다. 이번 한정 sacct 완료 snapshot만 확인했고 반복 감시 없음.
- main terminal wall18,464.642초. 20batch inclusive 합18272.541초; fit-inclusive 10076.594초, writer-inclusive 2088.329초, pre/post observer 983.501/3148.097초. inclusive/nested 값들을 다시 더하지 않는다.
- peak RSS 33.109 GiB, peak VRAM 42.553 GiB(원 terminal 계측 정의).
- W0는 exact 기존 관측 참조: 새 forward0/새 평가시간0. 원 job59721 W0관측930.845초 및 이전 failed/canceled jobs 비용은 이번 parent GPUh와 별도다. 동일 endpoint 중복 forward나 새 평가 없음.
- [realization.csv](realization.csv): mean/canonical/rewrite/KL별 normratio·directionalratio·cosine·error·zero-target leakage의 count/mean/min/max, Q 및 update norm. null은 0으로 바꾸지 않았다. role별 반복 Q를 합산하지 않으며 normratio 하나로 exact realization을 주장하지 않는다.
- [cost.csv](cost.csv): exclusive fit 단계와 inclusive timer를 구분한다.

## 근거·보존

config SHA `26096236ba0fe1a683c98d954904dbf0a048d4611f03cd62b1aef77f7c00091f`, lock SHA `78f4095399905ceb73864507ebee4a00a34b0457507ee67b5dc896013a20e385`. Model revision `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`, fixed first2000 order SHA `0b912d11659eb087ee71a391b7bea1e02ecc9d999a48254eb8559434965640f4`.

- [검산·완료·집계 provenance](../../../../../audits/servers/server4/jlz-interference-priced-l1-2k/w20-review/verification.json)
- [원본 read-set manifest](../../../../../audits/servers/server4/jlz-interference-priced-l1-2k/w20-review/input-manifest.json)
- [이전 W15 보고 보존](../w15/report-ko.md)

원 실행/raw: `/data/janghj/ODE-edit/local/jlz-interference-priced-l1-2k/repair-59721/`. 이번 CPU 결과: `/data/janghj/ODE-edit/local/jlz-interference-priced-l1-2k/w20-review-with-w0/` (W0 표 추가 전 CPU 산출물 `w20-review/`도 보존).

새 GPU/Slurm/모델 forward/실험 수리/재시도 0. noCP/exact_resume=NOT_AVAILABLE. source·compact 집계·manifest만 Git; raw/tensor/prompt/fullstdout는 local KEEP. NO_BROADCAST_NOT_REQUIRED. 기존 job·W&B run 변경 없음, monitoring_active=false/automatic_resume=false.
