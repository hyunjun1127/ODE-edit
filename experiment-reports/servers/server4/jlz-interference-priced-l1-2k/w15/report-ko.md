# PRICE 1,500-edit 중간 결과

USER recall: “1500edit까지 완료 되었는데 여기까지 결과를 기준으로 push하자”. 검토 범위는 PRICE B1–B15의 봉인된 commit/관측이다. W20 완료 보고가 아니다.

## 누적 결과

단위 %. R/P는 new NLL < true NLL, N은 true NLL < new NLL, 동률은 실패다. 각 행은 해당 시점까지의 모든 요청 occurrence를 포함한다.

| Endpoint | RS (분자/분모) | PS (분자/분모) | NS (분자/분모) | 조화평균 |
|---|---:|---:|---:|---:|
| W5_ALL_SEEN | 99.800 (499/500) | 92.800 (928/1000) | 87.320 (4366/5000) | 93.029 |
| W10_ALL_SEEN | 99.900 (999/1000) | 92.400 (1848/2000) | 86.370 (8637/10000) | 92.562 |
| W15_ALL_SEEN | 99.867 (1498/1500) | 91.467 (2744/3000) | 85.667 (12850/15000) | 91.970 |

W15 teacher-forced 정확도는 아래와 같다. R/P desired target은 new, N은 true다. 위 preference와 다른 지표이며 free generation 정확도가 아니다.

| 종류 | Strict | Token micro | Prompt macro |
|---|---:|---:|---:|
| R | 99.333 | 99.343 | 99.333 |
| P | 62.500 | 62.911 | 62.883 |
| N | 19.927 | 21.136 | 20.663 |

같은 first500에서 W5→W15: RS 99.8→99.8%, PS 92.8→91.6%, NS 87.32→84.76%. 중간 W10과 first100도 metrics.csv에 있다.

## Paired retention

- 동일 first1500 neighborhood의 W0→W15: 13276→12850 / 15000, 즉 88.5067→85.6667%, −2.8400pp. lost 672 / gained 246 / retained 12604. W0-correct retention 94.9382%.
- 각 birth cohort 편집 직후→W15의 neighborhood: 13057→12850 / 15000, lost 330 / gained 123 / retained 12727.
- 편집 직후→W15 rewrite: 1500→1498 / 1500, lost 2 / gained 0. Rephrase: 2743→2744 / 3000, lost 35 / gained 36.
- pre/post current와 birth cohort별 preference 및 TF-strict paired는 동일 case/prompt/token identity를 검산한 뒤 집계했다. 미측정 값을 0점으로 채우지 않았다.

## 실행·검산 범위

- task: jlz-interference-priced-l1-2k, 실제 arm PRICE. FLAT/REVERSE는 USER 취소 상태 유지, 재제출/조회 없음.
- 실행 source: `0415aba3c160170d306be8196792f198dad4d122`; 본 CPU 리뷰 source는 별도 publication commit이다. production source/launcher/frozen archive를 변경하지 않았다.
- config SHA: `26096236ba0fe1a683c98d954904dbf0a048d4611f03cd62b1aef77f7c00091f`.
- execution lock SHA: `78f4095399905ceb73864507ebee4a00a34b0457507ee67b5dc896013a20e385`.
- Meta-Llama-3-8B-Instruct revision `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`, frozen CounterFact first1500 of first2000, BS100, seed20261002, L4–L8. 원 runtime 기록: torch2.9.1+cu128/transformers4.57.1, NVIDIA RTX PRO 6000 Blackwell Server Edition, FP32 model/FP64 geometry/eager/TF32off/autocastoff.
- 15 fits/15 commits/14 own W-H-RNG-context-ledger joins/75 layer history appends를 확인했다. candidate BUILD 375, subject forward 375, subject backward 360, request-update 참여 36000. 물리 model-call 수와 같은 단위로 합산하지 않는다.
- terminal evaluated payload exact copy/no-resolve/no-double-add, history rewrite-only once/100요청 포함, fresh teacher-anchor-factor/entry-price 고정, FP64 KKT/FP32 cap·shared spend, active/update/expansion counter, 원래 B1 LOO receipt를 scalar로 재검산했다.
- 원본 R/P/N rows의 순서·case/prompt/token identity·유한값·분모·strict token 관계·margin 부호를 독립 stdlib reducer로 재집계하고 저장 summary와 비교했다. 참조한 511개 파일의 SHA를 검토 말미에 다시 확인했다.
- actual M/P/K/weight/activation tensor replay나 모델 forward는 하지 않았다. 원 receipt 기반 검증과 새 실제 GPU 검증은 구분한다. 별도 독립 reviewer 미사용(owner audit).
- 초기 리뷰 formatter가 mean/canonical 배열형 telemetry를 scalar로 취급하여 TypeError를 냈다. 리뷰 전용 집계에서 null을 제외하고 배열을 펼치도록 교정했다. 실험 실패나 production 수정이 아니며 원 raw는 불변이다.

## 비용·측정 요약

- B1–B15 batch-inclusive 합 13385.072초 (3.7181시간).
- fit-inclusive 7606.177초, writer-inclusive 1580.281초, pre observer 726.808초, post observer 2019.500초. nested/inclusive 항목을 중복 가산하지 않는다.
- fit 안 BUILD 1139.573초, subject 5758.748초, pullback 38.144초. 나머지 optimizer/projection/telemetry/price/LOO 항목은 cost.csv에 있다.
- 부모 job 한정 1회 snapshot: PRICE59768 RUNNING, elapsed13764초/GPU1. CPU59769 PENDING/GPU0. 이 부모 elapsed는 B16 이후 작업을 포함할 수 있어 W15 전용 비용으로 표시하지 않는다. job step 중복 계상/반복 polling 없음.
- W0는 기존 exact-identity 관측을 참조해 재사용: 새 forward 0 / 새 평가시간 0초. 원 job59721의 W0 관측930.845초는 repair 비용에 다시 가산하지 않는다.
- realization.csv는 batch×layer×mean/canonical/rewrite/KL별 normratio/directionalratio/cosine/relative error/zero-target leakage의 count·mean·min·max와 Q/update norm이다. null 비율을 0으로 채우지 않았고 normratio만으로 exact realization을 주장하지 않는다. Q는 같은 batch/layer 값이 role마다 반복되므로 role 간 합산하지 않는다. token/normratio 표본별 차이는 추정 tensor로 보완하지 않았다.

## 산출물·한계

- [전체 endpoint metric CSV](metrics.csv)
- [paired 및 birth cohort CSV](paired-cohorts.csv)
- [층별 실현 요약 CSV](realization.csv)
- [비용 CSV](cost.csv)
- [검산/로컬 산출물 SHA](../../../../../audits/servers/server4/jlz-interference-priced-l1-2k/w15-review/verification.json)
- [원본 read-set manifest](../../../../../audits/servers/server4/jlz-interference-priced-l1-2k/w15-review/input-manifest.json)

원자료: `/data/janghj/ODE-edit/local/jlz-interference-priced-l1-2k/repair-59721/PRICE/`. CPU 결과: `/data/janghj/ODE-edit/local/jlz-interference-priced-l1-2k/w15-review/`.

B16–B20 결과/현재 terminal/실패 파일은 이번 리뷰에서 읽지 않았다. W20은 NOT_REVIEWED_NOT_CLAIMED. noCP/exact_resume=NOT_AVAILABLE. 기존 runner/collector는 미변경, 새 GPU/Slurm/모델평가/자동retry 0. 능동 모니터링은 다시 중단한다. Git에는 compact 집계/source/manifest만, 원 raw/tensor/prompt/fullstdout는 local KEEP. NO_BROADCAST_NOT_REQUIRED.
