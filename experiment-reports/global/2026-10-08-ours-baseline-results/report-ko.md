# OURS와 baseline 중간 결과 2026년 10월 8일

2026-10-08T00:35:58.776646+09:00까지 서버별로 읽은 시점의 snapshot이다. 현재 세 모델 PRICE 18개 arm, 취소된 Qwen 6개 arm, 이전 baseline 12개 arm, 생성 평가를 추가한 baseline 18개 arm의 총54개 실행 구성을 구분한다. 실행 중인 결과의 한정 관측이며 이후 자동 갱신하지 않는다.

0.5K·1K·1.5K·2K 누적 평가 70개 지점(OURS 48, baseline 22)이 확보됐다. 표의 값은 RS / PS / NS (%)이며, 미산출을 0으로 대체하지 않았다. 각 endpoint의 분모는 R=N, P=2N, N=10N이다. 같은 이름의 기존 baseline과 generation 재실험은 별도 실행이며 값을 이어붙이지 않는다.

## OURS 누적 성능

| 모델 | Arm | 0.5K | 1K | 1.5K | 2K |
|---|---|---|---|---|---|
| GPT2-XL | MEMIT_CAP075 | 96.80 / 90.80 / 70.68 | 95.80 / 88.70 / 64.12 | 95.53 / 87.00 / 62.04 | 94.95 / 86.40 / 60.35 |
| GPT2-XL | MEMIT_CAP100 | 97.40 / 91.00 / 70.90 | 95.80 / 88.85 / 63.77 | 95.00 / 87.63 / 61.31 | 94.85 / 87.03 / 59.43 |
| GPT2-XL | MEMIT_FREE100 | 97.40 / 90.90 / 70.90 | 95.60 / 88.85 / 62.98 | 95.20 / 87.40 / 61.01 | 93.95 / 86.45 / 59.02 |
| GPT2-XL | ALPHAEDIT_CAP075 | 99.40 / 95.20 / 72.46 | 98.70 / 93.60 / 66.34 | 97.20 / 91.10 / 63.21 | 96.15 / 89.83 / 60.46 |
| GPT2-XL | ALPHAEDIT_CAP100 | 99.00 / 95.80 / 71.06 | 95.00 / 88.45 / 64.12 | 94.40 / 86.73 / 61.48 | 91.75 / 82.25 / 58.53 |
| GPT2-XL | ALPHAEDIT_FREE100 | 99.00 / 95.80 / 71.20 | 96.20 / 92.25 / 64.03 | 94.00 / 88.13 / 59.81 | 91.45 / 85.20 / 57.71 |
| Llama3 | LLAMA_CAP075 | 100.00 / 92.50 / 87.10 | 100.00 / 92.60 / 86.44 | 99.93 / 91.80 / 85.66 | 99.80 / 91.18 / 84.94 |
| Llama3 | LLAMA_CAP100 | 100.00 / 94.10 / 86.68 | 100.00 / 95.05 / 84.51 | 99.87 / 93.37 / 82.70 | 99.80 / 93.03 / 80.28 |
| Llama3 | LLAMA_FREE100 | 100.00 / 93.80 / 86.74 | 99.90 / 93.90 / 85.09 | 99.67 / 93.43 / 83.84 | 99.70 / 92.78 / 82.21 |
| Llama3 | LLAMA_AE_CAP075 | 100.00 / 95.50 / 83.30 | 99.90 / 95.85 / 78.94 | 99.80 / 95.97 / 76.79 | 99.75 / 96.33 / 75.01 |
| Llama3 | LLAMA_AE_CAP100 | 미산출 | 미산출 | 미산출 | 미산출 |
| Llama3 | LLAMA_AE_FREE100 | 미산출 | 미산출 | 미산출 | 미산출 |
| GPT-J | MEMIT_CAP075 | 99.60 / 93.50 / 79.58 | 75.40 / 69.50 / 62.79 | 83.40 / 76.50 / 60.71 | 미산출 |
| GPT-J | MEMIT_CAP100 | 99.60 / 95.30 / 78.88 | 91.50 / 84.80 / 64.62 | 92.47 / 85.20 / 62.99 | 미산출 |
| GPT-J | MEMIT_FREE100 | 미산출 | 미산출 | 미산출 | 미산출 |
| GPT-J | ALPHA_CAP075 | 98.20 / 92.10 / 74.14 | 96.30 / 89.35 / 64.93 | 미산출 | 미산출 |
| GPT-J | ALPHA_CAP100 | 미산출 | 미산출 | 미산출 | 미산출 |
| GPT-J | ALPHA_FREE100 | 미산출 | 미산출 | 미산출 | 미산출 |

## 이전 baseline 누적 성능

아래는 생성 평가 추가 전 실행이다. 저장된 R/P/N 성능을 게시하며 fluency·consistency 측정으로 표시하지 않는다.

| 모델 | Method | 0.5K | 1K | 1.5K | 2K |
|---|---|---|---|---|---|
| GPT2-XL | BASE_MEMIT | 95.40 / 89.60 / 71.86 | 94.90 / 86.80 / 66.11 | 95.13 / 85.80 / 63.06 | 93.45 / 84.00 / 60.20 |
| GPT2-XL | BASE_ALPHAEDIT | 99.40 / 96.70 / 71.70 | 99.40 / 96.20 / 67.86 | 99.40 / 94.83 / 66.62 | 99.10 / 94.35 / 65.01 |
| GPT2-XL | CAKE | 99.20 / 96.20 / 72.26 | 99.30 / 95.40 / 68.90 | 99.33 / 94.63 / 67.55 | 99.35 / 93.93 / 65.72 |
| GPT2-XL | ALPHAEDIT_BLUE | 99.40 / 94.40 / 73.20 | 99.50 / 93.85 / 70.65 | 미산출 | 미산출 |
| GPT2-XL | PRUNE | 미산출 | 미산출 | 미산출 | 미산출 |
| GPT2-XL | RECT | 미산출 | 미산출 | 미산출 | 미산출 |
| GPT-J | BASE_MEMIT | 99.40 / 95.80 / 75.70 | 99.10 / 95.15 / 69.84 | 98.47 / 94.90 / 65.37 | 97.65 / 94.73 / 61.09 |
| GPT-J | BASE_ALPHAEDIT | 99.60 / 96.50 / 79.16 | 99.60 / 96.25 / 76.98 | 99.53 / 95.83 / 75.95 | 99.65 / 95.48 / 74.28 |
| GPT-J | CAKE | 미산출 | 미산출 | 미산출 | 미산출 |
| GPT-J | ALPHAEDIT_BLUE | 미산출 | 미산출 | 미산출 | 미산출 |
| GPT-J | PRUNE | 미산출 | 미산출 | 미산출 | 미산출 |
| GPT-J | RECT | 미산출 | 미산출 | 미산출 | 미산출 |

## 생성 평가 재실험과 보존 결과

생성 평가 재실험의 편집 후 누적 지표와 완료된 generation endpoint는 이 snapshot에서 미산출이다. GPT2-XL의 개별 생성 관측 파일은 작성 중이며, 그 수만 inventory에 보존했다. 전체 cohort 점수로 평균 내거나 완료된 endpoint로 취급하지 않는다. W0에서 확보된 R/P/N은 metrics.csv에 별도 scope로 남겼다.

취소된 Qwen의 보존된 B1 및 W0 결과도 metrics.csv에 포함하며, 새 실행 또는 재개를 뜻하지 않는다. GPT-J OURS의 terminal TECHNICAL_BLOCKED와 완료 전 결과를 inventory에서 구분한다. terminal이 없는 항목은 NOT_RECORDED이며 이를 RUNNING으로 추정하지 않는다.

## 파일과 검산 범위

- [전체 관측 지표](metrics.csv): W0, 각 batch current/pre·current/post, 실제 milestone all_seen/post. NLL·TF token/prompt/strict 집계도 포함한다.
- [누적 지표와 분모](milestones.csv), [실행 상태 및 미산출](inventory.csv).
- [Batch 비용 및 source/config](batch-cost-provenance.csv), [층별 가격 분포와 최저가 층](price-distributions.csv), [종료 상태 및 예산 확장](fit-terminal-expansion.csv).
- [원자료 경로·SHA256](source-manifest.csv), 서버별 snapshot JSON. 원 prompt·생성문·토큰열·event stream·checkpoint는 각 서버 local에 보존한다.

이 게시 검산은 endpoint/cohort 크기, R/P/N 분모, numerator/rate 산술, finite 값, 원 observer의 no_mutation 표시 및 읽은 파일 SHA를 확인했다. 기존 summary를 사용한 게시 검산이며 모든 raw row의 독립 재평가나 method 정합 인증은 아니다. 비용은 원 receipt의 의미를 보존하며 중첩 시간을 합산하지 않는다. 가격 최저층 집계는 동률을 모두 포함하므로 요청수보다 합이 클 수 있다.

모델별 effective hparams/source와 이전 baseline의 writer·history 정책 차이는 원 실행 보고서에 따른다. 이 표만으로 완전히 통제된 인과 비교를 주장하지 않는다. 기존 Llama 10k baseline 게시 자료는 [통합 리뷰](../../servers/server4/blue-native-lifelong-comprehensive-review-2026-09-11-v1/diagnostic-report-ko.md), MEMIT-history는 [완료 리뷰](../../servers/server3/memit-history-fixed10k-20260928-v1/completion-review-r1/report-ko.md)에 보존되어 있다.

오프라인 표 재생성: `python3 audits/global/2026-10-08-ours-baseline-results/build_report.py`. 원 실험 코드·job·대형 raw를 변경하지 않았다. 수집 스크립트는 별도 read-only snapshot용이며 재실행 시 이미 저장한 서버 snapshot을 보존한다.
