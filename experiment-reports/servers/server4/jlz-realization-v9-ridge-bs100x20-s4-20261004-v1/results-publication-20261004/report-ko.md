# JLZ v9 2k — 저장 결과 게시 (2026-10-04)

사용자 직접 요청: “v11의 중간 결과, v9의 2000edit 모두 main에 push해”.
이 side conversation은 저장 결과만 CPU 검산·게시한다. 새 평가/fit/GPU/submit/cancel/retry는 없다.
원 제출 보고, 실행 source, 실패 기록과 raw는 보존한다.

## 완료 범위와 scheduler 상태

| 경로 | Job | 저장 commit | own-state join | W20 평가 | 기존 accounting |
|---|---:|---:|---:|---|---|
| A | 57899 | 20 | 19 | 2000요청 / 26000행 저장 | FAILED, exit 0:11 |
| B | 57900 | 18 | 17 | 없음 | CANCELLED by 1025 |

A runner의 terminal은 COMPLETED이며 W20 평가가 저장돼 있다. 이것과 Slurm FAILED는 다른 사실이다.
이번 게시에서 SIG11 원인을 새로 진단하지 않았으며 원인 NOT_IDENTIFIED다.
B의 1800-edit 부분 결과를 W20으로 표시하지 않는다. 두 arm 전체 상태는
PARTIAL_OR_TECHNICAL_FAILED이고 COMPLETED_2000_BOTH가 아니다.
[원 collector 요약](summary.json), [원 terminal](terminal.json), [scheduler 구분](scheduler-outcomes.json).

## W20: 같은 first2000 평가

RS는 rewrite new NLL < true NLL, PS는 paraphrase의 같은 기준,
NS는 neighborhood true NLL < new NLL이다. 동률은 실패다. 단위 %.

| 방법 | RS /2000 | PS /4000 | NS /20000 |
|---|---:|---:|---:|
| V9 A | 94.850 | 87.225 | 60.135 |
| BASE_ALPHAEDIT | 99.300 | 93.225 | 68.590 |
| AlphaEdit-BLUE L4+L8 | 99.600 | 97.150 | 76.585 |
| CAKE | 99.150 | 87.750 | 76.405 |
| MEMIT-H | 99.400 | 91.000 | 79.045 |
| BASE_MEMIT (history 없음) | 64.750 | 61.700 | 51.825 |

[추가 역사 비교표](comparison-W20-with-historical.csv)는 BLUE/CAKE의 기존 W20 공개 집계를 보완한다.
[원 collector 비교표](comparison-W20.csv)는 그대로 보존하며 그 BLUE NOT_AVAILABLE은
당시 collector 입력 상태다. 그 표의 CURRENT_MATCHED_A_B는 A/B 내부 비교 조건 label일 뿐,
B의 W20 존재를 의미하지 않는다.

모두 2000 edit 시점 first2000 분모이며 W100의 first2000 부분집합이 아니다.
역사 baseline은 같은 runtime/평가기 parity를 새로 증명한 matched 실험이 아니다.
BLUE L4+L8/L2=1, CAKE 설정, v9 L4–L8 및 실행 환경 차이를 보존한다.
MEMIT-H는 S3 H200/transformers4.44.2, 이번 v9는 S4/4.57.1이다.
BLUE/CAKE는 공개 aggregate 재사용이고 이번에 원 raw를 다시 전수 검산하지 않았다.
새 baseline fit/probe는 0이다.

## 시점별 저장 값

| V9 A 누적 endpoint | RS % | PS % | NS % |
|---|---:|---:|---:|
| W5 first500 | 99.60 | 94.90 | 82.86 |
| W10 first1000 | 99.20 | 92.95 | 71.36 |
| W20 first2000 | 94.85 | 87.225 | 60.135 |

동일 first500의 W20 값은 RS446/500, PS752/1000, NS3055/5000이다.
W20 전체의 at-write 합산→W20은 R1996→1897/2000, P3860→3489/4000,
N14469→12027/20000이다. At-write 합산은 서로 다른 endpoint이며 하나의 모델 평가가 아니다.
W0 N17711→W20 N12027/20000, lost6378/gained694는 기존 paired reducer 기록이다.
전체 paired/cohort 집계는 [paired-cohorts.json](paired-cohorts.json)에 있다.

## 게시 산출물과 검산

- [metrics.csv](metrics.csv): A/B의 저장된 current/all-seen/first500/active 등 분모와 NLL·TF.
- [candidate-summary.csv](candidate-summary.csv): 저장된 후보별 scalar 요약.
- [terminal-realization.csv](terminal-realization.csv): 층별 gamma/norm ratio/rho/planned·realized share/M diagonal.
- [tails.json](tails.json), [cost.json](cost.json), [baseline provenance](baselines.json).
- A allocated GPU 36710초, B32330초: 기존 accounting 재사용. 활용률이나 순수 fit 시간은 아니다.
- CPU 독립 재집계: A W20 26000행, case/kind/prompt 중복·누락·비유한 검사,
  preference/desired TF strict/token 분모가 저장 summary 및 게시 comparison과 일치.
- 이것은 저장 raw 검산이며 새 GPU numerical qualification 또는 독립 reviewer PASS가 아니다.

## Source·입력·원자료 보존

실행 source ab6f1bda4688ffc275dd5ff47649899d5a80093c,
config SHA633b8e27a063e6518bc6e7df0da5dcb65029146a9f01ecf5e2c51ca093090f6b.
모델 Llama3-8B-Instruct revision8afb486c1db24fe5011ec46dfbe5b5dccdb575c2,
CounterFact fixed10k first2000, seed20261002, BS100×20/arm, 별도 cold W0/H0.
설계는 [2k 계약](../../../../../plans/global/2026-10-04-jlz-realization-v9-ridge-2k/contract.json).

Local 원본:
`/data/janghj/ODE-edit/local/jlz-realization-v9/20261004-2k-v1/attempt-r2/`.
실행 code는 기존 main namespace에 이미 게시되어 있으며 이번 diff는 과학 code를 바꾸지 않는다.
83MB terminal-context-decomposition.json 및 full artifact-index.json은 원 local에 KEEP,
SHA/size는 [입력 manifest](../../../../../audits/servers/server4/v9-v11-result-publication-20261004/input-manifest.json)에 기록한다.
원 raw/tensor/prompt/fullstdout/model/checkpoint는 Git0.
NO_BROADCAST_NOT_REQUIRED: 사용자 요청은 Git 소형 산출물 게시이며 대형 전송은 하지 않는다.
monitoring_active=false / automatic_resume=false; 실험 프로그램과 기존 monitoring 정책을 변경하지 않는다.
