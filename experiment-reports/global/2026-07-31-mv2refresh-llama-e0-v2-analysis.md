# MV-2 단일 모델 refresh 독립 분석

- model: `llama3-8b-inst`
- verdict: `PIVOT_FIXED_DIRECTION_DYNAMIC_COEFFICIENT` — fixed-direction dynamic coefficient로 전환
- technical validity: `PASS`
- ITD denominator: `12` (실패 case `0`건 포함)

## 고정 분석 계약

- direction effect: `progress(A) - progress(B)`
- coefficient effect: `progress(B) - progress(C)`
- total refresh 기대효과: `progress(A) - progress(C)`
- 실패 case는 사후 제외하지 않고 세 contrast에 `0`을 기여한다.
- replay+sham envelope `e=0.0001`
- outcome-blind practical floor: `0.0001`
- bootstrap: seed `20260801`, resamples `4000`
- hierarchy: direction이 유일한 primary이고 coefficient는 direction 실패 시 conditional, total은 secondary 기대효과다.

## 효과

| contrast | mean | 10% trimmed mean | median | positive sign | paired bootstrap mean 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: |
| direction refresh | -0.00440808137 | -0.00347514153 | -0.00248718262 | 0/12 | [-0.00739453137, -0.00216433505] |
| coefficient refresh | 0.000103672345 | 5.34534454e-05 | 3.86238098e-05 | 12/12 | [3.48478556e-05, 0.00022093455] |
| total refresh (method 기대효과 proxy) | -0.00430440903 | -0.00342168808 | -0.00244069099 | 0/12 | [-0.0071797053, -0.00213002761] |

## Refresh opportunity oracle

- 정의: `max(A,B,C) - C`
- outcome-selected upper bound이며 실현 가능한 method 성능으로 해석하지 않는다.
- mean: `0.000103672345`
- 10% trimmed mean: `5.34534454e-05`
- median: `3.86238098e-05`

## Generic continuation gain

- 정의: `max(A,B,C) - partial_joint`
- 일반적인 second-half-step 이득이므로 refresh kill을 구제하지 않는다.
- mean: `0.480594416`

## Compute/NFE

- controlled NFE total: `516`
- proposal build total: `48`
- probe panel total: `36`
- wall seconds total: `8676.62`
- W1 controller incremental cost/case (diagnostic outcome forward 제외):
  - A: proposal build `1`, probe NFE `12`
  - B: proposal build `0`, probe NFE `12`
  - C: proposal build `0`, probe NFE `0`

## 기술 gate 실패

- 없음

## 해석 경계

- 이 문서는 한 모델의 사전 고정된 mechanism diagnostic만 판정한다.
- total refresh는 equal-C continuation의 absolute utility proxy이며 downstream 성능 예측이 아니다.
- secondary metric 하나나 bootstrap endpoint 하나만으로 핵심 판정을 뒤집지 않는다.
- technical gate 실패 시 scientific signal과 무관하게 판정을 차단한다.

## 실행·provenance 및 결과 방화벽

- raw input: `local/results/raw/session01_motivation/mv2refresh_llama_e0_v1/analysis_cases.jsonl`
- raw SHA-256(실행 전 확인): `c5c698ee7828000e4ecc0c5c15b86f0dca9ba014dbb5b5f678c59f5ab7d23e38`
- 이 분석은 `llama3-8b-inst` alias로 analyzer를 1회 실행해 exit `0`을 얻었다. JSON 산출물은 analyzer가 쓴 그대로이며, 본 문서에는 수치 재계산이나 판정 변경을 하지 않았다.
- GH 제공 Slurm 실행 메타데이터: `15610.0`, `COMPLETED`, exit `0`, elapsed `02:25:16`, MaxRSS `10196536K`; GPU allocated `40718852608` / reserved `43526389760`; host RSS `10761128KiB`; wall `8713.666184685193s`.
- GH가 확인한 launcher runtime metadata는 model `gpt-5.6-terra`, reasoning effort `ultra`이다. 이는 analyzer가 독립 조회한 값이 아니라 GH 제공 provenance로만 기록한다.

### 근거 상태 분류

- **Proposal:** `PIVOT_FIXED_DIRECTION_DYNAMIC_COEFFICIENT`는 이 단일-model diagnostic에서의 다음 실험 방향 제안이다.
- **Repo:** 위 raw hash와 analyzer JSON/Markdown 산출물은 이 workspace의 재현 가능한 증거다.
- **GH 추정:** launcher·Slurm·GPU/host 자원 메타데이터는 GH 제공 실행 기록이며, 본 분석에서 독립 재검증하거나 수치 해석하지 않았다.
- **사용자 확인 필요:** 이 기록을 외부 스케줄러 원장, 배포 환경, 또는 다른 실행과 연결하는 주장은 이 보고서 범위 밖이므로 사용자 확인이 필요하다.

### 결과 격리와 주장 범위

- **result firewall / no-peer:** Qwen, 기존 v1/v2 attempt report, pair audit, 그리고 다른 raw/result는 접근·비교하지 않았다.
- 측정값은 teacher-forced absolute rewrite utility proxy다. 따라서 accuracy, retention, 또는 full-method 성능에 관한 claim이 아니다.
