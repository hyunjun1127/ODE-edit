# MV-2 단일 모델 refresh 독립 분석

- model: `qwen2.5-7b-inst`
- verdict: `DIRECTION_REFRESH_CLEAR` — direction mechanism과 current-method 총 기대효과 모두 명확
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
| direction refresh | 0.00688737631 | 0.00458652973 | 0.00116252899 | 7/12 | [-0.00234320015, 0.0179910754] |
| coefficient refresh | 0.00166490177 | 0.00160557032 | 0.00139379501 | 11/12 | [0.000420762971, 0.00296467741] |
| total refresh (method 기대효과 proxy) | 0.00855227808 | 0.00559936166 | 0.00213193893 | 7/12 | [-0.000304655482, 0.019539207] |

## Refresh opportunity oracle

- 정의: `max(A,B,C) - C`
- outcome-selected upper bound이며 실현 가능한 method 성능으로 해석하지 않는다.
- mean: `0.0114911646`
- 10% trimmed mean: `0.00795847774`
- median: `0.00512933731`

## Generic continuation gain

- 정의: `max(A,B,C) - partial_joint`
- 일반적인 second-half-step 이득이므로 refresh kill을 구제하지 않는다.
- mean: `2.00896092`

## Compute/NFE

- controlled NFE total: `516`
- proposal build total: `48`
- probe panel total: `36`
- wall seconds total: `11564.4`
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

## v2 실행·provenance 및 결과 방화벽

- run: 지정된 Qwen analyzer를 raw SHA-256 검증 뒤 1회 실행했고 exit `0`이었다.
- raw input: `local/results/raw/session01_motivation/mv2refresh_qwen_e0_v1/analysis_cases.jsonl`
- raw SHA-256: `5fee05d7d6c3ec5870bdc3689f5422f38e588adfc23d45ab77401f9a11c5d618`
- Slurm: `15610.1`, `COMPLETED`, exit `0`, elapsed `03:13:40`, MaxRSS `16474488K`.
- 자원 기록: GPU allocated `44776698368`, reserved `47779414016`; host RSS `17354312KiB`; run wall `11617.541s`.
- analyzer JSON은 수정하지 않았다. 이 보강도 analyzer의 수치, verdict, threshold, denominator를 변경하거나 재계산하지 않는다.

### 근거 범주

| 범주 | 이 문서에서의 처리 |
| --- | --- |
| proposal | 사전 고정 분석 계약, hierarchy, threshold 및 해석 경계는 제안된 분석 규칙으로서 기록한다. |
| repo | 지정 raw 파일의 경로·SHA-256과 이번 analyzer의 exit/output은 이 작업 범위에서 직접 확인한 저장소 사실이다. |
| GH 추정 | launcher runtime metadata `model=gpt-5.6-terra`, `reasoning effort=ultra` 및 위 Slurm/자원 상태는 GH가 전달·확인한 runtime 정보로 구분하며, 이 독립 분석이 별도로 추론하거나 검증한 사실은 아니다. |
| 사용자 확인 필요 | launcher metadata와 실제 실행 정체성의 외부 대응, 그리고 이 diagnostic 밖의 운영·배포 판단은 사용자 확인 없이는 확정하지 않는다. |

### 결과 방화벽

- no-peer: 이 보고서는 지정된 MV-2 Qwen raw와 이번 analyzer output만 사용한다. peer 결과, pair audit, 타 모델 또는 다른 raw/result와의 비교·결합은 수행하지 않았다.
- 이 결과는 teacher-forced absolute rewrite utility proxy다. 따라서 accuracy, retention 또는 full-method 성능에 관한 claim이 아니다.
