# SH1 완료 main 결과 bounded 갱신 검산

nonce `USER-GH-ALL-SH-COMPLETED-MAIN-TABLE-REFRESH-20261009-R1-SERVER1` 수락 후
전용 clean worktree에서 origin/main `99ecdba4`와 main-results-policy 전체를 읽었다.
repo `hyunjun1127/ODE-edit`, SH1 session `01a04939-f93a-7b50-bca0-65438eab2062`.
이 nonce의 기존 own 출력은 없었다. root dirty/frozen source/job/raw/CP는 보존했다.
별도 runner 실행이 아니라 기존 완료 raw의 CPU 검산만 수행했다.

한 번의 exact own job accounting 조회와 local receipt/raw를 사용했다.
snapshot 기록 상한은 2026-10-09 23:05:52 KST (14:05:52 UTC)이며, 이후 상태를 재조회하거나
완료까지 polling하지 않았다. 이 시각은 receipt 기록 시각이며 scheduler의 정확한 상태 전환 시각이 아니다.

## 완료 수치 (단위 %, raw full precision은 JSON/CSV)

| Model / method | dataset / job | Score | Eff | Gen | Loc |
|---|---|---:|---:|---:|---:|
| Llama FT | CF 61771 | 56.586626 | 89.250000 | 73.300000 | 35.500000 |
| Llama SPHERE | CF 61770 | 86.870970 | 99.500000 | 94.950000 | 71.675000 |
| Llama MEMIT-FE | CF 61773 | 64.552174 | 80.450000 | 73.675000 | 48.850000 |
| GPT-J MEMIT_FE_HISTORY (별도 variant) | CF 61927 | 50.175886 | 50.350000 | 49.900000 | 50.280000 |
| Llama FT | zsRE 공개-query 재평가 61932 | — | 14.632401 | 11.816567 | 25.251388 |
| Llama MEMIT | zsRE 공개-query 재평가 61933 | — | 44.300714 | 39.947321 | 22.304913 |
| Llama AlphaEdit-BLUE | zsRE 공개-query 재평가 61935 | — | 95.872262 | 92.282183 | 32.828487 |
| Llama SPHERE | zsRE 공개-query 재평가 61937 | — | 95.126270 | 91.361488 | 31.381720 |

CF FT/SPHERE는 이전 본표와 원값 동일. CF MEMIT-FE는 이전 RUNNING에서 W20 검증완료로 변경 가능하다.
GPT-J history는 native MEMIT-FE 행에 합치지 않고 별도 variant 표에만 반영한다.
모든 CF Flu/Con은 **DEFERRED**, future checkpoint consumer pending이다.
공개-query zsRE는 기존 token-prefix 값 또는 W0 agreement 재명명이 아닌 새 실제 평가 raw다.
이전 raw의 동일 request-macro E/G/loc_ans와 old→new delta도 JSON에 보존했다.
query 정의가 달라진 결과 차이를 편집 성능의 인과적 변화라고 해석하지 않는다.

## 검산 범위

- CF: 정확 frozen runtime의 기존 `audit_factual`로 2,000 요청 / 26,000 prompt pairs /
  52,000 candidates, ordered cohort/token plan, token NLL의 FP32 평균·strict preference·정확도 및
  request-macro E/G/S/Score를 재검산. 원 source/config/COMPLETE/20 commit/cursor/최종CP SHA 검산.
- GPT-J history: 각20 commit의 native request digest, before→after H/W/context linkage,
  history append once 및 successful_calls, 최종 CP fullSHA와 원 factual evaluator 검산.
  raw source는 eaf78c33이며 새 Qwen memory repair로 relabel하지 않았다.
- zsRE: exact eval source b2806a60/CP source61716..21/새config/query proof/orderedcase/완료receipt/
  endpoint fullSHA를 결속했다. 저장된 predicted_token_id==target_token_id를 독립 재계산하여
  case별 평균을 2,000 요청에 macro 평균했다. Eff/Gen 각6,035 token, Loc12,465 token.
  summary 단순 rename 또는 tokenmicro 대체가 아니다. 숫자 GPU parity 주장은 하지 않는다.
- GPU forward/모델 복원/fit/새 Slurm/취소/hold/dep 변경/온라인 history 수정/CP이전삭제 모두0.
  audit 모델 로드는 없으며 tokenizer/소형 CPU query plan만 사용했다.

검산된 완료8행 중 기존 CF2행은 재확인, 나머지6행은 새 완료 점수 반영 대상이다.
zsRE AlphaEdit61934/MEMIT-FE61936은 PENDING으로 평가값 없음.
history Llama61928/Qwen61975는 RUNNING으로 W20 값 없음. 실패61929는 superseded 이력으로 보존.
Llama CF 취소61769/61772의 수치를 만들지 않으며 기존 historical42657/42658,
BLUE39283_1 및 PRICE FREE10060103 예외 행은 그대로 유지한다.
GPT2·tuning·collector·W0를 새 main 결과로 승격하지 않았다.

## 전달

GH sole README integration용 compact rows:
`audits/servers/server1/main-table-refresh-20261009/table-rows.json` 및 `table-rows.csv`.
각 행에 actual job name/ID/state, source/config, raw fullSHA, ordered cohort와 분모를 보존했다.
CPU reducer는 같은 디렉터리 `reduce.py`; CF 기존 검산기는
`audits/servers/server1/official-baselines-20261008/results-review-20261009/review.py`.
SHA는 rows manifest에 결속한다. own README 수정0.
`NO_BROADCAST_NOT_REQUIRED`: same-host raw/CP는 local KEEP, compact 검산결과만 Git/main 공유.
