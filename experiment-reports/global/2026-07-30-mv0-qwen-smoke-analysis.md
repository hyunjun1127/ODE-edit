# MV-0 Qwen 1-case smoke 독립 결과 분석

- 분석일: 2026-07-30
- 분석 역할: 실행에 참여하지 않은 result-analysis agent
- 대상 run: `mv0_qwen_smoke_v1`
- 대상 model: `qwen2.5-7b-inst`
- 대상 case: `18447` 한 건
- 최종 판정: **PASS**

이 판정은 Qwen 1-case implementation-fidelity/trace-neutrality smoke에만
유효하다. Qwen의 canonical 3-case MV-0, 양 model 전체 MV-0, MV-1 진입,
research motivation 또는 ODE-Edit의 성능·기제 claim을 승인하지 않는다.

## 1. 분석 경계

판정에는 다음 네 파일과 별도로 제공된 Scheduler fact만 사용했다.

- `local/results/raw/session01_motivation/mv0_qwen_smoke_v1/manifest.json`
- `local/results/raw/session01_motivation/mv0_qwen_smoke_v1/summary.json`
- `audits/global/2026-07-30-session01-mv0-qwen-execution-preflight.md`
- `plans/global/2026-07-30-session-01-motivation-validation.md`

`events`, `direct_z`, raw prompt/context, log, 실행 code와 다른 report는 읽지
않았다. 따라서 아래의 firewall과 trace 판정은 허용된 compact artifact가
표현하는 범위에 한정된다. 특히 summary에 기록된 `events_sha256`의 원문
재계산은 이 분석의 허용 범위 밖이다.

## 2. 판정표

| 점검 항목 | 관찰 | 판정 |
| --- | --- | --- |
| run/model identity | manifest와 summary가 모두 run `mv0_qwen_smoke_v1`, model `qwen2.5-7b-inst`를 기록한다. model/tokenizer revision은 모두 pinned Qwen revision `a09a35458c702b33eeacc393d103063234e8bc28`이다. | PASS |
| provenance/selection identity | 두 artifact의 provenance ID `d247b27edf638b7a9625954e8bf2aeac745b61247230f8bcf733e991a3ed9d16`와 selection manifest ID `186372487e0a79d64a4895dfa5ad142a5e9b1dd3f675c9a3468205147c38c4ce`가 일치한다. source는 21,919 rows, SHA256 `d017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f`로 canonical plan과 같다. | PASS |
| context/request identity | manifest context manifest ID와 summary `context_id`가 `e0c5f61d874334a2cab26f82fb3594d9ab88c9e5816be390274a0bb5e98c93bd`로 같다. 선택 case는 `18447` 한 건이고 request ID도 hash 한 건이다. | PASS |
| manifest hash binding | 실제 manifest SHA256을 재계산한 값은 `2f80784481c3451dcda991ce3b13fcfcbaa6149a581e4d98be33f2db12e3fc06`이며 summary의 `artifacts.manifest_sha256`과 정확히 같다. 참고로 분석 시점 summary SHA256은 `734c532285012196c708469dc103396b036199ae3b04087aee1a3550fdf083d3`이다. | PASS |
| fixed runtime/artifact identity | manifest는 tracked-clean ODE-Edit commit `598c5366bf4e355c4496e1b55b5b14043e5322bc`, Qwen hparams 및 fixed source/stats/projector hash·size, offline/float32/one-visible-GPU 상태를 기록한다. summary의 covariance load는 pinned layer 4–8 moments 5개이고 projector load는 0개다. | PASS |
| Slurm job-artifact binding | manifest와 summary 모두 `under_slurm=true`, job ID `15503`, job name `odeedit_mv0_qwen_smoke`, node `devbox`를 기록한다. 제공된 Scheduler fact도 job `15503`, `COMPLETED`, exit `0:0`이다. preflight가 허용한 exact job/run/node와 일치한다. | PASS |
| denominator | planned/attempted/case/pass가 모두 `1/1/1/1`이며 failure `0`, abort로 미실행 `0`, abort failure type `null`이다. case `18447`은 `attempted`이면서 pass다. 사전 선택된 한 건이 결과 후 denominator에서 빠진 흔적이 없다. | PASS |
| fidelity/trace-neutrality compact metric | `all_pass=true`; max final-weight relative L2 error, max teacher-logits relative L2 error, max teacher-NLL absolute error가 모두 정확히 `0.0`이다. 허용된 compact metric 표현에서는 native/adapter 및 trace-neutrality 차이가 관찰되지 않는다. | PASS |
| rollback | `all_rollbacks_exact=true`이고 final-weight relative L2 error도 `0.0`이다. | PASS |
| compact artifact firewall | manifest는 `case IDs/hashes/metrics only; no raw edit or eval fields`, `raw_templates_persisted=false`를 선언한다. 실제로 검토한 manifest/summary에는 raw prompt, target, context 문자열, paraphrase/locality/downstream evaluation field가 없고 request는 hash로만 남는다. `git_output_written=false`다. | PASS |
| no-recompute/direct-z policy | manifest는 covariance를 verified read-only로 두고 recompute/download를 차단하며 direct-z를 local run에서 한 번 계산해 freeze했다고 기록한다. summary는 direct-z artifact count `1`, covariance 5개 load, projector 0개 load를 기록한다. | PASS |

## 3. Scheduler와 자원

제공된 allocation은 GPU 1, CPU 8, host memory `65000M`이며 preflight의 exact
envelope와 같다. manifest와 summary의 visible GPU count는 1이고 GPU는
NVIDIA RTX A6000이다.

- Scheduler: elapsed `00:04:06`, MaxRSS `15463312K`
- summary: wall `240.8129 s`, host max RSS `17315736 KiB`
- summary GPU peak allocated: `43,809,407,488 B` (physical total의 약 86.07%)
- summary GPU peak reserved: `46,800,044,032 B` (약 91.95%)
- reserved 기준 GPU 여유: 약 3.82 GiB

Scheduler elapsed와 in-run wall의 차이는 약 5.19초이며 job wrapper/초기화
구간을 포함할 수 있는 일관된 크기다. Scheduler MaxRSS와 summary host
max RSS는 `1,852,424 KiB`(약 1.77 GiB) 차이가 난다. 두 계측의 집계
정의 차이일 수 있으나, 허용 자료만으로 원인을 단정하지 않는다. 다만 두 값
모두 `65000M` allocation보다 충분히 낮고 Scheduler가 `COMPLETED/0:0`이므로
이 차이는 이 smoke의 resource gate를 차단하지 않는다. GPU reserved peak가
physical memory의 약 92%로 높다는 점은 후속 규모 확장 시 별도 capacity
gate에서 다시 확인해야 한다.

## 4. 결론과 범위 제한

Exact preflight의 smoke 성공 조건인 Scheduler `COMPLETED/0:0`,
planned/attempted/pass `1/1/1`, `all_pass=true`, exact rollback,
compact native-adapter/trace error `0.0`, compact artifact firewall이 모두
충족됐다. 따라서 **Qwen case `18447` 한 건의 implementation-fidelity /
trace-neutrality smoke는 PASS**다.

다음은 이 PASS에서 도출할 수 없다.

- Qwen deterministic 3-edit canonical MV-0 완료
- Llama와 Qwen을 합친 양-model MV-0 완료
- statistical equivalence CI 또는 calibration bound 충족
- MV-1 이상의 routing/non-stationarity/rerouting/retention 가설 지지
- research motivation, ODE naming 또는 성능 개선 claim 지지

Case 확장과 다음 stage에는 canonical plan 및 preflight가 요구한 별도
post-run red gate와 새 실행 gate가 필요하다. 이 보고서의 PASS 자체는 그
확장을 자동 승인하지 않는다.
