# MV-2 Qwen refresh 독립 분석 — 기술 차단 보고

- 모델: 'qwen2.5-7b-inst'
- run ID: 'mv2refresh_qwen_e0_v1'
- 최종 상태: 'REJECTED_INPUT_SCHEMA' / technical validity 'BLOCK'
- 과학 판정: 미발행 ('NOT_EMITTED')
- 범위: Qwen 단일 모델의 사전등록 MV-2 mechanism diagnostic. Pair/cross-model 결론은 내리지 않는다.

## 잠긴 분석기 실행과 차단 근거

offline 실행에서 고정 분석기
'project/run_scripts/ode_edit_motivation/mv2_refresh_analysis.py'를 사전등록된
'qwen2.5-7b-inst' 인자와 변경 불가 상수(12 cases, bootstrap seed
'20260801', 4,000 resamples, practical floor '0.0001', matched-C
'rtol=3e-5', 'atol=3e-5')로 실행했다. 분석기는 결과 파일을 쓰기 전에
exit '2'로 fail-closed 했다.

    records[0].technical: exact key/order 위반
    expected = (exact_panel, lineage_exact, matched_second_c, rollback_exact,
                firewall_pass, receipt_before_outcome)
    actual   = (exact_panel, firewall_pass, lineage_exact, matched_second_c,
                receipt_before_outcome, rollback_exact)

12개 row 모두 같은 'technical' key order를 갖는다. 또한 raw JSONL에서
'lineage'와 'compute'도 analyzer가 요구하는 insertion order와 다른
alphabetical order로 기록돼 있다. 이 보고서는 raw를 재직렬화하거나
field-order를 사후 정규화하지 않았다. 따라서 이는 분석기 내부의
'BLOCK_TECHNICAL_INVALID' verdict도 아닌, 그 verdict에 도달하기 전
입력-schema 거부다.

## 고정 estimand와 미산출 상태

| 항목 | 고정 정의 | 잠긴 분석기 결과 |
| --- | --- | --- |
| 'Delta_direction' | 'A-B' | 미산출 |
| 'Delta_coefficient' | 'B-C' | 미산출 |
| 'Delta_total' | 'A-C' | 미산출 |
| 'e_m' | 'max_i(max(1e-4, abs(no_op), abs(sham-no_op)))' | 미산출 |
| oracle refresh opportunity | 'max(A,B,C)-C' | 미산출 |
| generic continuation gain | 'max(A,B,C)-partial_joint' | 미산출 |
| paired-bootstrap mean 95% CI | seed '20260801', 4,000 paired resamples | 미산출 |

사후 threshold 변경, field-order 정규화, retune, subset 교체, 또는
실패 case 제외로 이 차단을 구제하지 않았다. 'Delta_direction'은 유일한
primary, 'Delta_coefficient'는 conditional pivot, 'Delta_total'은 secondary
current-method 기대효과라는 hierarchy도 그대로 유지한다.

'A-C'가 산출되었더라도 이는 equal-C continuation에서 teacher-forced allowed
rewrite utility의 absolute progress일 뿐이다. accuracy, retention,
full-method 성능, downstream 성능, 또는 paper claim으로 해석하지 않는다.
Oracle은 outcome-selected upper bound이고, generic continuation gain은
refresh kill을 구제할 수 없다.

## NFE·wall·controller cost

- 잠긴 분석기는 aggregation 전 거부됐으므로 formal controlled-NFE total과
  case-level effect summary는 발행되지 않았다.
- 사전등록된 case별 contract: controlled NFE '43', proposal build '4',
  probe panel '3'.
- W1 controller incremental cost/case (diagnostic outcome forward 제외):
  - A (refreshed direction + refreshed coefficient): proposal build '1', probe NFE '12'
  - B (fixed direction + refreshed coefficient): proposal build '0', probe NFE '12'
  - C (fixed direction + fixed coefficient): proposal build '0', probe NFE '0'
- runner wall time: '11617.540922372602' seconds.

## Slurm 및 resource accounting

| source | field | recorded value |
| --- | --- | --- |
| runner summary | GPU peak allocated | '44776698368' bytes |
| runner summary | GPU peak reserved | '47779414016' bytes |
| runner summary | host max RSS | '17354312' KiB |
| runner summary | wall seconds | '11617.540922372602' |
| Slurm '15610.1' | state / exit | 'COMPLETED' / '0:0' |
| Slurm '15610.1' | elapsed | '03:13:40' |
| Slurm '15610.1' | allocated TRES | 'cpu=8,gres/gpu=1,mem=65000M,node=1' |
| Slurm '15610.1' | requested TRES | empty / not reported |
| Slurm '15610.1' | MaxRSS | '16474488K' |
| Slurm '15610.1' | node | 'devbox' |

Raw runner summary는 'completed', 'all_pass=true', 'all_rollbacks_exact=true',
'expected_counts_exact=true', 'failure_count=0'을 기록한다. 그러나 이 값들은
locked analyzer의 exact input-schema acceptance를 대체하지 않으므로
technical block을 해제하지 않는다.

## Provenance와 네 범주

- provenance ID: 'd247b27edf638b7a9625954e8bf2aeac745b61247230f8bcf733e991a3ed9d16'
- context manifest ID: 'e0c5f61d874334a2cab26f82fb3594d9ab88c9e5816be390274a0bb5e98c93bd'
- selection manifest ID: '47bb14e437ce193397dd4650a1bb2e98d5f48ba10821dc24c3e2723712741b66'
- 'analysis_cases.jsonl' SHA-256:
  '5fee05d7d6c3ec5870bdc3689f5422f38e588adfc23d45ab77401f9a11c5d618'
- model: 'Qwen/Qwen2.5-7B-Instruct' revision
  'a09a35458c702b33eeacc393d103063234e8bc28', 'torch.float32', offline 'true'
- ODE-Edit commit: 'cdc70db327ff264442cf26641c898bdefed3eaf3'
- fixed selection: salted ranks '[100:112]', 12 cases, seed '17', 'q=0.00390625',
  'h=0.5', layers '4,5,6,7,8'.

네 범주 경계는 다음과 같다.

- Proposal: fixed direct-z 아래 partial update 뒤 state-dependent non-stationarity와
  relinearization 필요 신호를 검정한다.
- Repo/protocol fact: canonical Qwen backbone, layers '4–8', rank slice
  '[100:112]', immutable covariance와 identity-only projector 정책이 고정돼 있다.
- GH inference: 동일 W1/equal-C의 A 대 B가 최소 direction 검정이며,
  coefficient-only 신호는 fixed-direction dynamic-coefficient pivot 후보이다.
- User confirmation: 없음. 다만 model별 독립 분석 뒤 pair red agent만 compact
  summary를 읽는 분리 경계는 유지한다.

## Result firewall 및 no-peer broadcast exception

manifest의 result firewall은 canonical request와 allowed rewrite contexts만
허용하며 evaluation prompts, generations, weights, logits, activations를
배제한다. Raw tensor는 Git-excluded이고 structured stream에는 IDs, hashes,
finite scalars, compact metadata만 남긴다. Covariance는 run-start pinned
immutable cache이고 miss/recompute/download는 차단되며, projector는 identity
preflight만 하고 deserialize하지 않는다. Direct-z는 W0에서 한 번 계산한 뒤
exact lineage의 W1 descendant에서만 허용된다.

No-peer broadcast exception: 이 분석은 Llama 디렉터리·artifact·summary를
읽거나 broadcast하지 않았다. Qwen-local technical block만 기록하며, raw peer
artifact 없이 이후 pair red review가 compact summaries만 읽을 수 있다는
사전등록 경계를 유지한다.
