# 회수 AlphaEdit BLUE L4-only checkpoint 기전 분석

작성 서버: Server2 / SH2. 상태: **최종 terminal 수집 보고**. scientific_promotion=false.
사용자 instruction ODEEDIT-S06-S2-CHECKPOINT-MECHANISM-AUDIT-20260920-V1에 한정해 H1–H4 판정이 허용되었다. 관측, 가능한 설명, 미분리 요인을 구분한다. 아래 archive 수치는 새 GPU parity의 증거가 아니다.

## 1. 현재 완료 범위와 핵심 수치

저장 current100 및 seen12를 독립 CPU reducer로 검사했다. 중복 15,600행은 scalar 일치를 확인한 뒤 한 번만 셌다. 유일한 요청은 10,000개이며, dedup 관측 837,200행과 at-write anchor 130,000행을 독립 표본수로 오해하지 않는다. 같은 요청의 반복 관측이다.
Cell 상태: `{"PASS": 30, "FAILED": 1}`. Geometry는 actual W0+12개 W/M, 동일 random vector256개를 사용하는11개 구간을 검사했다. C00/C01 등 모델·writer·evaluator gate의 성공 여부는 cell 표의 실제 status만 따른다.

### Actual gate와 차단 경계

최초 job51071은 P tensor hash header([d,d] 대 [1,d,d]) 오류로 FAILED1:0,230 allocated GPU-sec였다. P 원 bytes/fileSHA는 일치했다. Technical continuation51116은 이 metadata 검사만 교정하고 완료된 C00·W0/B1 평가를 SHA-bound 재사용했다. 새 평가 forward로 NS 값을 다시 시도하지 않았다. Continuation scheduler COMPLETED0:0이지만 numerical terminal은 **FAILED / NUMERICAL_CONTRACT_UNRESOLVED**다.

- C00: B1/B2/geometry32 K·bare K·h0 및 반복성 PASS.
- B1 성공 수: RS100/100,PS190/200,NS867/1000; 모든 success bit가 archive와 일치. W0 first100은5/100,20/200,886/1000이며10k W0로 확대하지 않는다.
- NS row237 최대 NLL 차이=0.00016117095947265625, margin 차이=0.00010061264038085938 > 고정1e-4. 모든 repeat row spread=0.
- M1 Gram:205,520,896원소 중4원소가1e-6+1e-5|ref| 초과, 최대 허용치 비율=1.64426349. 전체 최대 절대오차와 최대 비율 위치는 다르며 원JSON에 둘 다 있다.
- 실제 delta 상대오차=1.1321707e-05, 요청별 response 최대=9.56075246e-06 <1e-3. FP32 dense solve의 FP64 검산 최대 RHS 잔차=1.07472298e-06 <1e-5. 재solve 차이0; 실제 block affine와 key invariant PASS.
- W0 복원 및 nonselected parameter pointer/version guard PASS. Nonselected 전체 byte hash는 NOT_CLAIMED.



**최신 정책 변경:** 위 실패를 PASS로 바꾸지 않았다. C00/C01 재실행0. 수치 gate/검증 전용 GPU 호출을 제거하고 필수 artifact 가용성·source/shape/dtype/finite/I/O/상태복원/resource만 유지해 나머지 분석을 실행했다. 새 PASS는 계산 완료이며 numerical_validation=NOT_ESTABLISHED다. Tolerance를 확대해 통과시킨 실행이 아니다.

### B100 전체 seen 요청의 저장 평가

|metric_tag|numerator|denominator|rate|strict_numerator|strict_denominator|token_correct|token_denominator|
|---|---|---|---|---|---|---|---|
|RS|9939|10000|0.9939|9529|10000|9688|10163|
|PS|19136|20000|0.9568|13362|20000|13658|20326|
|NS|65348|100000|0.65348|8210|100000|9195|101270|

### 요청별 실제 at-write → B100 전이

|metric_tag|rows|start_success|end_success|n11|n10|n01|n00|loss_denominator|gain_denominator|margin_delta_mean|
|---|---|---|---|---|---|---|---|---|---|---|
|RS|10000|9993|9939|9938|55|1|6|9993|7|-2.03768|
|PS|20000|19403|19136|18980|423|156|441|19403|597|-0.814834|
|NS|100000|72505|65348|60958|11547|4390|23105|72505|27495|-1.08812|
|RP_JOINT|10000|9514|9285|9174|340|111|375|9514|486|-0.901802|

Loss 조건부 분모는 시작 성공, recovery 분모는 시작 실패다. `RP_JOINT`는 같은 request의 RS∧PS0∧PS1이며 strict도 세 prompt가 모두 strict일 때만 성공이다. Joint safety margin은 세 prompt safety margin의 최소값이다. At-write NS는 first-write-N이며 W0-correct-N 유지율이 아니다.

## 2. 원 실행·입력·수치 결속

원 arm은 AlphaEdit_L4_ONLY, 표시명 AlphaEdit_BLUE_L4_ONLY다. L4 down_proj FP32[4096,14336] 하나와 post-write cache_c[1,14336,14336]를 대상으로 한다. M은 static Wikipedia C0 또는 forward weight가 아니다. Base revision은 8afb486c1db24fe5011ec46dfbe5b5dccdb575c2이며 W0 저장 BF16 tensor를 FP32 cast한 identity를 사용한다. Source layers[4,5,6,7,8]의 P slot0=L4다.

원 실행 contract: Torch2.9.1+cu128, Transformers4.44.2, FP32/eager/autocast=false, TF32 matmul=false/cuDNN=true. Writer/evaluator BOS 차이, evaluator MB16·수동 left-padding·implicit position_ids를 유지한다. 이 문구는 실제 GPU 환경 PASS 선언이 아니며 원 요구사항이다. 실제 실행/환경 identity는 `context.json` 및 terminal receipt에 분리 기록한다. Migration39283_3/runtime40426 표기 차이는 숨기지 않고 W/M/model/source/order/context hash로 결속한다.

B001 archive 기대값은 RS100/100, PS190/200, NS867/1000이고 CPU 검산에서 일치했다. EN100/194/865를 사용하지 않았다. 첫 entry.context_hash는 JSON null의 digest였고 native lazy context 초기화 뒤 contexts.json/commit에는 봉인된 실제 context digest가 기록되었다. 이는 상태별 관측을 그대로 남긴 것이며 context를 다시 생성하지 않았다.

Source publication·실행 source·analysis source·report SHA는 서로 다른 identity다. Context 미제공 field는 NOT_RECORDED이며 최신 main SHA를 과거 실행 SHA로 대체하지 않는다. 새 W/M/checkpoint/복원 delta 저장은 0; 기존12CP는 읽기 전용 보존이다. 새 z fitting·editing chain·GSS·EN 실행은 없다.

## 3. 평가 정의와 종단 설계

Join은 (arm,metric_tag,identity), 관측 key에는 checkpoint batch를 추가했다. Identity는 원 evaluator의 [case_id,prompt_index,prompt,target_new,target_true] canonical JSON SHA다. Category가 digest에 없으므로 metric tag를 생략하지 않았다. 원 denominator는 request당 R1/P2/N10이다.

RS/PS safety margin=true_nll−new_nll, NS=new_nll−true_nll이며 양수만 성공이고 tie는 실패다. 각 NLL은 해당 target 자체의 teacher-forced sequence에서 target token 평균이다. Free generation이나 full-vocabulary classification accuracy와 동일하지 않다. RS/PS strict는 new target의 모든 token 정답, NS strict는 true target의 모든 token 정답이다.

각 request의 최초 anchor는 arrival batch current.json이다. 첫 sparse seen checkpoint나 entry.json을 at-write 또는 W0 평가로 바꾸지 않았다. First100은12경계, first1000은B10 이후, adjacent는 출발점에 이미 도착한 동일 rows만 비교한다. Age는 arrival batch를 고정한다. 처음 관측한 failure는 sparse interval-censored이며 관측 사이에 실패/회복이 없었다는 보장은 없다. W0 first100 새 평가가 결속되지 않은 범위는 W0→post loss로 표현하지 않는다.

## 4. Overwrite 및 전이 불확실성

원래 all-case를 유지한 census: `{"earlier_versions": 217, "groups": 9783, "groups_with_changed_target": 120, "repeated_groups": 125, "requests": 10000, "within_batch_conflict_group_batches": 5}`.
같은 batch의 서로 다른 target은 BATCH_INTERNAL_CONFLICT이며 배열 마지막을 실제 최신 정답으로 확정하지 않았다. Active-version은 별도 진단이고 원 benchmark를 대체하지 않는다. 같은 latest batch에서 target이 동일한 중복 요청은 함께 남을 수 있다. 두 상충 target이 original true보다 모두 우세하더라도 두 사실의 완전 동시 저장으로 해석하지 않는다.

|view|metric_tag|rows|start_success|end_success|loss_numerator|loss_denominator|gain_numerator|gain_denominator|
|---|---|---|---|---|---|---|---|---|
|original_all_case|RS|10000|9993|9939|55|9993|1|7|
|original_all_case|PS|20000|19403|19136|423|19403|156|597|
|original_all_case|NS|100000|72505|65348|11547|72505|4390|27495|
|interval_target_conflict_free|RS|9784|9779|9746|34|9779|1|5|
|interval_target_conflict_free|PS|19568|19035|18821|355|19035|141|533|
|interval_target_conflict_free|NS|97840|71092|64021|11271|71092|4200|26748|
|separate_active_version|RS|9779|9774|9741|34|9774|1|5|
|separate_active_version|PS|19558|19025|18811|355|19025|141|533|
|separate_active_version|NS|97790|71049|63992|11254|71049|4197|26741|

Request cluster bootstrap2000회, seed20260920. 같은 cohort membership에 같은 재표집을 사용하여 P/N rows와 observed times를 묶는다. Subject-relation sensitivity도 동일하게2000회다. 아래 구간은 pointwise conditional percentile95%이며 동시 band나 여러 edit 순서에 대한 보장은 아니다. Growing-cohort의 서로 다른 membership에는 각각 조건부 재표집이 적용된다. 분모0 replicate는 별도 finite count와 NA로 표시했다.

|metric_tag|cluster|clusters|replicates|net_rate_lo|net_rate_hi|loss_rate_lo|loss_rate_hi|gain_rate_lo|gain_rate_hi|
|---|---|---|---|---|---|---|---|---|---|
|RS|case_id|10000|2000|-0.0068|-0.004|0.00419847|0.00690898|0|0.5|
|RS|subject_relation_group_a|9783|2000|-0.00702193|-0.00389758|0.00400958|0.00710387|0|0.5|
|PS|case_id|10000|2000|-0.0159012|-0.0109|0.0195723|0.0240116|0.223282|0.298954|
|PS|subject_relation_group_a|9783|2000|-0.0157875|-0.010827|0.0194377|0.0242009|0.223656|0.299116|
|NS|case_id|10000|2000|-0.0750102|-0.0681397|0.155256|0.163605|0.154476|0.1652|
|NS|subject_relation_group_a|9783|2000|-0.0749825|-0.0681639|0.154996|0.163478|0.154151|0.164958|
|RP_JOINT|case_id|10000|2000|-0.0271|-0.0189|0.032085|0.0395333|0.191866|0.265663|
|RP_JOINT|subject_relation_group_a|9783|2000|-0.0270946|-0.0188471|0.032112|0.0397045|0.188906|0.26575|

## 5. Strict, token 및 NLL tail

Strict/token numden은 첫 표와 `functional_summary.csv`에 보존한다. 다음은 B100 all-seen의 per-prompt paired-target NLL 요약이다. Prompt를 독립 request로 재계수하지 않는다. 낮은 RS 또는 safety margin만으로 전체 pretrained capability 저하를 판정하지 않는다.

|metric_tag|new_nll_mean|new_nll_q05|new_nll_q50|new_nll_q95|true_nll_mean|safety_margin_q01|safety_margin_q05|safety_margin_q50|
|---|---|---|---|---|---|---|---|---|
|RS|0.230527|0.000512195|0.00709715|1.207|12.2384|0.966814|5.00953|11.9627|
|PS|1.58923|0.00494269|0.41793|6.79963|9.50584|-3.43903|0.387133|7.90979|
|NS|8.0201|1.38736|8.03193|14.3852|6.45812|-10.164|-6.07882|1.60519|

전체12점, first100/first1000, margin q01/q05/q50/q95/q99 및 true/new NLL은 CSV에 있다. 관측하지 않은 checkpoint·PS/NS를 보간하지 않았다.

## 6. Actual W/M와 stored-history action

W 차이는 FP64 cast 뒤 subtraction했고 norm/reduction도 FP64다. B100 ||W100−W0||/||W0||=1.3468, M trace=79876.3. M trace 증가를 layer capacity 소진으로 대체하지 않는다.

|batch|relative_w0_norm|history_trace|interval_delta_norm|stored_action_mean|stored_action_mc_se|stored_action_relative_halfwidth|stored_action_alignment_dimensionless|interval_update_total_cross_term|
|---|---|---|---|---|---|---|---|---|
|0|0|0|NOT_MEASURED|NOT_MEASURED|NOT_MEASURED|NOT_MEASURED|NOT_MEASURED|NOT_MEASURED|
|1|0.0982947|779.802|7.61019|NOT_MEASURED|NOT_MEASURED|NOT_MEASURED|NOT_MEASURED|NOT_MEASURED|
|5|0.224418|3955.06|15.6101|1.20423|0.031418|0.0511359|0.0908531|0.331228|
|10|0.329997|7925.16|18.7245|8.45344|0.220364|0.0510933|0.0873951|-0.00170759|
|20|0.490913|15917.8|28.1397|42.7381|0.60629|0.0278049|0.097633|-0.822795|
|30|0.625554|23820.2|30.0359|70.9922|0.704591|0.0194528|0.0708717|-0.777748|
|40|0.745416|31862.7|31.4211|120.603|1.0361|0.0168384|0.0735187|-0.511086|
|50|0.858315|39912.3|32.984|164.516|1.11196|0.0132477|0.0680372|-0.461817|
|60|0.96563|47931.6|34.3357|205.322|1.23226|0.0117631|0.0625555|-0.440956|
|70|1.06758|55912.3|35.3473|269.722|1.40441|0.0102055|0.064567|-0.582854|
|80|1.1648|63947.5|36.1737|358.925|1.69771|0.00927078|0.0703296|-0.489704|
|90|1.25819|71909.1|36.9589|418.914|1.91805|0.00897413|0.0687529|-0.783958|
|100|1.3468|79876.3|37.3494|521.836|2.27696|0.00855219|0.074578|-0.79002|

J=tr(D M_a Dᵀ)는 stored quadratic 작용이며 실제 forgetting 수치가 아니다. 11구간에 같은 Rademacher output vectors256개를 사용했다.64/128은 진단뿐이며 조기종료하지 않았다. J/n_a, J/tr(M_a), dJ/(||D||²tr(M_a)) 및 zero/negative flags는 원 CSV에 있다.1.96 MC SE는 근사적인 Monte Carlo 정밀도 지표이지 preservation threshold/엄밀한 유한표본 CI가 아니다. Stored M의 FP32 누적 오차는 actual historical key response와 차이를 만들 수 있다. 임의 방향 PSD checks는 PSD/rank 인증이 아니다.

구간 ||D||²−Σ||Δbatch||²는 내부 update의 총 교차항이다. 개별 pair의 상쇄나 single-batch 손상을 복원한 값이 아니다. Norm/angle만으로 output locality 손상을 확정하지 않는다.

## 7. 실제 모델·operator·demand 검증 경계

`fixed_probe_summary.csv`, `native_write_modes.csv`, `reconstruction.csv`, `counterfactuals.csv`가 없던 측정은 NOT_MEASURED status이며 0으로 채우지 않았다. Native700과 stream-ID-disjoint geometry512는 다른 panel이며 기존 reference512/G256을 교체하지 않는다. 동일 key를13history에 사용하고 각 native100은 자기 S/B를 유지한다.

일반 비대칭 H=λI+PM에는 LU/solve를 사용하며 명시 inverse/rawCG/Cholesky를 쓰지 않는다. Raw score를 ideal 범위에 clipping하지 않는다. R은 saved z와 physical FP32 entry h로 결속하며 mean-context K를 residual에 대입하지 않는다. Affine h0+(Wentry−W0)k_bare 차이는 저비용 관측일 뿐 차단 조건이 아니다. B1만 actual next-weight tensor와 비교 가능하고 나머지는 RECONSTRUCTED_NATIVE_WRITE다. 최신 사용자 지시로 수치 parity 인증·차단과 검증 전용 중복 호출을 제거했다. 원 C01 FAILED를 보존하며 새 계산은 numerical_validation=NOT_ESTABLISHED다. B1은 actual delta를 참조하되 재현 인증을 주장하지 않고, B91 dense-factor 중복 계산은 SKIPPED_USER_DIRECTED다.

고정 geometry512의 nu 평균은 M0 0.999999082 → M100 0.327913213; 감소 512/512, 증가 0/512. Raw 음수 0행, ideal 범위 밖 0행은 clipping하지 않았다. 최대 RHS 상대 잔차 1.20069828e-13는 계산 관측이며 통과 기준이 아니다.

Native demand의 raw B 직접 thin-SVD 요약:

|target_batch|modes|gain_min|gain_max|target_loading_sum|write_energy|
|---|---|---|---|---|---|
|1|100|0.190942|0.42565|536.214|57.915|
|2|100|0.243291|0.497927|542.632|59.5415|
|6|100|0.25177|0.410553|580.872|67.6485|
|11|100|0.143774|0.452777|626.346|76.9261|
|21|100|0.245739|0.458274|653.481|84.569|
|51|100|0.196867|0.449745|799.791|115.434|
|91|100|0.231933|0.433588|905.057|131.484|

이 에너지 분해는 factor write의 대수적 분해다. B1 actual residual/cross term을 별도로 보존하고, B2 이후를 실제 다음 checkpoint와 동일하다고 인증하지 않는다. 작은 key singular value만으로 gain을 추론하지 않았다.

선정된 NS panel 실제 margin/entry-gradient 요약:

|interval_start|interval_end|role|n|actual_margin_change_mean|predicted_mean|absolute_remainder_mean|lost|sign_agree|sign_denominator|DK_squared_mean|cross_mean|
|---|---|---|---|---|---|---|---|---|---|---|---|
|1|10|lost|16|-6.79388|-4.01764|3.92651|16|15|16|12.9627|1.67509|
|1|10|matched_retained|16|-1.61463|-0.53777|1.2915|0|11|16|10.3825|1.44975|
|50|100|lost|16|-4.05866|-2.79035|2.18707|16|13|16|57.3487|3.9799|
|50|100|matched_retained|16|0.206703|1.74392|2.65689|0|10|16|48.7024|3.42379|

Lost/retained는 archive outcome으로 사후 선정했다. 재측정 lost 수가16과 다를 수 있으며 원 panel을 교체하지 않았다. DK energy와 margin은 다른 값이고 부호가 있는 gradient 및 nonlinear remainder를 함께 본다. 새 backward는 실제 연구 분석이며 parity 전용 backward가 아니다.

고정 선분 s=0/.5/1의 실제 preference/TF strict(각 target 자체 TF 경로):

|interval_start|interval_end|s|role|n|ns_success|new_strict|true_strict|margin_mean|margin_min|margin_max|
|---|---|---|---|---|---|---|---|---|---|---|
|1|10|0|lost|16|16|0|2|3.9893|0.0214987|9.41957|
|1|10|0|matched_retained|16|16|0|0|4.50963|0.53426|9.6269|
|1|10|0.5|lost|16|10|0|2|1.10947|-4.42394|9.01309|
|1|10|0.5|matched_retained|16|16|0|0|3.99827|0.536294|6.65359|
|1|10|1|lost|16|0|0|0|-2.80459|-8.42333|-0.22775|
|1|10|1|matched_retained|16|16|0|0|2.895|0.100813|6.10503|
|50|100|0|lost|16|16|0|1|2.35438|0.496417|8.41976|
|50|100|0|matched_retained|16|16|0|2|2.51848|0.423827|8.06834|
|50|100|0.5|lost|16|9|0|1|0.428632|-3.77113|4.91409|
|50|100|0.5|matched_retained|16|15|0|1|2.87681|-0.00351238|10.6787|
|50|100|1|lost|16|0|0|0|-1.70428|-5.80797|-0.0221786|
|50|100|1|matched_retained|16|16|0|1|2.72519|0.084568|11.6437|

선정 panel의 parent R/P(이미 봉인된 평가 재사용):

|interval_start|interval_end|role|metric|rows|unique_requests|entry_success|endpoint_success|lost|recovered|mean_margin_change|
|---|---|---|---|---|---|---|---|---|---|---|
|1|10|lost|PS|28|14|28|28|0|0|-0.122844|
|1|10|lost|RS|14|14|14|14|0|0|-0.791552|
|1|10|matched_retained|PS|30|15|29|30|0|1|0.264945|
|1|10|matched_retained|RS|15|15|15|15|0|0|-0.0922824|
|50|100|lost|PS|32|16|32|31|1|0|-2.42982|
|50|100|lost|RS|16|16|16|16|0|0|-2.59991|
|50|100|matched_retained|PS|32|16|32|31|1|0|-1.25155|
|50|100|matched_retained|RS|16|16|16|16|0|0|-2.65498|

동일 parent가 복수 NS 또는 두 selection-role에 등장할 수 있다. 각 role 안 request/identity를 중복제거했고 role 사이 합을 독립 분모로 더하지 않는다.

고정 K/R × history 및 R 열 순열 대조(실제 편집 아님):

|history_batch|target_batch|kind|rows|energy_mean|energy_min|energy_max|target_error_mean|
|---|---|---|---|---|---|---|---|
|0|1|ORIGINAL_ALIGNMENT|1|57.915|57.915|57.915|0.130355|
|0|1|R_COLUMN_PERMUTATION|20|58.6248|58.3939|58.9745|0.13123|
|0|2|ORIGINAL_ALIGNMENT|1|57.7562|57.7562|57.7562|0.132616|
|0|2|R_COLUMN_PERMUTATION|20|58.5272|58.1576|58.8613|0.133366|
|1|1|ORIGINAL_ALIGNMENT|1|16.659|16.659|16.659|0.533647|
|1|1|R_COLUMN_PERMUTATION|20|16.867|16.7841|16.9797|0.534051|
|1|2|ORIGINAL_ALIGNMENT|1|59.5415|59.5415|59.5415|0.136786|
|1|2|R_COLUMN_PERMUTATION|20|60.1566|59.7477|60.4616|0.13712|
|10|1|ORIGINAL_ALIGNMENT|1|19.3268|19.3268|19.3268|0.542844|
|10|1|R_COLUMN_PERMUTATION|20|19.329|19.2056|19.4547|0.542258|
|10|2|ORIGINAL_ALIGNMENT|1|19.3395|19.3395|19.3395|0.54091|
|10|2|R_COLUMN_PERMUTATION|20|19.4648|19.2998|19.6592|0.541049|
|100|1|ORIGINAL_ALIGNMENT|1|28.2446|28.2446|28.2446|0.603656|
|100|1|R_COLUMN_PERMUTATION|20|28.1437|27.9383|28.3922|0.599146|
|100|2|ORIGINAL_ALIGNMENT|1|28.221|28.221|28.221|0.601836|
|100|2|R_COLUMN_PERMUTATION|20|28.2423|28.0411|28.4719|0.599874|

직접 thin SVD B의 gain²×||Rv||²는 factor write Frobenius energy를 분해한다. Actual/native 재현 잔차 및 cross term, near-degenerate band를 별도로 기록한다. 작은 K singular value가 필연적으로 증폭한다는 가정을 하지 않는다. 고정 K/R×history,20개 R permutation은 대수적 counterfactual이며 실제 editing 성과가 아니다.

## 8. Activation–margin 사후 panel

|interval_start|interval_end|selection_role|rows|distinct_requests|mean_entry_margin|mean_endpoint_margin|
|---|---|---|---|---|---|---|
|1|10|lost|16|14|3.9893|-2.80458|
|1|10|matched_retained|16|15|4.50963|2.895|
|50|100|lost|16|16|2.35438|-1.70428|
|50|100|matched_retained|16|16|2.51848|2.72518|

두 구간 모두 first100 NS1000에서 lost16+matched-retained16을 선정했다. Target conflict를 제외하고 target 길이 일치, relation 일치, entry margin 차이, seed hash 순으로 중복 없이 match했다. 선택은 outcome-dependent 사후 기전 표본이며 대표 성능 추정이나 controller threshold selector가 아니다.

실제 분석은 true/new 각각 all-valid-token TF 경로, E K/D K 및 2〈EK,DK〉, s=0/.5/1의 실제 margin·strict, entry gradient 선형예측과 remainder를 구분한다. 해당 CSV가 NOT_MEASURED이면 archive panel 선택만 완료한 것이다. 세 점과 한 entry gradient는 선분 전체의 인증이 아니다. Parent R/P 변화는 별도 observer이고 사례별 token patch를 구현 가능한 공통 weight update라고 부르지 않는다.

## 9. H1–H4 판정 및 H5

|hypothesis|basis|provisional|status|
|---|---|---|---|
|H1|이 단일 stream의 저장된 at-write/같은 cohort 전이에서 RS·PS와 NS의 시간적 양상이 다름. B100 all-case 조건부 loss는55/9993,423/19403,11547/72505이며 conflict-free 및 strict 전이를 별도 보고. 누적 history의 인과효과나 일반 capacity 판정 아님.|False|SUPPORTED|
|H2|동일512key의 M0→M100 nu 감소 512/512, 평균 0.999999082→0.327913213. 고정-key operator의 제한적 관측이며 raw nonsymmetric/수치동등성 미확립, 실제 forgetting 인과와 구분.|True|SUPPORTED|
|H3|7개 demand/700mode의 gain²×target-loading으로 factor write energy를 분해했다. 이는 정의상 대수관계이며 causal 설명의 독립 검증이 아니다. B1 actual reference 외는 RECONSTRUCTED_NATIVE_WRITE, dense-factor 검증 생략 및 원 C01 FAILED 때문에 actual write 전체에 대한 수치 인증은 미확립.|True|MIXED|
|H4|두 사후선정 구간 64개 NS문항에서 actual s0/.5/1와 all-valid-token gradient/EK/DK를 측정. 부호 일치/불일치와 remainder를 표에 모두 보존. Outcome선정에 의한 loss 자체를 독립 인과증거로 삼지 않고, 원 evaluator 동등성 미확립도 유지한다.|True|MIXED|

관측 사실과 가능한 설명을 구분한다. H1의 차등 시간양상은 이 단일 stream의 기술적 관측이다. History growth가 그 손상의 원인인지는 고정-key operator 및 실제 activation/signed margin 연결 없이는 분리되지 않는다. Current request composition, target demand, physical rounding, overwrite 및 nonlinear suffix가 경쟁 설명이다. H5는 current edit response를 유지하며 손상을 줄일 자유도 조사 필요성이라는 후속 질문이며 새 optimizer/GSS/EN/layer arm은 NOT_RUN이다.

## 10. 비용·실행·제한과 재현

Archive CPU wall=517.221661s. Geometry CPU wall=128.34606998693198s, original checkpoint full-hash 비용=15.29243337456137s, peak RSS=9580608KiB. 이 시간은 새 model/operator GPU 시간과 다르다. Worker 병렬 wall을 무조건 더해 end-to-end로 부르지 않는다. 실제 scheduler allocated GPU-sec/teacher-prefix/suffix F/B/solve/hash/I/O는 제공된 `context.json`의 cost ledger만 사용하며 미기록 값은 추정하지 않는다.

원 설계 RAM64GiB budget보다 Server2 admission ceiling60416MiB가 작다. 최신 task override는 프로젝트 cap2 안의1GPU×2lane, 각8CPU/exportNONE/Requeue0이다. 공통 key artifact 가용성만 선행하며 C01/pilot numerical PASS는 실행 prerequisite가 아니다. 의존성이 있는 단계는 slot을 채우려 중복 실행하지 않는다. 본 CPU report builder는 scheduler 조회/model/GPU/eval을 하지 않는다.

|allocated_gpu_seconds|job|numerical_status|runner_seconds|scheduler_state|stage|interval_seconds|compute_status|
|---|---|---|---|---|---|---|---|
|230|51071|FAILED_TECHNICAL_P_HEADER; NS parity FAILED|227.417|FAILED|original_gate|NOT_MEASURED|NOT_MEASURED|
|39|51116|FAILED_NUMERICAL_CONTRACT|35.0469|COMPLETED|technical_continuation|NOT_MEASURED|NOT_MEASURED|
|71|51137|NOT_ESTABLISHED|68.2844|COMPLETED|keys|NOT_MEASURED|PASS|
|262|51138|NOT_ESTABLISHED|259.82|COMPLETED|operator|NOT_MEASURED|PASS|
|174|51139|NOT_ESTABLISHED|NOT_RECORDED|COMPLETED|activation|155.621|PASS|

비용 ledger의 allocated GPU seconds는 scheduler allocation wall×GPU수이며 CUDA kernel busy time이 아니다. 포함된 load/hash/verification/evaluation 시간을 다시 더해 합계를 부풀리지 않는다. 미계측 세부 F/B·I/O는 NOT_RECORDED로 남기고 추정으로 분할하지 않는다. 실행한 C00/C01 capture/evaluation은 frozen parameter forward-only이며 backward0이다.

재현 명령:

```bash
python -B -m project.run_scripts.checkpoint_mechanism_audit.reporting --results /mnt/raid5/janghj/ODE-edit/local/checkpoint-mechanism-audit/20260920-v1/attempt-v1/results --output <NEW_EMPTY_OUTPUT> --cell-statuses /mnt/raid5/janghj/ODE-edit/local/checkpoint-mechanism-audit/20260920-v1/resume-no-gates-r1/publication-r1/cell-statuses.json --context /mnt/raid5/janghj/ODE-edit/local/checkpoint-mechanism-audit/20260920-v1/resume-no-gates-r1/publication-r1/report-context.json --final
```

4종 figure는 위 CSV와 본 Python code로 생성했다. PNG/PDF를 각각 두 번 생성하여 byte-exact 재현을 검사한다. Imagegen/수동수치수정은0. `figure-manifest.json`에 입력CSV·출력SHA를 기록한다. Raw prompts/weights/cache/tensor/prediction/log는 Git에 포함하지 않고 local 보존한다. NO_BROADCAST_NOT_REQUIRED. 이번 report 생성은 새 과학실험·무관 task recall·새 checkpoint 저장이 아니다.

## 11. Cell별 상태

|cell_id|stage|status|reason|
|---|---|---|---|
|A00|input_binding|PASS|Reused immutable historical evidence; not rerun|
|A01|functional_ledger|PASS|Reused immutable historical evidence; not rerun|
|B00|weight_history|PASS|Reused immutable historical evidence; not rerun|
|C00|prefix_parity|PASS|Reused immutable historical evidence; not rerun|
|C01|B1_reproduction|FAILED|Historical FAILED retained; removed as execution prerequisite by explicit user override|
|C02|key_bank|PASS|Native700 + geometry512 key artifact available; no parity certification|
|O000|history_operator_pilot|PASS|General LU/RHS computation; residual is observation only|
|O001|history_operator_pilot|PASS|General LU/RHS computation; residual is observation only|
|O010|history_operator_pilot|PASS|General LU/RHS computation; residual is observation only|
|O100|history_operator_pilot|PASS|General LU/RHS computation; residual is observation only|
|O005|history_operator_extension|PASS|General LU/RHS computation; residual is observation only|
|O020|history_operator_extension|PASS|General LU/RHS computation; residual is observation only|
|O030|history_operator_extension|PASS|General LU/RHS computation; residual is observation only|
|O040|history_operator_extension|PASS|General LU/RHS computation; residual is observation only|
|O050|history_operator_extension|PASS|General LU/RHS computation; residual is observation only|
|O060|history_operator_extension|PASS|General LU/RHS computation; residual is observation only|
|O070|history_operator_extension|PASS|General LU/RHS computation; residual is observation only|
|O080|history_operator_extension|PASS|General LU/RHS computation; residual is observation only|
|O090|history_operator_extension|PASS|General LU/RHS computation; residual is observation only|
|N001|native_demand_modes|PASS|Saved-target native demand/thin SVD computation, numerical equivalence NOT_ESTABLISHED|
|N002|native_demand_modes|PASS|Saved-target native demand/thin SVD computation, numerical equivalence NOT_ESTABLISHED|
|N006|native_demand_modes|PASS|Saved-target native demand/thin SVD computation, numerical equivalence NOT_ESTABLISHED|
|N011|native_demand_modes|PASS|Saved-target native demand/thin SVD computation, numerical equivalence NOT_ESTABLISHED|
|N021|native_demand_modes|PASS|Saved-target native demand/thin SVD computation, numerical equivalence NOT_ESTABLISHED|
|N051|native_demand_modes|PASS|Saved-target native demand/thin SVD computation, numerical equivalence NOT_ESTABLISHED|
|N091|native_demand_modes|PASS|Saved-target native demand/thin SVD computation, numerical equivalence NOT_ESTABLISHED|
|E00|algebra_controls|PASS|Fixed B1/B2 K/R ×4histories ×(original+20permutations); not editing arm|
|F00|activation_bookkeeping|PASS|Fixed geometry keys × actual11interval E/D algebra|
|G001_010|suffix_mechanism|PASS|Actual s0/.5/1 margins and entry-gradient paths; no duplicated parity forwards|
|G050_100|suffix_mechanism|PASS|Actual s0/.5/1 margins and entry-gradient paths; no duplicated parity forwards|
|Z00|report|PASS|Terminal collection and report, not numerical certification|

모든31cell의 성공을 가정하지 않는다. Final은 PASS/FAILED/BLOCKED/SKIPPED terminal 수집 후 생성하며 failed dependency만 관련 claim을 차단한다. Draft는 미실행 status를 terminal로 위조하지 않는다. 낮은 efficacy/큰slack/신호없음 때문에 관측을 제거하거나 threshold를 조정하지 않았다.
