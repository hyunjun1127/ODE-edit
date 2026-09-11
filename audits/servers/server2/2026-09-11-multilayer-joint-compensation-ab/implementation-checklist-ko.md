# B 및 shared kernel 구현 체크리스트

기준 설계 SHA a8f32937ddd2fc4f87cb3a2cfb9ef95de67d2fd39ce877142fa7b80a1884bf05. 사용자 instruction `ODEEDIT-S06-MULTILAYER-DAMAGE-COMPENSATION-B-SH2-V1`은 해석·실험·완료 scope main 통합을 승인한다. 과거 task/monitoring은 재개하지 않는다.

|설계 계약|신규 코드 위치|검증|
|---|---|---|
|token→context→request 평균, 전체sequence KL/NLL GGN|functional.py|CPU directional/PSD/cross-layer, actual model gate|
|full-space ordered tuple PCG, relres1e-4/max20|linear_solve.py|dense synthetic oracle, zero RHS, finite approximate typed|
|joint Current equality + 2 elastic channels|elastic_qp.py|4 active sets, exact nonnegative, primal/equality/stationarity/KKT|
|B WN4 byte固定, support8만; We/WN teachers|track_b/|actual restore/materialization + C8 removal exact WN|
|Frozen derivative만 고정, 실제 risks/RHS 갱신|track_b/|step/scaling/teacher identity fixture|
|native Alpha/MEMIT6 same-entry, hparams 그대로|native_baselines/|source import/write/history/5weight map|
|N4/M8/bank/teacher 공통 input|SH1 common files READONLY|SH1 seal→SH2 exact allowlist pull/full SHA/schema/order|
|short chain Middle B51..60 실제 W/M 누적|track_b/|history once, 자기 We 갱신, first B51 exact reuse|
|원시 지표/attribution/비용/그림/보고|track_b analysis|정확 분모/identity, code PNG, scientific_promotion=false|

SH1은 같은 저장소에서 자기 파일을 구현 중이다. Top-level init/contracts/common/track_a는 SH1 소유이며 수정·revert하지 않는다. SH2는 source/kernel commit을 먼저 교환하고 성능 결론을 기다리지 않는다. 아직 CPU/실모델 correctness 또는 endpoint 실행 PASS를 주장하지 않는다.
