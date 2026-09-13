# B 손상 보정 — Middle B-OS 사용자 recall 부분 보고

**부분 결과**: job45029 COMPLETED0:0. Finite endpoint/가중치·history 복원 검산 PASS. 두 PCG RHS 모두 20회 미수렴한 근사해이며 정확한 quadratic 최적해 또는 campaign 완료가 아니다.

## 1. 같은 Middle entry의 canonical 평가

|State|Panel|Metric|n/d|%|
|---|---|---|---:|---:|
|W0|Current100|RS|12/100|12.000|
|W0|Current100|PS|23/200|11.500|
|W0|Current100|NS|888/1000|88.800|
|W0|Fixed100|RS|5/100|5.000|
|W0|Fixed100|PS|20/200|10.000|
|W0|Fixed100|NS|886/1000|88.600|
|W0|Past100|RS|11/100|11.000|
|W0|Past100|PS|20/200|10.000|
|W0|Past100|NS|879/1000|87.900|
|We|Current100|RS|30/100|30.000|
|We|Current100|PS|53/200|26.500|
|We|Current100|NS|724/1000|72.400|
|We|Fixed100|RS|100/100|100.000|
|We|Fixed100|PS|194/200|97.000|
|We|Fixed100|NS|693/1000|69.300|
|We|Past100|RS|100/100|100.000|
|We|Past100|PS|192/200|96.000|
|We|Past100|NS|685/1000|68.500|
|N4|Current100|RS|100/100|100.000|
|N4|Current100|PS|197/200|98.500|
|N4|Current100|NS|711/1000|71.100|
|N4|Fixed100|RS|100/100|100.000|
|N4|Fixed100|PS|194/200|97.000|
|N4|Fixed100|NS|696/1000|69.600|
|N4|Past100|RS|100/100|100.000|
|N4|Past100|PS|193/200|96.500|
|N4|Past100|NS|686/1000|68.600|
|B-OS|Current100|RS|100/100|100.000|
|B-OS|Current100|PS|197/200|98.500|
|B-OS|Current100|NS|706/1000|70.600|
|B-OS|Fixed100|RS|100/100|100.000|
|B-OS|Fixed100|PS|192/200|96.000|
|B-OS|Fixed100|NS|695/1000|69.500|
|B-OS|Past100|RS|100/100|100.000|
|B-OS|Past100|PS|191/200|95.500|
|B-OS|Past100|NS|697/1000|69.700|

RS/PS=new NLL<true NLL, NS=true NLL<new NLL, tie failure. TF strict/token과 두 target NLL은 CSV에 별도 기록했다. Current/Fixed/Past는 각100 requests의 100/200/1000 prompts이며 총3900 pairs다. Fixed100/Past100은 새 BaseAudit/PastAudit bank와 같지 않다.

W0/We/N4는 같은 panel·순서·선택 weight bytes를 확인한 기존 S1 관측 재사용이다. 새 GPU forward0. Full cross-hardware numerical parity는 주장하지 않는다.

## 2. Controller 관측 및 solver

- Base128 risk: 0.0348951202304 → 0.113660569432.
- Past128 risk: 0.0103827207665 → 0.0194400029104.
- Current100_600contexts train_mean_NLL: 0.0474683861987 → 0.0628787853766.
- PCG a: 20 iterations, actual relative residual 0.15636985133, APPROXIMATE_PCG_NONCONVERGENCE.
- PCG u: 20 iterations, actual relative residual 1.79540589353, APPROXIMATE_PCG_NONCONVERGENCE.
- 일차 Current equality residual -2.50748e-11, 실제 train-NLL 변화 0.0154104. 일차 equality가 finite-step Current 보호를 보장하지 않았다.
- Stationarity norm 363.33; 실제 correction norm 5.07302, normalized native action 5.02402.

관측: 이번 B-OS에서는 Base/Past 위험이 모두 증가했고 Current 평균 NLL도 증가했다. 가능한 설명에는 큰 finite correction과 PCG 근사 오차가 함께 포함된다. 어느 요인이 주원인인지 이번 한 endpoint만으로 분리하지 못한다. 평균 Past NLL 감소와 양의 손상-tail risk 증가는 서로 다른 지표라 동시에 가능하다. 이 결과를 hard-Alpha capacity 불가능성이나 lifelong 열세로 확대하지 않는다.

## 3. 실제 비용과 잔여 범위
- Scheduler 3.967778 GPUh, 실패44991 98GPU-sec 포함 총 3.995000 GPUh. Cached common 준비비용과 native-z cold end-to-end 비용은 별도이며 여기에 포함하지 않았다.
- Edit core 13996.251s; terminal evaluation 147.342s; history 27.448s. Peak GPU 39527271936B; MaxRSS32779000KiB.
- Controller ledger {"forward_input_tokens": 1300992, "ggn_matvecs": 129, "gradient_backward": 792, "jvp_calls": 21156, "jvp_vectors": 21156, "logits_forward": 43296, "vjp_calls": 21156, "vjp_vectors": 21156}. Host transfer/runtime 전체 ledger는 partial-summary.json에 보존했다.
- L4는 exact WN4 고정; L8 node/endpoint equality, post-key FP32 outer-product append 각1회, 전체selected W0 restore 검산. GPU continuation replay는 수행하지 않았다.
- A0/A-OS 상대 canonical 결과 수신 대기. BLUE/native6/기타 B7/shortchain은 아직 이 보고서에 측정값 없음. 성능 때문에 제외한 것이 아니라 실행·관측 미완료다.
- BaseAudit/PastAudit와 L4-only/L8-only/joint signed interventions는 첫 runner에 누락됐다. 이를 측정한 것처럼 대입하지 않으며 후속 승인 범위에서 누락 관측만 보충해야 한다.
- 후속 Middle B-BF4의 보수적 wall 추정은 OS의 43 matvec 대비 최대85×4이므로 약30.8h + 관측/준비. 실측 보장이나 GPU-hour cap이 아니며 max20/relres1e-4/동일bank를 변경하지 않는다.

## 4. Provenance와 재현
- Frozen execution `3c0fd4f18189dac634db76f58b0e5e3b07f1ff8c` / tree `870d0b11875b00cf0f9ec5c23c740126a316a6b5`.
- Common READY `b04054714268677c7a4bb71d7dacd664a06069861bb05c1a74864262095c3064`.
- Local raw `/mnt/raid5/janghj/ODE-edit/local/multilayer-joint-compensation/20260911-v1/track_b/Middle-B-OS-tech-r1/output` (Git 제외, 원본 보존).
- Analysis source `7f70af238aa88d3dea3ae86635ff99f09f2b9d87`; analyzer SHA `2828740a33c3af67c749db149a534d857726c30cae198b79ac7a1b953498ddab`. Frozen execution과 analysis source를 구분한다.
- CPU 재현: `python -m project.run_scripts.multilayer_joint_compensation.track_b.analyze_partial --output <새 create-once 경로>`; frozen execution의 Git blobs와 실행 manifest를 대조하므로 이후 전용 branch 변경으로 원 실행을 덮어쓰지 않는다.
- INITIAL_VALID는 FD/GGN/적용 gate일 뿐 terminal 성능 증거가 아니다. 44991 기존 세 FD값과 실패는 보존했다. scientific_promotion=false.
- 새 후속 job은 최소 initial gate 뒤 다시 MONITORING_PAUSED_AWAITING_USER; 자동 terminal cascade 없음.
