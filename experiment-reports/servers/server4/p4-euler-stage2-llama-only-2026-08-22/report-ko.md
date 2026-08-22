# P4-Euler Stage2 Llama 전용 최종 보고서

> Scope: **LLAMA_ONLY_STAGE2_PLUS**. 공통 hyperparameter transfer가 Qwen 편집 성공을 만들지 못해 Qwen은 진단 전용으로 제외하고, Stage2+ 과학 분석은 Llama에만 한정한다.

## 최종 판정

- 상태: **SCIENTIFIC_HOLD**
- 사유: **Euler discretization refinement gate failure at locked T_z=1.25**
- 잠금: Llama `h=0.25`, `T_z=1.25`; Stage2 `M5,h=.25` 대 `M10,h=.125`.
- `Z+` d_i mean/median/p90/max = 0.599766/0.599819/0.819999/0.837496.
- `Z±` d_i mean/median/p90/max = 0.722633/0.737896/0.865267/1.022678.
- 사전 임계 median≤.10, p90≤.25, max≤.50를 두 arm 모두 초과했다. 이는 고정 T_z에서의 수치 refinement 실패이며 편집 성능 결론이 아니다.
- ZA/ZB 제출 0, 재시도·regrid·T_z·tolerance 변경 0, scientific promotion=false.

## Stage1 h=.25 strength curve와 Stage2 연결

Stage1은 동일 h=.25의 한 10-step trajectory에서 M1/M3/M5/M10 prefix를 재사용했다. `llama-stage1-h025-curve.csv`가 arm별 train new/true NLL, margin, displacement, raw-field norm과 request×microstep clamp 분모를 제공한다. Stage2 M5는 같은 h/T_z endpoint이고, M10은 같은 T_z를 h=.125로 세분화했다. M5와 M10의 endpoint가 크게 달라져 refinement gate를 통과하지 못했다.

Stage1의 M1/M3/M5/M10 간 request별 절대 이동은 `llama-stage1-prefix-distances.csv`에 있다. Stage2 실행은 중간 state를 저장하지 않았으므로 M5와 M10 trajectory가 **어느 최초 microstep에서** 갈라졌는지는 기록으로 판정할 수 없다. 원인은 추정하지 않는다.

## Endpoint 및 request-level 산출

- `llama-endpoint-summary.csv`: d_i 분포와 임계, clamp numerator/denominator/fraction.
- `llama-endpoint-requests.csv`: ordinal별 M5/M10 train new/true NLL, margin, state norm, M5↔M10 절대 endpoint 거리.
- request별 d_i와 ||z_M-y|| 벡터는 runner가 저장하지 않아 `NOT_RECORDED_CALIBRATION_RUNTIME_TELEMETRY_OMISSION`이다. authoritative d_i distribution만 Stage2 comparison receipt에 남아 있다.
- `train_new_beats_true`는 train target log-odds의 관측 필드이며 heldout Eff/Gen 성공 판정이 아니다.

## Microstep, objective, gradient, movement

`llama-microstep-ledger.csv`는 30개 실행 microstep(arm별 M5+M10)의 순서, h/T_z, autograd 1회/step 및 rooted step identity를 기록한다. 다만 runtime이 in-memory step receipt의 objective semantic/KL/decay, raw/projected gradient norm, movement, request clamp vector를 파일에 영속화하지 않았다. 해당 열은 `NOT_RECORDED_CALIBRATION_RUNTIME_TELEMETRY_OMISSION`이며 HOLD 이후 GPU replay를 하지 않았다.

## 불변식과 compute

- Stage2: 4 trajectories, 30 microsteps, autograd.grad 30, duplicate evaluation 0.
- W pointer/version/bytes 변화 0 및 terminal W0 restore PASS.
- FULL FP32, autocast/quantization 0; optimizer/Adam/backward/parameter-gradient 0.
- Z+ clamp M5 0/50, M10 0/100. Z± M5 6/50=.12, M10 0/100. 모두 <.50.
- writer/cache append/heldout/Native access 모두 0.
- `llama-compute.csv`에 trajectory별 inner wall time과 ledger가 있다. 성공 Slurm allocation은 Stage1 354초 + Stage2 152초 = 506 GPU초이며, inner trajectory 합은 299.982초 + 113.179초 = 413.161초다. 성공 실행의 setup/report overhead는 92.839초다. 선행 기술 실패 3회의 Llama GPU allocation은 47초이며 science delta는 0이다.

## Claim 경계

B1_CASE01은 영구 calibration-only다. heldout/rewrite/rephrase/locality/evaluator는 `NOT_RECORDED_CALIBRATION_TRAIN_ONLY`이며 confirmatory denominator나 성능 claim에 포함되지 않는다. Qwen은 별도 `qwen-archival-exclusion.json`에 root 무결성만 보존되며 이 분석의 scientific input 또는 numerical-lock input이 아니다.
