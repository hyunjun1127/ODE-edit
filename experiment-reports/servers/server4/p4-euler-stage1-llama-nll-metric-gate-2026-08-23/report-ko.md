# P4-Euler Stage 1 Llama NLL 및 metric 안정성 상세 보고서

> **판정 범위:** `LLAMA_ONLY`, `B1_CASE01_PERMANENT_CALIBRATION_ONLY`, train target-only. 이번 사용자 지시에 따라 기존 latent endpoint 수렴 임계는 비결정적 진단으로 유지하고, ODESteer식 **Euler step/strength 변화에서의 task metric 안정성**을 결정 기준으로 적용한다.

## 결론

- 개정 상태: **TASK_METRIC_GATE_PASS** (`USER_DIRECTED_GATE_POLICY_REVISION`).
- 선택값은 그대로 `h=0.25`, `T_z=1.25`, main `M=5`다. 수치나 실행 결과를 다시 계산하거나 변경하지 않았다.
- 선택 h=.25의 train target-new NLL median은 다음과 같이 감소했다.

| arm | M1 (T=.25) | M3 (T=.75) | M5 (T=1.25) | M10 (T=2.5) | M1→M5 감소 | M1→M10 감소 |
|---|---:|---:|---:|---:|---:|---:|
| Z+ | 7.655735 | 0.752718 | 0.109818 | 0.018935 | 98.57% | 99.75% |
| Z± | 8.668875 | 1.007112 | 0.128479 | 0.028574 | 98.52% | 99.67% |

- 같은 `T_z=1.25`에서 M5와 M10을 비교해도 두 arm 모두 10/10 request에서 positive new-vs-true margin을 유지했고, 기술적 설명용 cutoff `new NLL<1`도 10/10이었다. M10 NLL은 M5보다 더 낮아졌다.
- 따라서 **편집 목적의 train-target metric은 step refinement에 대해 안정적으로 성공 영역에 남았다**고 판정한다.
- 기존 `d_i` endpoint geometry 실패(Z+ median .599819, Z± median .737896)는 삭제하지 않는다. 이는 **강한 latent endpoint 수렴 증거가 부족함**을 뜻하지만, 이번 개정 gate에서는 task metric pass를 뒤집지 않는 진단이다.

## 실험 구조와 해석 주의

Stage 1은 각 `model×arm×h`에서 하나의 10-step trajectory만 실행하고 M1/M3/M5/M10 prefix를 재사용했다. 따라서 고정 h에서 M을 늘리면 `T_z=Mh`도 늘어난다. 이 표는 **수치 refinement 표가 아니라 strength/pseudo-time curve**다. 순수 discretization 비교는 Stage 2의 `T_z=1.25`, M5(h=.25) 대 M10(h=.125) 표에만 해당한다.

## 선택 h=.25의 NLL, margin, geometry

| arm | M | T_z | new NLL mean/median/p90/max | true NLL mean/median | margin mean/median | disp median | field median | clamp |
|---|---:|---:|---|---|---|---:|---:|---:|
| Z+ | 1 | 0.25 | 7.911791/7.655735/10.846969/14.676372 | 6.206138/6.545202 | -1.705653/-1.647710 | 1.156111 | 3.557465 | 0/10=0.000 |
| Z+ | 3 | 0.75 | 1.882282/0.752718/4.802225/5.002145 | 8.572370/8.621489 | 6.690086/7.712534 | 2.189766 | 2.817732 | 0/30=0.000 |
| Z+ | 5 | 1.25 | 0.261539/0.109818/0.828353/0.918343 | 10.394453/10.931762 | 10.132915/10.741758 | 3.221756 | 0.517206 | 0/50=0.000 |
| Z+ | 10 | 2.50 | 0.030453/0.018935/0.059423/0.120638 | 11.942139/12.601961 | 11.911687/12.540236 | 3.257528 | 0.196306 | 0/100=0.000 |
| Z± | 1 | 0.25 | 8.213417/8.668875/10.084655/14.183981 | 7.803952/8.518075 | -0.409464/-0.930922 | 1.978628 | 5.365330 | 0/10=0.000 |
| Z± | 3 | 0.75 | 2.104400/1.007112/4.752781/5.709442 | 9.477684/9.096126 | 7.373284/7.808984 | 3.615834 | 2.118460 | 2/30=0.067 |
| Z± | 5 | 1.25 | 0.286403/0.128479/0.878321/0.925239 | 10.960939/11.558947 | 10.674536/10.996335 | 3.879441 | 0.481568 | 6/50=0.120 |
| Z± | 10 | 2.50 | 0.041665/0.028574/0.071092/0.149207 | 12.706823/13.630969 | 12.665159/13.595558 | 4.002851 | 0.199167 | 21/100=0.210 |


NLL은 target-new token의 length-normalized train NLL이다. `true NLL`은 현재 request의 target_true/old object NLL이며, `margin=true NLL-new NLL`이다. M3부터 두 arm 모두 median margin이 양수이고, M5에서 new NLL median이 Z+ .1098, Z± .1285까지 내려갔다. M10은 추가 pseudo-time이므로 main M5보다 강한 endpoint이지 별도 선택값이 아니다.

## 전체 h×M 결과

`stage1-all-cells.csv`는 32개 cell 전부에 대해 new/true NLL와 margin의 mean/median/p90/max, displacement, raw field norm, clamp 분모, finite, target/root identity를 제공한다. 아래는 new NLL median 요약이다.

| arm | h | M1 | M3 | M5 | M10 | M10 clamp | Stage1 h 판정 |
|---|---:|---:|---:|---:|---:|---:|---|
| Z+ | 0.0625 | 10.159258 | 6.644109 | 2.730289 | 0.232934 | 0.000 | ADMISSIBLE |
| Z+ | 0.25 | 7.655735 | 0.752718 | 0.109818 | 0.018935 | 0.000 | ADMISSIBLE |
| Z+ | 1 | 6.673706 | 0.186999 | 0.034464 | 0.006939 | 0.540 | INADMISSIBLE |
| Z+ | 4 | 6.600191 | 0.757391 | 1.457679 | 1.052816 | 0.760 | INADMISSIBLE |
| Z± | 0.0625 | 9.663174 | 4.777327 | 2.189796 | 0.175915 | 0.000 | ADMISSIBLE |
| Z± | 0.25 | 8.668875 | 1.007112 | 0.128479 | 0.028574 | 0.210 | ADMISSIBLE |
| Z± | 1 | 7.957236 | 0.223189 | 0.024685 | 0.006825 | 0.590 | INADMISSIBLE |
| Z± | 4 | 7.814718 | 2.933648 | 1.268874 | 0.924545 | 0.790 | INADMISSIBLE |


관측상 h=1은 일부 NLL이 낮지만 M10 clamp가 Z+ 54/100, Z± 59/100이며, h=4는 각각 76/100, 79/100이다. 따라서 낮은 NLL만 보고 큰 h를 선택하지 않았고, 원래 Stage1 bracket 판정대로 h=.25를 유지한다.

## 같은 T_z에서의 metric 안정성

| arm | M5 new NLL mean/median/p90/max | M10 new NLL mean/median/p90/max | positive margin M5/M10 | NLL<1 M5/M10 | clamp M5/M10 | latent d median/p90/max |
|---|---|---|---|---|---|---|
| Z+ | 0.261539/0.109818/0.828353/0.918343 | 0.051360/0.034850/0.122406/0.157564 | 10/10, 10/10 | 10/10, 10/10 | 0.000/0.000 | 0.599819/0.819999/0.837496 |
| Z± | 0.286403/0.128479/0.878321/0.925239 | 0.068243/0.058835/0.133394/0.188966 | 10/10, 10/10 | 10/10, 10/10 | 0.120/0.000 | 0.737896/0.865267/1.022678 |


`NLL<1`은 관측을 읽기 쉽게 표시한 기술적 cutoff이며 사전 선언된 새 임계가 아니다. 결정의 근거는 사용자가 지정한 task-metric 안정성 판정이다. request별 값은 `stage2-same-T-request-level.csv`에 있다.

## M 효과, h 효과, request 표

- `stage1-M-effects.csv`: h 고정에서 M1→3, M3→5, M5→10, M1→10의 NLL 감소율과 true NLL/margin/displacement/field/clamp 변화. `T_z`도 변하므로 strength 변화로 표기했다.
- `stage1-h-effects.csv`: M 고정에서 인접 h 변화의 NLL/margin/displacement/clamp 차이. h와 `T_z`가 함께 변한다.
- `stage1-cell-rankings.csv`: arm별 16개 cell의 new NLL median 순위와 admissibility. 순위는 observation-only이며 큰 h 사후 선택에 쓰지 않는다.
- `stage1-M10-request-level.csv`: 모든 h와 arm에서 M10 request 10개씩, 총 80행의 new/true NLL 및 margin.
- `stage1-selected-h025-M5-M10-request-level.csv`: 선택 h=.25의 M5(T=1.25)와 M10(T=2.5) request-paired strength 변화.
- `stage1-cell-correlations.csv`: arm별 16-cell 집계에서 NLL median과 pseudo-time/displacement/field/clamp/margin의 Pearson 상관. 기술적 요약일 뿐 인과 해석은 하지 않는다.

## 계산량과 overhead

- Stage1은 8 trajectory, 80 Euler microstep, 80회 `autograd.grad`, duplicate evaluation 0이다.
- inner trajectory 합: **299.982s**. 성공 Llama Slurm allocation: **354s**. setup/model-load/report overhead: **54.018s (15.26%)**.
- 평균 inner microstep: **3.750s**. peak GPU memory: **30.933 GiB**.
- FULL FP32, W0 restore, finite, optimizer/Adam/backward/parameter-grad 0을 유지했다.

## ODE claim 경계

이번 PASS가 지지하는 범위는 **한 calibration unit에서 Euler step/pseudo-time 및 same-horizon refinement에 대해 train-target NLL 성공 영역이 유지되었다**는 경험적 주장이다. 기존 latent endpoint `d_i` 실패는 그대로 남으므로, 다음은 주장하지 않는다.

- M5와 M10 latent endpoint의 강한 수치 수렴
- 연속 ODE 해의 존재·유일성 또는 solver-order 증명
- heldout Eff/Gen/Loc, rewrite/rephrase 성능
- writer 적용 뒤의 편집 성능 또는 sequential editing 성능

B1_CASE01은 계속 permanent calibration-only다. heldout/writer/cache/Native는 0이며 Qwen은 이 보고서의 scientific input이 아니다. `scientific_promotion=false`, 추가 GPU/ZA/ZB 제출은 0이다.
