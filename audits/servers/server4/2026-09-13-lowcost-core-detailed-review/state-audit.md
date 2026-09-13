# Job46451 저장 상태·native fitting 독립 CPU 검산

범위는 이번 core의 봉인 source와 CPU `torch.load(weights_only=True, mmap=True)`이다. 신규 모델 로드·forward·평가·GPU replay·scheduler 조회·원격 raw 접근은 하지 않았다. 원본 source/raw는 변경하지 않았다. 후보 선택과 성능 원인 판정은 하지 않는다.

## 결과와 검증 수준

6개 endpoint와 공통 prepared snapshot 1개(총7개)의 selected W/M, context 및 RNG를 검산했다. 모든 W4/W8는 `[4096,14336]`, M4/M8는 `[1,14336,14336]`, dtype은 FP32이며 저장된 selected tensors에 nonfinite가 없었다. 각 endpoint의 실제 tensor SHA가 commit/state/evaluation의 SHA와 일치했다. 저장 state 간 비교에서는 잘못된 layer write, 다른 branch의 M 유입, 잘못된 endpoint 평가 결속을 발견하지 않았다. 이것은 독립 GPU 재현 또는 비계측 원본 대비 model-level parity PASS가 아니다.

|항목|실제 검산/기록|검증 범위·한계|
|---|---|---|
|core endpoint|N4/S875/S75/FULL8/RES8/REFIT4, 6/6|core만 완료; audit·suffix 미실행|
|first fit|N4 L4, compute_z100/keys1/readout1/solve1|같은 Server4 current-entry fresh-native; historical/S1 native 재사용 아님|
|second fit|FULL8 L8/RES8 L8/REFIT4 L4 각 z100/keys1/readout1/solve1|partial W와 공통 M/P/context/RNG hash 직접 대조|
|총 fitting|request-z400, solve4|저장 capture에도 각 fit의 z100/keys1/readout1 존재|
|fit 중 history|4회 모두 append0|source에서 final history loop만 제거, receipt의 M/P byte 불변 검사|
|endpoint history|총8 append: N4/S875/S75/REFIT4 각1, FULL8/RES8 각2|각 before M=공통 entry M, after M=해당 snapshot M; finalizer별 keys1, z/solve0|
|M8 setup|50 batches×100 events=5000, chronological, 재구성1회|공통 We의 key 재인코딩 source와 step receipt; 개별 과거 K tensor는 저장되지 않아 CPU Gram 재합산은 미실행|
|P mapping|물리 L4→source0→local0, L8→source4→local0|prepared metadata/runtime assertion과 source 확인; 이번 검산은 전체 5-stack 재해시를 중복하지 않음|
|eval state|6/6 actual endpoint state 일치, nonmutation=true|원실행의 parameter pointer/version + selected W/M/P/context/RNG guard; 새 model evaluation 없음|
|RNG/context|prepared 및6endpoint 실제 serialized digest가 공통 entry와 동일|CPU metadata 비교; GPU continuation replay 아님|
|process restore|selected_W0_exact=true, RNG_restored=true|M은 process-discarded; selected parameter version 복구는 주장하지 않음|

## 실제 stored-weight action

단위는 `||W_endpoint−W_entry||_F`이며 FP32 저장 tensor를 CPU float64 차분·norm으로 집계했다. Native metric/C-reg norm이나 target realization으로 이름붙이지 않는다. L4/L8의 서로 다른 target을 같은 activation potential로 비교하지 않는다.

|arm|L4 actual net Frobenius|L8 actual net Frobenius|
|---|---:|---:|
|N4|10.7440168052634|0|
|S875|9.401014704161975|0|
|S75|8.058012604281407|0|
|FULL8|10.7440168052634|5.214695775869617|
|RES8|8.058012604281407|6.044554162789381|
|REFIT4|9.285660014794022|0|

D4는 실제 N4 endpoint에서 prepared entry를 뺀 FP32 차분이다. CPU에서 `entry + alpha*(native-entry)`를 다시 계산하여 .875/.75의 materialization raw-byte SHA와 대조했다. REFIT4를 제외한 모든 arm의 final L4가 해당 materialization과 exact tensor equal이었다. REFIT4는 두 번째 L4 fit의 entry SHA가 .75 materialization이고 final SHA가 final snapshot과 일치했다. L8 donor가 없는 네 arm에서는 W8와 M8가 prepared와 exact equal이었다. Alpha1은 N4 tensor exact copy이며 alpha0은 이번 core endpoint에 없으므로 source 분기/기존 CPU test 수준으로만 확인한다.

Native 및 세 second-fit의100개 저장 z를 순서대로 hash한4개 root는 서로 달랐다. 이는 다른 captured target bytes와 호출 경로의 증거이지, target 품질 또는 loss 수렴을 증명하지 않는다. FULL8/RES8의 fit-entry M8는 같은 공통 M8이며 두 arm 모두 L8 fit-entry W8=We의 W8, 그러나 L4 partial W는 각1.0/.75 D4이다. REFIT4는 .75 D4의 실제 L4에서 다시 target을 계산했다.

## Source 근거와 의미론

실행 source는 `7ece056c33fbb4246245c15f5f7c2a678315c05c`, tree `352cdd8ca3f3d7b6f2b15f3ed6a6f775fb478443`이다. analysis commit과 구분한다.

|실행 파일|SHA256|주요 line 근거|
|---|---|---|
|`project/run_scripts/low_cost_write_donor_pilot/runtime.py`|`45f34c0014d86003aace41c869cfdbe0e554871c68d32295998c511b8000afff`|68–80 nonselected/eval guards;85–107 entry/M8;109–118 branch restore;122–145 fresh fit/materialize/finalize;146–148 initial gate;150–156 endpoint reload/eval;167–177 process restore|
|`project/run_scripts/low_cost_write_donor_pilot/fitting.py`|`859ee2fcfc673344e72c3af8380768304a49fb4682726e31ef62d82b5d5c0db7`|23–69 AST split;93–108 singleton policy;110–143 pass-through counters;145–175 fit;177–205 finalization;208–215 P mapping;218–237 FP32 materialization|
|BLUE `AlphaEdit/AlphaEdit_main.py`|`79da927aad5ab817fd008c5958768adcd00556989a8efbaa2c4bdc80d8fc842e`|89 native current-layer z;111–140 keys/residual/solve/materialize;237–239 final history|

Fitter는 module-global monkeypatch 대신 원본 module globals 사본에서 AST function을 compile한다. 원본 apply 마지막 history for-loop를 fit에서 제거하고 동일 loop를 finalizer로 옮긴다. 원래 target/solve 수식은 유지한다. blue=True singleton에서 residual divisor는1이고 L2=1이다. Native BLUE `compute_z.py`117–183은 최대25 loss evaluation, 마지막 iteration에서는 optimizer.step 전 break, 따라서 최대24 Adam update이며 loss<0.05이면 더 일찍 종료한다. `compute_z=100`은 optimizer iteration=100이 아니다. 실제 optimizer 반복 횟수는 structured fit receipt에 없고 별도 log 검산이 필요하다.

raw tensor hash convention은 두 가지다. Runtime fixtures hash는 dtype/shape header를 포함하고 fitting의 hash는 tensor bytes만 포함한다. 서로 다른 값이라는 이유로 mismatch로 판단하지 않고 CPU 검산에서 각각 같은 convention을 재현했다.

## 미검증·범위 제한

- 원본 uninstrumented writer와 logger-off/on actual model-level parity는 **NOT_TESTED**다. 원본 AST 수식 보존과 캡처/계수 비개입 source 검사는 이를 대체하지 않는다.
- Runtime guard가 확인하는 것은 named parameters의 object/pointer/version와 필요 시 byte SHA 및 명시적 selected M/P/context/RNG이다. 임의 non-parameter buffer, 모든 module global 또는 외부 library state를 포괄하는 proof는 아니다.
- `prepared.pt`/각 endpoint는 W4/W8+M4/M8+contexts/RNG와 base revision/sample metadata를 담은 selected-state snapshot이지 full pretrained model checkpoint가 아니다. 해당 base/source/P 참조가 필요하며 full GPU continuation은 이번에도 수행하지 않았다.
- process 종료에서는 selected W0/RNG를 복구하고 M은 process와 함께 폐기했다. common entry M을 process 종료까지 복구했다거나 parameter version이 원래값이라고 주장하지 않는다. Branch 사이 entry 복구와 종료 cleanup은 다른 계약이다.
- M8의5000event key마다 SHA와 시간은 있으나 K 자체의 전체 저장은 없으므로 raw tensor만으로 Gram을 재합산하지 않았다. 실제 source/실행 step receipts, 최종 M8 byte identity 수준이다.
- `INITIAL_VALID.json`은 이번 recall에서 사후 확인했다. 지난 pending/pause 시점에 agent가 actual gate PASS를 관측했다는 의미로 소급하지 않는다.
- 성능 null/저하와 correctness 오류는 구분한다. 이 감사는 어떤 후보도 선정하지 않으며 claim-decision은 PENDING_GH_REVIEW다.

## 재현

CPU 전용 `project/run_scripts/low_cost_write_donor_pilot/review_state.py --root <sealed output> --output <new create-once JSON>`을 실행한다. 원실행 root는 `/data/janghj/ODE-edit/local/low-cost-write-donor-pilot/20260913-v1/attempt-v1/output`이다. 첫 검산 결과는 같은 task control의 `state-cpu-review-v1.json`; whole-file hash inventory는 parent 분석의 독립 manifest와 함께 결속한다. 새 code에서는 중복 large-file SHA를 생략하고 tensor SHA/state 검산에 집중한다. 처음 실행한 code는7개 snapshot file SHA도 수행했으며, 이후 추가 CPU metadata 검사에서7개 RNG/context digest와 P mapping을 확인했다. 원본 raw/code 변경0, GPU0.
