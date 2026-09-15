# Saved-episode FD repair — bounded red preflight

2026-09-15. Source/CPU 판정이며 실제 Llama GPU 미분·FD PASS 또는 scientific G0가 아니다. 신규 독립 repair 지시/dispatch를 FULL_READ하고 지정 변경 파일만 검토했다. 이전 paused task를 재개하지 않았다.

## 판정

현재 검토된 코드에서 제출을 차단할 확정 math/state 결함은 발견하지 못했다. 실제 모델에서 E/D 각각 direct-route와 두 방향의 resolved FD가 통과하는지는 **NOT_TESTED_GPU**다. 낮은 성능을 실패 조건으로 추가하지 않았다.

| 항목 | source 근거 및 한계 |
|---|---|
| 저장 episode 재사용 | `repair_runtime.py:168` 진단 경로에는 native fitter/target/solve/map 재구성이 없다. 저장 Vp/A를 로드하고 Vp SHA/shape와 request order를 확인한다. postfit RNG/옛 gradient bytes/완전 continuation 복원을 주장하지 않는다. |
| 독립 미분 | `model_adapter.py:307` 직접 selected-weight leaf는 `_forward`/custom affine를 우회한다. E100 request mean과 S64 document mean은 기존 token/scoring 경로를 그대로 공유한다. `repair_checks.py`는 gC 대 gW Aᵀ 및 self/고정 독립 방향 bilinear를 검산한다. neural backward 전체의 독립 증명은 아니다. |
| 조기 저장 | `model_adapter.py:457` 및 `repair_runtime.py:37`은 각 완료 E/D gradient를 다음 단계 전에 tensor/rows/hash/RNG와 create-once 저장한다. probe C/input을 forward 전에, 완료 observation을 reducer 전에 저장한다. |
| FD 계약 | `repair_checks.py:29` 고정 10-scale grid/두 방향/두 objective, 최대80 signed observations, 상대 .15/절대1e-7/수렴분모 max(absAD,1e-7) 유지. 최소2개 인접 resolved smaller scales만 PASS; deterministic 첫 valid window를 선택한다. near-zero/동일 materialized amplitude는 UNRESOLVED다. E 두 방향 PASS가 D FD의 선행조건이다. |
| 방향/비용 | 독립 방향은 사전 고정 CPU Rademacher seed E2026091501/D2026091502이며 gradient 값으로 재선택하지 않는다. zero 관측3/objective 및 별도 materialization/direct/true-target 비용이 lock에 선언된다. |
| 실패 폐쇄 | `repair.sbatch`의 `set -euo pipefail` 후 technical process→fresh science process 순서다. `runner.py:70`은 model load 전에 양 objective/direction PASS와 exact scientific lock/source를 검증한다. 실패한 B1을 resume하지 않고 pretrained W0/cold M0에서 신규 단일 chain을 시작한다. |
| 과학식 보존 | 변경 diff에는 policy/ledger/native fitter/native map/원 BLUE source 변경이 없다. runner의 변경은 early diagnostics와 기술 검증 route/admission이다. 새 actual B1 Vp/A/gE/gD가 다르면 endpoint checks를 다시 수행하며 같을 때만 정확 evidence를 재사용한다. |
| 자원/중지 | script는 1 GPU/8 CPU/60416M/12h/export NONE. operations는 active/pending/configuring/completing 예약과 신규1의 합<=2를 검증한다. 실제 scheduler admission은 부모 소유이며 이 감사에서는 실행하지 않았다. PENDING/G0/terminal technical HOLD 후 agent 중지 정책을 유지한다. |

## WARN / 검증 한계

- 완료 objective 단위 early-save다. 한 sweep 내부 forward가 중간에 예외를 내면 그 미완성 sweep의 누적 gradient/rows는 저장되지 않을 수 있으나 직전 완료 objective와 실패 위치는 보존된다. 미완성 자료를 완성된 gradient라고 보고해서는 안 된다.
- 조기 materialization receipt WARN은 후속 좁은 수정으로 해소했다. `technical.py:107` optional recorder가 numerical FAIL raise 전에 receipt를 저장하고 `repair_runtime.py:115`가 create-once recorder를 제공한다. 계산/기존 default 동작은 불변이다. full forward 자체가 중간 예외면 완성 numerical receipt를 꾸며내지 않으며 상위 failure evidence가 남는다. exact-episode reuse의 추가 C0 확인은 기존 fail-closed 경로를 유지한다.
- full model/teacher의 기존 SHA+stable-stat 재사용은 새로운 full byte rehash가 아니다. CPU tiny fixtures는 실제 Llama FP32 backend parity의 대체물이 아니다.
- failed diagnostic process는 종료되고 science로 진행하지 않는다. W0 restore exact 선언은 성공 경로에만 존재한다. 실패 시 정상 restore 완료를 추정하지 않는다.

## 독립 CPU 확인

명령: `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /data/janghj/EasyEdit/.venv/bin/python -m unittest project.run_scripts.bg_tw_reference.ep_tw.test_direct_gradient project.run_scripts.bg_tw_reference.ep_tw.test_repair_checks project.run_scripts.bg_tw_reference.ep_tw.test_repair_runtime`

결과: **35 tests OK**, 1.423s. actual production custom Function의 작은 FP64 gradcheck, custom-node 우회, 조기 저장, 방향/FD reducer, E-before-D, exact technical receipt fail-close를 포함한다. 모델/GPU/Slurm/polling/push/commit=0.

## 검토 당시 파일 SHA256 (source freeze 전 draft)

| 파일 | SHA256 |
|---|---|
| model_adapter.py | 9e4d440e9aae260191d2adcfcc631c9d418311f1c4a8a76b7df46c8e9bb7dba5 |
| repair_checks.py | e34d6d5b87c9126dee1431494508b1243fb75056304cd378a33043791292bc0b |
| repair_runtime.py (materialization recorder 수정 후 좁은 재확인) | 17deff5c83ce925031e34300f32300e0678d0f907f5ba733fae6732b2583ab9a |
| technical.py (optional recorder만 추가) | 8ee4e0ac9b12c5c9149fd7034405cfa74142391b12224711ffacd35023b4b854 |
| repair_control.py | 8fccedcb092d3895ddc92ec37c92de9a5beb64f5eb758939dc58d3c697e7d906 |
| runner.py | 339a0b2fbc1fbf64b467dd6e43a59b861ce15ef5e87cb20b7af4d4cf81859442 |
| operations.py | bbe2031a44ae466ace6c054708da8afe5f6cf05117cc66b0e10d869518114d1b |
| repair.sbatch | fa1f1c535d9536a4cf2a56b63dc8504692d2eb89c73f401b8f959ad78e606552 |

Parent final freeze 이후 source identity는 이 draft와 구분하여 결속한다. 기술 GPU PASS·G0·완료 수치는 아직 없다.

35 CPU tests는 recorder 수정 전 독립 실행 결과다. 수정 후 전체 재실행은 부모 소유이며 그 결과를 본 감사의 독립 실행으로 중복 주장하지 않는다. 한정 감사 종료, 이후 자동 재검토/monitoring 없음.
