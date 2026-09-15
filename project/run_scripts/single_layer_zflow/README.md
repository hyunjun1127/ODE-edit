# Single-layer write-coupled z-flow: CPU reference와 SH2 runtime

고정 native writer를 이용해 actual-write loss와 변경 비용을 함께 최적화한다. 원 publication `69b467d3`은 CPU reference만 포함했다. 아래 SH2 runtime을 새로 구현했으며, 구현·CPU 테스트·실제 8B Llama 기술 검증·과학 실행 완료는 각각 별도 receipt로 구분한다. 파일 존재만으로 실제 Llama PASS를 주장하지 않는다.

## SH2 실행 경로 (2026-09-16)

| 파일 | 범위 |
|---|---|
| native_binding.py | pinned BLUE context/target/lookup/key group port; compute_z 호출 없음 |
| llama_adapter.py | 모든 token의 L4 affine cache, fresh L5–31 suffix, 필요한 위치의 full-vocabulary head, actual write parity |
| runtime.py / config.py | FP32 nonsymmetric native LU, batch당 B/S, 고정 main 설정과 telemetry |
| durable.py | W/M/X/B/S/K 및 config/context/RNG/ledger의 hash manifest + fsync + atomic non-overwrite publish |
| technical.py | actual 8B calibration/holdout parity, microbatch1/2, physical commit, 별도 Python process resume |
| runner.py | fresh W0/M0 B100×10, committed endpoint 관측 복구, 원분모 유지 |
| provenance.py / run.sbatch | original16/inherited12 및 실제 source/import/asset freeze, explicit60416M 제출 진입점 |

원본16개와 authority는 exact publication Git bytes에서 task-local `authoritative/`로 보존한다. 새로운 source HEAD/tree와 input.lock은 별도다. Pretrained shard는 기존 full-hash manifest와 현재 size/header/index를 결속하며, 중복 full-content 재해시나 cross-hardware bitwise parity로 표기하지 않는다.

실제 기술 검증 명령은 source freeze 이후 다음 순서다. `input.lock.json`과 output은 create-once task-local 경로다. task-only PYTHONPATH는 기존 portable transformers4.44.2를 가리키며 공유 환경을 수정하지 않는다.

```bash
python -m project.run_scripts.single_layer_zflow.provenance prepare --source-root "$PWD" --output /mnt/raid5/janghj/ODE-edit/local/single-layer-zflow/20260916-v1/inputs/freeze-r1
python -m project.run_scripts.single_layer_zflow.technical --phase initial --lock INPUT_LOCK --output TECHNICAL_OUTPUT
python -m project.run_scripts.single_layer_zflow.technical --phase resume --lock INPUT_LOCK --output TECHNICAL_OUTPUT
python -m project.run_scripts.single_layer_zflow.runner --lock INPUT_LOCK --technical TECHNICAL_OUTPUT --output MAIN_OUTPUT
```

위 명령의 기술 단계는 MAIN과 별도 allocation/비용/분모다. Actual technical receipt가 없으면 MAIN은 실행하지 않는다. MAIN은 기술 W/M를 carry하지 않는다. Barrier/Adam 및 조건 없는 N4 재실행은 이 launcher 범위가 아니다.

## 완료 결과의 CPU 검산과 그림

`analysis.py`는 완료된 10개 checkpoint의 file/tensor/RNG 및 parent/W/M/source 연결을 독립 검산하고, 저장된 per-item NLL에서 RS/PS/NS·strict/token·paired lost/gained를 재계산한다. 모델·evaluator를 호출하지 않는다. 미완료 chain을 완료 결과로 보간하지 않으며, canonical 분모는 1,000개다. At-write pooling, W10 전체, W5와 W10의 같은 first500은 별도 표다.

CPU에서 base snapshot의 selected W0 tensor만 읽고, 저장 W의 FP64 차이로 actual cost를 재계산한다. 각 M이 이전 M에 native CPU FP32 K@K.T를 한 번 더한 값과 정확히 같은지 확인한다. 모델 객체 생성·forward는 없으며 CPU threads=8을 실행과 맞춘다. 이는 saved end-state의 추가 검산이며 새로운 GPU resume/replay가 아니다.

```bash
python -m project.run_scripts.single_layer_zflow.analysis --root MAIN_OUTPUT --input-lock INPUT_LOCK --input-sha256 INPUT_SHA --output NEW_AGGREGATE_DIR --private-output NEW_LOCAL_PRIVATE_DIR --n4-raw N4_RAW --n4-sha256 N4_SHA
python -m project.run_scripts.single_layer_zflow.plots --aggregates NEW_AGGREGATE_DIR --output NEW_PNG_DIR
python -m project.run_scripts.single_layer_zflow.reporting --aggregates NEW_AGGREGATE_DIR --figures NEW_PNG_DIR --output NEW_REPORT_DIR --technical TECHNICAL_OUTPUT --n4-reuse N4_REUSE_MANIFEST --allocation TERMINAL_ALLOCATION_RECEIPT
```

`plots.py`는 집계 CSV만 읽고 고정 matplotlib 설정으로 PNG를 생성한다. 입력/code/output SHA와 재현 명령은 plot receipt에 포함된다. 그림의 current cohort 곡선을 final retention으로 해석하지 않는다. 같은 subject/relation에서 나중 batch의 다른 target이 있는 경우는 `SUPERSEDED_CANDIDATE_LATER_BATCH`로 별도 계수한다. 동일 batch 충돌과 relation 미기록은 별도 상태이며, 이 분류는 실제 forgetting 원인 인증이나 분모 제외가 아니다.

- [전체 method 파이프라인](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-16-single-layer-zflow-pipeline-v1.md)
- [실행 설정](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-16-single-layer-zflow-contract-v1.json)
- [선행 리뷰](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-16-single-layer-zflow-design-review-ko.md)

## 흐름

`W_entry/M_entry → K/B/S → all-token affine cache/entry teacher → X=0 → IMEX + fresh suffix L/g → 종료 → 실제 FP32 weight parity → history 1회 확정`

Main reference는 positive price=1, barrier off, 최대 25개 complete logical oracle sweep를 사용한다. Price는 미튜닝 초기값이다. 25회 내 최적 endpoint나 실제 편집 효능을 주장하지 않는다. `RESOURCE_STOP`도 마지막 accepted 후보를 반환하며 commit 결과와 별도로 기록한다.

## 파일

| 파일 | 역할 |
|---|---|
| flow_core.py | Native map, quadratic cost/metric, IMEX, barrier off/fixed/exponential, 상대 budget KKT, step 회복 |
| oracle.py | 전역 token weights, 고정 teacher reverse KL, all-token affine cache, complete X gradient |
| transaction.py | CPU FP32 weight/history materialization 및 in-memory exactly-once |
| demo.py | 비선형 causal toy suffix에 전체 경로 연결, JSON trace/receipt |
| tests/ | 수식·수치 종료·gradient·commit 회귀 검사 |

## CPU 실행

원 CPU reference만 실행할 때는 저장소 root에서 cached uv/PyTorch 환경을 사용한다. 실제 adapter를 포함한 전체 테스트는 아래 별도 pinned 의존성이 필요하다.

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --offline --no-project --with torch python -m unittest project.run_scripts.single_layer_zflow.tests.test_flow_core project.run_scripts.single_layer_zflow.tests.test_oracle_transaction project.run_scripts.single_layer_zflow.tests.test_pipeline -v
PYTHONDONTWRITEBYTECODE=1 uv run --offline --no-project --with torch python -m project.run_scripts.single_layer_zflow.demo --output /tmp/single-layer-zflow-demo.json
```

이미 PyTorch가 설치된 환경에서는 `uv run --offline --no-project --with torch python` 부분을 해당 Python으로 바꾸면 된다. Demo는 CPU tensor만 만들며 checkpoint 다운로드나 원격 실행을 하지 않는다. CPU suffix callback은 full logits를 반환한다. 실모델에서의 selected-position head 최적화는 별도 adapter에 속한다.

SH2 실제 adapter를 포함한 전체 CPU 회귀검사(작은 synthetic Llama, 다운로드/GPU 호출 없음):

```bash
PYTHONPATH=/mnt/raid5/janghj/ODE-edit/local/fixed10k-preedit-eval/attempt-v1/deps-transformers-4.44.2 \
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
/mnt/raid5/janghj/EasyEdit/.venv/bin/python -B -m unittest discover -s project/run_scripts/single_layer_zflow/tests -q
```

이는 새 의존성을 설치하거나 공유 환경을 바꾸는 명령이 아니다. Task-local sealed transformers4.44.2 경로가 없거나 다른 버전이면 actual adapter 검사는 fail-close한다. 이 CPU 테스트 통과를 실제 8B parity/SEQ1000 완료로 대체하지 않는다.

## 해석 경계

- Solver status와 commit status는 별개다. 수렴은 reduced-space 1차 조건이며 edit 성공 인증이 아니다.
- Fresh oracle 1회는 전체 logical request/context/token sweep다. Scalar barrier 계산은 모델 평가 횟수에 포함되지 않는다.
- 작은 budget의 잘못된 boundary 판정을 상대 slack으로 수정했다. Raw 및 normalized residual을 함께 기록한다.
- Teacher/prefix 준비와 terminal parity 비용은 oracle 횟수 밖에 별도로 기록한다.
- Candidate/reject에서 W/M를 변경하지 않는다. 실제 commit cost는 저장될 weight와 entry 차이로 다시 계산한다.
- 원 `transaction.py`는 in-memory reference다. 새 `durable.py`는 Linux local-filesystem single-writer, complete-bundle/parent 검증과 idempotent commit을 제공한다. 분산 multi-writer 보장을 주장하지 않는다. Actual separate-process Llama resume은 `TECHNICAL_VALID.json`이 있는 경우에만 주장한다.
