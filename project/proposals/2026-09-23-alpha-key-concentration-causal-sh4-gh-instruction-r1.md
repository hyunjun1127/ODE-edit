# GH 실행 지시 — AlphaEdit 원본 key 집중의 원인·결과·수정 연산 식별 / SH4

Instruction ID: `ODEEDIT-GH-SH4-ALPHA-KEY-CAUSAL-20260923-R1`

수신: `ODE-edit_GH (2)` / GH `01a04939-8873-7673-8dca-4c7fc5e31af0`.
실행 소유자: `ODE-Edit_SH4 (2)` / SH4 `01a04939-b5c7-7a03-ba2d-ef3343d62cfd`.
대상 repository: `hyunjun1127/ODE-edit`, remote `https://github.com/hyunjun1127/ODE-edit.git`.
GH CWD: `/mnt/raid5/janghj/ODE-edit`; SH4 CWD: `/data/janghj/ODE-edit`.

## 1. 사용자 명령과 실행 경계

사용자는 이번 실험을 GH가 SH4에 지시하고, **server2에 있는 원본 AlphaEdit checkpoint를 server4에 복사하여 실행**하도록 명시적으로 승인했다. **초기 gate 통과 후에는 모니터링하지 말라**는 지시도 함께 주었다.

GH는 아래 자료를 읽고 SH4에 실제 실행 envelope를 전달하라. 문서 작성 또는 task 생성만으로 완료 처리하지 말라. SH4가 구현·전송·초기 검증·실제 제출을 담당한다. 사용자에게 동일 실행·복사를 다시 승인받지 않는다. 다른 실행 중 작업은 중단하거나 변경하지 말고, SH4가 수행 중이면 관련 작업과 구분하여 이 지시를 인계한다.

이번 실행의 필수 범위는 설계의 **E0–E4 원인 식별 패키지**다. 아래 고정된 연산 대조는 모두 관측하며 PR/NS 향상을 성공 gate로 두지 않는다. 조건부 method 선택·W70부터의 수정 method sequential·W0 full10k는 기전 결과를 검토한 뒤 별도 명령으로 결정한다. 초기 gate 후 GH가 해석·arm 선택을 위해 모니터링해야 하는 종속 작업을 현재 job에 숨겨 넣지 않는다. 후속 단계는 결과물에 `FOLLOWUP_NOT_SUBMITTED`로 명시한다.

## 2. 전달할 원본 자료

이 자료는 현재 GH와 같은 host의 ignored local 아래 있으며 Git pull만으로 SH4에 전달되지 않는다. 문서와 JSON/CSV의 실제 bytes와 SHA256을 묶어 실행 위치로 복사하라.

입력 bundle root:
`/mnt/raid5/janghj/ODE-edit/local/alpha-key-concentration-causal-design-20260923-v1/`

필수 파일:

- `design-ko.md`, `contract.json`, `cells.csv`.
- `geometry-panels.json`, `history512.json`, `neighborhood512.json`, `order-controls.json`.
- `preparation-receipt.json`, `validation-receipt.json`, `algebra-checks.json`.
- `checkpoint-transfer-manifest.json`, `source-extraction.json`, `source/AlphaEdit_main.py`, `source/runtime.json`.
- `prepare-design.py`, `finalize-design.py`: CPU 준비/검증 도구이며 GPU runner가 아니다. Hardcoded source paths를 무단 실행하지 말고 원본 SHA를 유지하여 필요한 입력 경로를 명시적으로 결속한다.

추가 근거:

- `local/alpha-native-census-20260923-v1/review-ko.md`, `checkpoints/checkpoint-census.csv`, `checkpoints/validation.json`.
- `audits/global/2026-09-22-alphaedit-native-criticality-audit/target-native-execution.lock.json`, `target-native-bindings.json`.
- `local/datasets/counterfact-fixed-10k-v1/counterfact.json`.

SH4 입력 bundle 권장 경로:
`/data/janghj/ODE-edit/local/alpha-key-concentration-causal/20260923-r1/inputs/design/`.

Design source와 실제 execution source를 구분하여 각각 manifest를 만든다. 현 설계의 `cells.csv`는 조건부 후속과 대비 family까지 포함한 목록이다. 101행을 101개 GPU job으로 무조건 제출하지 않는다. 위 1절이 이번 제출 범위다.

## 3. Server2 → Server4 checkpoint 복사

Source host: `rke-server2`.

Source root:
`/mnt/raid5/janghj/ODE-edit/local/checkpoint-archives/server4-migration-20260911-v1/new177-v1/payload/local/fixed10k-native-baselines/attempt-v1/output/main-cell-1/`

Destination host: `rke-server4`.

Destination root:
`/data/janghj/ODE-edit/local/alpha-key-concentration-causal/20260923-r1/inputs/checkpoints/BASE_ALPHAEDIT/`

복사 대상은 B001/B005/B010/B020/B030/B040/B050/B060/B070/B080/B090/B100의 `W-method-state.pt` 12개다. 원본 terminal manifest 기준 **63,418,321,276 bytes, 약59.06 GiB**다. 각 파일의 expected bytes/SHA는 `checkpoint-transfer-manifest.json`에 있다.

GH가 사용자 승인 근거를 transfer approval에 기록하고, GH가 지정한 SH4 또는 source 담당 SH2가 서버 간 SSH/rsync로 복사한다. GH host에 checkpoint 전체를 중간 저장하지 않는다. 원본은 보존하고 `--delete`를 쓰지 않는다. 전용 destination의 부분 전송을 재개할 수 있게 하되 완성 파일은 checksum 확인 후 확정한다. 이미 server4에 같은 파일이 있다면 bytes/SHA가 일치하는 것만 재사용하고 `REUSED_VERIFIED`로 기록한다. 같은 이름이라는 이유로 대체하지 않는다.

함께 필요한 소형 sidecar `contexts.json`, `commit.json`, `native-observation.json`은 존재와 SHA를 확인해 명시적 allowlist로 복사한다. 과거 전체 output root를 통째로 복제하지 않는다. Target cache, P, dataset, base snapshot도 server4의 기존 자산이 정확히 일치하면 재사용한다. 누락 자산은 lock에 있는 정확한 파일만 추가 전송한다.

복사 전 server4의 실제 여유 공간을 검사한다. Checkpoint59.06GiB 외에 geometry context-key 약48.2GB, writer-mean key 약8.0GB, branch scratch·부분전송·평가 산출물이 필요하다. 원본/기존 실험을 삭제해 공간을 만들지 않는다. 부족하면 필요한/가용 bytes를 기록하고 pending한다. 전송 자체가 완료됐다는 것과 content 검증을 통과했다는 것을 별도 상태로 남긴다.

## 4. GH → SH4 권한 envelope

- Branch/worktree: 격리된 `codex/alpha-key-causal-sh4-20260923-r1`. 현재 사용자/타 작업의 dirty 파일을 reset, stash, overwrite하지 않는다.
- 구현 허용: `project/run_scripts/alpha_key_concentration_causal/` 안의 loader, source binder, geometry observer, native wrapper, branch runner, intervention, reducer, tests, launcher. Native 원본 파일은 보관본을 dependency로 사용하고 science 계산을 몰래 변경하지 않는다.
- SH4 문서 허용: `plans/updates/server4/alpha-key-causal-20260923-r1/`, `audits/servers/server4/alpha-key-causal-20260923-r1/`, `experiment-reports/servers/server4/alpha-key-causal-20260923-r1/`, `messages/server-heads/server4/alpha-key-causal-20260923-r1.md`, 대응 task/run metadata 및 transfer verification.
- Raw/output: `/data/janghj/ODE-edit/local/alpha-key-concentration-causal/20260923-r1/` 아래. Checkpoint, dataset, dense key/activation/logits, full log는 Git에 넣지 않는다.
- 제출: 이 사용자 명령에 따라 GH가 E0–E4 구현·초기 검증·본실행을 포함하는 envelope를 승인한다. SH4의 scoped Slurm 제출을 허용한다. 실행 source를 봉인하고 기술 검증을 끝내는 내부 절차는 유지하며 사용자 재승인 단계로 바꾸지 않는다.
- 자원: 기본 동시1job/1GPU. 현재 SH4 project GPU/host-memory cap과 기존 job을 실제 확인하여 요청한다. 지난 registry의 자원 관측을 현재 가용량으로 쓰지 않는다. Job wall time은 측정한 microbatch/solve 비용과 실제 허용 한도에서 결정한다.
- Boundary: 현재 session/CWD/remote 확인 후 전용 worktree와 실행 경로를 봉인한다. 모델 profile을 강제로 바꾸지 않는다.
- Git: GH가 clean integration checkout에서 검토한 소형 문서/source/manifest만 명시적으로 통합한다. Root의 다른 untracked 분석을 일괄 add/push하지 않는다.
- 보고: SH4는 사실·수치·분모·파일·기술 오류만 기록한다. 인과적 해석, method 승격, 새 arm 선택은 GH의 후속 리뷰 영역이다.

## 5. 변경하면 안 되는 과학 계약

대상은 **BASE_ALPHAEDIT 하나**다. blue=False, L4–L8, L2=10, 원본 P/nullspace_threshold=.02, batch-entry에서 L8 z 계산, 원본 residual divisor5/4/3/2/1을 유지한다. BLUE/L4-only/MEMIT arm으로 치환하지 않는다.

Model revision `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`, fixed10k dataset SHA `3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1`, 원본 FP32/eager, matmul TF32=false, cuDNN TF32=true를 결속한다. Stored bare1/generated5 context와 실제 group 평균(.5/.1)을 재사용한다. Prefix를 새로 생성하지 않는다.

- Geometry: E/M/O/L 각1000 × W0/W10/W50/W70/W80/W90/W100 =28 cells. Raw/centered/unit-normalized, mean energy, norm outlier, raw/projected, individual-context/actual-mean 통계를 모두 남긴다.
- 내부 calibration512와 assessment3488를 유지한다. N512와 H512를 서로 바꾸지 않는다.
- N512 전체는 observer-only이며 basis/gradient/strength/arm 선택에 사용하지 않는다. Current paraphrase도 observer-only다.
- H512 통계에는 superseded를 포함하여 all512 weight1. 기능 평가에서만 entry-time overwrite validity mask를 분리한다.
- Hybrid: W80/W100의 W4/W5/W6 base/checkpoint 8조합과 upper-layer negative control.
- 단일 batch: W50→B51, W70→B71, W80→B81, W90→B91. 각 entry z1회, 총400 requests. Post-z same-entry fork만 z를 공유한다.
- 모든 branch의 W/M/RNG/cursor를 독립 복원한다. Downstream K/residual은 해당 branch에서 재계산한다. 마지막 L8 write 후 residual도 측정한다.
- O-H: NATIVE/SHAM/H5/H6/H56/MASS56. Current/history penalty mismatch와 bank coverage를 기록한다. M_eff는 임시 solve operand이며 persistent cache는 native timestamp append만 수행한다.
- O-W: no/mean/centered/full component, full-hook=weight-write parity. 정의된 common direction이 nonzero이면 rank1 실제 weight ablation과 norm-matched control도 수행한다. 정확히 정의되지 않으면 `NOT_APPLICABLE`과 이유를 남기고 N에 유리한 다른 방향을 고르지 않는다.
- K/R 2×2: 같은 P/M/λ 및 receiving state에서 operand 효과와 interaction을 측정한다. Inference key patch와 writer operand 교체를 구분한다.
- 이 패키지는 고정된 반사실 대조의 관측을 수행한다. 성능이 낮다는 이유로 arm을 누락하거나 method를 새로 튜닝하지 않는다. 설계의 과학적 조건부 승격은 이후 리뷰에 남긴다.

## 6. 초기 gate의 정확한 정의

초기 gate는 **기술적 유효성과 자율 실행 가능성**이다. RS/PS/NS/PR 개선을 요구하지 않는다.

1. G0: session/source/config/model/data/context/P/checkpoint content identity, 전송 SHA, 자원·공간 및 native arm 결속 PASS.
2. G1: source/CPU 검사와 작은 actual-model prefix/full parity, key position/padding/현재 tokenizer, own/upper-layer key invariance, hook OFF/full Δ parity, W/M/RNG 복원 PASS. H512 timestamp 재구성의 native key SHA 또는 사전 반복에서 정한 수치 envelope를 확인한다. 불일치를 PSD clipping·dtype 변경·bank 재선택으로 숨기지 않는다.
3. G2: **W50→B51의 첫 full native100 batch와 SHAM 대조**가 완료되고 필수 K/R/Δ/stage/observer/source/cost 산출물이 생성된다. N512/H512/current R/P 전부와 overwrite mask, final L8 residual, branch state 격리가 확인된다. 이 B51을 뒤에서 다시 최적화하지 않고 유효 산출물을 재사용한다.
4. G3: 나머지 E0–E4 실행 queue와 reducer/report 생성이 agent의 반복 개입 없이 진행 가능하고, 실행 source/전체 argv/input manifest/job ID/작업 순서가 봉인된다. 실패 시 원인·상태를 파일로 기록하고 자동으로 scope를 바꾸지 않는다.

GH는 SH4의 실제 gate receipt와 제출 ID를 한 번 받아 검증한 뒤 `INITIAL_GATE_PASS_MONITORING_STOPPED`를 기록하라. Job을 제출했거나 PENDING이라는 이유로 gate PASS라고 쓰지 말라. Gate 전에 기술 실패가 나면 해당 attempt를 분리하여 scope 내 기술 수정은 가능하지만 개선된 성능을 만들기 위한 정책 수정은 금지한다.

## 7. 초기 gate 이후 모니터링 금지

사용자 지시는 여기서 기존의 계속 모니터링하는 운영 관행보다 우선한다.

- **GH와 이 task를 위한 SH4/worker의 능동적인 주기 모니터링을 초기 gate PASS 후 종료한다.**
- 이후 `squeue/sacct`, log tail, metrics 반복 조회, `wait_threads` 반복 대기, SSH polling, heartbeat/cron/monitoring automation을 새로 만들지 않는다. 이미 이 task 전용으로 만든 감시가 있다면 해당 감시만 종료한다. 다른 task의 감시는 변경하지 않는다.
- GH는 gate receipt·실제 job ID·계약 경로·최종 결과 예정 경로를 보고하고 turn을 끝낸다. 사용자 호출 전 상태 확인이나 추가 scientific submission을 하지 않는다.
- **모니터링 종료는 실행 중단이 아니다.** 사전에 승인한 Slurm runner와 dependency/reducer는 자연 진행한다. 프로그램 내부의 finite/shape/checksum 검사, 오류 기록, 파일 flush, 정상 종료·사실 보고서 자동 생성은 계속한다.
- Gate 이후 기술 실패도 무한 자동 재시도하지 않는다. 증거를 보존하고 terminal status를 기록한다. 완료/실패 결과는 파일에 남기며 GH를 주기적으로 깨우지 않는다.
- 사용자 재호출 시에만 완료 상태와 산출물을 회수·해석한다. 현재 지시를 모니터링 자동화 생성 요청으로 해석하지 않는다.

## 8. 필수 결과와 보고 위치

초기 receipt: `experiment-reports/servers/server4/alpha-key-causal-20260923-r1/initial-gate-ko.md`와 대응 JSON.

최종 사실 보고: `experiment-reports/servers/server4/alpha-key-causal-20260923-r1/report-ko.md`.

Raw에는 설계의 state/cohort geometry, per-request key, operator stage, writer mode, history mismatch, same-entry effects, component interchange, K/R effects, protection transitions, cost breakdown, source/state receipt를 남긴다. 평균만 남기지 말고 재계산 가능한 K/R와 delta factor, timestamp/current bank를 보관한다. All-valid-token hidden tensor와 full-vocabulary logits를 무조건 전량 저장하지 않는다.

Report에 각 예정 cell의 `COMPLETED / TECHNICAL_FAILED / NOT_APPLICABLE / FOLLOWUP_NOT_SUBMITTED`를 기록한다. 미실행을 유효한 negative result로 세지 않는다. W70 suffix/method sequential, reordered suffix, W0 full10k 등 후속 cell은 이번 E0–E4 범위와 명확히 구분한다.

소형 보고/manifest만 GH에 공유하고 대형 checkpoint를 다른 모든 서버로 재방송하지 않는다. **현재 명령의 완료 경계는 GH의 SH4 실지시·checkpoint 이관 준비/실행·초기 gate PASS 및 본작업의 자율 진행 확인 후 모니터링 종료**다. 전체 실험 완료 여부는 gate 시점에 미관측이면 그대로 기록한다.
