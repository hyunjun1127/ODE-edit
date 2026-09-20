# SH3 → SH4 EN adaptive-nullspace 인계

사용자 migration nonce `ODEEDIT-GH-SH3-EN-ADAPT-MIGRATE-TO-S4-20260920-R1`에 따라 S3의 이 task GPU 실행권한을 중지했다. **실제 GPU 제출·allocation·T0·B1·B300은 모두 0**이다. `sbatch --test-only`가 표시한 51258은 실제 제출/예약 job이 아니다. 취소할 본 task job이 없었으며 다른 job과 전역 cap1은 변경하지 않았다. 과학 결과나 모델 수치 PASS를 주장하지 않는다.

## 소스와 검증

- 실행 source commit: `2ff2393f19b74195ed52d016d4d73ae0878d9946`, tree `992958cd40f898cd24c01b56692b75fc4243c99a`.
- Branch: `codex/server3-en-adaptive-nullspace-b300-v1`. 위 commit은 push 완료. 이 인계 commit은 reporter 보조 코드와 인계 문서를 추가하며 과학 실행 source는 위 commit으로 구별한다.
- Source snapshot: `/data/janghj/ODE-edit/local/en-adaptive-nullspace/20260920-v1/attempt-v1/source/`; 2,486 files / 26,682,367B. `attempt-v1/execution.lock.json`에 member SHA/size를 봉인했다. 실행 output은 없다.
- `project/run_scripts/en_adaptive_nullspace/`: geometry/selector/controller, native/current, objective, metrics, technical, runtime/runner/run.sbatch. 공통 S2 z-hook을 read-only import한다.
- 핵심 CPU 테스트43개 PASS, reporter 추가4개 PASS. GPU/실제 수치 동등성의 증거가 아니다. Command: `python -m unittest project.run_scripts.en_adaptive_nullspace.test_geometry_selector project.run_scripts.en_adaptive_nullspace.test_native_current project.run_scripts.en_adaptive_nullspace.test_metrics project.run_scripts.en_adaptive_nullspace.test_technical project.run_scripts.en_adaptive_nullspace.test_objective project.run_scripts.en_adaptive_nullspace.test_report`.
- Metrics contract: 같은 namespace `metrics-contract.json`; BLUE preference/ties-failure, desired TF token-micro/prompt-macro/strict, true/new/desired NLL, paired IDs, request-cluster bootstrap10000/20260920. TF를 generation 정확도로 부르지 않는다.
- T0 재사용 z 카운터, current identity, N4 B2/B3 L_R/L_H observer, 요청별 alias-weighted response, byte-exact observer alias와 G SHA/단계 타이머 누락은 commit 전에 수정했다.

## 입력 인계와 역전송 금지

S4 원 reference: `/data/janghj/ODE-edit/local/en-execution-reuse/20260919-v1/PREP/attempt-v1/output/generated/manifest.json`. 동일 `generated/` 아래 R512/Dev128 capsule/key/residual/logp를 그대로 재사용한다. 원 입력: `/data/janghj/ODE-edit/local/bpcw512/20260918-v2/reference-inputs-v2/inputs.json`, SHA `507b202108acc90e10810455f57da60df14e9e9e31ba567c8c75f51cf76afaeb`.

S3 수신본: `/data/janghj/ODE-edit/local/en-adaptive-nullspace/20260920-v1/inputs/generated-v1/`. 2,560 files / 101,519,959,223B. R512 512문서/130,235 valid positions, Dev128 128문서/32,473 positions. 전송은 migration 수신 직전에 완료됐으며 신규 구성원 전부 full SHA/size/CPU shape/dtype 검산 PASS, 미완료·누락·불일치·삭제0. transfer process0. S4 원본과 S3 수신본을 보존한다. **이 101.5GB를 S4로 역전송하지 않는다.**

S3-only receipt 패키지는 `handoff.json`의 정확 경로/size/SHA로 SH4가 선택 pull할 수 있다. 약596KB이며 allowlist, 신규 파일 검산 receipt, source lock, Pstar derivation receipt와 기존 ACK만 포함한다. Teacher/model/dataset/Pstar tensor/edited state를 포함하지 않는다.

Fixed10k JSON SHA `3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1`, ordered root `5b01356928ad2e2e46effad5a2c5621c87746b610992b7b92bf3ace16b6f5729`, 정확 앞300을 사용한다. S3 READY Llama/native/context/P/C0는 기존 정본과 결속했고 공유 dirty EasyEdit는 보존했다.

S3 Pstar는 기존 P4를 CPU eig로 복원한 immutable 입력이다: shape [14336,14326] FP64, basis.npy 1,643,020,416B/SHA `515f7d9947b5c7e177ca4c8d9ba0529823f06db9e714fb07c2b7eb63c36e4e05`, CPU 8threads 약178.59초, CUDA 초기화0. S4의 기존 Pstar provenance를 우선 재사용하되 다른 basis 좌표를 같은 tensor SHA로 혼동하지 않는다. Native에는 원 FP32 P4를 그대로 사용한다.

## SH4가 이어서 처리할 정확한 경계

과학 math/controller/native hook를 재구현할 필요는 없다. 다만 현재 runtime/launcher는 S3에 봉인되어 있어 **그대로 SH4에서 실행하면 안 된다.** 아래 destination binding을 task-local adapter로 바꾼 뒤 새 실행 source를 봉인해야 한다.

1. `runtime.py`의 READY_MANIFEST, ubuntu Slurm guard, Python 절대경로/정확버전, asset_inventory의 S3 inode/device/mtime, source_allowlist를 SH4의 승인된 실제 native/model/P/C0/context/env closure에 결속한다. S3 stat receipt는 SH4 로컬 inode 검증을 대체하지 않는다.
2. `metrics.py`와 `metrics-contract.json`의 S3 READY evaluator 절대경로를 SH4의 동일 source SHA closure로 결속한다. evaluator tokenization/MB16/manual left-padding 의미는 유지한다.
3. `runner.py` ROOT/input/basis, `run.sbatch` node/Python/host memory/output/source, `prepare_basis.py`의 로컬 P path를 SH4 task namespace로 이식한다. S4 원 generated teacher를 직접 참조한다. 다른 venv 디렉터리 복사0.
4. S3 계획은 1GPU/8CPU/121856MiB/6h wall, host peak90GiB·GPU peak100GiB·wall1.5–3h **추정**이었다. 실제 실행 검증값이 아니다. SH4의 더 낮은 memory ceiling을 별도로 admission하고 필요하면 동일 의미의 prefix streaming 등으로 CPU peak를 줄여야 한다. 전체640 prefix payload 약18.04GB, Pstar1.64GB, geometry/branch W/M/G 및 model-load 임시 peak가 더 필요하다. 4-request T0는 full z chunk16 peak를 검증하지 않는다.
5. 원 design T0를 실제로 새 실행한다. 과거 waiver를 PASS로 승격하지 않는다. T0 코드는 precision NOT_ESTABLISHED를 명시적으로 보고하도록 되어 있으며 finite/identity failure는 exception이다. Batch16 z 수치 차이와 실제 GPU memory는 미측정이다.
6. Reporter `report.py`는 보조4테스트 PASS지만 실제 output을 소비하지 않았다. **미구현:** current/active-past/all-seen paired 분리 재구성의 일부, active-past preference, 실제 Slurm accounting schema 결속. 현재 runner는 all-seen/first100 paired와 active-past TF aggregate 및 rawcompact를 저장하므로 추가 forward 없이 후처리할 수 있다. 이 누락을 완료로 숨기지 않는다.

save_checkpoints=false, edited W/M/delta/optimizer disk0, exact crash-resume NOT_AVAILABLE. B1 네 endpoint 후 N4/EN_EXACT/EN_ADAPT 세 own trajectory를 B300까지 진행하는 계약은 유지한다. S3는 이 인계 뒤 실행하지 않는다.

## S3 비용·보존 상태

본 task GPUh0. Reference 수신은 101.5GB, 검산 wall662.13초이며 전송과 겹친다. 이 시간을 합산해 독립 wall처럼 세지 않는다. 과거 S4 teacher 생성/native 비용을 신규 S3 GPU allocation으로 세지 않는다. 마지막 검산 후 diskfree17,586,044,928B는 비독점 관측이다. S4에 이미 있는 reference의 신규 storage 요구는0이며 실행 scratch와 환경 peak는 SH4가 실제 관측으로 판단한다.

Root/기존 bootstrap/readiness/dirty EasyEdit와 다른 task를 보존했다. 이 보고는 **S3 인계 상태**이며 B300 과학 완료 보고가 아니다.

Migration 정본 `938db0b765e1a3d4a1e6edc8da702d2486ad6e24` FULL_READ 완료. SH4 상한60416MiB/59GiB와 새 destination `local/en-adaptive-nullspace/20260920-v1/server4-migration-r1/`를 확인했다. S3 추가 구현/검산 실행은 중지했고, reporter4개 테스트는 중지 지시 직전 안전한 작업 경계에서 완료된 증거다. 인계 commit에 모든 source/report 변경을 보존하며 기존 dirty를 되돌리지 않았다.
