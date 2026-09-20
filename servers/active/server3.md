# Server3 active record

## EN adaptive-nullspace 실행 이관 — 2026-09-20 최신

사용자가 S3 GPU 여건 때문에 이 실험을 S4에서 실행하도록 지시했다.
SH3 본task 신규GPU submit/release/retry 중지, 미제출 ACK 수신.
기존 구현/검산/전송 partial은 보존하고 SH4에 인계한다.
아래 B300 실행승인은 이 task에 한해 SH4로 이전되며 S3에서는 유효하지 않다.
전역 cap1/기존 readiness 자산과 다른 task 상태는 불변.
정본: [migration](../../messages/head/2026-09-20-en-adaptive-nullspace-migrate-sh3-to-sh4.md).

## 최신 과학 실행 승인 — 2026-09-20 / project cap1

사용자 요청으로 server3의 project GPU 상한은 **1**로 변경한다.
아래 cap2/bootstrap 제출금지는 역사이며 이 최신 범위에는 적용하지 않는다.
Llama readiness 완료를 재사용하고
[EN adaptive-nullspace envelope](../../messages/head/2026-09-20-sh3-en-adaptive-nullspace.md)의
T0 → cold B100 네 arm → 세 own-trajectory B300 → 보고/main 게시를 승인한다.
Fixed10k 동일 순서, server2 공통 z-hook/지정 hooking.py 참조, 추가 TF accuracy/NLL 분석.
save_checkpoints=false / exact crash-resume NOT_AVAILABLE, 기존 자료 삭제0.
현재 실행 제출/모델성능은 아직 미확인; 새 task의 M0/source/input/resource lock 뒤 진행.
그 밖 과학 task/자동 B1000·10k 확장 권한은 없다.

## 최신 준비 승인 — 2026-09-19

사용자가 데이터·모델·Covariance·Projector를 실제 실험 가능한 상태로 만들도록
추가 지시했다. 아래 bootstrap 미설치/미전송/제출금지는 당시 관측으로 보존하고,
이번 [readiness envelope](../../messages/head/2026-09-19-sh3-experiment-readiness.md)에서
격리 설치·exact 선택 전송·최대1GPU bounded smoke를 승인한다. Project ceiling2,
save_checkpoints=false. 과학 arm/chain은 여전히 미승인이다.
현재 EXPERIMENT_READY는 아직 미달성; 실측 완료 전 준비 성공으로 바꾸지 않는다.

## 등록 authority — 2026-09-19

- 상태: **REGISTERED / BOOTSTRAP_RUNTIME_NOT_READY / SUBMISSION_NOT_AUTHORIZED**.
- 사용자 등록 요청: `codex://threads/01a0b9c8-d12c-7133-b336-cc1b5fc2a6b3`를 새 SH3로 확인·repo 등록·최신화.
- 논리 역할: `head-server3`, `server-head`, server3.
- 실제 hostname: `ubuntu`, 실제 사용자 `janghj`.
- App host: `remote-ssh-codex-managed:lab123`.
- Session: `01a0b9c8-d12c-7133-b336-cc1b5fc2a6b3`.
- CWD/repository: `/data/janghj/ODE-edit`, `hyunjun1127/ODE-edit`.
- Origin: `https://github.com/hyunjun1127/ODE-edit.git`.
- GH: `01a04939-8873-7673-8dca-4c7fc5e31af0`.
- Direct local endpoint: `/data/janghj/.codex/app-server-control/app-server-control.sock`.
- 사용자가 SH3에 준 원 지시: `ODEEDIT-SH3-BOOTSTRAP-20260919-V1`.
- GH 등록/동기화 envelope: [등록 인계](../../messages/head/2026-09-19-sh3-registration-and-sync.md).

GH는 app read의 session/host/CWD, 기존 SSH alias의 hostname/origin/socket,
SH3 M0 및 실제 direct nonce ACK를 대조했다. Source 등록은 준비 완료나
실험 성능 검증을 뜻하지 않는다.

## 저장소와 통신

초기 root main은 clean, HEAD `8836ff564b5844d412d43533a3ce4ecbfcecc9d1`,
tree `6f671b63f63a5af4fe3730e7e7bf099423cfd943`였다.
전용 bootstrap branch는 `codex/server3-bootstrap-20260919-v1`,
worktree는 `/data/janghj/ODE-edit/local/state/sh3-bootstrap-20260919-v1/worktree`.
기존 root/다른 worktree/공유 Git identity를 덮어쓰지 않는다.

GH→SH3 direct bootstrap turn `01a0b9c8-d2df-73b1-a486-f1f87181931c`에서
nonce `ODEEDIT-GH-SH3-REGISTER-20260919-R1` ACK 수신.
SH3→GH M0 nonce `ODEEDIT-SH3-BOOTSTRAP-GH-0eeb54f4756c` 수신.
SH3의 최초 unrelated-active guard HOLD와 이후 관련 onboarding 전달을 분리한다.
SH1/SH2 ACK 수신은 SH3 보고이며, SH4 active로 인한 보류는 연결 실패 확정이 아니다.
SSH hostname 확인/agent ACK/artifact 전송 준비는 서로 다른 검증이다.

## 자원과 권한

- Tracked prospective project ceiling: **2 GPU** (`control/gpu-concurrency-policy.tsv`).
- 기존 ignored local cap0는 미활성/미제출 상태로 보존 가능하다. 물리4개를 cap으로 사용하지 않는다.
- Repo host-memory ceiling: GPU당 **119GiB = 121856MiB**; scheduler 정책120GiB.
- Slurm node `ubuntu`, partition `gpu`; SH3 initial read 관측96CPU/512000MiB/`gpu:h200:4`.
- 위 자원은 point-in-time 사실과 상한이며 예약/가용성 보장/실험 허가가 아니다.
- 이번 범위 신규 GPU/model/eval/Slurm submission·job mutation·환경 설치·대용량 전송·삭제0.
- 실험별 source/config/input/resource preflight 및 별도 승인 task 전 제출 금지.

## 실행 환경 및 입력 준비

| 항목 | 확인된 실제 상태 | 허가/한계 |
| --- | --- | --- |
| EasyEdit | `/data/janghj/EasyEdit`, origin zjunlp/EasyEdit, HEAD `3488a66ee988d83ee7891a8abbbe6bcb24a77daf` | 기존 수정·삭제·untracked 다수; read-only 보존. HEAD만으로 baseline 승인하지 않음 |
| Python | `/usr/bin/python3`, 3.10.12 | 제어/CPU metadata용 관측. 과학 runtime 승인 아님 |
| 통상 venv 후보 | EasyEdit .venv/venv, janghj miniconda3/anaconda3의 지정 python 경로 부재 | 서버 전체에 환경이 없다는 전수조사 결론 아님 |
| Dependency overlay | 새 task용 pinned torch/transformers 환경 미결속 | 다른 서버 환경 존재를 추정하지 않음, 설치/upgrade0 |
| HF cache | `/data/janghj/.cache/huggingface/hub` 디렉터리 존재 | pinned model/revision/shards 확인 미완료, model load0 |
| fixed10k | `/data/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1` 부재 | 예정 경로만 등록, raw 전송/재생성0 |
| Context/P/stats | EasyEdit examples의 P 후보와 data/hparams 존재; root stats 부재 | 정확 task provenance/shape/content 결속 미완료 |

기존 EasyEdit 변경은 사용자 자료다. baseline commit으로 checkout/reset하거나 설치로
덮지 않는다. 환경·pinned model·fixed10k/context/P/statistics 준비는 별도 계획에서
정확한 원본/대상/담당자/용량/설치 또는 전송 허가를 정해야 한다.

## 기본 운영 정책

- 별도 저장 명시가 없는 새 실험은 `save_checkpoints=false`.
- 기존 checkpoint 삭제 권한 없음. 미저장 시 exact crash-resume 불가를 기록한다.
- SH는 사실/수치/계약 gate만 보고, 과학적 해석은 GH 소유다.
- Onboarding local session-boundary 설정은 실제 이 session/CWD/repo로만 허용하며
  shared global/user config나 다른 agent identity는 수정하지 않는다.
- 원 보고·nonce/오류·historical source는 보존하고 최신 registry를 clean ff-only로 동기화한다.


## 2026-09-19 실제 readiness 완료 (SH3 소유 갱신)

Instruction `ODEEDIT-S06-SH3-EXPERIMENT-READY-ASSETS-20260919-V1`. **Llama EXPERIMENT_READY**, job50986 COMPLETED/0:0.
Qwen model/assets/evaluation READY, native context/shadow NOT_VERIFIED.
원 bootstrap 미준비 기록은 역사로 보존한다. Root/전용 boundary PASS, isolated runtime
`/data/janghj/ODE-edit/local/runtime/server3-experiment-ready-v1`, manifest `agents/server3/experiment-ready-paths-20260919-v1.json`.
상세 `experiment-reports/servers/server3/experiment-readiness-2026-09-19-v1/report-ko.md`.
Project cap2/준비동시1, 기존 source KEEP, save_checkpoints=false.
이번 기술 준비 이외 새 과학task 제출은 별도 지시가 필요하다.
