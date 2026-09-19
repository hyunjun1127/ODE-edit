# SH3 초기 설정 사실 보고

- instruction_id: `ODEEDIT-SH3-BOOTSTRAP-20260919-V1`
- 관측: 2026-09-19 KST. 최초 자원 관측 22:09:57, 환경 metadata 22:14:02.
- 상태: `BOOTSTRAP_BLOCKED` / `SUBMISSION_NOT_AUTHORIZED`.
- 범위: 초기 설정, 읽기 전용 준비 상태 확인, 명시적으로 요청된 peer 통신. 과학적 해석/실험 없음.

## 최신 요약

- server3 registry 등록 완료, root ff-only 동기화 및 root/전용 worktree session boundary PASS.
- GH M0 nonce 수신 ACK는 GH 게시 문서/직접 commentary로 확인. GH direct stream terminal은 timeout 기록 유지.
- SH1/SH2/SH4는 exact session·nonce·동일 stream `turn/completed` ACK 모두 PASS.
- SSH hostname GH/SH1·SH2·SH4 PASS. Artifact mapping 미설정, 실제 전송0.
- 미해결: 승인 실행 interpreter/source/overlay/model revision 및 fixed10k/context/P/statistics identity.
- `BOOTSTRAP_BLOCKED`, cap0, `SUBMISSION_NOT_AUTHORIZED`, `save_checkpoints=false` 유지.

이하 최초 관측과 후속 해소 내역을 구분하여 보존한다.

## 서버·세션·저장소 결속

| 항목 | 확인 결과 |
| --- | --- |
| 실제 hostname / 사용자 | `ubuntu` / `janghj` (uid/gid 1025) |
| 실제 session | `01a0b9c8-d12c-7133-b336-cc1b5fc2a6b3`; CODEX_THREAD_ID와 CODEX_SESSION_ID 일치 |
| app host | app list의 `remote-ssh-codex-managed:lab123`, 동일 session/CWD 확인 |
| root CWD | `/data/janghj/ODE-edit`, janghj 소유/읽기·쓰기 가능 |
| origin | `https://github.com/hyunjun1127/ODE-edit.git`; repository identity PASS |
| 시작 HEAD | `8836ff564b5844d412d43533a3ce4ecbfcecc9d1` |
| 시작 tree | `6f671b63f63a5af4fe3730e7e7bf099423cfd943` |
| root branch/dirty | `main`, tracked/untracked clean; ignored local scratch 별도 |
| 전용 branch | `codex/server3-bootstrap-20260919-v1` |
| 전용 worktree | `/data/janghj/ODE-edit/local/state/sh3-bootstrap-20260919-v1/worktree` |
| 초기 registry 대조 | GH/SH1/SH2/SH4 exact session 일치. SH3 session 미지정/future target. SH1 registry CWD와 app resume CWD는 다름: 아래 기록 |
| boundary helper | exit 4: `BLOCK missing local session boundary config`; 초기 `NOT_CONFIGURED`; 아래 등록 후 검사 PASS |
| AGENTS.md | `/`, `/data`, `/data/janghj`, repo 및 tracked 하위 경로에서 없음 |

읽은 문서: `PROTOCOL.md`, `servers/connection-inventory.md`,
`messages/head/2026-08-29-app-server-direct-protocol.md`,
`servers/slurm-memory-policy.tsv`, `control/gpu-concurrency-policy.tsv`,
`scripts/check-session-boundary.sh`, `scripts/check-agent-access.sh`.
초기 `servers/active/server3.md`는 없었다. GH가 정확 session 등록과 최신화를 게시했으며 SH3는 해당 파일 및 global plan을 수정하지 않는다.
GH가 보낸 `ODEEDIT-GH-SH3-REGISTER-20260919-R1` nonce 지시를 수신했고 ACK했다.

## 통신 검증

SSH는 기존 구성 alias만 사용하여 BatchMode/StrictHostKeyChecking/8초 timeout으로
hostname만 확인했다. GH/SH1 서버 `devbox`, SH2 `server2`, SH4 `server4` 응답 PASS.
private 접속값/키/비밀번호/token은 기록하지 않는다.

App-server는 SSH 안의 target Unix socket에서 WebSocket Upgrade → initialize →
initialized → exact thread/resume → start 또는 related steer → terminal 수집 절차를
사용했다. [공식 절차](https://developers.openai.com/codex/app-server)와 repo protocol을
참조했다. 승인값과 ACK는 별도로 기록한다.

| Peer | session | nonce | 요청 수락 | 실제 응답 |
| --- | --- | --- | --- | --- |
| GH | `01a04939-8873-7673-8dca-4c7fc5e31af0` | `ODEEDIT-SH3-BOOTSTRAP-GH-0eeb54f4756c` | M0 steer 수락, turn `01a0b9cb-758a-7723-a475-0040820af88d` | terminal 수집 180초 timeout: `COMMUNICATION_HOLD`; GH 등록 nonce 지시 별도 수신 |
| SH1 | `01a04939-f93a-7b50-bca0-65438eab2062` | `ODEEDIT-SH3-BOOTSTRAP-SH1-37fc8541b50e` | start 수락, turn `01a0b9cc-3a4a-7b43-954d-6427398b0ac1` | `completed`, 동일 nonce와 SH1 exact session ACK PASS |
| SH2 | `01a0493a-074c-7f91-9a13-769116326fef` | `ODEEDIT-SH3-BOOTSTRAP-SH2-06073a09cdde` | start 수락, turn `01a0b9cc-3dec-75c2-a56d-7b5227f58ae8` | `completed`, 동일 nonce와 SH2 exact session ACK PASS |
| SH4 | `01a04939-b5c7-7a03-ba2d-ef3343d62cfd` | `ODEEDIT-SH3-BOOTSTRAP-SH4-f342bb5ff61b` | 미전송 | `COMMUNICATION_HOLD`, `active_turn_guard`; 관련 없는 active 작업에 주입하지 않음 |

GH 최초 접속은 active 관련성 미확인으로 중단했다. 이후 read_thread에서 사용자 요청이
SH3 등록임을 확인했고 GH 직접 지시도 수신하여 exact expectedTurnId에 관련 M0를
한 번 전달했다. SH4는 bounded status snapshot에서도 active라 보류했다.
SH1은 registry의 예전 worktree CWD 대신 resume에서 `/mnt/raid5/janghj/ODE-edit`를
반환했다. session은 동일하며 registry CWD 정합성 갱신은 GH 소유다.

Artifact 전송: `/usr/bin/rsync`, repository helper 존재. 그러나
`servers/local/rsync-targets.tsv`, session boundary 등 local inventory 없음.
전체 승인 source/destination mapping은 `NOT_CONFIGURED`, 실제 전송/rsync 실행 0.
SSH 성공은 artifact 경로 검증으로 승격하지 않는다.

## EasyEdit·Python·dependency·공통 asset

| 항목 | 관측 및 상태 |
| --- | --- |
| EasyEdit | `/data/janghj/EasyEdit`, janghj 소유/읽기 가능, origin `https://github.com/zjunlp/EasyEdit.git`, 관측 HEAD `3488a66ee988d83ee7891a8abbbe6bcb24a77daf` |
| EasyEdit dirty | modified 19, deleted 141, untracked 18. 기존 상태 보존; HEAD만으로 실행 source identity 확정 불가 |
| 승인 EasyEdit 기준 | `NOT_CONFIGURED`; GH 선택 대기 |
| 검사 보조 Python | `/usr/bin/python3` 3.10.12. `importlib.metadata` 기준 torch/transformers/numpy/accelerate/sentencepiece 미설치. 모델 library import 없음 |
| 실험 interpreter | `NOT_CONFIGURED`; system Python은 승인 실행환경이 아님 |
| dependency overlay | `NOT_CONFIGURED`; 정확 버전/설치 계획은 GH가 기준 interpreter/source를 지정한 후 작성 가능 |
| 모델 cache 후보 | `/data/janghj/.cache/huggingface/hub`, GPT-J/Llama3/Qwen2.5 snapshot 디렉터리 존재/소유/읽기 확인. 상세 metadata는 `environment-observations.json` |
| pinned revision | `NOT_CONFIGURED`; 관측 snapshot 이름은 승인 revision이 아님. shard 완전성 `NOT_VERIFIED` |
| CounterFact 기존 데이터 | `/data/janghj/EasyEdit/data/counterfact` 존재/읽기 가능. fixed10k identity는 `NOT_VERIFIED` |
| fixed10k/sample order/context | server3 승인 경로 `NOT_CONFIGURED`; raw CounterFact로 대체하지 않음 |
| P/statistics | 기존 EasyEdit의 null_space_project 파일 3개 metadata만 확인. 공통 자산 identity `NOT_VERIFIED`; `EasyEdit/stats` 없음 |

대용량 파일 load·전체 hash·copy 없음. GH에 6항목(EasyEdit commit, interpreter,
overlay, pinned model, fixed10k/order/context/P/statistics, 누락 시 설치/선택 전송
담당자)을 요청했다. GH는 아직 경로를 확정하지 않았다고 지시했다. 설치 0.

## 자원·Slurm·storage

- GPU: NVIDIA H200 NVL 4개, 각 143771 MiB. 최초 관측 GPU0/1/2/3 사용 메모리
  122493/0/43853/122493 MiB, utilization 96/0/20/95%.
- Slurm: 명령 사용 가능. `ubuntu` node, `gpu` partition, `MIXED`, GRES `gpu:h200:4`.
  CPUTot 96, CPUAlloc 42, RealMemory 512000 MiB, AllocMem 327680 MiB,
  AllocTRES GPU 3. 다른 사용자의 running job 4개 관측.
- `squeue -w ubuntu -u janghj` 출력 0행. project job pattern 설정이 없으므로
  모든 사용자의 job을 포괄하는 project 귀속과 충돌 부재는 `NOT_VERIFIED`.
- GH 지시: 기존 disabled cap0 유지. tracked prospective ceiling 2 GPUs는 활성화 허가가 아님.
- memory: tracked scheduler maximum 120 GiB/GPU, repo request maximum
  119 GiB/GPU = 121856 MiB/GPU. 실제 task/job memory request `NOT_CONFIGURED`.
- CPU 96 logical, RAM total 540546285568 B, available 501396537344 B.
- 작업 `/data` filesystem available 379381481472 B (95% 사용).
- bootstrap 출력은 허용된 ignored state와 전용 보고서 경로.
  신규 실험 output root는 `NOT_CONFIGURED`.
- 활성 cap과 job당 memory 조건을 GH에 요청했으며 `SUBMISSION_NOT_AUTHORIZED` 유지.

## 수행 변경·보존·다음 담당

- 전용 branch/worktree와 허용된 3개 경로의 문서/작은 metadata만 생성.
- `git config --worktree`를 command-scoped extension flag와 사용하려던 시도는
  Git에서 거부되어 identity 변경 없음. GH 허용대로 이후 commit/access check는
  command-scoped `user.*`/`agent.*`만 사용; shared/global Git identity 변경 0.
- GH가 전용 branch push를 명시 허용. main 통합/registry는 GH 소유.
- 원 root/EasyEdit dirty, 기존 job·checkpoint를 보존.
- 신규 GPU/model/Slurm 제출·cancel·hold·requeue·scheduler 변경·환경 설치·artifact
  전송·대용량 copy·삭제·background polling·다른 task 재개는 전부 0.
- `save_checkpoints=false`. periodic/best/final weights 및 W/M/optimizer/RNG/
  복원 delta 자동 저장 없음. 기존 checkpoint 삭제 정책 아님. 평가/NLL/log와
  source/config/sample identity·비용·작은 provenance 보존. 미저장 상태의 exact
  crash-resume 보장 없음. 저장 예외는 해당 task의 권한 출처/대상/시점 필요.

미해결/담당:
1. GH: server3 registry main 게시와 local ignored boundary 설정 범위/결속 확정.
2. GH: 승인 EasyEdit source/interpreter/overlay/model/data/common assets 및 누락 시 담당자 지정.
3. GH/SH4: SH4 idle 이후 nonce ACK; unrelated active task를 침범하지 않는다.
4. GH: 다음 과학 task가 필요할 때 활성 cap과 구체적 memory를 별도 승인.
5. SH3: GH 게시 commit 수신 후 clean clone ff-only sync와 boundary/registry ACK.

이번 bootstrap은 누락된 실행환경 및 통신 항목 때문에 `BOOTSTRAP_BLOCKED`이며,
보고 후 자동 실험·모니터링·기존 task 재개 없이 다음 지시를 기다린다.


## 등록 및 local boundary 후속 확인

GH 등록 commit `0ff1e41cc225bb45518b5b9518aea66d7a19ae1f`의
`ODEEDIT-GH-SH3-REGISTRATION-SYNC-20260919-V1`을 읽고 허용된 설정을 적용했다.
`ACK nonce=ODEEDIT-GH-SH3-REGISTER-SYNC-20260919-R1`.
Root를 clean 확인 후 `21a368689bc6227a2d68da6b97f0ff97600ca034`로 ff-only sync,
tree `6cf2adca84053d39373387d2775c3f72899cf6fe`, ahead/behind 0/0, clean 확인.
Registry의 실제 host/session/CWD/repo 대조 PASS. 전용 branch도 origin/main을 merge하여
공유 source를 동기화했고 SH3 소유 report 변경만 추가했다. Main 통합 작업은 수행하지 않았다.

Root와 전용 worktree의 `servers/local/session-boundary.env`를 각각 실제 CWD로
신규 작성했다. 기존 파일 없음/ignored 확인, 두 boundary helper PASS,
model/profile 미고정, shared/global Git config 변경 0. 해당 추가 write는 GH envelope의
명시 허용 범위다. 최초 boundary 미설정 BLOCK은 해소됐다.

GH가 게시한 등록 문서에는 M0의 정확 nonce ACK가 있다. 별도 app commentary에서도
GH가 동일 nonce 수신을 확인했다. 최초 M0 direct stream의 180초 terminal timeout은
보존하며 durable ACK/별도 직접 응답 증거와 구별한다. Report commit
`50de0a8c7d32492242b44077bb9d4a2d22428784` branch push와 GH related steer 수락 확인.

등록/경계 설정 미해결 항목은 해소됐다. 과학 runtime/interpreter/asset identity는
아직 미준비로 `BOOTSTRAP_BLOCKED`를 유지한다. 활성 cap0는 GH 지시상의 권한이며
local `gpu-caps.tsv`를 임의 생성하지 않았다. 설치/전송 담당은 다음 준비 계획에서
SH3 inventory/명령 제안 → GH 별도 승인으로 정해졌다. 실행 승인은 이번 범위에 없다.


## SH4 통신 완료 후속

SH4가 idle로 바뀐 사실을 compact snapshot으로 확인한 뒤 최초 미전송 nonce를
한 번 전달했다. Turn `01a0b9d3-7415-7f63-8800-3e6ba8d886f7` start 수락,
동일 stream `completed`, `ACK nonce=ODEEDIT-SH3-BOOTSTRAP-SH4-f342bb5ff61b`,
정확 session `01a04939-b5c7-7a03-ba2d-ef3343d62cfd` 수신 확인.
SH4 active 보류 항목은 해소됐다. 실험 재개/scheduler/파일 전송 요청 없음.
GH report turn도 요청 수락 후 180초 terminal 수집 timeout으로 기록했다.
이는 GH M0 수신 ACK나 이후 등록 게시를 취소하는 상태가 아니며 terminal 검증과 구별한다.
