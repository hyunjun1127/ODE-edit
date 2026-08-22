# server4 active record

## 현재 authority

- server: `server4`
- physical hostname: `server4`
- repository: `hyunjun1127/ODE-edit`
- repository CWD: `/data/janghj/ODE-edit`
- 담당 server-head: `server4-server-head`
- Codex session ID:
  `01a028a7-9e3c-7541-81ba-efb40555d17d`
  (`codex://threads/01a028a7-9e3c-7541-81ba-efb40555d17d`)
- local hard boundary: checker PASS, model/profile은 user-managed
- 갱신 시각: `2026-08-22T17:51:38+09:00`
- current GH directive: onboarding/resource audit 완료 후 신규 실험 job을
  제출하지 않고 GH task 명령 대기

이 session은 이전 `registered-pending-clone` 상태를 supersede한다. 과거 task,
report, audit와 experiment provenance는 변경하지 않는다.

## Repository 및 Git/Agent 상태

- remote: `https://github.com/hyunjun1127/ODE-edit.git`
- `git pull --rebase origin main`: fast-forward 완료, conflict 없음
- audit HEAD/tree:
  `96494a05c37ffb32f101d53d14e0e61b2f25664d` /
  `58c07ad6ef33571650d92c28c58fb31871915a9d`
- Git identity: `server4-server-head <server4-server-head@lab.local>`
- agent identity: `server4-server-head`, role `server-head`, hostname `server4`
- heartbeat: `agents/server4/server4-server-head.json`
- ignored local boundary:
  `servers/local/session-boundary.env`, checker PASS
- ignored local cap:
  `servers/local/gpu-caps.tsv`, `PROJECT_GPU_CAP=2`

## Protocol 및 task 상태

- `PROTOCOL.md`, root `README.md`, `messages/README.md` 전체 확인
- `PROTOCOL.md`: 51,147 bytes, 1,125 lines, SHA256
  `a54a4e7c00c36ec9f3b0fe122e5d8735dacc2c21c8604bc598593eb396936663`
- broadcast/global-head command는 worker claim 대상이 아니며 server4는
  `tasks/status/<task_id>/server4.json`으로 상태를 보고한다.
- global-head만 모든 target server가 `done` 또는 명시적 `waived`일 때 task를
  닫고 `tasks/done/<task_id>.closure.md`를 작성한다.
- 점검 시 `tasks/pending/` task는 0건이다. 따라서 보강할 server4 status
  파일도 0건이며 task ID 없는 orphan status는 만들지 않았다.

## Slurm 및 resource

- Slurm query/submit command: available; 이번 온보딩에서는 submit 0건
- partition: `gpu`, default time `04:00:00`, max time `30-00:00:00`
- node state: `MIXED`
- CPU: 128 logical CPUs, Slurm alloc 18 CPUs
- host memory: OS 503 GiB, available 약 406 GiB; Slurm configured 500 GiB,
  alloc 120,000 MiB
- physical GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition 8개,
  각 97,887 MiB
- Slurm 전체 alloc GPU: 2개; non-project job 1건이 사용 중
- ODE-Edit project pattern 실행 GPU: 0개
- project GPU cap: 동시 최대 2 GPU
- host-memory request cap: GPU당 65,984 MiB
- `scripts/check-slurm-resource-cap.sh server4 1 65984`: PASS
- 현재 GH 지시 때문에 cap PASS 여부와 무관하게 신규 experiment submit은 HOLD

## Storage

- `/data`: 7.0 TiB 중 6.4 TiB 사용, 약 193 GiB 가용, 사용률 98%
- inode 사용률: 3%
- repo-local ignored `local/` write probe: PASS 후 즉시 삭제
- 판정: `WARN`; 신규 대용량 artifact 생성 전 용량 정리 또는 예상 출력량
  확인이 필요하다.

## 판정

- control-plane onboarding: `PASS`
- resource/storage onboarding: `WARN` (storage 98%)
- session/repository boundary: `PASS`
- scientific job submission: `HOLD` (현재 GH 지시 및 향후 task별 red-team
  preflight 필요)
- next owner: `server4-server-head`, GH task 명령 대기
