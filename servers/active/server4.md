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
- 갱신 시각: `2026-08-22T18:22:14+09:00`
- current GH directive: `S4-M1` server4 artifact path seal; shared scientific
  launcher 변경이 필요하면 구현 전 차단 보고, scientific submit HOLD

이 session은 이전 `registered-pending-clone` 상태를 supersede한다. 과거 task,
report, audit와 experiment provenance는 변경하지 않는다.

## Repository 및 Git/Agent 상태

- remote: `https://github.com/hyunjun1127/ODE-edit.git`
- `git pull --rebase origin main`: fast-forward 완료, conflict 없음
- S4-M1 base HEAD/tree:
  `9e1943d29e19bf1429a99ec86fce0b13c878b513` /
  `e702159897b8de363ee957e46e92b172532b98d1`
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

- `/data`: 7,619,770,974,208 bytes 중 7,028,782,764,032 bytes 사용,
  206,896,500,736 bytes(약 193 GiB) 가용, 사용률 98%
- inode 사용률: 3%
- repo-local ignored `local/` write probe: PASS 후 즉시 삭제
- bounded ODE-Edit clone + EasyEdit dependency + HF cache:
  103,290,650,624 bytes(약 96.2 GiB)
- 판정: `STORAGE_SCOPE_OK`; 현재 2-GPU job에 즉시 치명적인 local-space
  blocker는 bounded 범위에서 확인되지 않았다. Exact task output contract가
  주어지면 submit 전 point-in-time 재확인이 필요하다.

## AlphaEdit reusable artifact

- Llama/Qwen projector 2개와 Wikipedia stats 10개는 canonical lock의
  size/SHA와 exact MATCH하며 `PROTECTED_REUSABLE=22,575,859,404 bytes`다.
- server4 실제 자산 root: `/data/janghj/EasyEdit`
- current lock root: `/mnt/raid5/janghj/EasyEdit` (server4에서 missing)
- reusable asset verdict: Llama `READY_REUSE`, Qwen `READY_REUSE`
- S4-M1 focused rehash: projector 2개와 stats 10개 size/SHA exact MATCH;
  projector CPU mmap shape/dtype PASS
- current legacy launcher path: `BLOCKED_SHARED_SOURCE_CHANGE`;
  `ODEBFArtifactGuard`와 `P0ArtifactGuard`가 runtime root override를 받지 않고,
  현재 ODE-BF launcher도 local runtime config를 소비하지 않는다.
- server4 canonical runtime root `/data/janghj/EasyEdit`는 승인됐지만 실제
  launcher-consumed seal은 아직 생성하지 않았다. shared source 승인 전 재계산,
  다운로드 또는 submit하지 않는다.

## 판정

- control-plane onboarding: `PASS`
- resource/storage readiness: `STORAGE_SCOPE_OK`
- session/repository boundary: `PASS`
- artifact content: Llama/Qwen `READY_REUSE`
- artifact path seal / launcher: `BLOCKED_SHARED_SOURCE_CHANGE`
- scientific job submission: `HOLD`
- next owner: global-head가 shared resolver/API 변경 범위와 target launcher를
  지정·승인하면 server4가 재개
