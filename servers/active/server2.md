# server2 active record

## 현재 authority

- server: `server2`
- physical hostname: `server2`
- host ID: `remote-ssh-codex-managed:lab121`
- repository: `hyunjun1127/ODE-edit`
- 갱신 시각: 2026-08-22
- server-head (SH2):
  `01a0278d-6e24-7a81-974a-96e4addd7ca5`
  (`codex://threads/01a0278d-6e24-7a81-974a-96e4addd7ca5`)
- repository CWD: `/mnt/raid5/janghj/ODE-edit`
- local hard boundary: `Sol Ultra`, checker PASS

새 session은 2026-08-17 registry의 SH2 authority를 supersede한다. 과거
experiment provenance는 변경하지 않는다.

## Repository 상태

- remote: `https://github.com/hyunjun1127/ODE-edit.git`
- connection audit branch: clean `main`
- audit HEAD/tree:
  `0d63ad4ec4978be6d04aabb640e17917bd1348d7` /
  `652b84489825371558002e24b381dcb777327a7f`
- cached `origin/main`과 audit HEAD가 일치했다.
- registry push 뒤 GH가 별도 direct inbox로 fetch 및 ff-only 최신화를
  명령한다. 실행 중인 experiment job과 result root는 건드리지 않는다.

## 연결 및 runtime

- GH→SH2 direct inbox/ACK: PASS
- SH1→SH2 ping/ACK와 SH2→SH1 ping/ACK: PASS
- `PROTOCOL.md` full-read identity:
  SHA256 `a54a4e7c00c36ec9f3b0fe122e5d8735dacc2c21c8604bc598593eb396936663`,
  51,147 bytes, 1,125 lines
- EasyEdit, `.venv`, `uv`, Hugging Face Llama/Qwen cache: available
- Slurm query/submit commands: available
- SSH/rsync local inventory와 client: available; 이번 audit에서는 remote
  authentication/transfer를 실행하지 않았다.

## Slurm 및 resource

- audit 시 `22600`: `PENDING(Resources)`
- audit 시 `22601`: `PENDING(ReqNodeNotAvail)`
- server2 registry cap: 4; active P3 task-specific override: 2
- job 변경·취소·재제출, model/GPU/result mutation: 0

## 판정

- session routing/direct inbox: PASS
- SH1↔SH2 bidirectional inbox/ACK: PASS
- clean main 및 hard session/repository boundary: PASS
- next: registry push 뒤 fetch 및 `git merge --ff-only origin/main`
