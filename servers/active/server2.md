# server2 active record

## 현재 authority

- server: `server2`
- physical hostname: `server2`
- host ID: `remote-ssh-codex-managed:lab121`
- repository: `hyunjun1127/ODE-edit`
- 갱신 시각: 2026-08-29 (session authority rotation)
- server-head (SH2):
  `01a0493a-074c-7f91-9a13-769116326fef`
  (`codex://threads/01a0493a-074c-7f91-9a13-769116326fef`)
- repository CWD: `/mnt/raid5/janghj/ODE-edit`
- local hard boundary: session ID user-confirmed; app list/read PASS

새 session은 2026-08-28 registry의 SH2 authority를 supersede한다. 과거
experiment provenance는 변경하지 않는다.

이번 갱신은 session authority만 교체했다. 아래 repository/runtime/Slurm 관측값은
별도 표기가 없으면 2026-08-22 마지막 audit 기준이다.

## Repository 상태

- remote: `https://github.com/hyunjun1127/ODE-edit.git`
- connection audit branch: clean `main`
- audit HEAD/tree:
  `0d63ad4ec4978be6d04aabb640e17917bd1348d7` /
  `652b84489825371558002e24b381dcb777327a7f`
- cached `origin/main`과 audit HEAD가 일치했다.
- registry push 뒤 GH가 별도 app-server direct turn으로 fetch 및 ff-only 최신화를
  명령한다. 실행 중인 experiment job과 result root는 건드리지 않는다.

## 연결 및 runtime

- GH에서 SH2 session list/read: PASS
- GH→SH2 app-server direct request-response: PASS, turn
  `01a04994-0ed3-7253-a155-e5aacdaa9943`
- direct-mode ACK nonce
  `ODEEDIT-APPSERVER-PROTOCOL-20260829-SH2-R1`: PASS
- direct sync turn `01a04998-3f36-7630-92b1-4738e576fca6`:
  `6d9e4e625c7ed016742ca3516299eae40d9b4af1`, clean,
  ahead/behind `0/0`
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

- session authority ID: user-confirmed
- session routing/read: PASS
- app-server direct coordination: PASS
- unsolicited reverse inbox는 protocol 범위 밖이며 사용하지 않음
- runtime session/repository boundary: pending revalidation
- protocol registry ff-only sync: PASS
