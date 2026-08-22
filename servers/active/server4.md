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
- 갱신 시각: `2026-08-23T01:30:45+09:00`
- current GH directive:
  `ODEEDIT-S05-P4-EULER-PROJECTED-SEMANTIC-ODE-V1`
  Llama ZA h=1 B2–B10 terminal report complete; `POST_ZA_PAUSE_ACTIVE`,
  `IDLE_AWAITING_GH_CALL`

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
- S4-M1-R1 EasyEdit path seal: `PASS`
  - path: `agents/server4/alphaedit-runtime-path-seal.json`
  - id: `ODEEDIT-S4-M1-R1-SERVER4-ALPHAEDIT-PATH-SEAL-V1`
  - logical `/mnt/raid5/janghj/EasyEdit` -> runtime `/data/janghj/EasyEdit`
  - Llama/Qwen 각 hparams/P/stats 7개 SHA/size/shape/dtype PASS
- shared guard interface: `PASS`; default `None` canonical lock identity 유지,
  immutable seal receipt를 P0/ODE-BF guard에 전달
- generic HF mapping: `WITHHELD` 유지; shared/cache 정책은 변경하지 않음
- P4 server4 HF consumed-closure seal: `PASS`
  - exact pinned absolute snapshot/revision과 config/tokenizer/index/all shard
    size/SHA full-read PASS
  - Llama/Qwen extras exact list/count/bytes/root observation-only,
    load/decision influence count 0
  - P4 loader는 offline/local-only FULL-FP32이며 stream/evaluator + final
    PRE-GPU receipt 전 model import/load를 거절
- AlphaEdit evaluator mapping: `WITHHELD`; server4 canonical root 미확인
- no-model launcher dry-plan: EasyEdit `/data` resolution PASS,
  `launcher_ready=false`, model/GPU/Slurm action 0

## 판정

- control-plane onboarding: `PASS`
- resource/storage readiness: `STORAGE_SCOPE_OK`
- session/repository boundary: `PASS`
- artifact content: Llama/Qwen `READY_REUSE`
- EasyEdit artifact path seal/interface: `PASS`
- P4 HF readiness: `PASS`
- full launcher/sealed deployment: `PASS`
- P4-Euler ZA h=1: `TERMINAL_TECHNICAL_PASS / PROJECTION_DOMINATED_EXPLORATORY_RUN`
- scientific job submission: `POST_ZA_PAUSE_ACTIVE`; ZB/Qwen/tuning/rerun 0
- scientific promotion: `false`
- next state: `IDLE_AWAITING_GH_CALL`

## P4 Phase 0 heartbeat

- authoritative contract: `FULL_READ_PASS`, SHA
  `0e6b0ad5110ba8bc758a92ffa126ab094afff57aa18223c4f2d1d16de1170611`
- P4 focused gates 13/13, reused P1R52/path-seal gates 34/34 PASS
- current `/data` available: `207310778368` bytes
- current host available memory: `435844132864` bytes
- server4 current Slurm occupancy: other-user 2-GPU job 1; SH4 project job 0
- P4 HF seal SHA/root: `02db39b6...8854` / `7616e511...893b`
- HF consumed closure: Llama/Qwen `PASS`, extras influence count 0
- blockers: `BLOCKED_EVALUATOR_PACKAGE`, `BLOCKED_SEALED_STREAM_TRANSFER`
- model load / GPU use / Slurm submit: `0 / 0 / 0`
- P4 verdict: `BLOCKED_READINESS`; scientific submit `HOLD`

## P4 ZA case01 pilot heartbeat

- transfer full-read: `PASS`; member root `28189ad3...9843`, stream root
  `74d68965...89e6`, order `abe62c07...cd5c`, evaluator identity
  `72b8ecb7...07d`
- final PRE-GPU: `PASS`, identity `257c74ec...4a71`
- source HEAD/tree: `8229219c01f92e9c1d279b07504e88fc4b95f7ba` /
  `12f56a35b7f2c8bb074444fcd1c67ab94c8718e9`
- Slurm: `22793_[0-1%2]`, Llama/Qwen case01 both `COMPLETED 0:0`,
  `00:09:55`; current active SH4 project GPU 0
- execution: both models 8 outer × 5 inner, target/gradient nonfinite 0,
  writer 0, retry 0, W0 unchanged/restored true
- science: Llama Gen tie but Z± Gen NLL mean worse; Qwen Z± Gen 18/20 vs
  Z+ 19/20 and Gen NLL mean worse; Native-Z mean Gen NLL best on both
- overhead: successful `0.3306 GPU-h`, pre-success fail-close `0.045 GPU-h`;
  final receipts 1,720,393 bytes, logs 3,173 bytes
- report: `experiment-reports/servers/server4/2026-08-22-p4-za-case01-pilot-detailed.md`
- current status: `TECHNICAL_PILOT_PASS / SCIENCE_HOLD /
  INCOMPLETE_TELEMETRY`; automatic promotion 0, subsequent submit HOLD

## P4 ZA M1 budget-ablation heartbeat

- user-authorized delta: target inner Adam update `M=5→1`; all other
  scientific/runtime bindings fixed
- source HEAD/tree: `26dc6007...4bcb6` / `8ecedf0c...65c8`
- final PRE-GPU: `PASS`, identity `9e24d098...9cebe`; focused tests 23 PASS
- Slurm: `22818_[0-1%2]`, Llama/Qwen `COMPLETED 0:0`, `4:16 / 4:06`;
  current active SH4 project GPU 0
- execution: M1 update 8/arm, selected-final observation 8/arm, FULL-FP32,
  writer0, nonfinite0, W0 unchanged/restored
- barrier Gen NLL delta: Llama `-0.9889` with discrete Gen -1/20; Qwen
  `-0.8075` with Gen tie 20/20
- overhead: `0.1394 GPU-h`, 57.8% below M5 successful pilot
- resource snapshot: `/data` available `206722387968` bytes; host available
  memory `436516229120` bytes; other-user job 22352 uses 2 GPU, SH4 uses 0
- report: `experiment-reports/servers/server4/2026-08-22-p4-za-m1-vs-m5-case01.md`
- current status: `M1_BUDGET_ABLATION_SUPPORTED_FOR_FURTHER_VALIDATION /
  PROGRESSIVE_EDIT_NOT_ESTABLISHED / HOLD`; automatic promotion 0

## P4-Euler ZA h=1 Llama B2–B10 terminal heartbeat

- user/GH-authorized exploratory setting: Llama only, `h=1`, `M=5`,
  `T_z=5`; sealed B2–B10, B1 calibration-only excluded
- source HEAD/tree: `f4fbfb5de132de9b986e4b291c5b41784c182743` /
  `8c3f444014907f69b1bb793674f7a5a9e2b9ee0e`
- Slurm: `22854_[2-10]%2`, `9/9 COMPLETED 0:0`; current SH4 project GPU 0
- terminal receipts: case 9, arm 27, action-freeze 18, microstep 90;
  mismatch/nonfinite/W mutation/duplicate/failure 0
- primary limitation: Z+ clamp `286/450=.635556`, all-5 `19/90`;
  Z± clamp `381/450=.846667`, all-5 `54/90`
- classification: `PROJECTION_DOMINATED_EXPLORATORY_RUN`; clamp correlation
  causal claim 0; h=.25 calibration and Stage2 geometry diagnostic remain separate
- report:
  `experiment-reports/servers/server4/p4-euler-za-llama-h1-b2b10-2026-08-23/report-ko.md`
- current resource snapshot: `/data` available `205303037952` bytes; host
  available memory `435321729024` bytes; other-user job 22352 uses 2 GPU;
  SH4 project job 0
- current status: `POST_ZA_PAUSE_ACTIVE / IDLE_AWAITING_GH_CALL`;
  Qwen/ZB/tuning/rerun/new GPU/Slurm/model action 0; promotion false
