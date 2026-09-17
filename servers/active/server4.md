# server4 active record

## 최신 GPU cap override — 2026-09-17

- 사용자 원문: “R-GD와 R-Q 만 실험 진행시키자. SH4에게 전달하는 것으로 해”, 이어 “cap 2로 늘리자.”
- 현재 Server4 project GPU cap은 **2**이며 아래 2026-09-16 cap1보다 우선한다. 다른 서버 cap은 불변이다.
- 정본: `control/gpu-concurrency-policy.tsv`. 기술/prep/teacher/science 및 다른 admitted project capacity를 합쳐 최대2GPU다.
- 공통 기술 검증 뒤 R-GD/R-QP 두 arm은 각각1GPU로 두 slot을 활용한다. 다른 점유가 있으면 합계 cap2를 지키며 기존 job은 임의 변경하지 않는다.
- 인계: `messages/head/2026-09-17-sh4-l4-preserving-repair-twoarm.md`. 과거 cap 초과 기록을 새 cap으로 소급 정당화하지 않는다.
- 아래 정책·온보딩·실험 수치는 당시 기록을 보존한다.

## 최신 GPU cap override — 2026-09-16

- 사용자 원문: “server4 gpu cap은 1이다.”
- 현재 Server4 project GPU cap은 **1**이며 아래 역사적 cap4 및 이전 cap2보다 우선한다.
- 정본: `control/gpu-concurrency-policy.tsv`; 신규 기술/prep/teacher/과학 job을 합쳐 단일 slot으로 admission한다.
- 이미 할당된 GPU와 admitted pending의 동시실행 가능 용량을 포함한다. 신규 array는 `%1` 또는 동등한 단일 dependency lane을 사용한다.
- 기존 job 취소·재시작·원source 변경 허가는 아니며, 기존점유가 있으면 신규실행을 대기시킨다. 다른 서버 cap은 불변이다.
- 인계: `messages/head/2026-09-16-sh4-local-z-adaptive-allocation.md`.
- 아래 온보딩·실험 수치는 당시 관측을 보존한 기록이다.

## 현재 authority

- server: `server4`
- physical hostname: `server4`
- repository: `hyunjun1127/ODE-edit`
- repository CWD: `/data/janghj/ODE-edit`
- 담당 server-head: `server4-server-head`
- Codex session ID:
  `01a04939-b5c7-7a03-ba2d-ef3343d62cfd`
  (`codex://threads/01a04939-b5c7-7a03-ba2d-ef3343d62cfd`)
- local hard boundary: session ID user-confirmed; app list/read PASS;
  app-server direct request-response PASS
- 갱신 시각: `2026-08-29` (session authority rotation; 이하 runtime/실험 heartbeat는 기존 관측 보존)
- current GH directive:
  `ODEEDIT-S05-P1R54-ENERGYFREE-LOCALZ-B100-V1`
  FZ/PDZ job23375 terminal factual report complete; next instruction HOLD

이 session은 이전 server4 session `01a028a7-9e3c-7541-81ba-efb40555d17d`를 supersede한다. 과거 task,
report, audit와 experiment provenance는 변경하지 않는다.

GH에서 새 SH4 session의 list/read와 app-server direct request-response를
검증했다. Direct-mode 전환 ACK turn은
`01a04994-02d7-7482-8ec7-4706e96ab414`이며 nonce
`ODEEDIT-APPSERVER-PROTOCOL-20260829-SH4-R1`이 PASS했다.
`01a04998-37e0-7300-a9e8-91b5f16c03ec` direct turn에서 protocol commit
`6d9e4e625c7ed016742ca3516299eae40d9b4af1`로 ff-only sync도 PASS했고,
worktree는 clean, ahead/behind `0/0`이다.
`send_message_to_thread` dynamic wrapper와 unsolicited reverse inbox는
사용하지 않는다.

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
  `servers/local/gpu-caps.tsv`; current user-registered `PROJECT_GPU_CAP=4`

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
- current user-directed project GPU cap: server4의 `janghj` allocation 동시 최대 4 GPU
- public scheduler host-memory ceiling: GPU당 60 GiB
- repository request ceiling with headroom: GPU당 59 GiB (`60416M`)
- `scripts/check-slurm-resource-cap.sh server4 1 60416M`: required before submit
- 현재 `janghj` active GPU는 0개이며 신규 experiment submit은 HOLD

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
- scientific job submission: target-timescale/Native terminal; active `janghj` GPU 0,
  신규 submit HOLD
- scientific promotion: `false`
- next state: `IDLE_AWAITING_GH_CALL`

## P1R53 request-local target-speed B100 terminal heartbeat

- contract: `FULL_READ_PASS`, SHA256 `fde535a1...da322`, 17,305 bytes, 808 lines
- source base: `8d39bd253d196a016cc65518af3df75644b561d8` / tree `33d27e97a8f75bef867b774182a5f1cb06f637d9`
- model/input: Llama only, transferred SINGLE_CANONICAL_STREAM B1 100 requests
- arms: LP-S / LFD-E; K8, h=1/8, microstep1, T=1, C3 writer8
- external references: existing Z0-COARSE / Native AlphaEdit / Native MEMIT, execution0
- registered janghj/project GPU cap: 4; task array concurrency: 2
- execution source: `715df5845a312d10f4825560829ea271e9ef854b` / tree `cbf6a961aaed6df09c80126028de7121add107b9`
- valid array: `23318_[0-1]%2`, LP-S/LFD-E both `COMPLETED 0:0`; runtime 1150s/1101s
- completeness: each field eval8, writer8/layer40, cache reuse8/append1, W0 byte/pointer restore exact, FULL-FP32
- technical exclusion: `23314_[0-1]` pre-model session-boundary missing, scientific denominator influence0
- LP-S: W Rewrite 100/100, Rephrase prompt 181/200, Loc 881/1000, clamp 34/800
- LFD-E: W Rewrite 95/100, Rephrase prompt 153/200, Loc 888/1000, clamp 0/800
- canonical report: `experiment-reports/servers/server4/p1r53-request-local-speed-llama-b100-2026-08-24-v2`
- report/manifest/receipt SHA: `f25e397a...77bb` / `1d5aa605...c510` / `575b7c76...423a`
- current action: terminal report complete, active project GPU0, no new submission
- scientific promotion: false

## P1R54 energy-free Local-Z B100 heartbeat

- contract: `FULL_READ_PASS`, SHA256 `6fcfc7df...3d58`, 12,866 bytes, wc-lines610
- source base: `ec7c25e2c42816718ac2eda1a43efae82c466970` / tree `d557cc7edbcb5eaa71faddf365c52880713ae366`
- implementation commit/tree: `cf6d1dc673d84129c46a742948dcf83703fa07f9` / `05b4b8b0a1f41b73d92a6ec38360b4314e06faf7`
- arms: FZ `F_i=m_i||z0_i||d_i`; PDZ `F_i=m_i||z0_i||[-expm1(-ell_i)]d_i`
- scope: Llama sealed B1 100, K8, h=1/8, C3 writer8/layer40, FULL-FP32
- numerical lock root: `b9f7ca3beac529491b4cf16237f8b7887a3f02913110639ed2c29a63dc4286ad`
- focused mechanical/legacy suite: 38/38 PASS; GPU/Slurm/model action0
- registered janghj/project GPU cap4; planned array `0-1%2`
- execution source/tree: `27339f987c4f4147a6bf6347061444f53af095a4` / `3683ae0bdb7004148516514ffbf567c566dfab94`
- valid array: `23375_[0-1]%2`; FZ/PDZ 모두 `COMPLETED 0:0`, elapsed 1267s/1264s
- completeness per arm: field eval8, writer8/layer40, cache reuse8/append1, W0 restore, FULL-FP32, additional F/B0
- FZ W: Eff 100/100, Gen prompt 183/200, strict85/100, Loc890/1000
- PDZ W: Eff 100/100, Gen prompt179/200, strict82/100, Loc883/1000
- FZ accepted-z Rewrite/Rephrase NLL mean: `0.048273 / 1.274926`; post-W `0.050604 / 1.983023`
- PDZ accepted-z Rewrite/Rephrase NLL mean: `0.093820 / 1.584130`; post-W `0.092770 / 2.421488`
- both arms clamp0/800; FZ hard-tail/barrier-writer transfer and PDZ under-edit pacing remain factual limitations
- canonical report: `experiment-reports/servers/server4/p1r54-energyfree-localz-llama-b100-2026-08-24-v3`
- report/manifest/receipt SHA: `5132ae8f...a510` / `a1765961...d685` / `673933a3...a5ce`
- current action: terminal factual report complete, active project GPU0, no new submission
- scientific promotion: false

## P1R52 target-timescale B100 terminal heartbeat

- target array: `23132_[0-4]`, Z0/Z1/Z15/Z20/Z30 5/5 `COMPLETED 0:0`
- same-horizon: Z0/Z1, `T_z=1`, resolution-only report complete
- longer-time: Z1/Z15/Z20/Z30, fixed `dt=1/16`, report complete
- Native reference: AlphaEdit `23140_0` valid; MEMIT `23140_1` technical
  exclusion 뒤 TECH-R2 `23142_1` valid
- raw roots: ignored local, immutable reference only
- report roots:
  - canonical: `experiment-reports/servers/server4/p1r52-target-timescale-same-horizon-b100-2026-08-23-native-v3`
  - canonical: `experiment-reports/servers/server4/p1r52-target-timescale-longer-time-b100-2026-08-23-native-v3`
  - v1/v2 roots preserved as superseded report lineage
- Native endpoint coverage: accepted-z + post-W, z→W transfer and compute in v3
- v3 layout: final W Eff/Gen/Loc + z/W NLL overview first; Rewrite,
  Rephrase, same-horizon/longer-time special tables separated
- final Tz selection: false
- scientific promotion: false

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
