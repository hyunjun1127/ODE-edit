# GH Stage 0 Diagnostic Plan — BF-ODE-Edit

- 작성 시각: 2026-07-30
- 작성 agent: head-server1-gh (global-head)
- 상태: `blocked_on_server_onboarding_and_user_context`
- proposal: `project/proposals/00.proposal`
- proposal-side rationale: `project/proposals/sections/01-stage-0-diagnostic.md`
- global evidence index: `experiment-reports/global/2026-07-30-stage0-diagnostics.md`

## 1. 인수 판정과 운영 경계

| 구분 | 내용 |
| --- | --- |
| repo/protocol에서 확인한 사실 | server1에 GH clone이 있으며 `agent.role=global-head`, `agent.hostname=server1`이다. 별도 server-head, inbox, task, run, audit, tracked 실행 script는 없다. private GitHub remote `hyunjun1127/ODE-edit`는 등록됐고 첫 push 전이다. |
| GH 추정 | 이 clone은 template로부터 시작한 독립 연구 repo 초기화 단계다. 실제 연구 remote로의 전환은 아직 이루어지지 않았다. |
| 사용자 확인 필요 | server1의 server-head 겸임 여부, model/dataset 접근, baseline revision, Stage 0 compute budget. GPU cap과 memory cap은 local-only config로 이식하되, 실제 job 전 current availability를 재확인한다. |
| 이번 GH 조치 | proposal을 canonical input으로 보존하고, Stage 0 diagnostic·claim boundary·server registration 전 instruction envelope를 만든다. Slurm/SSH/rsync는 실행하지 않는다. |

추가 read-only 환경 점검에서 이 GH clone은 Slurm client와 controller 응답을
확인했다. server1의 local GPU cap은 3, GPU당 host-memory request cap은
198117 MiB로 private config에 이식한다. 이는 server-head ownership,
model/dataset readiness 또는 Slurm 제출 권한을 뜻하지 않는다.

`PROTOCOL.md`와 user prompt 사이의 실질 충돌은 발견하지 못했다. 다만 prompt의
"각 SH에게" 지시는 active SH가 없으므로 실제 inbox 대신 등록 후 즉시 사용할
server별 envelope 초안으로 해석했다. 등록 전 실행 지시는 protocol의 server
onboarding/GPU-cap gate와 충돌한다.

또한 `PROTOCOL.md` 내부에서 experiment-section layout은
`experiment-reports/experiments/`를 권장하지만 GH access matrix와 access
checker는 `experiment-reports/global/`만 GH에게 허용한다. 이 plan은 더 좁은
권한을 우선해 global evidence index를 사용한다. section index의 GH 소유권은
사용자 확인 후 protocol/template을 함께 바꿔야 한다.

## 2. Canonical research direction

### proposal에서 온 내용

- MEMIT의 layer별 low-rank write를 final endpoint가 아닌 local actuator로
  보고, synchronous READ–PROPOSE–CONTROL–COMMIT 후 재선형화한다.
- controller는 rewrite prompt와 MEMIT의 5 prefix만 보며, paraphrase,
  neighborhood/locality, downstream metric은 edit-time signal로 쓰지 않는다.
- 장기 sequential stream에서 covariance-weighted cumulative capacity load를
  측정하고, 낮은 marginal cost layer로 write를 routing한다.

### GH 재정리

핵심 가설은 “dynamic relinearization과 capacity routing이 필요하다”가 아니라,
**static allocation이 설명하지 못하는 layer-state 변화와 load concentration이
관측될 때에만 dynamic method를 고려한다**이다. Stage 0가 그 전제의 최소
증거를 판정한다.

### 왜 final paper plan이 아닌가

proposal에는 구현, 데이터, 비교군, 성공 narrative가 제시되어 있으나,
same-snapshot utility가 측정 가능한지, ranking이 실제로 변하는지, capacity
proxy가 장기 손상과 관련되는지, 최신 baseline과 공정 비교가 가능한지는 아직
run artifact나 audit로 검증되지 않았다. 따라서 proposal의 superiority,
causality, novelty, full-stream feasibility는 claim으로 승격하지 않는다.

## 3. Stage 0 scope와 정보 firewall

### 최소 실행 단위

1. **S0-preflight (실행 전):** frozen baseline revision과 CounterFact atomic
   subset manifest를 기록한다. edit-time 허용 prompt와 evaluation-only prompt
   handle을 코드 수준에서 분리한다.
2. **S0-baseline trace:** MEMIT sequential baseline을 100 edits × 2 fixed edit
   orders로 실행하여 layer별 key/residual/proposal/capacity trace를 `local/`에
   기록한다. 이 run은 controller 성능 비교가 아니다.
3. **S0-perturbation diagnostic:** 성공하거나 아직 rewrite deficit이 있는
   edit에 대해 same-state proposals를 기록하고, 사전 고정된 small joint
   partial update 하나 뒤 동일 허용 context에서 다시 측정한다. 평가 prompt는
   이 loop에 접근하지 못한다.
4. **S0-replication:** 서로 다른 두 primary backbone에서 동일 diagnostic을
   반복한다. second backbone이 준비되지 않으면 결과는 `single-backbone
   diagnostic`으로만 표시하고 Stage 1로 승격하지 않는다.

### Stage 0에서 증명해야 할 최소 신호

아래 수치는 **GH 사전등록 operational threshold**다. 논문 claim이나 보편적
자연 법칙이 아니다. viable edit는 base editor가 direct-z를 만들고, 허용
rewrite context에서 finite proposal/metric을 산출한 edit다.

| Signal | 기록 metric | provisional pass criterion |
| --- | --- | --- |
| H1 heterogeneity | C-normalized `a_l`의 edit별 coefficient of variation, top/median ratio | viable edit의 60% 이상에서 `CV(a_l) >= 0.15` 및 `top/median >= 1.10` |
| H2 non-stationarity | pre/post partial update Spearman rank, top-2 Jaccard, proposal cosine | accepted partial update의 20% 이상에서 `Spearman < 0.90` 또는 top-2 membership change; 동일 edit-layer proposal cosine도 함께 보고 |
| H3 concentration | sequential baseline의 `Psi_l` Gini, max-load share | 각 backbone/order에서 마지막 checkpoint의 `Gini(Psi) >= 0.20` 또는 max-load share가 균등 share의 1.5배 이상 |
| Firewall | controller input access log와 metric call trace | edit loop에서 paraphrase/neighborhood/locality/downstream prompt 또는 label 접근 0건 |
| Reproducibility | manifest, seed/order/subset/config/commit | 두 order와 가능한 두 backbone에서 동일 metric schema·실행 path를 재현 |

threshold가 전체적으로 충족되지 않아도 raw metric, failure, viable-edit
denominator를 숨기지 않는다. 비율의 uncertainty는 edit-level bootstrap CI로
보고한다.

## 4. Kill, pivot, next-stage criteria

### 즉시 중단 (`block`)

- edit-time loop가 evaluation-only prompt, label, metric을 읽거나 step-size,
  layer choice, stopping에 사용한 경우.
- dataset split/subset provenance, baseline revision, seed/order, code commit,
  local artifact path 중 하나라도 복원할 수 없는 경우.
- GPU cap이 unknown/exceeded, server onboarding audit가 `block`, 또는 raw
  output/checkpoint/credential이 Git에 들어간 경우.

### 연구 방향 kill 또는 pivot

- 두 backbone에서 H1 또는 H2가 사전 기준에 일관되게 미달하고, static
  layer-wise scaling이 matched rewrite efficacy에서 동등하거나 낫다:
  **ODE/relinearization claim kill; static capacity routing만 별도 연구
  후보로 재정의**.
- H3가 없거나 `Psi_l` concentration과 retention/capability proxy의 관계를
  subsequent small validation에서 지지하지 못한다: **capacity-routing
  mechanism claim kill**. routing 없이 state-adaptive scheduler만 분리하여
  재평가하거나 종료한다.
- average accepted macro-round가 2를 지속적으로 초과할 것이 확실하고
  compute-normalized signal이 없다: **BF-ODE method track kill**.

### Stage 1 진입 (`GO`, 아직 성능 claim 아님)

모든 firewall/reproducibility gate가 pass이고, 두 backbone 각각에서 H1–H3 중
적어도 두 signal이 pass하며, 특히 H2가 한 backbone 이상에서 pass해야 한다.
그 뒤에만 1K-edit의 matched-efficacy 비교군(MEMIT, naive K-step,
global alpha, static alpha_l, sequential-small-step, BF-uniform)을 제안한다.
Stage 1 proposal은 별도 red pre-flight audit 후 GH가 `tasks/pending/`으로
승격한다.

## 5. Blue team checklist

- [ ] baseline source revision/license, model revision, CounterFact subset
  manifest와 allowed 5-prefix semantics를 명시한다.
- [ ] `project/run_scripts/`에 deterministic Stage 0 wrapper와 dry-run을
  만들고, output은 `local/results/raw/<run_id>/`로만 쓴다.
- [ ] base MEMIT을 먼저 1–3 edit smoke test로 재현하고, controller를 끈
  instrumentation-only trace를 만든다.
- [ ] same snapshot에서 모든 editable layer의 key, residual, normalized
  proposal, `a_l`, `Psi_l`을 record한다; sequential write 중간값과 혼동하지
  않는다.
- [ ] perturbation h, accept/reject definition, viable-edit denominator,
  checkpoint schedule을 실행 전 고정한다.
- [ ] `Naive K-step`, global alpha, static alpha_l가 same direct-z, layer set,
  covariance, edit order, stopping definition을 공유하는지 확인한다.
- [ ] wall-clock, forward/backward count, NFE, requested/actual GPU를
  reportable metadata로 기록한다.
- [ ] raw trace/full log/checkpoint는 `local/`에만 두고 compact manifest와
  checksum/size만 Git에 남긴다.

## 6. Red team kill-test checklist

- [ ] controller module과 callback이 paraphrase, neighborhood/locality,
  downstream prompt/label/metric object에 import 또는 access하지 않는지
  static check와 runtime trace로 확인한다.
- [ ] direct-z, layer set, covariance cache, model revision, edit order가
  baseline/variant 간 동일한지 확인한다.
- [ ] partial update가 fixed update split인지, actual relinearization인지
  proposal hash/direction and utility pre/post records로 판별한다.
- [ ] `a_l` normalization, target token position, viable-edit exclusion,
  tie/top-k rule, Gini implementation이 사전에 고정되었는지 확인한다.
- [ ] empty/zero utility, QP infeasible, rejected step, failed edit를
  denominator에서 사후 제거하지 않았는지 확인한다.
- [ ] 100-edit subset이나 edit order를 positive signal 기준으로 고르지
  않았는지 확인한다; subset hash와 order seed를 확인한다.
- [ ] raw dataset/generation/checkpoint/full log/credential이 Git staged
  paths에 없는지와 artifact broadcast exception을 확인한다.
- [ ] server onboarding, Slurm GPU cap, resource collision, pre/post-run
  audit 판정이 모두 기록되었는지 확인한다.

`block`은 실행/승격을 중지한다. `warn`은 caveat를 message/report에 남기고
GH waiver 없이는 long-horizon claim으로 승격하지 않는다.

## 7. Server-head instruction envelope 초안

server1은 GH clone만 있는 `pending-onboarding` 상태이고 server-head가 없다.
아래는 향후 `server=<등록명>`에 대한 **초안**이며
`messages/inbox/<server>.md`로 아직 발행하지 않는다.

```text
명령 ID: stage0-onboard-and-preflight-<server>
대상: <server>의 server-head
대상 Codex session ID: <servers/active/<server>.md에 등록된 session ID>
대상 repository CWD: <servers/active/<server>.md에 등록된 repo clone 경로>
대상 Git repository identity: hyunjun1127/ODE-edit
사전 boundary check:
  scripts/check-session-boundary.sh <target-session-id> 가 pass여야 한다.
목적과 배경: BF-ODE-Edit Stage 0를 시작하기 전에 MEMIT diagnostic의
  실행 가능성·firewall·재현성을 확인한다. 이 명령은 성능 실험 또는 논문
  claim을 승인하지 않는다.
허용 write path:
  servers/active/<server>.md
  agents/<server>/
  plans/updates/<server>/
  tasks/proposed/<server>/
  messages/server-heads/<server>/
  messages/acks/<server>/
  audits/servers/<server>/
  experiment-reports/servers/<server>/
  runs/<run_id>/ (compact metadata only)
  project/run_scripts/ (parent-reviewed tracked wrapper only)
  local/ (raw dataset/output/log/checkpoint only)
Slurm 제출 권한: not allowed. GH가 red pre-flight pass, active GPU cap,
  proposed task를 확인한 뒤 별도 envelope로만 allowed가 될 수 있다.
GPU cap: `servers/local/gpu-caps.tsv`의 server별 private cap과
  `mem_mb_per_gpu`를 사용한다. server1은 3 GPU / 198117 MiB per GPU,
  server4는 3 GPU / 65984 MiB per GPU다.
  `scripts/check-slurm-resource-cap.sh <server> <requested_gpus>
  <requested_mem_mb>`가 통과할 때까지 pending_resource_cap으로 둔다.
red-team gate: onboarding audit와 S0 pre-flight에서 data/eval, logic/evidence,
  git/protocol auditor 모두 pass 또는 GH 기록 waiver. block이면 즉시 중단.
artifact broadcast 의무: smoke/preflight가 ordinary local artifact를 만들면
  scripts/rsync-artifact-broadcast.sh를 사용한다. active peer가 없거나
  artifact가 없으면 exception/reason을 server-head message와 run metadata에
  기록한다. private/sensitive/repo-external/--delete transfer는 금지한다.
완료 보고 경로:
  messages/acks/<server>/
  messages/server-heads/<server>/2026-07-30-stage0-onboarding.md
  audits/servers/<server>/stage0-onboarding.preflight.md
  plans/updates/<server>/stage0-feasibility.md
금지 사항: evaluation prompt/label을 controller에 주입하지 말 것; raw
  credential/connection detail/dataset/checkpoint/log를 Git에 쓰지 말 것;
  red block 무시, unapproved Slurm submission, destructive rsync, direct
  global plan/task/inbox 수정 금지.
예상 산출물: onboarding record, agent heartbeat, local path/cap feasibility,
  baseline revision and dry-run command, pre-flight audit, proposed (not
  approved) Stage 0 task.
중단 조건: missing model/dataset access, unknown/exceeded GPU or memory cap,
  missing baseline provenance, session boundary mismatch, firewall failure,
  red block, Git conflict, private artifact exposure.
```

## 8. Git, protocol, artifact, credential risk controls

| 위험 | 방지책 |
| --- | --- |
| template remote를 연구 remote로 오인 | remote 변경은 user confirmation 후에만 수행; 현재 URL은 template로 기록 |
| uncommitted proposal/template 변경과 충돌 | 기존 변경을 덮어쓰지 않고 GH 파일만 추가; commit 전 staged access check |
| server1에 SH 없는 실행 지시 | inbox/task/Slurm 대신 onboarding envelope 초안만 유지 |
| 다른 repo Codex session 오조작 | target session ID + CWD + Git identity를 instruction에 쓰고 boundary helper failure는 `block` |
| evaluation leakage | static + runtime access audit, forbidden prompt objects 분리, red pre-flight block |
| raw artifact/credential Git 유입 | `local/`, `servers/local/` ignored boundary, staged scan, compact manifest만 tracked |
| broadcast 누락 또는 sensitive transfer | ordinary `local/` broadcast 기록, exception reason, manual `transfers/` approval for sensitive/destructive/external path |
| sync conflict | `scripts/sync-agent.sh`의 dirty-tree skip 및 protocol fail-stop; conflict 시 `control/sync-paused` 판단 |

## 9. 사용자 확인 필요

1. server1에서 GH가 server-head를 겸임할지, 별도 server-head Codex session을
   만들지.
2. Stage 0를 맡을 server 이름/SH와 해당 server의 model·CounterFact·baseline
   access, Slurm GPU/memory cap, compute budget.
3. baseline source repository/revision 및 Llama-3-8B/GPT-J 사용 승인·접근 조건.
4. GH가 `experiment-reports/experiments/<section>/` index를 작성하도록
   protocol access matrix/checker를 확장할지, 현재처럼 global index만 쓸지.

이 확인 전에는 Stage 0 문서화만 완료된 상태이며, 실험은 pending이다.
