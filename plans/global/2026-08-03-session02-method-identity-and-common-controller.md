# Session 02 — ODE-Edit Method Identity and Common-Controller Validation

- 개시: **2026-08-03 14:26 KST**
- GH: `019fb1ea-03cb-7c20-bb3b-eba5f8d6f5f2`
- 상태: **`DESIGN_REFERENCE; PRIMARY_EXECUTION_SUPERSEDED`**
- primary compute-aware execution:
  [`2026-08-03-session02-compute-aware-main-table-spec.md`](2026-08-03-session02-compute-aware-main-table-spec.md)
- 실행 owner: SH1 primary implementation/run, SH2 independent reproduction
- 방법명: **ODE-Edit**
- 모델: `Llama3-8B-Instruct`, `Qwen2.5-7B-Instruct`
- actuator family: MEMIT, canonical-projector/history AlphaEdit
- 공통 원칙: 두 모델과 두 family에 동일한 controller rule; model-specific rescue 금지

이 plan은 Motivation 결과를 더 retune하는 문서가 아니다. Session 01에서 분리된
share–magnitude implementation 원인을 상속하되, ODE-Edit의 기술적 정체성, dynamic
feedback 필요성, capacity geometry의 functional relevance를 순서대로 검증한다.

> **Supersession note:** 첨부 method-direction 및 compute review를 반영해 첫 main
> table은 MEMIT-first compute-aware progression으로 축소했다. 평균 accepted round 목표는
> 삭제했고, P0/P1 profiler 뒤 10-edit five-arm identity와 100-edit four-arm table로 간다.
> 아래의 두-family/7-arm/MI-S 내용은 broad design reference이며 actionable instruction이
> 아니다. 충돌 시 위 compute-aware execution spec을 우선한다.

## 1. 네 범주

### Proposal에서 온 내용

- fixed direct-z가 유일한 parameter endpoint를 정하지 않을 수 있다.
- multi-layer finite write 뒤 key, residual, proposal efficiency와 marginal capacity가
  변하므로 entry-state allocation은 stale해질 수 있다.
- 같은 current snapshot에서 모든 layer actuator를 다시 만들고 joint partial update를
  반복하면 더 낮은 cumulative capacity endpoint를 찾을 수 있다는 가설이다.

### Repo/protocol에서 확인한 사실

- Session 01 C3는 relative layer share와 global magnitude를 분리해야 한다는
  directional signal만 남겼다.
- applied hard barrier, trust/reject/rollback, first-hit, ODE necessity와 medium
  sequential benefit은 아직 검증되지 않았다.
- EasyEdit source는 read-only이며 covariance, Alpha projector/history, Wikipedia 계열
  precomputed artifact를 재계산하지 않는다.
- GH는 설계·관리 owner이고 실제 구현·Slurm·raw artifact는 SH가 담당한다.

### GH 추정

- 핵심 estimand는 waypoint 수가 아니라 `Full ODE-Edit - best frozen/static
  synchronous controller`의 matched-terminal frontier 차이다.
- small Method identity run에서는 유의성보다 두 모델에서 같은 방향의 paired signal과
  catastrophic regression 부재가 적절하다.
- local counterfactual probe로 routing headroom을 추정할 수 있지만 이는 long-horizon
  성능 예측이나 method claim이 아니다.

### 사용자 확인 필요

- 현재 없음. Method 정식 개시, ODE-Edit 명칭, 두 모델 공통 method, GH/SH 역할 분리는
  사용자 직접 지시로 확정됐다.
- 구체 tolerance, trust threshold, capacity envelope와 compute ceiling은 결과를 보기 전
  locked execution spec으로 별도 고정한다. 이는 사용자 선택이 아니라 GH 설계 후
  preflight 대상이다.

## 2. Canonical objective

한 문장으로 고정한다.

> MEMIT은 바뀐 model state를 뒤 layer 계산에만 전달하지만, ODE-Edit은 바뀐 joint
> state를 모든 layer의 다음 proposal과 allocation 계산에 다시 전달한다.

ODE가 필요한 이유는 write를 느리게 적용하기 때문이 아니다. Joint partial update가
다음 순간의 layer proposal과 allocation을 바꾸기 때문에, model-state-dependent edit
vector field를 feedback으로 반복 적분해야 한다는 가설 때문이다. Best static
controller가 같은 frontier를 만들면 ODE necessity는 기각한다.

## 3. Method contract

Outer edit \(t\)에서 다음 순서를 바꾸지 않는다.

1. **TARGET:** entry model \(W_{t,0}\)에서 direct-z \(Z_t^\star\)를 정확히 한 번 계산한다.
2. **READ:** current snapshot \(W_{t,s}\)에서 residual과 모든 edit-layer key를 읽는다.
3. **PROPOSE:** weight mutation 없이 모든 layer low-rank proposal을 같은 snapshot에서
   생성한다.
4. **CONTROL:** allowed rewrite context의 smooth directional progress와 cumulative
   covariance capacity로 실제 applied coefficient \(y_{t,s,l}\)를 공동 결정한다.
5. **TRIAL:** QP output과 동일한 coefficient로 모든 layer에 joint partial update를
   임시 적용한다. 사후 radial rescale은 없다.
6. **ACCEPT/REJECT:** predicted/actual progress, hard event non-worsening, capacity
   barrier를 확인한다. Reject면 weights, history, RNG lineage를 exact rollback한다.
7. **RELINEARIZE:** accept 뒤 모든 layer proposal과 allocation을 새 joint state에서
   다시 계산한다.
8. **FIRST HIT:** preregistered rewrite event를 처음 만족하면 추가 proposal/write 없이
   종료한다.

같은 fixed direct-z를 쓴다는 것은 모든 arm이 같은 request-derived target으로 actuator를
만든다는 뜻이다. \(H^L(W_T)=Z^\star\)를 강제로 terminal condition으로 두지는 않으므로,
direct-z residual/cosine은 output rewrite event와 별도 축으로 반드시 보고한다.

## 4. Research questions와 반증 조건

| ID | 질문 | 살아남는 최소 신호 | 반증 또는 pivot |
|---|---|---|---|
| M1 | partial joint write 뒤 vector field가 실제로 변하는가 | residual/key/actuator/share 변화가 numerical tolerance를 넘고 ranking turnover가 관찰됨 | 변화가 noise 수준이면 ODE를 kill하고 static routing |
| M2 | refresh가 static allocation보다 유용한가 | matched event에서 Full ODE-Edit의 capacity–retention frontier가 두 모델에서 같은 방향 | dynamic이 두 모델 공통으로 동등/열세면 static pivot |
| M3 | cumulative capacity routing이 기능적 preservation과 연결되는가 | capacity/Gini 감소와 prior retention non-worsening이 함께 관찰됨 | proxy만 줄고 retention/locality가 악화하면 geometry claim kill |
| M4 | first-hit이 excess write를 줄이는가 | matched event에서 accepted path, C-distance 또는 NFE 감소 | full-budget과 차이가 없으면 terminal contribution 제거 |
| M5 | method가 model/family를 넘어 같은 rule로 작동하는가 | 동일 threshold/sign/fallback으로 Llama/Qwen×MEMIT/Alpha가 non-collapse | 한 모델 전용 rescue가 필요하면 common method gate fail |

## 5. Required arms

| Arm | proposal direction | allocation | step/terminal | 식별 대상 |
|---|---|---|---|---|
| Native | native ordered | native | native full write | EasyEdit baseline |
| Frozen K-split | entry frozen | entry frozen | equal subdivision/full endpoint | waypoint-only negative control |
| Global scaling | native/frozen | native share | prelocked common scalar | global norm control |
| Static synchronous | entry frozen | entry synchronous | common trust/first-hit | static routing |
| Dynamic share only | frozen proposal | current-state share | common trust/first-hit | allocation feedback |
| Dynamic direction only | refreshed proposal | entry share | common trust/first-hit | relinearization |
| Full ODE-Edit | refreshed proposal | refreshed share | adaptive trust/first-hit | full method |

`Static synchronous`와 `Full ODE-Edit`은 terminal rule과 allowed information이 같아야
한다. Static baseline을 결과를 본 뒤 model별로 고르거나 scalar를 재조정하지 않는다.
AlphaEdit은 별도 method가 아니라 projected actuator instantiation이며 controller
threshold, sign rule, stopping rule은 MEMIT과 동일하다.

## 6. Metric와 information firewall

### Controller가 보는 것

- rewrite request, subject, target object
- native MEMIT direct-z context
- current hidden/key/residual
- precomputed covariance 또는 canonical Alpha projector/history
- hard controller-side target-token event와 smooth directional surrogate

### Controller가 보지 못하는 것

- CounterFact paraphrase/generalization outcome
- neighborhood/locality answer
- downstream benchmark
- held-out retention outcome
- 다른 arm의 terminal result

### 평가 축

- **rewrite:** controller-side hard event, evaluation efficacy, generation/generalization
- **preservation:** locality, prior-edit retention final/AUC, edit-age curve
- **target fidelity:** \(R_z\), \(C_z\), prefix별 residual
- **capacity:** \(\sum_l\Psi_l\), max-layer share, Gini, accepted C-distance,
  endpoint displacement
- **dynamics:** actuator/key cosine, efficiency rank turnover, share/support change,
  trust ratio, accept/reject
- **compute:** direct-z count, proposal NFE, accepted/rejected round, QP time,
  wall time, GPU-hours, peak memory

Teacher-forced smooth utility는 controller proxy이며 `efficacy`나 `generalization`으로
이름 붙이지 않는다. Statistical unit은 request/order이고 동일 case의 model/family
반복을 독립 sample로 세지 않는다.

## 7. Expected-effect forecast

Method identity screen에서 다음 두 값을 retrospective diagnostic으로만 계산한다.

1. **local routing headroom:** 같은 snapshot에서 layer별 tiny counterfactual trial을
   exact rollback하고, best admissible progress-per-capacity와 static allocation의 차이를
   측정한다.
2. **refresh value:** 같은 trust radius에서 refreshed proposal/allocation과 entry-frozen
   proposal/allocation의 actual controller-progress 차이를 paired로 측정한다.

이 값은 “이 snapshot에서 dynamic routing이 얻을 수 있었던 local headroom”의 근사다.
장기 eff/gen/loc 개선폭으로 외삽하지 않으며 threshold, arm 선택, early rescue에
사용하지 않는다. Headroom이 두 모델에서 거의 0이면 대규모 실행 전에 ODE를 kill한다.

## 8. Session progression

### MT-A — deterministic technical identity

GPU 결과를 보기 전에 unit/contract test로 다음을 검증한다.

- QP equality residual과 active-slope support
- zero/negative-slope layer write 금지
- applied coefficient = solver output; post-QP rescale 0
- capacity quadratic과 barrier의 direct recomputation 일치
- forced reject의 weight/history/RNG exact rollback
- first-hit 뒤 proposal/write 0
- direct-z once/edit
- same-snapshot proposal hash와 proposal-build 중 mutation 0
- Alpha history는 accepted outer terminal 뒤 한 번만 append
- evaluation data import/access가 action freeze 이전 불가능

하나라도 실패하면 scientific outcome을 보지 않고 기술 수리 후 MT-A만 반복한다.

### MT-B — fresh two-edit GPU smoke

- 두 모델×두 family의 fresh 2-edit stream
- `Static synchronous`와 `Full ODE-Edit`만 실행
- technical invariant와 artifact schema 확인만 수행
- superiority, preservation 또는 ODE necessity를 판정하지 않음

### MI-S — fresh four-edit common signal screen

- C3 case를 재사용하지 않은 fresh 4-edit stream, 한 개 locked order
- Native, Static synchronous, Dynamic share only, Dynamic direction only,
  Full ODE-Edit
- 두 모델과 두 family에 동일 policy
- non-stationarity, direct-z fidelity, routing headroom, matched-event
  capacity/retention 방향을 확인

Lenient open 조건은 family-average `Full - Static` 방향이 두 모델에서 material하게
음수가 아니고, 적어도 한 모델에서 strict positive이며, 어느 cell도 catastrophic
acquisition/retention collapse가 없는 것이다. 두 모델 중 하나만 살리기 위한 rule
변경은 허용하지 않는다.

### MI — ODE identity common pilot

- fresh 10 edits, 최소 두 locked order
- 7개 required arm 전체
- 두 모델×두 family 공통 controller
- C3/Motivation case는 sample 또는 tuning set으로 재사용하지 않음

다음이 모두 필요하다.

1. 모든 cell이 native current acquisition 대비 locked tolerance 안에서 non-collapse
2. prior retention final/AUC가 native-matched tolerance 안
3. `Full - best preregistered Static/Frozen` family-average 방향이 두 모델 모두 양수
4. matched rewrite event에서 cumulative capacity와 max-layer concentration 감소
5. first-hit이 accepted distance 또는 NFE 감소
6. non-stationarity와 direct-z fidelity panel 완결
7. accepted macro-round distribution을 관측값으로 보고하고, 추가 `N_field`/GPU time을
   정당화하는 명확한 non-dominated frontier gain

Small pilot이므로 formal superiority significance를 요구하지 않는다. 다만 반대 방향을
“sample이 작다”는 이유로 양성으로 바꾸지 않는다.

### MS — medium sequential validation

MI 통과 후에만 fresh 100-edit, 여러 order/seed를 연다. Edit-age retention, collapse
onset, capacity/Gini, locality/downstream proxy, matched efficacy와 compute frontier를
검증한다. Capacity proxy가 낮아도 retention/downstream이 공통으로 나빠지면 current
capacity geometry를 kill하거나 새 proposal로 분리한다.

### ML — large/lifelong

MT, MI, MS를 모두 통과한 뒤에만 1K+를 연다. 10K/full stream은 별도 사용자 승인
대상이다. 그 전에는 lifelong superiority, preservation guarantee, deployable efficiency
claim을 열지 않는다.

## 9. 빠른 실행 원칙

- Llama 완료 뒤 Qwen을 올리지 않는다. 각 승인된 phase에서 두 model job을 같은
  submission batch로 올린다.
- server1 cap 4 안에서 one-GPU job이 preflight를 통과하면 model×family 네 cell을
  병렬화할 수 있다. 실제 memory request는 `198117 MiB/GPU` 이하로 SH1이 검증한다.
- server2 cap 3에서는 Llama/Qwen reproduction pair를 함께 올리고 나머지 GPU는 cap
  overflow를 만들지 않는다. `66017 MiB/GPU` 이하로 SH2가 검증한다.
- direct-z는 diverged stream state 때문에 arm 간 무조건 재사용하지 않는다. 반면
  covariance, Alpha null-space/projector/history seed, Wikipedia 계열 artifact는 기존
  precomputed 파일을 read-only 재사용한다.
- audit은 session boundary, leakage, EasyEdit write, resource cap, artifact schema,
  technical invariant에 한정한다. 결과 해석은 각 model run 뒤 별도 Terra Ultra
  analysis agent가 작성하고 SH가 검토한다.

## 10. SH 역할과 HOLD instruction envelope

아래는 onboarding blocker가 해소된 뒤 보낼 실행 envelope의 초안이다. 현재는
`Slurm submission: not allowed`이며 구현도 시작하지 않는다.

### SH1 / server1

- **목적과 배경:** reusable ODE-Edit-side hook, controller, contract test와 MT primary
  run 구현
- **target session:** `019fc5e0-eb7e-78a3-9436-93885621b8dc`, Sol Ultra,
  repository `hyunjun1127/ODE-edit`
- **필수 선행:** GH와 분리된 dedicated worktree/CWD, SH1 session boundary, server-head
  registry, clean task base, cap 4 재-ACK
- **허용 write path 초안:** `project/run_scripts/ode_edit_method/`,
  matching `project/run_scripts/session02_*`, `plans/updates/server1/`,
  `audits/servers/server1/`, `experiment-reports/servers/server1/`, `runs/`,
  ignored `local/results/raw/session02-*`와 `local/logs/session02-*`
- **Slurm:** 현재 not allowed; locked execution spec과 red preflight 뒤 allowed로 전환
- **GPU/memory cap:** 4 GPU, 198117 MiB/GPU
- **red gate:** technical identity, leakage, resource, EasyEdit read-only,
  precomputed-artifact reuse pass; block이면 stop
- **artifact broadcast:** terminal raw/log/manifest를 helper로 active peer에 broadcast,
  불가능하면 exception record
- **완료 보고:** `plans/updates/server1/`, `audits/servers/server1/`,
  `experiment-reports/servers/server1/`, `runs/`와 direct Codex ACK
- **금지:** EasyEdit 수정, artifact 재계산/다운로드, direct-z 별도 track 열람/병합,
  model-specific rule, eval leakage, Git push without SH-reviewed scope
- **예상 산출물:** reusable package/tests, locked command, compact manifest,
  model별 별도 analysis report, pair synthesis
- **중단 조건:** boundary mismatch, dirty overlap, cap fail, Terra analysis agent mismatch,
  technical invariant fail, red block

### SH2 / server2

- **목적과 배경:** SH1 implementation의 clean-source independent reproduction,
  second order와 portability 확인
- **target session:** `019fc5ec-f85b-7770-a73a-1d19be1cd491`, Sol Ultra,
  repository `hyunjun1127/ODE-edit`
- **필수 선행:** SH2/server2 identity envelope 정정, session boundary/registry,
  local method-runtime config, exact source commit 확인
- **허용 write path 초안:** `plans/updates/server2/`,
  `audits/servers/server2/`, `experiment-reports/servers/server2/`, `runs/`,
  ignored `local/results/raw/session02-*`와 `local/logs/session02-*`;
  scientific implementation 변경은 GH 재승인 전 금지
- **Slurm:** 현재 not allowed; SH1 MT source lock과 red preflight 뒤 allowed로 전환
- **GPU/memory cap:** 3 GPU, 66017 MiB/GPU
- **red gate:** source hash, boundary, leakage, resource, artifact reuse,
  independent command reconstruction pass; block이면 stop
- **artifact broadcast:** terminal raw/log/manifest broadcast, 불가능하면 exception record
- **완료 보고:** `plans/updates/server2/`, `audits/servers/server2/`,
  `experiment-reports/servers/server2/`, `runs/`와 direct Codex ACK
- **금지:** server1 identity impersonation, EasyEdit 수정, artifact recompute/download,
  model-specific rescue, SH1 raw result를 보고 threshold 변경, direct-z track 접촉
- **예상 산출물:** independent reproduction manifest, second-order result,
  model별 analysis report, discrepancy report
- **중단 조건:** boundary/runtime/source mismatch, cap fail, artifact absence,
  Terra agent mismatch, red block, SH1 spec divergence

## 11. 현재 execution hold

다음 네 항목이 해결되기 전에는 구현·Slurm 제출을 열지 않는다.

1. SH1 dedicated worktree와 cap 4 session-boundary 재-ACK
2. SH2 identity/session/runtime/registry blocker 해소
3. exact tolerance, capacity envelope, trust threshold, case/order, command를 담은
   locked MT/MI-S execution spec
4. Terra Ultra pre/post analysis agent availability 확인

Direct Codex inbox를 instruction/ACK plane으로 사용하고, Git에는 승인된 source,
small manifest와 compact report만 남긴다. Raw output, generation, checkpoint, model,
dataset, full log, credential은 Git에 두지 않는다.
