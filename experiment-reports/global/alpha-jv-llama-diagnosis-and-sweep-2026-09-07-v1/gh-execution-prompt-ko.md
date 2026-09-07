# GH 실행 지시문: Llama 원인 진단 + 모델별 hparam sweep 독립 병렬 실험

당신은 ODE-edit의 GH다. 아래 범위의 구현·실험을 독립 작업으로 준비하고, 기존 운영 계약에 따라 별도 자원을 배정받은 SH에게 실행을 맡겨라. 이 작업은 Server4의 다음 두 task와 **병렬**이며, 해당 task의 종료를 기다리는 후속 run이 아니다.

- `odeedit_orbode_cum_s4_r1`
- `odeedit_alpha_l8_s4_takeover`

목표는 두 가지다.

1. **Track D:** Llama JV B10의 near-stall이 어떤 원인으로 발생했는지 같은 state의 관측과 제한된 intervention으로 진단한다.
2. **Track S:** 정상 development state에서 Llama/Qwen 각각의 λ·T·N이 actual write와 RS/PS/NS에 미치는 영향을 병렬 분석한다.

Track S는 Track D의 성공·exact replay 확보를 기다리지 않는다. Track D는 sweep에서 좋은 설정을 찾았다는 이유로 생략하지 않는다. 최종 method, normalization, lifelong 성능 claim은 이번 실험에서 성급히 확정하지 않는다.

## 0. 입력 문서·source·격리

먼저 다음 문서와 실제 source를 읽어라.

- 상위 연구 계획: `experiment-reports/global/alpha-jv-parallel-research-plan-2026-09-07-v1/report-ko.md`
- 이번 상세 설계: `experiment-reports/global/alpha-jv-llama-diagnosis-and-sweep-2026-09-07-v1/design-ko.md`
- GH 통합 분석: `experiment-reports/global/alpha-jv-sequential-mechanism-audit-2026-09-07-v2/gh-mechanism-review-ko.md`, 특히 §8–9·13.
- 같은 폴더의 `sweep-candidates.csv`, `sweep-candidate-policy.json`, `supporting/llama-audit.md`.
- SH1 완료 보고: `experiment-reports/servers/server1/alpha-jv-sequential1000-layer-review-2026-09-07-v1/factual-report-ko.md`와 checkpoint/normalization/endpoint 관련 publication.

작성 시 확인한 references:

```text
repository: hyunjun1127/ODE-edit
main_at_design: e25a5685a4ad7f8606a8c265abf070b0d77a83d1
alpha_scientific_runtime: 77358b1546d1baf83b3e251afcce663b08d7bfd7
completed_alpha_publication: 0d0a0131e4a6a2a645dfa6530377d420a084d136
l8_takeover_contract: 44602a1a80554c67da0ef9646b43d843104785f2
orbode_cumulative_contract: 8610faf0e114059a5e08116f1164baf31c573e80
```

실제 시작 시 remote/local HEAD와 diff를 확인하고 source와 report reference를 별도로 pin하라. Main이 갱신되어도 임의의 파일들을 여러 commit에서 섞지 말라. 로컬 경로에 없는 문서를 읽은 것처럼 기록하지 말고 위 pinned Git 자료 또는 전달받은 문서를 사용하라.

격리 계약:

- 별도 worktree/branch/source/output/process를 사용한다. Live Server4 source에 hook을 넣거나 설정·환경·checkpoint·cache를 변경하지 않는다.
- 두 Server4 GPU에 새 process를 동시 탑재하지 않는다. 해당 작업을 취소·재시작·선점하지 않는다.
- 비교용으로 완료·봉인된 publication만 읽는다. 작성 중 JSON/checkpoint로 원인이나 진행률을 추정하지 않는다.
- 모델 객체와 AlphaEdit module globals/cache를 track·branch 간 공유 mutation하지 않는다. 같은 process의 순차 비교는 매 branch entry를 정확히 복원하고 새 Family를 만드는 경우에만 허용한다.
- SH가 승인받은 독립 host/GPU/hour cap을 `resource.lock.json`에 기록한다. 미배정 값은 임의로 채우지 않는다. 미배정 시 CPU 구현·검증은 진행하고 GPU 단계만 `WAITING_FOR_ISOLATED_RESOURCE`로 남긴다.
- 기본은 독립 GPU 한 개에 한 process다. 이미 별도로 배정된 GPU 두 개가 있으면 D/S를 각각 실행할 수 있다. 자동 자원 확대는 금지한다.
- GPU 한 개면 D의 artifact 확인 중 S를 준비한다. D 복원이 막히면 S를 진행한다. 하나의 결론을 기다리느라 독립 작업 전체를 대기시키지 않는다.

권장 신규 경로:

```text
runtime ownership: project/run_scripts/alpha_jv_llama_diagnosis/
local artifacts: local/alpha-jv-llama-diagnosis-sweep/<attempt-id>/
publication: experiment-reports/servers/<assigned-server>/alpha-jv-llama-diagnosis-sweep-2026-09-07-v1/
global synthesis: experiment-reports/global/alpha-jv-llama-diagnosis-sweep-results-2026-09-07-v1/
suggested task names:
  odeedit_llama_diag_parallel_v1
  odeedit_jv_sweep_parallel_v1
```

경로는 create-once다. 기존 디렉터리를 덮어쓰거나 실패 attempt를 재사용하지 않는다. Raw tensor/prompt/context/cache/checkpoint는 local-only다. Git에는 source·raw-free aggregate·manifest·checksum·보고서만 올린다. Main merge/push는 기존 GH/SH 권한 범위를 따른다.

## 1. 과학적 출발점과 금지된 shortcut

Llama final RS 실패77건 중75건은 B10 신규 edit 실패이고, 후속 rewrite 실패는2건이다. 정상 B1–B9에서도 PS/NS는 모두 Official보다 낮았다. 두 문제를 분리하라. B1은 cold M0이므로 그 열세를 historical cache 부족만으로 설명하지 말라.

B10 case4228은 tiny positive entry residual과 stock compute-z delta0 관측을 갖는다. 기존 inverse-weight 수치는 실제 requestwise Gram 기여율이 아니다. 기존 H condition number는 약3.19이고, λ .001–1 shadow는 B10 field를 거의 바꾸지 않았다. 이미 완료된 1,040 CPU shadow를 GPU 결과로 다시 세거나 반복 생성하지 말라.

이번 실험에서 고정할 것:

- 모델별 원본 AlphaEdit compute-z, target loss/contexts/iterations/learning rate/early-stop .05.
- Tokenizer/padding/subject-token position/readout L8, editable L4–L8, native P/L2/keys/history finalization.
- FULL-FP32 model/storage/forward, source의 기존 연산 예외, pinned attention backend/TF32/autocast 정책.
- JV/NRMS/S의 native direction은 full residual/divisor1, NNLS c≥0와 upper bound 없음. O-SAME/O_NATIVE는 stock remaining-layer allocation을 그대로 유지한다.
- Inner z 재최적화 없음. h는 integrator에만 적용. First-hit/backtracking/rollback/adaptive gain OFF.

금지:

- 특정 case 삭제, tiny residual clip/floor의 자동 도입, case4228을 hardcode한 write policy.
- Native writer에 normalized residual이나 active-only requests를 넣는 것.
- 성능이 나쁜 endpoint를 Official 또는 더 긴 T로 replacement하는 것.
- 정상 λ sweep의 개선을 원래 B10의 원인 확정으로 보고하는 것.
- 새 history-output controller, causal layer weights, external reference bank, dynamic-z를 이번 D/S에 섞는 것.
- L8 재배분 자체를 성공 기준으로 삼거나 1k/10k full-grid를 자동 제출하는 것.

Independent branch 간 entry restore는 필수 실험 제어다. 이를 integrator의 adaptive rollback과 혼동하지 말라. 후자는 사용하지 않는다.

## 2. 즉시 두 트랙으로 나누어 준비

### D owner

Sealed artifact inventory, exact-state restoration, raw capture, N0/NRMS same-state attribution, actual anomaly intervention을 담당한다.

### S owner

Immutable λ/T/N configuration, 새로운 common-entry dev sample, 5λ와 Euler/horizon actual sweep, 비용·endpoint 분석을 담당한다.

공유 구현은 fixture binder·normalization adapter·configurable trajectory이며 한 owner가 통합한다. 서로 기존 변경을 되돌리지 않는다. Native writer/JVP/NNLS kernels는 그대로 재사용한다. 같은-entry raw bundle과 fixed-z는 immutable artifact로 공유하고 mutable Family/overlay는 공유하지 않는다.

기존 1k main launcher/provenance를 그대로 호출하지 말라. 기존 two-model smoke/main-table prerequisite, 고정1000 sample, Server2 cap 등을 새 소형 실험에 통째로 상속하면 불필요한 run이나 잘못된 자원 계약이 생긴다. 검증된 primitives를 재사용하되 신규 runner/manifest는 이번 D/S의 실제 sample·자원·범위에 결속한다.

## 3. D0 — 복원 가능성부터 닫아라

원인 진단 target은 원래 **Llama JV W9/M9 + B10 100requests + 원래 z/context**다. 필요한 항목을 실제 파일·bytes·SHA·source identity와 함께 조사하라.

```text
base model and tokenizer snapshot
selected editable weights W9 (or W5)
Alpha cache M9 (or M5), cache_c_new
projector P and native hparams
fixed z tensor, or original z hash for validated recomputation
request records/order and original context JSON
runtime source/environment/dtype/attention/RNG identities
intermediate entry/commit W/M/z hashes
increment/history journal availability and replay status
```

Production inventory는 W1/M1,W5/M5,W10/M10만 기록한다. Fixed-z tensor 저장은 해당 source에서 확인되지 않았고 journal replay는 NOT_TESTED다. W9/z가 있다고 가정하지 말라. Context 원문 없이 재생성한 값을 exact라고 부르지 말라.

복원 순서는 아래로 고정한다.

1. 추가 sealed W9/M9/z가 실제 있으면 사용한다.
2. 충분한 sealed journal이 있으면 replay 후 hash를 검증한다.
3. 없으면 W5/M5에서 B6–B9를 한 번 source-exact bounded replay한다. 이 경로의 비용을 먼저 resource ledger에 예약한다.
4. 중간 W/M/z와 B10 z까지 일치해야 `EXACT_STATE_REPLAY`다.
5. 첫 identity mismatch가 발생하면 위치와 원인을 기록한다. 명백한 source-binding 오류를 고친 새 technical attempt 외에 반복 replay로 맞추지 않는다. 다른 state로 계속 관측할 경우 처음부터 `NEW_REPLAY_ANALOGUE`로 이름을 바꾼다.
6. W5/context도 없으면 `HISTORICAL_EXACT_STATE_UNAVAILABLE`. S와 접근 가능한 정상 controls는 계속 진행한다.

W10을 W9로 사용하지 말라. 작은 weight 차이와 무관하게 M10에는 B10 history가 append되어 있다. W5 복원 실패를 W0→B9 전체 replay로 자동 확대하지 말라.

정상 retrospective controls는 C-B1(W0/M0, 원 B1), C-B6(W5/M5, 원 B6)로 둔다. B9 control을 위해 추가 replay하지 않는다. W5→W9 replay를 수행한다면 B6의 정상 raw capture와 baseline endpoint를 observer 비개입 검증 후 재사용한다.

## 4. 최소 구현 명세 — 이 경계를 지켜라

새 package에 다음 책임을 분리하라. 이름은 구현 계약이며 이미 존재하는 실행 CLI라고 가정하지 않는다.

| 모듈 | 책임 | 재사용할 기존 primitive |
|---|---|---|
| `contracts.py` | frozen config, IDs, λ/T/N/h와 normalization contracts | 기존 source constants/SCIENCE를 baseline으로 확인 |
| `fixture.py` | W/M restore, warm/cold bind, fixed-z와 semantic inventory attach | `ObservedFamily`, `FixedZArtifact`, 기존 state/hash 함수 |
| `capture.py` | raw residual/JVP/dictionary/reference와 repeatability | `NativeDictionary`, `GroupedFP32Overlay`, `TerminalResponseObserver` |
| `normalization_views.py` | source-exact N0 delegate, NRMS_ENTRY | `FrozenNormalization` |
| `requestwise.py` | g_i/H_i, all-row influence, same-state shadows | 기존 FP64 NNLS/metric |
| `trajectory.py` | config를 받는 최소 joint adapter, branch/prefix lifecycle | 기존 `run_joint`의 수학·overlay·finalization 순서 |
| `runner.py` | D/S 별도 manifest·실행·실패 ledger | 기존 model/asset/evaluator binding |
| `tests.py` | 아래 최소 계약 검사 | 기존 small algebra/fidelity helpers |

읽어야 할 source 지점:

```text
ordered_response_barrier_ode/runtime.py:
  compute_fixed_z, terminal_graph, terminal, build_layer, run_official, finalize
ordered_response_barrier_ode/terminal_jvp.py:
  TerminalResponseObserver.observe
native_response_ode_v31/algebra.py:
  FrozenNormalization, nnls_response
native_response_ode_v31/native_binding.py:
  NativeDictionary.build, capture_reference, whiten
alpha_native_response_ode_v31_sequential/trajectory.py:
  current response assembly, physical append, node telemetry, endpoint/reset
alpha_native_response_ode_v31_sequential/telemetry.py:
  separately imported LAMBDA and history/single-layer shadows
alpha_native_response_ode_v31_sequential/state.py:
  W/M checkpoint and commit identities
```

주요 구현 규칙:

1. Sealed W/M를 model/module에 restore한 후 새 Family를 생성한다. Warm에서는 `bind_existing_method_state()`를 쓰고 cold reset하지 않는다.
2. Saved z를 attach할 때 `FixedZArtifact`의 tensor/order/context hashes와 semantic inventory를 함께 복원한다. `.values`만 덮어쓰지 않는다.
3. Target/terminal은 FP32 `[D,B]`; primary residual은 `target32-terminal32` 후 FP64 weighting이다. 먼저 double subtraction하지 않는다.
4. Primary N0는 기존 `FrozenNormalization`에 delegate한다. Active는 entry FP32 scale>0으로 동결한다. Exact-zero row도 native B100 writer에는 남긴다.
5. Raw D에 대해 JVP를 구하고, response FP32 값을 FP64로 올린 뒤 normalization/whitening한다. Raw D를 먼저 거대 whitening scale로 model에 넣는 새 경로를 만들지 않는다.
6. Per-request raw bundle은 `[m,D,B]`. Production flattened vector는 `[D,B_active].reshape(-1)`. 별도의 request-major 순서로 e/Psi를 엇갈리게 만들지 않는다.
7. qref는 entry5 directions에서 한 번 동결한다. q_l은 current direction의 비용이므로 다음 node에서 바뀔 수 있다.
8. Physical velocity coefficient는 `c_l/sqrt(q_l)`, actual step coefficient는 `h*c_l/sqrt(q_l)`다.
9. T/N/λ를 하나의 immutable config로 trajectory·telemetry·shadow·lock에 전달한다. `h=T/N`, actual clock=`sum(h)`를 검사한다. JSON의 T만 바꾸거나 import-time constants를 일부만 monkeypatch하지 않는다.
10. 기존 trajectory의 hardcoded N0 label을 NRMS 결과에 재사용하지 않는다. Normalization ID와 실제 denominators를 config에서 저장한다.
11. 각 actual branch는 자기 current state에서 keys/D/JVP를 재계산한다. N0의 coefficient sequence를 NRMS나 다른 λ branch로 복사하지 않는다.
12. Endpoint capture/history append/reset의 source lifecycle을 유지한다. Branch는 독립 model trajectory를 배포하지 않고 entry로 복원한다. 두 arm의 endpoint policy를 다르게 만들지 않는다.

Inactive request의 weighted g_i/H_i는0, applied denominator는NA로 기록한다. Division by zero는 금지한다. Zero native direction이 제외되어 m<5이면 실제 active-layer mapping을 보존하고 publication만 L4–L8 순서로 zero-pad한다. q=0인 direction을 whitening하지 않는다.

### 최소 CPU tests

새 대형 audit 프로젝트로 확대하지 말고 다음 오류를 막는 소형 테스트를 먼저 통과시켜라.

- Source N0 rounding/weight parity, exact-zero와 tiny-positive 구분.
- NRMS active inventory 동일성, native RHS·request inventory 불변.
- Production g/H와 requestwise sum 일치.
- Whitening/raw coefficient roundtrip, layer permutation/coordinate scaling.
- Same-entry branch identity와 qref 고정, per-node q_l 갱신.
- Config λ가 모든 telemetry/shadow에 동일; T=N*h.
- Empty active set/zero response는 finite no-action/NA 처리, nonfinite input은 typed boundary.
- All-row influence가 actual write path에 전달되지 않음.
- Prefix observation의 W/M/RNG/overlay 비개입.
- Stored normalization label과 applied denominator 일치.

Well-conditioned FP64 합성 재구성·KKT는 scaled residual1e-10을 engineering 기준으로 사용하되 source의 기존 tolerance를 바꾸지 말라. 실제 ill-conditioned/zero-response는 별도 상태로 기록하고 ridge로 통과시키지 않는다.

GPU fidelity는 정상 신호 fixture의 최초 entry와 한 nonzero node에서 최소 확인한다. 기존 raw FD grid `2^-7,2^-8,2^-9`, 인접 epsilon 두 점의 충분한 signal, cosine≥.99/relativeL2≤.05, 기존 overlay parity 기준을 재사용한다. Raw signal이 envelope 이하이면 `JVP_NUMERICALLY_UNRESOLVED`; tiny response의 정확성이 검증됐다고 하지 않는다. D-B10의 zero-field를 비정상 구현으로 처리하지 말고 정상 smoke와 구분한다. 한 번 검증한 동일 kernel을 candidate마다 FD로 재검증하지 않는다.

## 5. D1 — 같은 state의 raw attribution

Entry capture3회를 하되, 다음 세 가지 차이를 분리하라.

- 동일 primitive의 반복 편차.
- Compute-z의 native capture/context/token/batching과 controller canonical capture 사이의 차이.
- `float32(z-Phi)`와 `double(z)-double(Phi)`의 arithmetic-order 차이.

Primary target/readout은 바꾸지 않는다. Stock target log의 delta0·loss·stop reason을 수집하되 없으면 `NOT_OBSERVED`로 두고 추정하지 않는다. Repeated capture가 같아도 deterministic rounding/context bias가 없다는 뜻은 아니다.

Normalization의 entry 기준은 최초 source-equivalent capture로 봉인한다. 세 capture의 평균이나 마지막 capture로 기준을 바꾸지 않는다. 반복 관측은 observer이고 normalization initialization이 아니다.

공통 W/M/z/D/qref의 raw5-direction JVP를 한 번 저장한다. 각 i에 대해 다음을 계산하라.

$$
\Psi_i=P_i\operatorname{diag}(q^{-1/2})/d_i,\quad e_i=R_i/d_i,
\quad H_i=\Psi_i^\top\Psi_i,\quad g_i=\Psi_i^\top e_i.
$$

검사: `sum(H_i)==H`, `sum(g_i)==g`를 FP64 scaled tolerance로 production solve와 대조한다. Bit equality를 강제하지 않는다.

필수 항목:

```text
case_id, entry residual norm, active flag, applied denominator
raw response norm by layer, weighted/whitened response norm
g_i[5], H_i[5,5], signed gain/cancellation
trace(H_i)/trace(H), top1/top5 share, effective request count
global spectrum/condition, error-span explanation, NNLS support/KKT
native coefficients c, raw coefficients c/sqrt(q)
physical native/Frobenius field norm and angle
requestwise actual-vs-predicted response where measured
```

원래 tiny case는 보고서의 지정 diagnostic case일 뿐 runtime selection rule이 아니다. 전체100행에 대해 동일 attribution을 저장하라. `trace(H)=0` 또는 zero field의 ratio/angle은 NA/명시된 zero로 처리한다.

### All-row influence observer

모든 i에 대해 `H_minus_i=H-H_i`, `g_minus_i=g-g_i`를 계산해 coefficient/physical field 영향을 본다. 원래 B_active, D, z, qref, q_l, λ는 고정한다. 99개 기준 재정규화를 하지 않는다. 뺄셈 cancellation이 문제면 나머지 H_i/g_i를 다시 합산한다.

기존 `nnls_response`는 e/Psi를 받으므로 새 Gram-input solver를 만들 필요가 없다. 저장된 production e/Psi의 해당 request 전체 activation block을 zero-weight하여 같은 solver를 호출하고, 그 g/H가 위 leave-one-row 식과 일치하는지 검사한다. Original denominators와 `[D,B_active]` index mapping은 유지한다.

이 coefficient를 실제 model write에 사용하지 말라. 명칭은 `ALL_ROW_OBJECTIVE_INFLUENCE_OBSERVER`이며 request deletion 실험이 아니다.

### N0 대 NRMS shadow

NRMS는 source-frozen FP32 scales의 FP64 RMS다.

$$
s_{RMS}=\sqrt{\operatorname{mean}_{i\in A}(s_{i,32}^2)},\qquad
d_i=\sqrt{|A|}\,s_{RMS}.
$$

제곱·평균은 FP64에서 수행한다. Scale을 FP64 residual norm으로 새로 정의하지 않는다. 두 weight views에서 e/Psi 모두 같은 denominator를 적용한다. V0는 rounding 아래 약.5이며 exact.5 강제 rescale은 금지한다.

기존 λ shadow는 재사용한다. 새 raw bundle에서 λ 그림이 필요하면 CPU solve만 한다. B10에서 후보λ가 무효였다는 결과를 정상 S sweep의 금지 근거로 사용하지 않는다.

## 6. D2 — 제한된 actual intervention

Exact 또는 명시된 analogue D-B10 entry에서 다음 세 arm을 실행하라.

| Arm | 변경 |
|---|---|
| D-N0 | Source-exact JV, λ.1/T2/N4/h.5 |
| D-NRMS | 동일 계약에서 objective/JVP normalization만 NRMS_ENTRY |
| D-O-SAME | 동일 JV entry/target에서 pinned Official one-pass |

O-SAME은 baseline-native write path를 유지한다. qref/observer 정보를 Official controller에 삽입하지 않는다. 이미 계산한 frozen z를 소비하는 기존 `run_official(fixed_z=...)` 계약을 재사용한다. 원래 Official chain의 B10 수치를 이 arm 대신 쓰지 않는다.

D-N0와 D-NRMS는 각각4 nodes다. 공통 entry raw captures를 재사용하되 실제 raw JVP 호출/재사용을 ledger로 구분한다. Canonical current100 평가를 세 arm 모두 수행한다. 상태가 finite이면 poor endpoint도 그대로 기록한다.

Endpoint 관측:

- Entry 평가1회, 각 final의 current RS/PS/NS 분모100/200/1000.
- Current100 request별 new/true NLL, margin, strict/token secondary.
- Old B1–B9 rewrite900: entry1회와 각 final. Entry-success→failure, entry-failure→recovery, initially-successful의 cumulative loss를 구분.
- Intended hF, virtual activation progress, dense materialized delta norm/changed FP32 fraction, actual-predicted error.
- Native raw/normalized path work와 endpoint net action을 구분. Microstep 제곱합을 path work라고 부르지 않음.
- 각 endpoint의 N0-V와 NRMS-V를 모두 cross-score하고 raw residual 분포·semantic 결과도 함께 기록.
- Tiny case와 나머지99개의 변화는 보조 분해이며 전체100 denominator를 대체하지 않음.

Old full PS/NS는 첫 진단의 필수 범위가 아니다. 저장한 endpoint에서 후속 evaluation이 필요하면 별도 비용으로 한다. Controller에는 old evaluation·PS/NS를 넣지 않는다.

**Endpoint 활성화 계약:** 기존 `run_joint`/`run_official`은 반환 시 entry로 복원한다. Current/old900/cross-score는 동일한 actual materialized endpoint가 활성화된 scope에서 수행하거나, 검증된 endpoint W를 별도 shadow에 복원해 수행한다. 평가 직전 endpoint W hash를 확인하고 평가 후 entry로 복원한다. 반환 직후의 model을 무조건 final로 간주하지 말라. 평가를 위한 writer 재실행·history 재append는 금지한다.

N0가 원래 state의 near-stall을 재현하지 않으면 `ORIGINAL_ANOMALY_NOT_REPRODUCED`로 보고 원래 원인 claim을 보류한다. 이 경우에도 새로운 state에서의 결과는 버리지 않는다. NRMS만 잘 나왔다고 original failure를 수정하거나 normalization을 production에 배포하지 않는다.

해석:

- Raw Gram 지배와 field 억제, actual write/semantic 변화가 함께 바뀌면 해당 state의 weighting causal contribution 근거다.
- NRMS가 유효한 tiny request를 희생했다면 trade-off를 공개한다.
- N0/NRMS 둘 다 실패하고 O-SAME만 성공하면 현재 normalized control/cone/time 문제가 남는다.
- O-SAME도 실패하면 target/native feasibility 후보가 남지만 target-invalid가 확정된 것은 아니다.
- 필요 시 후속으로 기존 direct-z intervention의 training-context efficacy만 관측할 수 있다. 이번 기본 실행에 새 target 재학습/held-out z intervention을 넣지 말라.

## 7. S0 — 독립 sample과 source-exact common-entry 준비

S는 D 복원·원인 결론과 무관하게 시작하라. N0/기존 model-specific baseline compute-z를 유지한다.

Sample 제안:

```text
seed_namespace: ODEEDIT-LLAMA-DIAG-SWEEP-20260907-V1
selection_key: SHA256(seed_namespace + "|" + canonical_case_id)
selection_order: hash ascending; canonical_case_id tie-break
S_DEV: first eligible100
S_AUDIT_RESERVED: next eligible300, fixed3×B100
models: same IDs and order for Llama and Qwen
```

Eligibility는 기존 dataset/schema의 유효 record이며, 완료 main1000·기존 pilot/dev·두 live reserved inventory를 제외한다. 제외는 immutable sample manifests로만 하며 현재 성능/teacher success/residual 크기로 고르지 않는다. 부족하면 `SAMPLE_POOL_INSUFFICIENT`다. 임의 규칙 완화나 실패 request 교체는 금지한다.

S_DEV는 cold W0/M0의 정상 운영점 비교다. N0가 여기서도 stall하면 정상 sample을 교체하지 않고 새 failure로 남긴다. S_AUDIT_RESERVED는 이번 full grid에 쓰지 않는다.

Sample seal 때 canonical evaluation inventory와 실제 분모도 기록한다. D의 원래100/200/1000과 달리 새 sample의 prompt 개수가 다르면 record를 버리거나 복제해200/1000으로 맞추지 않는다. Pinned evaluator가 허용하는 actual request/prompt denominator를 모든 후보에 동일 적용하고 차이를 명시한다. 기존 evaluator schema 자체를 만족하지 못하면 입력 boundary로 보고한다.

모델별 entry W/M/context/P/hparams, fixed z와 normalization/qref를 한 번 계산·봉인하고 모든 후보가 공유한다. Candidate마다 compute-z를 다시 실행하지 말라. Llama와 Qwen의 원본 z hparams는 서로 다른 값을 그대로 유지한다. λ_response는 native L2가 아니다.

## 8. S1 — GH 후보를 그대로 actual sweep

아래9점을 **양 모델**에서 비교하라. Llama가 우선이지만 결과가 좋은 모델만 확장하지 않는다. 중간 .0316228/.316228을 CPU-only로 낮추지 않는다. 이들은 이번 사용자 추가 요청을 반영한 actual development 범위다.

| ID | λ_response | T | N | h |
|---|---:|---:|---:|---:|
| JV-BASE | .1 | 2 | 4 | .5 |
| JV-LAM-001 | .01 | 2 | 4 | .5 |
| JV-LAM-00316 | .03162277660168379 | 2 | 4 | .5 |
| JV-LAM-0316 | .31622776601683794 | 2 | 4 | .5 |
| JV-LAM-1 | 1 | 2 | 4 | .5 |
| JV-RES-N2 | .1 | 2 | 2 | 1 |
| JV-RES-N8 | .1 | 2 | 8 | .25 |
| JV-HOR-T1 | .1 | 1 | 2 | .5 |
| JV-HOR-T4 | .1 | 4 | 8 | .5 |

O_NATIVE one-pass를 모델별 common-entry reference로 한 번 추가한다. JV-RES-N16(λ.1/T2/N16/h.125)은 원래 후보 pool에 보존하되 기본 sweep은 아니다.

### 실행 최소화와 순서

모델별 실제 paths는7개다.

1. λ.1,h.5,T4:8nodes. T1/T2/T4 세 endpoint.
2. λ.1,h1,T2:2nodes.
3. λ.1,h.25,T2:8nodes.
4. λ.01,h.5,T2:4nodes.
5. λ1,h.5,T2:4nodes.
6. λ.03162277660168379,h.5,T2:4nodes.
7. λ.31622776601683794,h.5,T2:4nodes.

합34nodes/170 main target JVP per model, 양 모델68nodes/340JVP를 nominal 상한으로 계측한다. 공통 entry 재사용으로 절약한 실제 호출은 따로 기록한다. Official/compute-z/entry solves/evaluator/FD/history finalization은 이 수에 포함하지 않는다.

Prefix materialization/evaluation은 기존 `derived_observation_only=True` 경로를 사용한다. History append0, persistent endpoint capture0, W/M/RNG/overlay state 비개입을 검사한다. Prefix observer가 counter를 늘리면 diagnostic ledger로 분리하되 controller state를 바꾸지 않는다. Prefix를 primary finalize한 뒤 continuation하는 구현은 금지한다.

T4/N8 parent의 T2 prefix를 JV-BASE로 저장할 때 `effective_T=2,effective_N=4,h=.5`와 `parent_T=4,parent_N=8`을 나누어 기록한다. Vector field가 total T/N을 입력으로 사용하지 않는 autonomous current-state field임을 확인한다. Parent config를 child endpoint label에 그대로 복사하지 않는다.

Derived prefix는 성능을 평가할 수 있는 W이지 committed-history checkpoint가 아니다. 그 prefix의 M에는 current batch keys가 append되지 않았다. 이번 S는 common-entry 성능 비교에만 사용하며, 나중에 sequential 시작점으로 쓸 경우 해당 W에서 별도로 정확히 한 번 finalization한 M과 새 checkpoint identity가 필요하다.

Resource 부족 시 성능을 보고 cell을 제거하지 않는다. 먼저 양 모델의 base/time axes와 coarse λ.01/1을 같은 우선순위 묶음으로 완료하고, 다음 두 intermediate λ를 진행한다. 기본9점이 미완이면 `PARTIAL_BUDGET`으로 명시하고9점을 완료한 것처럼 보고하지 않는다. 이 순서는 budget 대응 순서이며 범위 자동 축소 승인이 아니다.

N16은 N8까지의 실제 refinement/roundoff를 검토하고 남은 승인 budget 안에서 별도 manifest로 켠다. 결과를 보고 특정 모델만 N16으로 rescue하지 않는다. λ×T×N 전체 Cartesian product는 만들지 않는다.

## 9. S2 — 성능·기하·수치 결과를 함께 평가

모든9개 endpoint와 Official에서 current100 canonical RS/PS/NS, requestwise new/true NLL/margin, strict/token secondary를 평가한다. Entry/W0도 한 번 평가한다. 합계 점수뿐 아니라 Official 대비 paired improvement/loss와 entry 대비 손실·회복을 분리한다.

Node에는 다음을 기록하라.

```text
actual pseudo-time, lambda, T, N, h, normalization ID
V, raw residual norms, training semantic margin/first-hit observation
g/H/G, c, raw direction coefficients, KKT residual
native raw/normalized velocity action, path work, endpoint net action
actual barrier increment and finite-step defect (separate definitions)
actual/predicted activation change, physical materialization discrepancy
absolute layer action and share, physical native/Frobenius angle
main/diagnostic/evaluator time and calls
```

First-hit은 기존 training predicate의 관측일 뿐 stopping/selector가 아니다. Full semantic evaluation을 매 node 추가해 비용을 숨기지 말고 관측 주기·counter를 lock하라. Prefix/current endpoint 평가와 diagnostic training observations를 구분한다.

분석은 다음 세 축으로 작성하라.

1. λ: 동일 T2/N4에서 amplitude·angle·quality/action 변화.
2. Resolution: T2 고정 N2/N4/N8의 W/Phi/residual endpoint distance와 dissipation/roundoff.
3. Exposure: h.5 고정 T1/T2/T4의 progress·semantic quality·locality 변화.

같은 scalar V가 비슷해도 requestwise residual/semantic quality가 다를 수 있다. 사전 고정 progress levels V/V0=.75/.50/.25, tolerance band .02에서 **실제 saved states만** matched-progress 보조 비교에 사용하라. 미도달은 NOT_REACHED/UNMATCHED다. Weight interpolation으로 새 endpoint를 만들거나 NS가 좋은 prefix를 고르지 말라.

더 큰 N의 낮은 점수를 곧바로 ODE 실패라고 하지 말라. Coarse Euler의 과도한 contraction과 정확한 continuous integration은 다르다. 더 긴 T의 개선은 더 많은 exposure/compute가 만든 것일 수 있다. NFE와 wall time을 같이 보고하라.

S는 N0 고정으로 닫는다. D에서 NRMS가 유망해도 진행 중 S의 weight view를 바꾸지 않는다. 후속 normalization×선택운영점 교차는 별도 source/science ID로 제안한다.

## 10. 시행착오를 줄이는 단계별 종료 정책

| 상태 | 조치 |
|---|---|
| Source/context/P/model identity 불일치 | 해당 fixture HOLD; 다른 독립 track은 진행 |
| W9/z 미확보 | D exact unavailable; bounded route 또는 명시된 analogue; S 계속 |
| g/H row reconstruction 또는 coefficient transform 실패 | 새 adapter technical fix 후 새 attempt; GPU sweep 확대 금지 |
| FD 신호가 envelope 이하 | NUMERICALLY_UNRESOLVED; 정확도 PASS 금지, 원인 claim 제한 |
| 충분한 신호에서 FD/parity 실패 | Hook/index/dtype/overlay 조사; tolerance 임의 완화 금지 |
| Native metric nonpositive/unresolved | 대체 metric 없이 HOLD |
| Finite zero-field/near-stall/horizon miss | Endpoint 평가, scientific observation으로 포함 |
| Finite Euler barrier defect | 기록; fixed-grid primary를 사후 reject/backtrack하지 않음 |
| OOM/resource cap | Technical/budget status; 실패 case replacement 금지 |
| NRMS rescue | Development intervention 근거; 즉시 production 수정/원래 final 점수 갱신 금지 |
| λ/T/N 개선 | Operating-point 후보; B10 원인 해결·lifelong 성공으로 승격 금지 |

원인 진단의 충분조건을 single PASS로 압축하지 않는다. Exact-state raw attribution, numerical-origin 근거, actual intervention, normal-control, fresh development, independent audit를 서로 다른 단계로 보고하라.

## 11. 저장·재현·결론 제출

각 track에 source/runtime/science/sample/resource lock, run registry를 둔다. 필수 trajectory identity:

```text
track, fixture, provenance_mode, model, arm, candidate_id
source_sha, model/tokenizer/P/hparams/context identities
entry_W_sha, entry_M_sha, cache_c_new
fixed_z_sha, request_order_sha, semantic_inventory_sha
normalization_id, source_scales_sha, active_mask_sha, applied_denominator_sha
qref and metric identity, lambda, T, N, h, state_version
parent trajectory and prefix endpoint identity when reused
```

필수 raw/local artifact:

- 실제 fixed-z와 entry activation/raw JVP bundle.
- 재현 가능한 selected W/M/context 지원 및 chronological low-rank increment journal.
- Actual endpoint snapshots 또는 exact reconstruction을 검증한 기록.
- Prompt-level evaluation과 all-row matrices.

필수 raw-free publication:

```text
artifact_availability.json
replay_identity.csv
capture_repeatability.json
request_response_contributions.csv
all_row_influence.csv
normalization_same_state.csv
diagnosis_nodes.csv
diagnosis_endpoint_metrics.csv
diagnosis_prompt_transition_summary.csv
sweep_candidate_metrics.csv
sweep_endpoint_distances.csv
sweep_layer_action.csv
physical_materialization_checks.json
mutation_checks.json
compute_accounting.csv
diagnosis_summary.json
factual-report-ko.md
```

Full W/M copy를 매 node 영구 저장하는 식으로 비용을 늘리지 않는다. Low-rank journal과 필요한 endpoint를 저장하고 재구성 parity를 실제 검증한다. Journal이 있다는 이유만으로 replay 가능하다고 주장하지 않는다.

최종 보고서에는 다음 질문에 각각 답하라.

1. 원래 B10 exact state를 확보·재현했는가?
2. 실제 weighted Gram/field를 어느 request가 얼마나 지배했는가?
3. Tiny anchor는 numerical capture 차이인가, 실제 보호해야 할 작은 target displacement인가, 미확정인가?
4. N0→NRMS가 같은 state에서 field·physical write·100-request semantic 결과를 어떻게 바꿨는가?
5. Tiny request 자체 및 old900의 손실 대가는 무엇인가?
6. Same-entry Official은 같은 target을 실현할 수 있었는가?
7. 정상 state의 λ·T·N 각각이 Llama/Qwen 품질에 얼마나 영향을 줬는가?
8. 좋은 결과가 감속/under-edit, 더 긴 exposure, numerical refinement 중 무엇으로 설명되는가?
9. 원본 main1000과 새 development 결과를 구분했는가? 비용은 얼마인가?
10. 다음에 검사할 최소 후보와 아직 못한 판단은 무엇인가?

S의 전 후보 결과를 공개하고 Pareto/trade-off를 먼저 보고하라. Scalar 총점으로 숨기지 않는다. 최종 후보를 골라 audit로 넘기기 전 new-edit noninferiority/locality 목표·예산·선택 규칙을 별도로 lock한다. Audit/후속 Server4 결과를 본 뒤 같은 sample을 재사용해 tuning하지 않는다.

이번 완료 조건은 높은 점수가 아니라 **B10 원인 연결과 정상-state hparam 효과를 서로 구분하는 재현 가능한 결과**다. Server4 두 run과 결론을 합치는 작업은 봉인 publication이 나온 뒤 수행하되, 그 대기 때문에 D/S 자체를 멈추지 말라.
