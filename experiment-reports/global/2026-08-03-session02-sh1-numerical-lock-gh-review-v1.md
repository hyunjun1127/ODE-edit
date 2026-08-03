# Session 02 SH1 numerical lock — GH review v1

- 작성: **2026-08-03 KST**
- GH session: `019fb1ea-03cb-7c20-bb3b-eba5f8d6f5f2`
- reviewed SH1 session: `019fc63e-5217-7250-9c22-c5b2ec4248f0`
- reviewed branch/head: `codex/odeeditsh1` / `8275f6537c4ffff6b7bc1acae159184a7aa1c9ef`
- 판정: **`REVISION_REQUIRED; NUMERICAL_LOCK_NOT_APPROVED; GPU_SLURM_HOLD`**
- scientific outcome inspected: **0**
- canonical revision envelope: `ODEEDIT-S02-NUMLOCK-REVISION-V1`
- revised execution spec:
  [`../../plans/global/2026-08-03-session02-compute-aware-main-table-spec.md`](../../plans/global/2026-08-03-session02-compute-aware-main-table-spec.md)

## 결론

SH1의 35개 CPU invariant test와 common-controller skeleton은 유지할 가치가 있다. 그러나 현재
head는 실제 Llama/Qwen에서 P0를 실행할 수 있는 backend가 아니라 `MethodBackend` protocol과
test-only `_ToyBackend`까지만 구현했고, launcher도 의도적으로 dry plan만 출력한다. 따라서
“implementation prep complete”는 **invariant skeleton complete**로만 인정하며 numerical lock은
승인하지 않는다.

가장 큰 계산량 오류는 Native/Scalar의 ordered proposal build가 timer 밖에 있어
`Full/Native` ratio의 분모를 과소계측하는 점, 매 accepted step마다 전체 target weight를 CPU로
복사하는 점, terminal C-energy와 checkpoint/restore가 controller timer 밖에 있는 점이다. 또한
절대 `h_0=h_max=0.25`는 모델과 edit별 C-scale을 무시해 Motivation에서 관측한 under-write를
다시 만들 수 있다.

## Repo/code에서 확인한 사실

1. SH1 보고의 네 commit과 clean head를 확인했다.
2. GH가 동일 head를 detached 상태에서 다시 실행한 focused CPU suite는 `35/35 PASS`였다.
3. CounterFact와 두 모델 hparams의 제안 hash/size, 네 case ID의 존재를 read-only로 확인했다.
4. `runtime.py`에는 `MethodBackend` protocol만 있고 concrete model backend는 없다.
   실제 구현은 `tests/test_runtime_invariants.py`의 `_ToyBackend`뿐이다.
5. `session02_compute_aware_p01.py`는 “never executes a job”이라고 선언하고 `--dry-run`을
   필수로 받는다.
6. Native는 entry event를 측정한 뒤 이미 hit인지 확인하지 않고 direct-z/native proposal과
   write를 수행한다. Existing first-hit test는 adaptive path를 검증하지만 Native regression은
   잡지 못한다.
7. Native/Scalar의 `build_native_terminal`은 `field` 또는 `native_proposal` timer/counter 없이
   직접 호출된다. 반면 dynamic field는 별도 timer를 사용한다.
8. 현재 `N_state_fwd`는 event의 `nfe`와 field forward를 함께 더하고 `N_trial`도 별도 기록한다.
   기존 spec의 “trial과 별도 state forward” 정의와 일치하지 않아 비용식에서 double counting
   위험이 있다.
9. `measure_event`는 old/new target을 두 개의 full forward로 따로 실행한다.
10. `apply_accepted_factors`는 accepted transition마다 모든 target parameter의 full CPU backup을
    만든다.
11. `terminal_net_c_energy`는 entry/current dense difference와 covariance product를 terminal에서
    계산하지만 controller timer에 포함되지 않는다.
12. `events.py`의 현재 event는 inference float measurement이며 concrete grad-enabled smooth event
    생성 경로는 아직 없다.

## Proposal에서 온 내용과의 관계

- 고정 direct-z, current joint state에서의 all-layer synchronous relinearization, allocation
  recomputation, joint partial commit, first-hit terminal이라는 핵심 method 정의는 유지한다.
- Compute 감소는 proposal의 과학 claim이 아니라 사용자가 추가한 co-primary engineering
  objective다. 따라서 좋은 retention만으로 untimed/hidden compute를 허용하지 않는다.
- `D_native`는 terminal distance budget이 아니다. 다만 C-unit이 edit마다 달라 절대 step이
  noise가 되는 문제를 막기 위한 **초기 trust-radius scale diagnostic**으로만 비교한다.

## GH의 추정

- 한 번의 activation backward와 low-rank functional trial은 dense gradient/trial copy보다
  싸질 가능성이 높다.
- Old/new event를 combined batch로 실행하고 unchanged-state reject에서 field를 재사용하면
  fixed multi-hop보다 실제 overhead를 줄일 가능성이 있다.
- 반대로 current accepted-step CPU backup과 dense terminal geometry가 유지되면 GPU kernel보다
  PCIe/host synchronization이 병목일 수 있다.
- Concrete backend가 없으므로 SH1의 `0.5--2 GPUh/model`, `3--10 GPUh/model`, peak
  `42--48 GiB`는 forecast일 뿐 검증된 resource fact가 아니다.

## GH revised lock

다음은 provisional accept한다.

- cases `2022, 12498, 20964, 768`, canonical order/hash, seed `17`
- `tau=0.1`, event/hard worsening tolerance `1e-6`
- `S_max=6`, One-refresh cap `2`, state당 reject cap `4`
- 기존 trust accept/expand/shrink 상수와 QP float64/tolerance
- scalar grid/bisection, hook/functional identity tolerance
- 두 모델 동일 backend/controller/fallback, direct-z once, no post-QP rescale
- `Full/Native <=3x` green, `(3x,4x]` yellow, `>4x` P3 HOLD

다음은 수정 전 reject한다.

- absolute `h_0=h_max=0.25`
- ambiguous `N_state_fwd` 및 Native proposal cost가 빠진 compute schema
- accepted transition마다 full CPU backup
- dry-only launcher를 executable P0 path로 간주하는 것

Trust radius는 entry raw synchronous joint proposal의 unit-C normalization 전 norm
`D_sync_entry`로 정규화한다.

\[
h_0=0.25D_{sync,entry},\qquad h_{max}=0.50D_{sync,entry}.
\]

P0에서 `h_0/D_native`가 두 모델 중 하나라도 `[1/8,1/2]` 밖이면 P1을 HOLD한다. 이 판정은
효과/retention을 보지 않고 geometry만 사용하며 모델별 rescue를 금지한다.

## Main table을 늦추지 않는 실행 순서

1. **Revision gate:** concrete ODE-edit-side backend, executable P0 runner, fair timer/counters,
   entry-only rollback, Native first-hit와 event batching identity를 CPU/dry test로 닫는다.
2. **P0 technical profiler:** 한 case/model을 Llama/Qwen 동시 제출해 numerical identity,
   memory, component time, radius scale만 본다. 과학 우열에는 쓰지 않는다.
3. **P1:** 공통 backend가 lock되면 기존 네 case/네 arm만 실행한다. 이 review 때문에 별도
   scientific arm을 추가하지 않는다.
4. **P2/P3:** 기존 10-edit mechanism table과 100-edit four-arm main progression을 유지한다.

즉, 이번 revision은 실험 수를 늘리는 것이 아니라 잘못된 P0를 한 번 돌리고 다시 구현하는
비용을 막는 최소 gate다.

## Red-team kill/hold gate

- concrete backend 또는 grad-enabled event 없음: P0 HOLD
- Native와 Full의 timer 범위가 다름: compute comparison invalid, P0 HOLD
- per-accepted full CPU backup 또는 dense functional trial copy 존재: P0 HOLD
- first-hit 이후 proposal/write 발생: method contract fail
- exact rollback 실패: execution kill
- combined event가 reference와 tolerance 밖: batching fix 전 HOLD
- `D_sync_entry` degenerate 또는 common radius diagnostic fail: P1 HOLD
- 한 모델 전용 controller/hyperparameter rescue 필요: common-method gate fail
- `Full/Native >4x`: P3 HOLD; retention 결과만으로 override 금지

## Git/protocol/artifact/resource 위험

- EasyEdit는 read-only이며 모든 구현은 ODE-edit-side hook/backend에 둔다.
- precomputed covariance/null-space/Wikipedia artifacts는 hash preflight 뒤 재사용하고 재계산하지
  않는다.
- raw output/log/model artifact는 ignored local path에 두며 Git에는 schema, manifest, summary만
  남긴다.
- server1 tracked active record는 cap `4`를 기록하지만 과거 SH ACK에는 cap `3`이 있었다.
  현재 revision은 GPU cap `0`이라 충돌 영향이 없다. P0 submit envelope 전 SH local boundary가
  current cap과 일치하는지 다시 확인한다.
- Terra Ultra subagent runtime mismatch는 해결되지 않았다. Sol agent로 대체하지 않으며 실제
  scientific result interpretation은 승인된 Terra Ultra review 또는 GH의 별도 boundary 판정 전
  claim HOLD다.
- direct-z temporary session의 artifact, job, report는 이 track에서 읽거나 모니터링하지 않는다.

## 사용자 확인 필요

현재 없음. GPU/Slurm은 HOLD 상태이며, SH1 revision 결과를 받은 뒤 GH가 별도 submit envelope를
발행한다.

## Claim boundary

현재 확인된 것은 reusable invariant skeleton과 CPU oracle뿐이다. ODE-Edit의 model-scale
implementation correctness, compute 절감, MEMIT 대비 retention, 두 모델 공통 작동은 아직
검증되지 않았다.
