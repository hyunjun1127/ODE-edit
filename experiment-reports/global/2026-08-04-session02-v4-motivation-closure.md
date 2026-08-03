# Session 02 V4 Motivation closure 및 Method-stage handoff

- 최종 갱신: **2026-08-04 03:45 KST**
- 방법명: **ODE-Edit**
- 모델: `llama3-8b-inst`, `qwen2.5-7b-inst`
- model dtype: checkpoint-original `torch.bfloat16`
- 최종 SH1 source checkpoint: `7b1241d4de67abf9c5ef07161ca0358d4064464c`
- V3 proposal ID: `bc8fc256737d4e82b83f1937633ca94709d59e51969612611e20981aa70c4c23`
- V3 lock SHA-256: `cd4ffeb2310e1d261cea0457cd683293e53cf92b136b6d6eb01cd5c4af579882`
- V4 proposal ID: `ca18f3c27c1e7a347eac4a569d94b43373641c05fd1fa00f44b9875595c5d83c`
- V4 lock SHA-256: `b2834fdfe394b10984618f3af0791b6d1c442b13548debeb75c3b24d007ec5f0`
- 최종 motivation gate: **`MOTIVATION_CLOSED_POSITIVE`**
- 다음 단계: **`METHOD_MAIN_TABLE_PILOT_READY`**
- evaluation/generation count: `0`
- scientific superiority claim: **아직 없음**

## 0. Executive decision

Motivation 단계는 **closed-positive**로 판정한다.

이 판정은 ODE-Edit이 MEMIT 또는 AlphaEdit보다 이미 우수하다는 뜻이 아니다. 다음의 더
제한된 결론을 뜻한다.

1. 기존 worst-context relative event가 실제 positive-mean candidate를 rollback시키는
   under-commit을 만들었다는 직접 증거가 있다.
2. 이를 uniform-mean relative floor와 direct-z-oracle absolute-new floor로 교체한 V3는
   약한 relative-only endpoint와 한 context의 veto를 동시에 제거했다.
3. V3 뒤 남은 Llama 실패는 event가 아니라 macro-step과 unchanged-state retry의 controller
   문제로 분리됐다.
4. 동일한 V4 controller를 두 모델에 적용하자 Llama `2/4`, Qwen `4/4`, pooled `6/8`
   first-hit을 얻었다. 성공한 여섯 endpoint의 diagnostic C-energy는 모두 해당 validated
   Native reference보다 작았다.
5. V4 Full/Native aggregate GPU ratio는 Llama `2.0789x`, Qwen `0.9233x`이고, 최대
   per-case ratio도 `3.3860x`로 predeclared `4x` stop line 아래다.

따라서 “state-dependent joint field를 반복 재선형화하면서 더 낮은 capacity endpoint를
찾을 수 있는가”라는 연구 방향은 양 모델에서 다음 method-stage pilot을 열 만큼 살아남았다.
다만 efficacy, generalization, locality, retention과 fresh-panel generalization은 이번
실험에서 측정하지 않았으므로 main-table superiority는 열지 않는다.

## 1. 네 범주의 분리

### 1.1 Proposal에서 온 내용

- 한 edit의 direct-z는 고정하지만 finite multi-layer write는 다음 residual, key,
  downstream activation, layer proposal과 marginal capacity cost를 바꾼다.
- MEMIT은 ordered one-sweep이고, ODE-Edit은 joint partial update 뒤 모든 edit layer의
  proposal과 allocation을 current joint state에서 다시 만든다.
- ODE가 필요한 이유는 write를 천천히 적용하기 위해서가 아니라 state-dependent edit
  vector field를 반복 적분하기 위해서다.
- Proposal은 검증된 claim이나 확정 paper plan이 아니다.

### 1.2 Repo/protocol에서 확인한 사실

- EasyEdit source는 수정하지 않았고 ODE-edit-side hook/runtime만 사용했다.
- 두 모델 모두 original checkpoint BF16, 고정 revision/context/case order/seed와 동일한
  공통 event/controller를 사용했다. Model-specific rescue와 fallback은 `0`이다.
- direct-z는 실제 current edit state에서 edit당 정확히 한 번 계산됐다.
- covariance, Wikipedia와 null-space 계열 artifact는 기존 precomputed 파일을 read-only로
  사용했고 재계산하지 않았다.
- raw artifact와 logs는 ignored `local/`에만 있다. Git에는 code, lock, test와 이 compact
  report만 남긴다.
- server1 GPU cap은 `3`이며 paired jobs는 aggregate `2/3`을 사용했다.
- server2는 onboarding HOLD이므로 `BROADCAST_EXCEPTION_SERVER2_NOT_READY`로 rsync하지
  않았다.

### 1.3 GH 해석

- V4의 양 모델 positive signal과 compute 감소는 method-stage pilot을 정당화한다.
- Llama에서 V3 `0/4`가 V4 `2/4`로 바뀐 것은 event threshold 완화가 아니라 macro-step과
  retry semantics 교정의 결과이므로, 사용자가 지적한 “너무 작은 write가 noise가 될 수
  있다”는 가설과 방향이 맞는다.
- Qwen `4/4`와 aggregate compute `<1x`는 특히 유망하지만 네 case의 development panel
  결과라 expected benchmark gain으로 외삽할 수 없다.
- Order position `1+`에서는 arm별 prior edit history가 달라진다. 해당 C-energy/Native
  ratio는 useful diagnostic이지 causal same-entry 우위가 아니다.

### 1.4 사용자 확인 필요

- 없음: 현재 사용자 지시는 동일 공통 method로 motivation을 닫고 main table을 빠르게
  여는 것이며, V4는 그 최소 gate를 충족했다.
- 미래 model-specific calibration, lifelong scale, AlphaEdit-ODE 확장은 별도 stage에서만
  연다.

## 2. Event redesign

### 2.1 기존 문제

기존 event는 약한 relative threshold와 과도한 worst-context aggregation을 함께 썼다.

\[
\min_c \{\ell_{new,c}-\ell_{old,c}\}\ge 0
\]

이 정의는 두 방향의 오류를 동시에 만들 수 있다.

- 평균적으로 target이 좋아져도 한 context가 음수면 전체 edit을 rollback한다.
- 모든 margin이 간신히 양수이기만 하면 absolute target likelihood가 충분히 올라오지 않아도
  성공할 수 있다.

실제 legacy Llama Full case `12498`에서는 마지막 trial의 mean margin이
`+0.6979167`인데 min margin이 `-0.25`여서 rollback됐다. 이는 old all-context hard gate의
under-commit을 직접 입증한다.

반면 retained legacy successful Native rows에서는 absolute-new under-realization이 직접
입증되지 않았다. 따라서 “기존 성공 endpoint가 약했다”는 문장은 claim하지 않는다.

### 2.2 V3 primary event

Context는 동일한 uniform weight를 쓴다.

\[
M(W)=\frac1C\sum_c\left(\ell_{new,c}(W)-\ell_{old,c}(W)\right)
\]

Direct-z oracle은 Native endpoint 복제 목표가 아니라 absolute-new floor를 정하는
calibration으로만 사용한다.

\[
L_{req}=L_{new}(W_0)+\rho\left(L_{new,z^\star}-L_{new}(W_0)\right),
\qquad \rho=0.5
\]

V3 first-hit은 다음 두 조건을 모두 만족한다.

\[
M(W)\ge0,\qquad L_{new}(W)\ge L_{req}
\]

Decision smooth objective는 두 deficit의 smooth maximum이다. 다음은 decision에서 제거했다.

- per-context minimum 또는 worst-context veto
- oracle margin fraction `q_margin`
- Native NLL equality
- model-specific threshold와 fallback

Calibration accounting은 모든 terminal row에서 total model forwards `4`, 그중 oracle extra
forwards `2`, oracle backward `0`으로 검증됐다. Ordinary event는 두 separate
teacher-forced forwards를 쓴다.

## 3. V3가 분리한 controller 병목

V3 P1 결과는 Llama Full `0/4`, Qwen Full `2/4`였다. Qwen의 두 hit은 V3 정의가 실제
작동함을 보였지만 Llama 실패를 해결하지 못했다.

Read-only trajectory RCA에서 다음이 확인됐다.

- Llama case `12498`은 세 번째 accepted step 뒤 `M=-0.3854167`, `q_new=.61398`까지
  도달했다.
- coefficient L2는 `.002941`에서 `.000257`, `.000221`로 급감했다.
- 다음 nonpositive trial 뒤 radius를 줄였지만 QP의 trust constraint가 non-binding이라
  동일 coefficient tuple을 네 번 재생했다.
- Llama `2022/20964/768`은 여섯 accepted round 동안 progress가 단조 증가했으나 작은
  macro-step 때문에 cap에 도달했다.
- Qwen near misses도 `12498: M=-.03125,q_new=.77372`,
  `20964: M=.33333,q_new=.49373`로 두 constraint의 경계에 있었다.

따라서 V3 뒤의 병목은 event를 더 약하게 만들 문제가 아니라 finite-BF16 macro-step과
retry가 실제 candidate를 바꾸도록 만드는 문제였다.

## 4. V4 common controller

V4는 V3 event를 그대로 동결하고 다음만 공통 변경했다.

| 항목 | V3 | V4 |
|---|---:|---:|
| `h0_fraction * D_sync_entry` | `.25` | `.50` |
| `h_max_fraction * D_sync_entry` | `.50` | `1.00` |
| `kappa` | `1.0` | `1.25` |
| `S_max` | `6` | `6` |
| `beta` | `.8` | `.8` |

P0에서 관찰된 native-relative scale로 환산하면 common formula의 initial radius는 대략
Llama `.23365 D_native`, Qwen `.33860 D_native`다. 모델별 상수는 사용하지 않았다.

Unchanged-state rejection retry는 radius contraction뿐 아니라 requested progress 전체를
명시적으로 줄인다.

\[
r_s=\gamma_{down}^{s},\qquad
p_s=r_s\min\{\kappa d,\,\beta h_s\lVert g\rVert_2\}
\]

동일 state와 cached field를 재사용하되 requested progress 또는 coefficient tuple이 이전
reject와 같으면 fail-close한다. 추가 field build나 backward는 없다.

Llama case `768`에서 실제 retry는 다음처럼 달라졌다.

| retry scale | requested progress | coefficient L2 |
|---:|---:|---:|
| `1.0` | `.03682168` | `6.82648e-5` |
| `.5` | `.01841084` | `3.41324e-5` |
| `.25` | `.00920542` | `1.70662e-5` |
| `.125` | `.00460271` | `8.53311e-6` |

Field rebuild는 `0`이었다. 이는 V3에서 동일 candidate를 네 번 반복한 구현 결함을 닫는다.

## 5. Executed evidence

### 5.1 V3 technical calibration과 P1

| stage/model | Slurm job | 상태 | elapsed | terminal manifest SHA-256 |
|---|---:|---|---:|---|
| V3 P0 Llama | `16287` | `COMPLETED 0:0` | `00:05:32` | `c0ac3f192c05d2badeb1866cc585762ad7aca0e7ebbd8836dc4da159043f1b84` |
| V3 P0 Qwen | `16288` | `COMPLETED 0:0` | `00:07:32` | `32a0b67e347d4fde7fe1666c8c85e926817c3fd760e76067035e58e82ad1a37f` |
| V3 P1 Llama | `16289` | `COMPLETED 0:0` | `00:17:46` | `a02568fe4ea4c41cdb6a1cadb2f56a775ccca891dfdbbd17c23613f3db4f3a75` |
| V3 P1 Qwen | `16290` | `COMPLETED 0:0` | `00:23:33` | `29fae415b6f285083ebe3a82750ffa538865f611e7e1c2e1e1b2d3e9f0ee3269` |

V3 P0는 두 모델 모두 original BF16, context, oracle, hook/reference, functional/commit,
rollback과 terminal hash gate를 통과했다. V3 P1은 모델별 `12/12` controller/compute/
mechanism/event rows, direct-z `12`, evaluation `0`이다.

### 5.2 V4 terminal artifacts

| 모델 | Slurm job | 상태 | elapsed | terminal manifest SHA-256 |
|---|---:|---|---:|---|
| Llama | `16295` | `COMPLETED 0:0` | `00:07:45` | `3c5cec556da0f0d404bcfd1ec86e1b1d26e509278588c39aacfe698bde371fa7` |
| Qwen | `16296` | `COMPLETED 0:0` | `00:07:14` | `3de2dd6c5e30f75da2422312b9fcf8f725fbaeddf00cdf094129b328ae623c17` |

Local roots:

- `local/results/session02-v4-macrostep-retry-p1-r1-llama3-8b-inst-ca18f3c2`
- `local/results/session02-v4-macrostep-retry-p1-r1-qwen2.5-7b-inst-ca18f3c2`

각 모델은 Full-only 4-edit sequential stream, controller/compute/event/mechanism `4/4`,
direct-z `4`, evaluation `0`이다. Terminal listed-file hash/size, target guard, T/C,
hook/reference, request/target-weight rollback, Omega chain과 arm isolation은 모두 PASS다.

첫 V4 제출 `16293/16294`는 첫 controller row 전 calibration metadata attribute 오타로
fail-close했다. Method outcome은 `0`이었고 raw partial root는 보존했다. R1은 real dataclass
shape를 재현하는 focused regression 뒤 동일 controller/event/lock hash로 실행됐다. 이
중간 실패에 대한 별도 tracked report나 commit은 만들지 않았다.

## 6. V4 결과

### 6.1 Strict trajectory table

| 모델 | case | status | K / rejects | final 또는 last-trial `(M, q_new)` | C-energy / Native | GPU / Native |
|---|---:|---|---:|---|---:|---:|
| Llama | `2022` | cap unresolved | `6 / 0` | `(-.447917, .594450)` last trial | n/a | `3.385958` |
| Llama | `12498` | **hit** | `2 / 0` | `(1.208333, .737614)` | `.262940` | `1.231473` |
| Llama | `20964` | **hit** | `2 / 0` | `(2.937500, .974151)` | `.295733` | `1.276772` |
| Llama | `768` | trust rejection | `3 / 4` | `(.750000, .492412)` last trial | n/a | `2.443273` |
| Qwen | `2022` | **hit** | `2 / 0` | `(2.625000, .623597)` | `.453370` | `1.377454` |
| Qwen | `12498` | **hit** | `1 / 0` | `(.927083, .828972)` | `.081352` | `.739195` |
| Qwen | `20964` | **hit** | `1 / 0` | `(1.572917, .603540)` | `.046489` | `.792993` |
| Qwen | `768` | **hit** | `1 / 0` | `(4.739583, .677424)` | `.086000` | `.790708` |

Summary:

- Llama Full: `2/4` hit
- Qwen Full: `4/4` hit
- pooled: `6/8` hit
- 성공한 여섯 endpoint의 diagnostic C-energy/Native: 모두 `<1`
- aggregate Full/Native GPU: Llama `2.078865x`, Qwen `.923277x`
- max per-case GPU ratio: `3.385958x`, `4x` stop line 미만
- peak allocated: Llama `18.586 GiB`, Qwen `20.190 GiB`

Llama의 두 실패는 서로 다른 constraint에 걸렸다. Case `2022`는 `q_new>.5`이지만 mean
margin이 음수였고, case `768`은 mean margin이 양수지만 `q_new=.492412`로 absolute floor를
근소하게 못 넘었다. 이 사실 때문에 threshold를 post-hoc 변경하지 않았다.

### 6.2 Capacity 해석 경계

성공한 여섯 trajectory의 C-energy ratio는 모두 positive directional evidence다. 그러나
sequential arm은 성공/실패 history가 달라질 수 있으므로 order position `1+` ratio는
causal same-entry 비교가 아니다.

가장 강한 matched-entry evidence는 order position `0`, Qwen case `2022`다. Native와
V4 Full이 같은 baseline state에서 시작해 모두 hit했고, Full C-energy는 Native의
`.453370`이다. Llama의 두 성공은 method가 해당 모델에서도 작동함을 보이지만 Native보다
causally 낮은 endpoint를 확정하는 evidence로 과장하지 않는다.

### 6.3 Compute 해석

V3 P1 Full aggregate는 Llama `3.5182x`, Qwen `2.1664x`였다. V4는 각각 `2.0789x`,
`.9233x`다. 이 비교는 동일 V3 event와 validated Native timing을 사용한 operational
comparison이며 model load는 제외했다.

V4가 빨라진 핵심은 round 수다.

- Llama successful cases: `K=2`
- Qwen successful cases: `K=1,1,1,2`
- V3의 작은 late-step과 duplicate retry를 제거했다.

Fresh-panel end-to-end throughput은 아직 검증하지 않았으므로 deployable speedup으로
claim하지 않는다.

## 7. Gate 판정

| gate | 판정 | 근거 |
|---|---|---|
| original-BF16/common-method technical | PASS | 두 모델 terminal integrity, model branch `0` |
| event redesign | PASS | mean aggregation + absolute-new floor, legacy under-commit direct evidence |
| vector-field nontriviality | PASS | prior mechanism diagnostic에서 direction-ID transition mean `1.0` |
| cross-model directional | PASS | Llama `2/4`, Qwen `4/4`, pooled `6/8` |
| capacity direction | PASS with caveat | 6 hit ratios `<1`; matched-entry strongest evidence는 Qwen `2022` |
| compute stop line | PASS | max `3.386x < 4x` |
| compute-positive motivation | PASS | aggregate Llama `2.079x`, Qwen `.923x` |
| eff/gen/loc/retention superiority | OPEN | evaluation `0` |
| deployable/lifelong claim | CLOSED | fresh scale evidence 없음 |

최종 판정은 **`MOTIVATION_CLOSED_POSITIVE`**다. 이전 보고서의
`COMMON_CONTROLLER_REDESIGN_REQUIRED`는 V4로 해소됐다. 이 판정은 method section과 sealed
small main-table pilot을 열지만 paper-level claim을 열지는 않는다.

## 8. Main-table pilot 방향

다음 실행은 development 4-case를 재사용한 tuning이 아니라, 실행 전에 고정한 fresh small
panel이어야 한다.

### 8.1 Main arms

- EasyEdit Native MEMIT
- ODE-Edit V4
- AlphaEdit (EasyEdit precomputed null space/stat을 read-only 재사용)

Static synchronous는 mechanism ablation으로 별도 열에 두되 primary baseline으로 부르지
않는다. 모든 모델은 Llama3-8B-Instruct와 Qwen2.5-7B-Instruct original BF16을 쓴다.

### 8.2 필수 metric

- efficacy, generalization, locality
- sequential prior-edit retention
- terminal C-distance/C-energy
- wall/GPU sec per edit와 peak memory
- direct-z/model forward/field/backward/trial/reject/write counters
- success-conditioned와 all-request aggregate를 함께 보고해 rollback survivor bias를 막는다.

### 8.3 Pilot gate

- 동일 공통 V4 controller, model-specific rescue/search/fallback `0`
- 두 모델 모두 Native 대비 rewrite metric이 non-inferior한 방향
- pooled capacity가 낮고 locality 또는 prior-retention이 악화되지 않는 signal
- Full/Native aggregate GPU `<=3x` 목표, 어느 모델도 `>4x`면 HOLD
- direct-z once/edit, original BF16, rollback, first-hit freeze, EasyEdit read-only 유지

이 pilot을 통과하기 전에는 100-edit/lifelong scale, AlphaEdit-ODE extension 또는 큰 main
table로 확장하지 않는다.

## 9. Claim boundary

현재 허용되는 문장은 다음과 같다.

> ODE-Edit은 고정 direct-z 아래에서 joint state에 따라 모든 layer proposal을 반복
> 재선형화한다. Uniform-mean/absolute-new first-hit event와 finite-BF16-aware common
> macro-step controller를 사용한 4-edit development diagnostic에서 Llama와 Qwen 모두
> first-hit signal을 보였고, Qwen matched-entry case 하나에서는 Native보다 낮은
> C-energy endpoint를 찾았다. 이 결과는 method-stage evaluation을 정당화하지만 성능
> 우위를 확정하지 않는다.

다음은 주장하지 않는다.

- MEMIT/AlphaEdit 대비 efficacy, generalization, locality 또는 retention 우위
- 모든 edit 또는 두 모델 평균의 causal capacity 우위
- benchmark-level compute speedup
- sequential/lifelong collapse 방지
- model-specific hparam 필요성

## 10. Git/protocol/artifact boundary

- SH1의 intermediate checkpoint를 개별 push하지 않고 최종 code diff를 한 commit으로
  squash한다.
- 첫 V4 plumbing failure에 대한 별도 tracked code/report commit은 만들지 않는다.
- Git에는 source, tests, numerical lock과 이 report만 포함한다.
- raw logs, direct-z tensors, event traces와 manifests는 ignored `local/`에 보존한다.
- credentials, raw IP/username/port/key/token/private secret은 Git에 기록하지 않는다.
- server2 onboarding 완료 전 artifact rsync는 하지 않는다.
