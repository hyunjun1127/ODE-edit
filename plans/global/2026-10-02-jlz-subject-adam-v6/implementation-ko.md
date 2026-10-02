# JLZ v6 구현 계약

2026-10-02. [Method 정본](method-ko.md)을 구현하기 위한 파일 책임과 실행 순서다. Production 구현과 GPU qualification은 아직 수행하지 않았다. 새 구현은 `project/run_scripts/jlz_subject_adam/`에 두고 기존 v4/v5 실행 경로를 수정하지 않는다.

## 1 파일 책임과 재사용

| 새 모듈 | 책임 | 재사용 근거 |
|---|---|---|
| `profile.py`, `inputs.py` | model/benchmark native 입력·readout·fact version·eligible set | `jlz_native_joint/inputs.py`, `jlz_pilot/prompts.py` |
| `entry.py` | clean anchors, own-entry KL, raw context keys, 두 종류 cache | `jlz_native_joint/entry.py`, `jlz_writer_coupled/entry.py` |
| `geometry.py` | whole logical batch fixed P, PSD G/E와 factor | `jlz_writer_coupled/geometry.py` |
| `subject.py` | 모든 층 subject δ joint NLL/KL, 선택 current teacher capture | `jlz_native_joint/adapter.py`, `oracle.py` |
| `allocation.py` | A/B root norm 비용과 absolute D gradient | 신규; v4 quadratic policy 대체 |
| `physical_aux.py` | 실제 공유 U, current 정합/distillation/KL, past 보존 | v5 `physical.py` direct VJP; objective는 신규 |
| `optimize.py` | 한 Adam, 고정 native LR, full-batch 누적, pulse, clamp | v4 optimize 구조 재사용, 정책·일정은 신규 |
| `writer.py`, `memory.py` | terminal candidate exact commit, actual key history, native memory | v5 writer/memory의 상태 규약 |
| `observe.py`, `run.py` | 평가의 학습 비개입, receipts, source/profile binding | 기존 collector/evaluator 계약 |

v5 oracle를 절대 D oracle로 그대로 호출하면 안 된다. 그 구현은 `v.detach()*anchor`를 새 leaf로 만들고 반환 gradient에 anchor를 곱한다. v6는 absolute δ leaf를 공유하거나, 반환된 absolute δ gradient를 명시적으로 누적해야 한다. Anchor scaling을 이중 적용하거나 native graph를 detach하지 않는다.

## 2 Batch entry

1. 실제 요청 B와 모든 native token IDs·mask·position·subject lookup·target/readout을 identity로 결속한다. Rewrite와 KL의 문맥·정규화는 baseline 그대로다.
2. Clean W_t, H_t, 실제 current fact 집합을 고정한다. Teacher memory에서 current 재편집 fact를 제외한다.
3. Clean native forward에서 canonical full block anchors, native own-entry KL, 모든 rewrite context의 subject key와 두 cache를 수집한다.
4. Whole current B에 대한 P/G/E를 FP64로 계산하고 batch 동안 고정한다. 모든 context를 padding 포함 위치와 정확히 결속한다.
5. Gram에 큰 B가 부담이면 정확한 blocked/implicit operator를 사용한다. Request·context 일부만으로 P를 만드는 근사를 숨기지 않는다.
6. `σ_l=sqrt(mean_r a_lr²)`를 고정한다. Native 개인 clamp에는 σ가 아닌 a_lr를 쓴다.
7. Seeded current permutation과 four balanced partition을 저장한다. Past S를 batch당 한 번 뽑아 four partition으로 나눈다. Arm 간 선택 ID는 같고 teacher/b는 각 arm의 실제 상태다.

## 3 정확한 optimizer 순서

```text
D[l] = zeros(d_out[l], actual_B), requires_grad=True
Adam(D, betas=(0.9,0.999), eps=1e-8, weight_decay=0)
for candidate k in 1..25:
    zero all D.grad
    update = (k < 25)
    primary: full logical batch native subject forward
       compute request-SUM NLL and current||entry KL
       if update: accumulate native task gradient over all microbatches
       if pulse/terminal: detach required native teacher snapshots before step
    compute request-SUM native norm and B * allocation penalty on whole D
       if update: backward and accumulate in same absolute D.grad
    if k in {5,10,15,20}:
       materialize full-current-B actual weights once for this D
       actual selected current I_j and past J_j; use all native contexts
       compute B * (|I_j|/B * Ccurrent + |J_j|/M * Rpast)
       backward actual auxiliary; add absolute D gradient
    if update:
       assert all accumulated gradients finite
       set common lr = lr_native  # constant from the first update; no warmup
       Adam.step() exactly once
       project each delta[l,r] into c_native * a[l,r] ball
       release this candidate's teacher and materialized weight caches
    else:
       materialize final full-current-B actual weights
       actual no-grad full-current native forward; capture actual input keys
       record final native/actual gaps and actual per-context NLL
       commit exact materialized weight tensors with no later update/re-solve
       append history once; update native memory
```

Teacher capture와 geometry/policy 계산은 별도의 optimizer step을 추가하지 않는다. Candidate25에는 backward·Adam update·후보 선택이 없다. Clamp나 작은 layer norm을 이유로 Adam momentum을 초기화하지 않는다.

위 current/past auxiliary는 method의 평균 및 cohort 가중치를 적용한 뒤 전체를 B배 한다. Task만 B배 하고 past를 누락하면 안 된다. 빈 partition은 해당 항과 forward를 생략하며 division by zero를 만들지 않는다.

## 4 Native와 actual branch의 의미

- Subject branch에서 δ는 지정된 block output의 subject lookup 위치에만 더한다. 동일 요청의 모든 native context가 같은 δ를 공유한다.
- Native NLL hook은 해당 층의 모든 지정 intervention **이후** 값을 읽는다. 현재 Llama profile은 loss layer가 모든 edit layer 뒤에 있지만, generic profile에서 같을 때 pre-injection 값을 읽는 기존 hook 등록 순서 문제를 재사용하지 않는다.
- KL readout은 native lookup이다. `"{} is a"`의 마지막 token으로 임의 이동하지 않는다.
- Actual current subset을 관측할 때도 U는 전체 B의 D/P로 만든다. Subset별 writer solve는 금지한다.
- Current subject teacher는 post-injection block output이다. Current distillation teacher는 native NLL readout의 full vocabulary 확률이다.
- Distillation은 teacher‖actual, native/physical 보존 KL은 current‖entry/admission이다. 각각 이름과 방향을 telemetry에 남긴다.
- Actual current NLL은 terminal audit에 기록하지만 추가 task gradient로 직접 최소화하지 않는다. Past NLL 악화 hinge는 별도 보존 항이다.

## 5 Cache와 수치 의미

Native strict-prefix cache는 subject 이전 causal 상태까지 재사용한다. Actual branch는 첫 modified down_proj 직전까지만 재사용한다. 모든 토큰의 weight write 뒤에 달라지는 upper-layer prefix/KV를 actual branch에 재사용하지 않는다.

입력 identity·model/adapter·dtype·readout·context가 바뀌면 관련 cache를 폐기한다. Commit 후 다음 batch의 P, anchors, own-entry teachers, prefix는 갱신한다. 과거 memory teacher는 cache가 아니라 고정 보존 기준이므로 일반 cache처럼 갱신하지 않는다.

Materialization은 FP64 D/P 곱, FP32 cast, FP32 entry addition이다. Forward는 같은 materialized weight의 linear를 한 번만 수행한다. Backward는 `g_D=Gᵀ(XP)`와 input gradient `g_X=G W_eff`를 유지한다. Direct VJP의 reassociation은 dense gradient와 bitwise 같다고 주장하지 않고 qualification 허용오차로 검증한다.

Geometry G/E는 대칭화 후 FP64 PSD factor로 계산한다. 상대 scale 기준1e-10 이내 음의 고유값은 roundoff로 보고0으로 처리하고 크기·개수를 기록한다. 이를 넘으면 geometry numerical error다. 이 tolerance는 layer 효율 gate가 아니다. Production 실제 SPD residual·gradient tolerance는 qualification receipt에 고정하고 P/N로 바꾸지 않는다.

## 6 Telemetry

후보별로 다음을 기록한다.

- native NLL/KL/norm, separate physical/realization policy 값.
- Arm norm shape, λ·constant lr·warmup_updates=0, 후보 번호·Adam step counter.
- 층/요청 δ norm·relative norm, zero/clamp 비율. 전체값과 subgroup 분포를 모두 유지.
- Native·policy·actual auxiliary gradient norm 및 가능한 범위에서 inner product. Microbatch 전체가 끝난 gradient를 사용하며, raw sum과 mean 단위를 구분.
- Probe ID·current/past partition·teacher identity·teacher candidate·stop-gradient·각 pulse weight.
- Physical hidden consistency, distillation, current native KL, past KL/NLL degradation.
- Branch별 실제 forward/backward 수, processed tokens·layers·head positions, materialization 횟수, 시간·peak memory.

후보25에서는 gradient를0으로 채우지 않고 `gradient_measured=false, gradient=null`로 기록한다. 최종 full actual subject realization, actual ΔW norm, native-vs-actual NLL/KL gap, key shift와 materialized weight hash를 기록한다. Group norm/weight norm 비중을 causal contribution으로 표기하지 않는다.

## 7 Technical qualification과 해석

CPU algebra validation과 실제 모델 qualification을 구분한다. CPU fixture만 통과해 GPU 동등성을 주장하지 않는다.

실제 모델에서 필요한 검증은 clean zero route, 고정 nonzero 후보의 native reference parity, auxiliary direct VJP와 dense gradient, finite-difference tiny fixture, microbatch 분할 불변성, post-injection readout, teacher detach, G/E off-diagonal, no future/evaluation input, final exact commit/history 정합이다. Layer 사용 비율·RS/PS/NS·realization 크기로 technical pass/fail을 정하지 않는다.

Time profile은 native joint fit과 같은 모델·precision·B·입력·장비에서 측정한다. Extra40%는 목표일 뿐 pass를 맞추려고 context를 삭제하거나 평가를 축소할 권한이 아니다. 비용 초과는 보고하고 별도 profile revision으로 다룬다.

이번 문서는 method와 구현 명세다. 첫 두 arm의 BS100×5(각 500 edits), model/benchmark별 concrete stream, 장비 배치, GPU job 제출은 별도 실험 설계·dispatch의 책임이다.
