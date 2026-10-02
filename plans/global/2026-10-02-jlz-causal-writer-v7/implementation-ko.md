# JLZ v7 구현과 수치 검증 계약

2026-10-02. [Method 정본](method-ko.md)을 실제 코드로 옮기는 계약이다. 새 namespace는 `project/run_scripts/jlz_causal_writer/`다. v6의 고정 P 경로와 별도 revision으로 식별한다. 이 문서는 production 구현 또는 GPU qualification 완료 증거가 아니다.

## 1 모듈 책임과 기존 코드의 한계

| 신규 모듈 | 책임 |
|---|---|
| `profile.py`, `inputs.py`, `entry.py` | native 입력/lookup/readout, clean anchors/teachers, A factor와 첫 write site cache |
| `subject.py` | frozen-entry native subject δ 공동 loss 및 같은 후보 teacher capture |
| `dynamic_solve.py` | full-context current key로 differentiable solve, dual/primal reference, implicit VJP |
| `causal_builder.py` | whole-current RW layer-major barrier, candidate별 실제 K/P/U와 중간 상태 |
| `physical_linear.py` | 동일 FP32 materialization, input/D/P 세 방향 VJP |
| `allocation.py` | dynamic G/E 및 A/B aggregation, 전체-B gradient |
| `physical_aux.py` | 기존 selected current/past native loss, builder를 통한 모든 D로의 역전파 |
| `optimize.py` | native+policy+aux gradient 누적, 단일 Adam, 사후 clamp |
| `writer.py`, `memory.py`, `observe.py`, `run.py` | exact terminal commit/history, native memory, 평가·실행 receipts |

`jlz_writer_coupled/geometry.py::solve_basis`는 `no_grad`, `keys.detach`, 호출마다 A factorization을 하므로 그대로 반복 호출하지 않는다. `physical.py::PhysicalLinear.backward`는 g_P=None이므로 그대로 재사용하지 않는다. `entry.py`의 clean key detach/CPU 저장은 dynamic key capture 구현이 아니다. 기존 module hook을 무비판적으로 연결하는 대신 각 경계의 reference를 먼저 정의한다.

## 2 Candidate 구성 순서

```text
entry = bind W_t, H_t, native tokens, anchors, own-entry teachers, memory sample
A_factor[l] = factor(lambda_C * C0[l] + H_t[l]) once per logical batch
K_first, P_first = compute exact first-site basis once
D[l] = zeros(d_out[l], actual_B), requires_grad
optimizer = Adam(D, lr=native_lr, betas=(.9,.999), eps=1e-8, weight_decay=0)
for k in 1..25:
    assert no active δ hook on actual path
    zero all D.grad
    update = k < 25
    native = full-logical-B subject-injected frozen-entry forward
    if update: accumulate request-SUM native gradient (teacher snapshots detached)
    builder = candidate-local immutable graph from the SAME pre-update D
    stage_states = first modified down_proj input/residual caches for all RW rows
    for l in ordered eligible write sites:
        gather graph-connected subject K_l across ALL current RW microbatches
        P_l = P_first if first site else differentiable_full_context_solve(K_l,A_factor[l])
        G_l, E_l = current differentiable geometry; never detach
        W_eff[l] = entry32[l] + cast32(D64[l] @ transpose(P64[l]))
        if a later write site exists:
            apply own W_eff to all tokens and execute intervening frozen operations
            advance ALL stage_states to the next site's pre-write input
    policy = actual_B * allocation_loss(D, current_G_E, arm)
    aux = 0
    if k in [5,10,15,20]:
        aux = actual_B * (len(I_k)/B * C_current(I_k)
                       + len(J_k)/M_p * R_past(J_k))
        # Every aux actual model uses complete current-B causal W_eff;
        # D→lowerW→upperK→upperP→aux remains connected.
    if update:
        backward policy + aux (all chunks/candidate checkpoint work complete)
        Adam.step exactly once at constant native LR
        per-layer/request native post-update clamp; keep moments
        release all candidate K/P/U/activation/teacher caches
    else:
        actual full-current no-grad forward with candidate25 W_eff
        compare builder keys with final-model pre-own-write keys
        audit native/actual gaps and record actual RW context NLL
        commit the identical materialized FP32 tensors
        update history once and native memory from this actual candidate
```

Native norm을 primary gradient에 포함했다면 policy 쪽에서 중복 가산하지 않는다.Primary native graph의 backward를 먼저 마쳐도 같은 D leaf의 값은 update까지 고정한다.Teacher capture에는 추가 forward를 요구하지 않는다.Candidate25의 native/actual forward에는 추가 update가 없다.

Fit 중 master 모델 가중치는 W_t로 유지한다. Actual candidate는 functional weight 또는 예외 시에도 복구되는 임시 적용 범위에서만 실행한다. 다음 후보의 native 경로가 직전 actual candidate 가중치를 읽지 않도록 entry weight 및 hook 복구를 검사한다. History와 memory의 실제 변경은 terminal commit 이후 한 번만 수행한다.

## 3 Layer barrier와 microbatch

각 층에서는 전체 B 요청의 모든 native RW context key가 모일 때까지 solve하지 않는다. Partial batch의 실제 B, 가변 context 수, zero-weight context, 요청별 소유 순서를 처리한다. K는 graph-connected `cat` 또는 동등한 reduction으로 모은다. `detach().cpu()`로 저장한 뒤 새 leaf로 만드는 것만으로는 기존 의존성이 유지되지 않는다.

Context 수가 달라도 native task mean과 writer α를 보존한다. 전체 B의 policy scalar를 한 번 계산하며, 이를 microbatch별 root의 합으로 바꾸지 않는다. 다른 요청의 D가 shared weight를 통해 key를 바꾸는 경로도 유지한다. 층마다 처음부터 full prefix를 다시 실행하거나 microbatch별로 solve하는 방식은 reference가 아니다.

Writer는 current RW key만으로 구성한다. KL 입력과 past 입력은 actual loss의 관측 대상이며, current writer의 fitting key 열에 섞지 않는다. History H_t와 C0는 이 batch 동안 상수이며, key를 갱신할 때마다 추가하지 않는다.

## 4 미분 가능한 solve

Method의 whitened dual을 reference 식으로 구현한다. A=LLᵀ는 고정하고, K→F→S/V/T/P는 후보별 graph로 구성한다. 동등한 primal blocked solve도 모든 context와 gradient를 보존하는 경우 사용할 수 있다. A의 SPD 검사가 실패했다고 임의로 jitter를 추가하거나, context/low-rank 근사 또는 Ω 재정규화를 수행하지 않는다.

Implicit VJP는 Q→Y=M⁻ᵀQ→g_K를 사용하며, policy E의 직접적인 K 의존성과 activation에서 유래한 g_K도 합산한다. G=TᵀT를 사용하면 T의 graph도 유지한다. G/E 값을 갱신하더라도 factor/eigenvector/P/K를 detach하면 v7이 아니다.

Quadratic은 FP64에서 q=tr(DᵀD G/E)로 계산한다. E에는 동등한 비음수 표현인 `||D(PᵀK-Z)sqrtΩ||²`도 사용할 수 있다. 원점에는 명시적으로 zero-norm subgradient 0을 적용하여 `sqrt(0)`에서 NaN이 전파되지 않게 한다. Trace 음수의 크기가 `1e-12 * max(1, sum(abs((DᵀD)*Qᵀ)))` 이하이면 roundoff로 보고 0으로 clip하며 횟수와 크기를 기록한다. 이를 초과하는 음수는 수치 불일치로 처리하고 reference와 대조한다. Clip은 ε 평활화도 품질 gate도 아니다.

## 5 Direct D/P VJP와 graph 분할

입력 X와 출력 adjoint Gout에 대해 g_D=Goutᵀ(XP), g_P=Xᵀ(Gout D), g_X=Gout W_eff를 반환한다. D/P 방향의 reassociation은 FP64로 수행하고, D gradient는 leaf dtype으로 되돌린다. W_eff의 forward는 FP64 product→FP32cast→FP32entry addition 순서를 보존한다. Custom VJP로 D/P 경로를 명시한다면, 별도의 W argument 경로에서 같은 gradient를 중복 가산하지 않는다.

메모리를 절약하려고 actual 관측 chunk에서 D/P를 temporary leaf로 만드는 경우, 각 chunk의 D/P adjoint를 모아 **원래 causal builder outputs**으로 역전파해야 한다. Policy에 명시적인 K 의존성이 있으면 K adjoint도 모은다. G를 original T에서 계산했는데 g_P만 수집하고 T graph를 버려서는 안 된다. Leaf P에서 `PᵀAP`로 G를 재구성하거나, T adjoint도 원래 graph로 돌려보내거나, reference의 전체 graph를 유지한다. 어느 방식이든 같은 후보에서 전체 graph autograd와 비교한다.

## 6 Cache와 checkpoint

Native와 actual의 cache 경계는 다르다. Native subject 이전의 strict causal-prefix는 native 용도로만 재사용할 수 있다. Actual에서는 첫 modified down_proj 이전까지만 entry cache로 사용한다. 후보에 의존하는 상층 K/P/hidden/KV를 다음 후보에 재사용하지 않는다.

Checkpoint closure는 immutable candidate D/P/W_eff를 인자로 받는 pure function으로 실행하거나, closure 안에서 같은 weight를 설치하고 복원한다. 종료된 `install()` context에 의존하여 backward 재실행 때 baseline weight가 사용되는 구현은 금지한다. Adam은 모든 재계산과 backward가 끝난 뒤 실행한다.

Key builder의 subject-prefix truncation은 causal adapter에만 허용한다. 원래 token IDs·mask·RoPE/position·lookup을 보존하면서 각 context를 해당 subject 위치까지 처리한다. 모든 context를 유지한다. Full-row reference와 K 및 total D gradient가 일치하는지 검증한다. Native primary와 actual full-prompt loss에는 원래 입력을 사용한다. Builder에 남아 있는 subject 이전 hidden만으로 NLL readout을 대체하지 않는다.

A factor는 CPU 또는 GPU에 보관할 수 있으며 layer별 streaming도 가능하다. Device 이동과 cast를 통과하는 gradient를 보존한다. Memmap/host 저장을 사용하면 candidate state와 autograd adjoint의 소유 관계를 기록한다. GPU 용량에 맞추기 위해 암묵적으로 stop-gradient mode로 전환하지 않는다.

## 7 기술 검증과 초기 허용오차

CPU 합성 대수 검증과 실제 모델 GPU qualification을 구분한다. 아래는 사전에 고정하는 최초 비교 기준이며, RS/PS/NS를 보고 완화하지 않는다.

| 비교 | 초기 기준 |
|---|---|
| FP64 dense/dual·trace identity·implicit VJP 대수 비교 | tiny fixture에서 atol1e-9, rtol1e-8 |
| FP64 central finite difference total-D/solve/linear VJP | fixture에서 atol2e-7, rtol5e-5 |
| Production SPD residual | ‖MP−Kbar‖F / max(‖Kbar‖F,1e-12) ≤1e-8 |
| FP32 native/actual 동일 후보 loss parity | request-mean loss 절대차≤5e-5 |
| FP32 staged/reference key parity | componentwise atol1e-4, rtol1e-5 |
| FP32 total D gradient parity | RMS(g−g_ref) ≤1e-6 +2e-3 RMS(g_ref) |
| Commit 일치 | actual terminal에서 평가한 materialized tensor와 commit tensor의 hash 일치 |

Dense reference와 optimized path는 같은 D·입력·dtype·teacher·model state를 사용한다. Adam trajectory의 bitwise 일치를 assert해서는 안 되지만, 짧은 실행에서 candidate/step 수, 유한값, 동일한 loss 정의는 확인한다. 기술적 불일치는 reference로의 동등한 fallback 또는 수정으로 해결하며, 후보의 품질을 선별하지 않는다.

필요한 fixture는 비영 하층 D에 따른 upper K/P 변화, P-gradient를 의도적으로 차단한 경로와의 total-D gradient 차이, all-B barrier/microbatch 분할·request permutation, zero D, 불균일 context/B1/partial batch, native post-intervention readout, checkpoint on/off, causal-prefix/fullrow, terminal staged/final key 일치, history/memory once-only다. Stop-P 비교는 검증용 negative control이며 본 실험의 세 번째 arm이 아니다.

## 8 기록과 구현 완료 판정

각 후보에서 native NLL/KL/norm, dynamic G/E policy, pulse별 actual loss, 층별 δ/U/clamp 비율, K_l^a−K_l^entry와 P_l(D)−P_l(entry)의 상대 norm, fullgraph와 qualification에서 P 경로를 차단한 경우의 차이를 구분하여 기록한다. Key/P 변화량은 품질 gate로 사용하지 않는다.

필요한 dependency receipt는 `key_source=actual_lower_writer`, `P_candidate=k`, `K_candidate=k`, `all_current_columns=true`, `grad_P_enabled=true`, `grad_K_solve_enabled=true`, `direct_K_policy_gradient=true`, `builder_delta_hook=false`다. 단순한 flag assert에 더해 수치 gradient 검증을 첨부한다.

Cost는 native forward/backward, builder layer-token forward/backward, checkpoint replay, A factorization, dynamic context solve, solve adjoint, materialization, aux suffix/head, transfer, official evaluation, peak VRAM/RSS로 나누어 기록한다. Logical candidate 수와 실제 forward/backward 호출 수를 혼동하지 않는다.

Production 구현 완료에는 새 source SHA, 위 수치 검증, 실제 small pilot, bounded B100의 시간·memory receipt가 필요하다. 문서와 CPU toy만으로 GPU 구현 PASS나 속도 향상을 선언하지 않는다.
