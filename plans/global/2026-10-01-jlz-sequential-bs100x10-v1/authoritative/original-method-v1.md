# JLZ: MEMIT-H 위의 공동 local-z — method·pipeline 명세 v1

작성 2026-10-01. 상태는 설계 명세다. GPU 실행, job 제출, dispatch는 없다.
실험 구성(arm, 규모, 판정)은 이 문서의 범위가 아니며, 결과를 보고 정한다.
참조 구현은 [`project/run_scripts/jlz_ref/`](../../project/run_scripts/jlz_ref/)이다. NumPy와 toy 모델로 쓴 실행 가능한 명세이며, CPU 검사 9개가 통과했다.

## 0. 요약

- **목표.** 층마다 local-z를 직접, 그리고 모든 층을 동시에 최적화한다.
  각 층의 z는 원 compute_z와 같은 손실로 정하고, 실현은 MEMIT-H의 solve를 그대로 쓴다.
- **핵심.** fit 중 forward가 MEMIT-H가 실제로 commit할 update $\Delta_l=R_l\,\mathrm{adj}_l^\top$를 이미 적용한 모델에서 손실을 잰다.
  그래서 fit이 최적화하는 대상과 commit되는 대상이 같다. shrinkage도 fit 안에 들어간다.
  보존 비용을 벌점 항으로 넣지 않으므로 NLL과 보존 사이의 새 가중치가 필요 없다.
- **원 목적과의 차이.** 원 compute_z의 항(NLL, KL 0.0625, decay 0.5, clamp 0.75, zero-step 0.05)은 그대로다. 바뀌는 것은 세 가지다.
  1. z가 L8 하나에서 L4–L8의 층별 z로 늘어난다.
  2. 층별 z가 공동으로 풀린다.
  3. 최적화기가 Adam에서 block proximal 방법으로 바뀐다.
- **위치.** 앞서 합의한 MEMIT-H 기반, 모든 층 기본 사용, 원 목적 유지라는 조건을 따른다.
  MEMIT-HJ(L8 z 하나와 κ 몫 분배, [실험 설계](2026-09-30-memit-hj-experiment-design-v1/design-ko.md)와 `project/run_scripts/memit_hj/`)는 z를 공동화하지 않은 별도 방법이다.

## 1. 정의

batch 요청 B개, 편집 층 $l\in\{4,\dots,8\}$의 진입 상태에서 다음을 정한다.

| 기호 | 정의 | 근거 |
|---|---|---|
| $K_l$ | context 평균 subject key (d_in×B) | pinned `compute_ks` |
| $h_{l,r}$ | 층 l block 출력, canonical prompt의 subject 마지막 token | native compute_z의 `target_init` |
| $A_l$ | $\lambda C_0^{(l)}+H_l$, λ = 15000 | MEMIT-H |
| $\mathrm{adj}_l$ | $(A_l+K_lK_l^\top)^{-1}K_l$ | pinned MEMIT-H solve 식 그대로 |
| $R_l$ | 층 l의 local-z target 변위, 열 r이 요청 r의 $z_{l,r}-h_{l,r}$ | native δ와 같은 양 |
| $\Delta_l(R_l)$ | $R_l\,\mathrm{adj}_l^\top$ (divisor 없음) | pinned materialization 식 그대로 |

**목적.** 요청별 native 항을 그대로 쓰고, 층별 decay만 합친다.

$$\min_{R}\ \sum_{r\in\mathcal A}\Big[\mathrm{NLL}_r(W+\Delta(R))+0.0625\,\mathrm{KL}_r(W+\Delta(R))\Big]+\sum_{l}\sum_{r\in\mathcal A}0.5\,\frac{\|R_l[:,r]\|}{\|h_{l,r}\|^2}$$
$$\text{s.t.}\ \ \|R_l[:,r]\|\le 0.75\,\|h_{l,r}\|$$

- $\mathrm{NLL}_r$: 요청의 여섯 context prompt에서 target token 평균 NLL의 평균(native).
- $\mathrm{KL}_r$: native KL prompt `"{} is a"`의 subject 위치에서, 진입 분포 대비 KL(native teacher).
- $\mathcal A$: 진입 손실 $\mathrm{NLL}_r(W)\ge0.05$인 요청. 나머지는 native zero-step 규칙대로 target을 0으로 고정하고 목적에서 뺀다.
  다만 그 key는 adj의 batch Gram에 남는다. MEMIT-H가 zero-step 요청을 잔차 0으로 solve에 넣는 것과 같다.

**환원 관계.**
- 한 층만 자유로 두면, 그 층의 MEMIT-H 실현을 반영한 local-z가 된다.
- BLUE L4+L8은 층별 local-z를 순차(L4 먼저)로, 실현을 고려하지 않고 푼 것이다.
- MEMIT-H는 L8 z 하나를 divisor로 나눈 것이다.

## 2. 왜 이 형태인가

1. **실현을 fit 안에 넣어야 한다.**
   원 MEMIT의 두 단계는 서로를 상쇄해 왔다. z 단계는 clamp 경계에서 과하게 잡고(L4 local-z fit NLL 0.0011), 실현 단계는 shrinkage로 일부만 쓴다(Official MEMIT post-L8 미실현 28%).
   z만 정확히 풀면 실제 편집이 약해진다. 실현을 forward에 넣으면 필요한 만큼 target을 키우는 일이 같은 손실 안에서 정확히 일어난다.
2. **원 Adam은 쓸 수 없다.**
   Adam의 step norm은 약 lr·√d라 모든 block이 clamp 구에 붙는다.
   toy(d = 512, 2요청 × 5층)에서 원 루프는 block 대부분이 반지름의 0.98 이상에 붙었고, 목적값은 1.380으로 proximal 해 0.515보다 컸다(`test_native_adam_pins_blocks_and_loses_to_prox`).
3. **원 decay는 norm에 선형이라 쓰지 않는 층은 정확히 0이 되어야 한다.**
   gradient에 사영만 하는 SPG는 0 근처에서 진동하고, 정확한 0을 만들지 못한다.
   그래서 decay와 clamp를 묶은 정확한 radial prox를 쓴다. 이 prox는 반경 방향으로 줄인 뒤 자르는 형태다.
   toy에서 15 block 중 9개가 정확히 0이었고, 이들의 subgradient 비는 0.988 이하였다.
   이는 toy의 결과이며 Llama에서 어느 층이 쓰일지에 대한 증거가 아니다.
4. **배분이 목적 안에서 읽힌다.** 단일 요청이면 층 l의 실현 변위는 $s_lR_l$이고 $s_l=\kappa_l/(1+\kappa_l)$, $\kappa_l=k_l^\top A_l^{-1}k_l$이다.
   KKT에서 층 l이 쓰이는 조건은 다음과 같다.
   $$s_l\,\|\nabla_{d_l}(\mathrm{NLL}+\mathrm{KL})\|\ \ge\ 0.5/\|h_l\|^2$$
   즉 **편집 민감도 × MEMIT-H 실현 비율 ÷ 원 decay 가격**이 층 선택과 몫을 정한다. 원 decay가 norm에 선형이라 해는 일부 층만 쓰는 희소한 해일 수 있다.

## 3. pipeline (batch 한 번)

| 단계 | 연산 | 정밀도 | 재사용 | 산출 |
|---|---|---|---|---|
| P0 상태 | W, H, C0, context, RNG, ledger | W·H FP32 | `memit_hj/state.py`, `engine.py` | 진입 identity hash |
| P1 문항 | 요청마다 rewrite 6 + KL prompt 1, lookup, target | – | `z_hook.prepare_batch`(B요청) | batch spec, input identity |
| P2 진입 forward | 아래 네 가지를 한 번에 수행 | FP32 | `memit_hj/oracle.py`의 teacher 방식, `z_hook.capture_prefix` | teacher, 앵커, $\mathrm{NLL}_r(W)$, prefix cache |
| P3 실현 인자 | $K_l$ ← pinned compute_ks(5회). $A_l$ ← `mom2_update_weight·get_cov(...).double()+H_l.double()`. $\mathrm{adj}_l$ ← pinned solve 식(5회) | FP64 | `memit_hj/writer.py`의 `A()`, pinned 식 | adj, $G_l$ 요약(`algebra.capacity`, 복원 fallback 포함) |
| P4 zero-step | $\mathrm{NLL}_r(W)<0.05$인 요청을 고정 | – | native 규칙 | `POLICY_ZERO_STEP` 목록 |
| P5 oracle | 아래 별도 설명 | FP32 (probe FP64) | `z_hook.native_losses` closure, `memit_hj/precision.py` | 값, 기울기, 항별 값 |
| P6 solver | block proximal spectral gradient (§4) | FP32 | `memit_hj/spg.py`의 상태·상한·plateau 규약 | 해 R*, 상태, 호출 수, KKT |
| P7 commit | 반환 평가에 쓴 $W_{\mathrm{eff}}$ 텐서를 그대로 W에 기록 | FP32 | – | W SHA, ΔW SHA |
| P8 history | post-write key(pinned compute_ks 5회) → pinned append 식 | MEMIT-H와 동일 | pinned `cache_c[i] += …` | H SHA, L4 key 불변 확인 |
| P9 관측 | at-write·all-seen·W0 기준 관측, JLZ 영수증 | – | `engine.observe`, `reducer.py` | 표준 지표, §6 기록 |

**P2 진입 forward의 네 산출물.**
- native KL teacher.
- 층 L4–L8 앵커 $h_{l,r}$. 출력 hook으로 얻는다.
- 진입 $\mathrm{NLL}_r(W)$.
- 층 4 입력의 prefix cache. 즉 층 3 출력이며, `capture_prefix`의 층 인자 의미는 T0에서 확인한다.

**P5 oracle의 한 번 호출.**
1. $\Delta W_l=(R_l^{(64)}\,\mathrm{adj}_l^{(64)\top})$을 FP32로 바꿔 $W_{\mathrm{eff},l}=W_l+\Delta W_l$를 만든다. pinned materialization 식을 그대로 쓴다.
2. prefix에서 층 4부터 끝까지 forward한다. 층 4–8의 down_proj만 forward hook으로 $W_{\mathrm{eff}}$를 쓴다.
3. prompt를 microbatch로 나누고 gradient를 누적한다.
4. `native_losses`로 NLL·KL을 계산한다(delta = 0을 넘겨 decay는 0). decay는 별도로 더한다.
5. 반환값은 $\sum_{r\in\mathcal A}(\mathrm{NLL}_r+0.0625\,\mathrm{KL}_r)$와 $\nabla_R$(autograd)다.

**commit 정합.** P5와 P7이 같은 식, 같은 텐서를 쓰므로 commit된 모델은 마지막 oracle forward와 같다. 참조 구현에서는 이 차이가 비트 단위로 0이다(`test_commit_reproduces_the_fit_and_appends_history`).

## 4. solver

변수는 자유 block $(l,r)$의 열들이다. 각 block에는 decay 계수 $c_{l,r}=0.5/\|h_{l,r}\|^2$와 반경 $\rho_{l,r}=0.75\|h_{l,r}\|$가 붙는다.

- **trial 점.** $x^+=\mathrm{prox}_t(x-t\nabla f)$. block별 prox는 $v\mapsto \hat v\cdot\min(\rho,\max(0,\|v\|-tc))$이다.
- **수락.** $F(x^+)\le\max_{10}F-\frac{\sigma}{2t}\|x^+-x\|^2$, σ = 1e-4(SpaRSA형). 실패하면 t를 반으로 줄이고, 최대 50회 시도한다.
  t의 초기값은 Barzilai–Borwein 값 $s^\top s/s^\top y$이다.
- **종료.** prox-gradient 잔차 $\|x-\mathrm{prox}_1(x-\nabla f)\|/\max(1,\|\nabla f_0\|)\le$ tol.
  tol과 호출 상한은 교정으로 정한다.
- **상태.** `CONVERGED`, `STALLED_AT_PRECISION`(수락점 6개 plateau 규칙), `NOT_CONVERGED`(상한), `LINESEARCH_FAILED`, `POLICY_ZERO_STEP`.
  반환점은 다시 평가하고, 그 평가도 호출 수에 포함한다.
- **KKT 기록(block별).**
  - 0인 block: $\|\nabla_b f\|/c_b$(≤ 1이 최적).
  - 내부 block: 접선·반경 잔차.
  - 경계 block: clamp multiplier와 접선 잔차.
- **교정(T0-JLZ).** `memit_hj/calibration.py`와 `precision.py`를 다층 $W_{\mathrm{eff}}$ oracle로 확장한다.
  - FP64 probe로 정규화 오차를 재고, tol = max(10·ε₃₂, 5 × p95 오차)로 둔다.
  - 호출 상한은 1.25 × p95 호출 수로 둔다.
  - 차단 조건은 기존과 같다: median 호출 > 200, 상한 > 400, 정밀도 바닥 > 1e-3.
  - 결합 문제이므로 실제로 쓸 batch 크기로 교정한다.
- **위험.** toy의 수렴 profile을 보면 일부 batch는 차단 조건에 걸릴 수 있다. 교정이 차단되면 가속 proximal 방법(FISTA + restart, proximal quasi-Newton)을 새 버전으로 정의한다. 기준을 사후에 완화하지 않는다.

  | tol | 1e-3 | 1e-4 | 1e-6 | 1e-8 |
  |---|---|---|---|---|
  | 15 block toy 호출 수 | 191 | 463 | 589 | 845 |
  | 10 block toy 호출 수 | 97 | 425 | 847 | 1505 |

  목적값은 tol 1e-3에서 이미 1% 안이다. 활성 집합은 1e-4–1e-6에서 안정된다.

## 5. 기술 gate

| ID | 조건 |
|---|---|
| J1 진입 parity | P2 teacher와 $\mathrm{NLL}_r(W)$가 같은 요청의 native compute_z 첫 손실(δ = 0)과 일치 |
| J2 adj parity | P3 adj가 같은 입력에 대한 pinned solve와 비트 단위로 일치 |
| J3 prefix parity | R = 0에서 층 4부터의 suffix 출력이 native full forward와 일치(허용치는 T0 실측) |
| J4 materialization | commit된 W가 반환 평가의 $W_{\mathrm{eff}}$와 비트 단위로 일치 |
| J5 commit 정합 | commit 모델의 native forward NLL·KL이 반환 평가와 J3 허용치 안에서 일치 |
| J6 feasibility | 모든 block ≤ ρ(1+1e-6), 0 block은 정확히 0 |
| J7 명제 3 | commit 뒤 L4 key가 진입 key와 비트 단위로 일치 |
| J8 history | append가 pinned 식과 같은 입력에서 일치 |
| J9 유한성 | 모든 값 유한. 아니면 `NONFINITE`로 기술 중단 |

## 6. 기록

- **batch 단위.**
  - zero-step 수, solver 상태·호출·backtrack·잔차.
  - 목적의 항별 값(NLL·KL·decay).
  - J1–J9 결과, 단계별 timer(P2–P8), 최대 메모리.
- **층 단위.**
  - target norm $\|R_l\|$, 진입 key에서의 실현 변위 $\|\Delta_lK_l\|$와 그 에너지 몫.
  - 활성 block 수.
  - 자기 key 실현 비율 $\mathrm{diag}(G_l(I+G_l)^{-1})$, $G_l$ 고유값 요약.
- **block 단위.** norm/ρ, 상태(0·내부·경계), §4의 KKT 값.

## 7. 비용과 메모리

- **호출 1회의 비용.** batch의 rewrite·KL prompt(B × 7)를 층 4부터 forward·backward하는 것이다.
  native compute_z가 batch 전체를 한 iteration 도는 비용의 약 28/32다.
- **z 단계 비용.** 따라서 대략 (호출 수 / 25) × 0.875 × native z 시간이다. MEMIT-H 10k BS100의 native z 시간은 5.31h다.
  - 호출 100회면 약 18.6h.
  - 호출 400회면 약 74h.
  - 그 밖의 비용(key, 평가 등)은 MEMIT-H와 비슷한 약 7.1h다.
  - 실제 호출 수는 교정에서 정해진다. 결합 batch에서는 이 항이 주된 비용이다.
- **BS1.** 호출당 prompt가 7개라 훨씬 싸다.
- **메모리.**
  - $W_{\mathrm{eff}}$ 5개(FP32 약 1.2GB)와 FP64 임시 곱 1개.
  - adj(FP64, d_in × B).
  - prefix cache(B·7 prompt × 층 4 입력).
  - activation은 microbatch로 제한한다.

## 8. 진단 모드

본 방법이 아니라 원인 분해용이다. RAM 분기에서만 쓰고 영수증에 표시한다.

- `native_adam`: 같은 목적에 원 Adam 루프를 쓴다. 고정 정도와 목적값 차이를 본다.
- `injection`: 실현을 고려하지 않는 주입형 공동 z를 쓴 뒤, 순차 target tracking으로 실현한다. 실현을 fit 밖에 둔 경우의 부족분을 본다.
- `free_layers=[l]`: 단일 층으로 환원한다(`test_single_layer_reduction`).
- `decoupled`: 요청별 B = 1 실현 사상으로 따로 푸는 근사다. 비용용이며, J5 차이를 함께 보고한다.

## 9. 코드 배치 (구현 대상)

[`project/run_scripts/memit_hj/`](../../project/run_scripts/memit_hj/)의 engine에 writer 하나를 더하는 형태다.

| 새 파일 (`project/run_scripts/jlz/`) | 내용 |
|---|---|
| `geometry.py` | P3: key, A, adj, G 요약. pinned 식은 AST로 가져와 쓴다 |
| `oracle.py` | P2 진입 forward와 P5 다층 $W_{\mathrm{eff}}$ oracle, FP64 probe 확장 |
| `solver.py` | §4 block proximal solver와 native Adam 진단. `jlz_ref/prox.py`의 torch 이식 |
| `writer.py` | `JLZWriter.run(requests, count, solver=…)`. Engine.step이 기대하는 기록 형식(layers, z, keys, append, timers, calls)에 JLZ 필드를 더한다 |
| `test_cpu.py` | 작은 fixture로 J2·J4·J6–J8, `jlz_ref` 검사의 torch 판 |

- cell에는 `writer: jlz` 필드를 더한다.
- state, CP, 관측, reducer, technical, history refresh(v3)는 기존 모듈을 그대로 쓴다.
- pinned BLUE/EasyEdit 원본은 수정하지 않는다.

## 10. 정하지 않은 것

실험 결과를 보고 정한다.

- **batch 크기와 비교 arm.** BS1과 BS100 중 무엇으로 할지, MEMIT-H·MEMIT-HJ·BLUE와 어떻게 비교할지.
- **희소 해 처리.** 원 decay 때문에 해가 희소할 때, 모든 층 사용을 강제하는 제곱 decay 변형을 둘지. 이는 원 목적에서 벗어난다.
- **solver 가속.** 교정이 차단될 때의 가속 solver.
- **history refresh(v3) 결합.**

## 11. 파일

| 파일 | 내용 |
|---|---|
| `jlz_ref/toy.py` | 잔차 MLP toy와 수동 backprop. 한 prompt에 위치 하나 |
| `jlz_ref/problem.py` | 진입 측정, adj, zero-step, 목적·기울기, prox, KKT·배분 보고 |
| `jlz_ref/prox.py` | block proximal spectral gradient, native Adam 진단 |
| `jlz_ref/writer.py` | P1–P9 참조 pipeline |
| `jlz_ref/test_jlz_ref.py` | 검사 9개(약 5초) |

검사 9개가 확인하는 항목은 다음과 같다.
- 기울기 유한 차분
- prox 정확성
- 수렴과 KKT
- commit 정합과 history
- zero-step
- 요청 결합
- Adam 고정
- 단일층 환원
- 상태와 호출 상한

실행은 `python3 -m unittest project.run_scripts.jlz_ref.test_jlz_ref -v`이다.
