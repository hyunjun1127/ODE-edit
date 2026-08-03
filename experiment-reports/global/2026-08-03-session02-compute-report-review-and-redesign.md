# Session 02 compute report 점검과 실험 재설계

- 작성: **2026-08-03 KST**
- 작성자: GH `019fb1ea-03cb-7c20-bb3b-eba5f8d6f5f2`
- 상태: **DESIGN REVIEW; NO SCIENTIFIC OUTCOME**
- 입력 SHA-256:
  `028f712d9e69cc5a5fe8ee1fea620562931b8afa0b618b7527e11fa0b1913eba`
- revised execution spec:
  [`../../plans/global/2026-08-03-session02-compute-aware-main-table-spec.md`](../../plans/global/2026-08-03-session02-compute-aware-main-table-spec.md)

## 결론

보고서의 핵심 지적은 맞다. 평균 accepted macro-round를 `1--2`로 미리 압박하면
ODE-Edit을 static write와 한 번의 correction으로 축약할 수 있다. 따라서 round 수는
first-hit과 trust가 만드는 관측값으로 바꾸고, 계산량 목표는 `K` 자체가 아니라 비싼
`N_field`와 field당 비용을 줄이는 것으로 재정의했다.

## 외부 문헌 확인 사실

- [ODESteer](https://arxiv.org/html/2602.17560v2)는 nonlinear activation field를 multi-step으로 적분하며 default Euler 10 step을
  사용한다. TruthfulQA throughput 표의 Original 대비 ODESteer 감소는 세 backbone에서
  약 7.0--8.9%다. RK4의 이득은 Euler 대비 작다고 보고한다.
- [ODE-M](https://arxiv.org/html/2605.19409v3)은 calibration set 1024, Euler `h=0.05`를 사용하며 merge cost를
  `O(ceil(t_k/h))` first-order gradient evaluations로 설명한다.

두 논문은 각각 activation steering과 CLIP continual merging이므로 ODE-Edit의 예상
overhead나 `S_max`의 직접 근거가 아니다. `cheap RHS`, `short/event terminal`, `Euler-first`
원칙만 가져온다.

## GH가 추가로 보정한 사항

1. 보고서의 `1<=K<=6` 대신 entry-hit을 포함해 `0<=K<=6`으로 기록한다.
2. `S_max=6`은 provisional common safety cap이며 최적 step 수 claim이 아니다.
3. no-grad trial은 accept 뒤 backward를 위해 state forward를 다시 해야 한다. 따라서
   accepted trial graph reuse와 no-grad mode를 P0에서 수치 동일성·time·memory로 비교한다.
4. 별도 6-arm 4-edit screen은 profiler와 합쳐 네 arm으로 줄인다.
5. 100-edit main table은 Ordered를 P2 ablation으로 옮겨 네 arm만 실행한다.
6. 900--1000 edit age 분석은 1K stage로 미룬다.
7. SH1의 첫 CPU milestone은 12 tests를 통과했지만, 현재 directional derivative reference가
   target weights에 대한 단일 `torch.autograd.grad`를 사용한다. 이는 정확성 oracle로는
   유효해도 dense weight-gradient materialization을 피한다는 compute contract는 아니다.
   Main backend는 module input/grad-output hook contraction으로 분리한다.

## Repo/protocol에서 확인한 사실

- 기존 method 문서에는 평균 `1.5--2` 목표와 `average refresh <=2` gate가 남아 있었다.
- C3 fixed four-hop은 native 대비 4.80--6.93배 비용을 사용했으나 functional retention은
  개선하지 못했다.
- SH1은 scientific outcome, numerical lock, launcher, Slurm 없이 backend-neutral skeleton과
  12개 CPU invariant test를 통과한 상태에서 compute redesign partial hold를 적용했다.

## GH 추정

- direct-z once, one-backward all-layer slope hook, functional low-rank trial, reject field reuse,
  accepted graph reuse가 결합되면 fixed four-hop보다 overhead를 크게 낮출 가능성이 있다.
- 실제 overhead와 필요한 `K`는 아직 실험 전이므로 수치 개선을 사실로 주장할 수 없다.

## 사용자 확인 필요

없음. 사용자가 계산량 감소를 core objective로 두고 보고서 점검 및 재설계를 직접 지시했다.

## Claim boundary

현재 열린 것은 compute-aware implementation hypothesis와 실행 설계뿐이다. ODE-Edit이
Native보다 빠르다거나, 3--6 refresh가 필요하다거나, retention--compute frontier를
개선한다는 claim은 아직 열리지 않았다.
