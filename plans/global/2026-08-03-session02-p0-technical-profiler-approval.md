# Session 02 P0 technical profiler — GH numerical-lock approval

- 작성: **2026-08-03 KST**
- GH session: `019fb1ea-03cb-7c20-bb3b-eba5f8d6f5f2`
- canonical SH1: `019fc63e-5217-7250-9c22-c5b2ec4248f0`
- reviewed SH1 head: `f0db6743bed05be4c7073dbfe4d2d0ec11ea0961`
- main code-integration head: `4a54893`
- lock file SHA-256:
  `c0d74f6d614ea0b644488f4eaa9cbfacc1551def3b4f2516c6877fa0f3f8c710`
- proposal ID:
  `7d15789d5952f281166db8579db1cd7397d3dbb419b447178eeb0c7976d06a95`
- 판정: **`P0_TECHNICAL_LOCK_APPROVED; P1_AND_SCIENTIFIC_CLAIMS_HOLD`**

## GH 검증 사실

- SH1 final worktree는 clean이며 GPU/model load, Slurm, scientific outcome, EasyEdit write,
  artifact download/recompute와 direct-z temporary-track access가 0건이다.
- GH가 SH1 head에서 46개 focused CPU test를 독립 재실행해 모두 통과했다.
- 동일 head와 main integration 뒤 모두 paired dry plan, shell syntax, `git diff --check`,
  lock/spec hash를 재검증했다.
- Concrete backend는 두 모델에 동일한 `EasyEditMemitBackend`와 controller schema를 사용한다.
- Native/Scalar proposal, checkpoint/restore, terminal geometry, setup과 controller wall/GPU timer가
  공통 compute 범위에 포함된다.
- Functional trial은 low-rank hook이며 accepted step별 full CPU backup이 없다. Outer entry
  checkpoint rollback test는 통과했다.
- Runner에는 Slurm submit 기능과 scientific evaluation이 없으며, output은 ignored
  `local/results/session02-*` 아래로 제한된다.

## `N_z` canonical reconciliation

Compute-aware spec의 `N_z=1 per outer edit`은 **entry event가 miss여서 실제 edit action이 필요한
outer edit**에 적용한다. Entry가 이미 semantic first-hit이면 edit는 no-op이고

\[
N_z=0,\quad K_{acc}=0,\quad N_{proposal}=N_{trial}=N_{write}=0
\]

로 종료한다. 그 외에는 `N_z=1`이며 edit 안에서 direct-z를 재계산하지 않는다. 이 문서가
기존 spec의 unconditional 문구에 대한 canonical addendum다. Hash-locked spec 파일 자체는
변경하지 않아 outcome-free proposal identity를 보존한다.

## 승인된 P0 범위

- models: `Llama3-8B-Instruct`, `Qwen2.5-7B-Instruct`
- case: `2022`, seed `17`, one order
- arms: Native, Static, One-refresh, Full
- repetitions: warm-up 1 + recorded 3
- purpose: loaded-model numerical identity, fair compute, memory와 trust-radius geometry만 측정
- evaluation/generation/scientific superiority comparison: 금지
- Llama/Qwen은 별도 1-GPU job 두 개를 같은 submission batch로 동시에 제출
- job당 8 CPU, `65000 MiB`, `04:00:00`; aggregate 2 GPU, server1 project cap 4 이하

## P0 pass/hold gate

다음이 모두 통과해야 P1을 검토한다.

1. offline model/tokenizer, context suffix, fixed artifact와 covariance preflight PASS
2. combined old/new event와 two-forward reference identity PASS
3. Full warm-up의 hook/finite-difference identity PASS, target dense grad 0
4. functional trial과 committed event identity PASS
5. scoped covariance-preflight replacement가 `finally`에서 복원되고 terminal hash PASS
6. 두 모델 모두 `h0/D_native`가 `[1/8,1/2]` 안에 있음
7. output/terminal manifest와 counter/timer schema complete
8. 한 모델 전용 rescue, artifact recompute/download, EasyEdit write 0

다음은 자동 후속 조건이다.

- terminal geometry가 controller wall 또는 GPU time의 10% 초과: P1 전 low-rank cross-term
  evaluator 구현
- `Full/Native >4x`: P1 HOLD 후 compute path 최적화; retention 결과로 override 금지
- 모든 arm이 entry-hit해 hook gate를 만들 수 없음: 다른 case를 임의 선택하지 않고 P0 HOLD
- Terra Ultra analysis runtime mismatch: raw technical artifact는 보존하되 interpretation과 claim
  promotion HOLD, Sol agent 대체 금지

## Claim boundary

이 승인은 1-case technical profiler 실행만 허용한다. ODE-Edit의 성능 개선, compute 절감,
MEMIT 우위, 두 모델 공통 scientific success는 아직 주장할 수 없다. P1, P2, P3와 main table
제출은 별도 GH envelope가 필요하다.
