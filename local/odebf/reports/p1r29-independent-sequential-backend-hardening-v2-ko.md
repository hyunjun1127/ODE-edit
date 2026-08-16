# P1R29-independent Sequential/Historical Backend Hardening v2

- instruction: `ODEEDIT-S05-P1R29-INDEPENDENT-SEQUENTIAL-BACKEND-HARDENING-V1`
- status: `BACKEND_HARDENED_WAITING_VIABLE_ATOMIC_ADAPTER`
- checkpoint: `fa3d570bdcea49cdaa1d705dfd399648a4920778`
- parent scaffold: `b0acdb995942b3561da09312eb67899b2d41af5e`
- tree: `c623b13882e10ec3a3255de2d2065e2f17ab04c9`
- exact contract SHA-256: `d4ad648da38ec3348970d68b759f67e8cad2508c832ef9cd70f00f275ce29676`

## 1. 상태 분리

### FACT

- `SEQUENTIAL_BACKEND_IMPLEMENTATION=CONTINUE_COMPLETE_MODEL_FREE`
- `SCIENTIFIC_SEQUENTIAL_EXECUTION=HOLD_UNTIL_VIABLE_ATOMIC_ADAPTER`
- model/GPU/Slurm/scientific result-root action = `0/0/0/0`
- failed P1R29 Atomic checkpoint import = `0`
- 실제 Round1/B10×2/B10×10 실행 = `0/0/0`

기존 Stage-A 보고서와 `b0acdb9` 코드는 이제 명시적으로 `SCAFFOLD`다. 해당 보고서와 v1 lock의 바이트는 보존됐다. 이 backend hardening은 P1R29 Atomic의 과학적 실패를 감추거나 성공으로 합산하지 않는다.

P1R29 Atomic 결과는 별도 사실이다: `SCIENTIFIC_FAIL_NEUTRAL_STRENGTH_GATE / TERMINAL_DEBT_REMAINS / WRITE_UNDER_REALIZED`. 따라서 과학 실행 gate는 닫혀 있다. 반면 교체 가능한 sequential backend의 model-free 구현은 완료했다.

## 2. Adapter-neutral interface

backend는 선택될 Atomic adapter로부터 다음 typed state를 받는다.

- current BF16 `W`, current `z`
- layer proposal `B_l`
- authoritative physical signed slope `a_l`
- semantic command `rho`
- current raw keys
- terminal physical-state keys
- adapter-owned preprocessing, request order, tokenizer/target-normalization identities

backend에는 target/debt/token normalization 재구현이 없다. 동일 preprocessing은 미래 Atomic adapter가 공급해야 한다. adapter 및 H/P backend가 추가하는 model forward/backward는 `0/0`이다.

## 3. Historical solve와 cache

raw `K_H`는 AlphaEdit solve에, projected `Z_H=P K_H`는 Structural-H risk에만 쓴다. history-only block은 outer history identity별로 한 번 LU factorization하고 K8에서는 current B10 Schur block만 갱신한다.

cache key는 다음 전부를 포함한다.

1. history version
2. active-record digest
3. raw key hash
4. projected key hash
5. projector hash

active column 수는 `10×version`이 아니라 ledger의 active snapshot에서 읽는다. collision fixture에서는 version 2이면서 active columns 10인 상태를 정확히 처리했다. stale cache identity는 geometry를 변경하거나 단순 중단하지 않고 exact uncached Woodbury backend로 fallback하고 receipt에 남긴다. raw/projected semantic drift 자체는 integrity error로 fail-close한다.

## 4. Cumulative Structural-H

v1 scaffold의 self-term-only H를 폐기하고, round 안에서 다음 상태를 누적한다.

`A_H,l = sum_{j<k} h v_{l,j} B_{l,j} Z_H,l`

candidate quadratic은

`H(v)=sum_l ||A_H,l + h v_l B_l Z_H,l||²`

이며 offset, signed cross, self를 모두 직렬화한다. negative cross도 허용된다. dense fixture에서 두 accepted step의 직접 계산과 quadratic expansion이 일치했다. history가 비어 있는 round1은 모든 v에서 H가 정확히 0이다. H의 budget/veto/retry/strength attenuation은 0이다.

Historical solve와 Structural-H의 역할은 분리된다.

- historical solve: proposal `Q_l` 자체를 과거 key geometry로 교정
- Structural-H: 같은 semantic strength 안에서 layer allocation만 교정

## 5. Cumulative Structural-P와 계산량

v1의 prior factor 이중 loop를 제거했다. 각 accepted update마다 raw cumulative `P_prior` scalar를 cross+self로 증분 갱신하고, layer별 weighted low-rank factors/right/covariance action을 concat cache한다. candidate cross는 concat cache에 대한 batched Gram으로 계산한다.

router에는 constant `P_prior`를 넣지 않는다. 다음을 분리한다.

- raw cumulative P: terminal/physics telemetry
- marginal `Delta P(v)=P(v)-P_prior`: router input
- same-state Neutral marginal Delta P: normalization과 비교 telemetry

`P_prior=0`과 매우 큰 constant offset fixture가 동일 Soft allocation을 산출했다. 50 prior accepted updates 뒤 candidate query의 `prior_pairwise_replay_count=0`이고, batched cross call 증가는 layer 수와 동일했다. dense prior outer+prior inner cross/self parity도 통과했다.

## 6. Exact-strength SoftHP

우선순위는 고정됐다.

1. `a^T v=rho`, `v>=0`
2. Neutral-relative numerical/trust validity
3. controllable normalized H/P minimax
4. H/P tie 안에서 capacity 최소화

Neutral/Soft는 동일 `rho`를 사용한다. DOF=0이면 Soft=Neutral이다. Soft numerical failure는 retry/backtracking 없이 same-strength Neutral fallback이다. `1e-8` energy relative tolerance는 과학 budget이 아니라 Neutral보다 수치적으로 더 위험한 energy를 허용하지 않는 certificate tolerance이며, trust-bound active 여부를 별도 receipt한다.

## 7. All-or-nothing transaction

한 transaction이 다음을 함께 commit한다.

- persistent BF16 weights
- raw/projected keys와 records
- cumulative-P scalar/factors
- round H state
- factorization cache invalidation
- completed round/version

`after_weights`, `after_history`, `after_p`, `after_h`, `after_cache`, `after_round` 여섯 fault injection 모두에서 전체 state identity가 이전 값으로 정확히 복원됐다. verified=false는 mutation 0이다. 동일 transaction ID 재실행은 append 0의 idempotent receipt다. verification 범위는 numerical/provenance/physical endpoint이며 Eff/Gen/Loc 또는 functional metric hard gate는 0이다.

collision은 obsolete solve/H record를 active registry에서 제거하지만 cumulative physical P/load를 감소시키지 않는다.

## 8. Terminal key identity

terminal keys는 다음 6개 identity가 모두 같을 때만 재사용한다.

1. committed BF16 W
2. request order
3. tokenizer/target normalization
4. layer set
5. capture method
6. hook-disabled physical state

각 field의 단독 mismatch fixture는 정확히 한 번 commit-state recapture를 발생시켰다. 전체 일치 시 recapture는 0이다.

## 9. Model-free gate 결과

- 신규 focused: `24/24 PASS`, warnings-as-errors
- focused+impacted: `52/52 PASS`, warnings-as-errors
- Woodbury parity: history `0/10/50/90` PASS
- collision active-count 및 stale exact fallback PASS
- cumulative H/P dense parity PASS
- same-rho/DOF0/Neutral fallback PASS
- six-phase transaction rollback/idempotency PASS
- O(N²) replay operation-count negative PASS
- adapter/preprocessing ownership 및 no-extra-F/B PASS
- compile/AST/bash/source-manifest/dry-plan PASS

## 10. 예상 compute 분해

한 future B10×10 trajectory는 outer 10×K8=80 current-state fields를 가진다. history-only factorization은 round당 최대 1회, K-step에서는 B10 current block/Schur 갱신만 수행한다. Structural-H는 현재 proposal과 cached `Z_H` action, Structural-P는 concat low-rank cache와 candidate Gram을 사용한다. 이들 router telemetry의 추가 model F/B는 0이다.

outer commit 이후 current+prior evaluation은 총 55 B10 evaluations이며 pure edit ledger와 evaluation ledger가 분리된다. 실제 wall/F/B/token/MaxRSS는 모델을 실행하지 않았으므로 `NOT_RECORDED`다. 이는 model-free adapter integration contract이지 실제 B10×2 smoke receipt가 아니다.

## 11. 경계

### FACT

backend component와 CPU contract는 hardened checkpoint로 봉인됐다.

### INFERENCE

O(N²) prior replay와 stale-cache geometry 위험을 제거했으므로, viable Atomic adapter가 선택될 경우 sequential runtime의 계산·트랜잭션 기반으로 재사용할 수 있다.

### TECHNICAL_FAIL

현재 unresolved technical failure는 없다. 미래 adapter가 preprocessing/state identity를 충족하지 않으면 integration technical hold다.

### SCIENTIFIC_FAIL

기존 P1R29 Atomic은 strength/debt/realization gate에서 실패했다. 이 checkpoint는 그 결과를 바꾸지 않는다.

### NOT_RECORDED

- viable Atomic adapter: `NOT_RECORDED`
- actual Round1 numerical equivalence: `NOT_RECORDED`
- actual B10×2/B10×10 result: `NOT_RECORDED`
- retention/locality/proxy alignment: `NOT_RECORDED`

최종 상태는 `BACKEND_HARDENED_WAITING_VIABLE_ATOMIC_ADAPTER`이며 scientific promotion은 false다.

