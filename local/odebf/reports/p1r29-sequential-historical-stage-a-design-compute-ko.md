# P1R29 Sequential/Historical Stage-A 설계·계산 보고서

- instruction: `ODEEDIT-S05-P1R29-SEQUENTIAL-HISTORICAL-PREPARATION-V1`
- 상태: `PREPARED_WAITING_P1R29_ATOMIC_GATE`
- Stage-A checkpoint: `b0acdb995942b3561da09312eb67899b2d41af5e`
- parent: `7ffd70169f19cfcd3de798053b4f8f418cabd1a9`
- tree: `7bcf7a00de6a6263ca120660a0c0d91c6d42ef59`
- exact contract SHA-256: `7b65be938dba17d8ab932555436bbfc152e2a9f690ecc319239e97a8fee227f9`
- 범위: CPU/source/test/report only
- model/GPU/Slurm/result-root action: `0/0/0/0`

## 1. FACT — 구현된 범위

### 독립 arm 상태와 트랜잭션

Neutral/Soft를 포함한 각 arm은 서로 공유되지 않는 `SequentialArmState`를 갖는다. 이 상태는 다음을 함께 유지한다.

- persistent BF16 weight snapshot
- raw historical solve key `K_H`
- projected historical risk key `Z_H=P K_H`
- history version별 Woodbury factor cache
- 이전 outer round의 layer별 low-rank update 목록과 cumulative Structural-P 상태
- collision/audit metadata와 append transaction identity

현재 B10은 자신의 K8 동안 history에 들어가지 않는다. verified endpoint commit 이후에만 B10 10개 record를 한 번 append한다. verification 실패/rollback은 append 0이며, 같은 transaction replay는 idempotent append 0이다. collision 발생 시 obsolete record/key는 active solve/H registry에서 빠지지만 cumulative physical P/load는 감소하지 않는다.

### AlphaEdit historical Woodbury

일반 projector 경로를 대상으로

`A_H = lambda I + K_H^T Z_H`

를 history version별로 LU factorization한다. 각 K-state에서는 current B10 block만 갱신해 block/Schur solve를 수행한다. raw `K_H`와 projected `Z_H`는 함께 검증하며 projected key가 raw key를 대체하지 않는다.

history `0/10/50/90` 및 Llama/Qwen 형상 대리 fixture에서 cached path와 기존 exact Woodbury backend의 Q를 동일 residual tolerance `1e-8`로 비교했다. projected-key drift가 단 1 ULP라도 있으면 cache construction을 fail-close하여 cached path를 사용하지 않는다. cache 자체는 model forward/backward를 추가하지 않는다.

### Incremental Structural-H

구현된 H는 정확히 신규 accepted update의 historical projected-key movement만 측정한다.

`H_inc(v) = sum_l ||h v_l B_l Z_H,l||_F^2`

offset과 linear term은 0이다. `W_entry-W0` baseline, absolute H budget, veto, retry, target/strength attenuation은 없다.

### Sequential cumulative Structural-P

이전 outer/K-step의 accepted low-rank factor와 covariance action을 layer별로 보존한다. 현재 candidate에 대해 다음 세 항을 모두 구성한다.

- prior cumulative offset
- prior–candidate covariance cross term
- candidate quadratic self term

focused dense parity fixture에서 이 quadratic form이 dense `tr(D C D^T)`와 일치했다. scalar `committed_load`는 telemetry일 뿐 P 계산의 대체물이 아니다.

### Exact-strength SoftHP router

우선순위는 다음과 같다.

1. `a^T v = rho_cmd`, `v>=0`
2. Neutral-relative energy envelope 안에서 normalized H/P worst score 최소화
3. H/P tie `1e-8` 안에서 capacity 최소화

Neutral과 Soft는 동일한 `rho_cmd`를 사용한다. positive active layer 수가 `n+`이면 DOF는 `n+-1`이다. DOF=0이면 Soft=Neutral이다. Soft solver numerical failure는 retry/backtracking 없이 같은-strength Neutral로 fallback하며 strength를 줄이지 않는다.

### Terminal key 재사용과 outer skeleton

terminal virtual state와 committed BF16 endpoint weight identity가 같으면 K8 capture의 `keys_by_layer`를 append에 재사용한다. 다르면 명시적 recapture callback만 허용하고 count를 기록한다.

B10×10 skeleton은 다음을 고정한다.

- round-entry history count: `0,10,...,90`
- 매 round K8
- debt는 각 outer B10 시작에서 0으로 reset, batch 간 이월 0
- inner evaluator access 0
- post-commit cumulative B10 evaluation: `1+...+10=55`
- history maximum 100 records

## 2. 검증 결과

- 신규 focused tests: `15/15 PASS`, warnings-as-errors
- 신규+직접 영향 tests: `43/43 PASS`, warnings-as-errors
- Python compile: PASS
- bash syntax: PASS
- AST firewall: PASS
- source-manifest rehash: PASS
- source entries root: `1f130b55621b895fa059a6921823ed4e68133a9c91f34470d42ab42deaa1b570`
- dry-plan identity: `fb9fd7ea183d1e40bdb09e9d9cae71c2a6ae503e1ceedab52d8185fb90dde19e`
- retry/backtracking/inner-evaluator/model-F/B added by telemetry: `0/0/0/0/0`

핵심 negative fixtures는 projected-key drift, incomplete/invalid geometry, Soft numerical failure, unverified commit, duplicate transaction, inner evaluator access, stale terminal-key identity를 fail-close하거나 계약에 정해진 동일-strength fallback/append0으로 totalize한다.

## 3. 예상 계산량 분해

### Pure edit core

한 arm의 full sequential edit는 outer 10 rounds × K8 = 80 field refresh다. 각 field에서 historical Woodbury current block과 H/P routing을 갱신한다.

- `P K_H`는 매 K-step 재계산하지 않고 commit 시 1회 생성한다.
- history-only Gram/LU는 version별 1회 생성하고 K8에서 재사용한다.
- K-step에서는 B10 current block/Schur solve만 갱신한다.
- Structural-H/P router는 저장된 factor/key/covariance action만 사용하므로 model F/B 추가 0이다.
- terminal key identity가 일치하면 별도 `compute_ks` recapture 0이다.

history가 M개일 때 dense full refactor의 반복 대신 history factor cache + current B10 Schur를 사용한다. 실제 alias별 wall time, F/B/token, MaxRSS는 Stage A에서 모델을 실행하지 않았으므로 `NOT_RECORDED`다.

### Evaluation

outer commit 뒤 current+prior B10을 한 번씩 평가하므로 한 trajectory당 B10 endpoint evaluation은 총 55회다. theta0 teacher-KL/locality 같은 outer observation은 후속 frozen runtime contract에서 지정된 cadence로 수행하며 pure K8 edit core와 별도 phase에 기록한다. Stage A는 evaluator를 호출하지 않았다.

따라서 이번 최적화는 early stop이 아니라 historical proposal/routing의 중복 계산 제거다. 실제 pure-edit/evaluation 비율은 Stage B 결과가 있어야 측정할 수 있다.

## 4. INFERENCE

- history factor cache는 동일 exact backend parity gate를 통과할 때만 활성화되므로, 계산 절감이 scientific geometry 변경으로 연결되지 않도록 경계가 생겼다.
- cumulative P cross term을 보존했기 때문에 SoftHP는 이전 outer damage와 현재 candidate의 상호작용을 볼 수 있다. 다만 이 proxy가 실제 locality/history retention을 개선한다는 결론은 Stage A에서 낼 수 없다.
- H/P allocation influence가 관측되더라도 성공 판정이 아니다. 후속 동일-strength Neutral 대비 retention/P/locality를 terminal 결과로 비교해야 한다.

## 5. NOT_RECORDED / 후속 gate

- frozen P1R29 Atomic winning checkpoint와 arm/allocation: `NOT_RECORDED`
- 실제 round1 Atomic endpoint equivalence: `WAITING_FROZEN_HANDOFF`
- B10×2/B10×10 model 결과와 runtime: `NOT_RECORDED`
- actual historical retention, locality, teacher-KL, P/H proxy alignment: `NOT_RECORDED`
- scientific promotion: `false`

Stage B는 GH가 SH2의 frozen winning checkpoint를 create-once handoff하고 명시적으로 release한 뒤에만 열린다. 그때 target normalization, reachability/full-strength, alias별 promotion cell을 확인하고 round1 empty-history equivalence → B10×2 smoke → B10×10 순서로 진행한다. 현재는 어떠한 B1/B10×2/B10×10 submission도 생성하지 않았다.

## 6. Claim boundary

이번 checkpoint는 sequential/Historical 실행 결과가 아니라 실행 전 reusable Stage-A 준비물이다. SH2 live P1R29 source/job/result를 읽거나 변경하지 않았고, P1R28도 변경·재실행하지 않았다. 결론 상태는 정확히 `PREPARED_WAITING_P1R29_ATOMIC_GATE`다.

