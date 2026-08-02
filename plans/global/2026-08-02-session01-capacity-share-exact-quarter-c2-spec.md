# Motivation — capacity-share exact-quarter c2 실행 계약

- 날짜: 2026-08-02
- 방법명: **ODE-Edit**
- 상태: implementation/test 완료, 제출 전
- 모델: `llama3-8b-inst`, `qwen2.5-7b-inst`
- editor family: MEMIT, canonical-history AlphaEdit
- 공통 policy: `capacity-share-history-exact-quarter-k4-v3`

## 결론과 목적

c1은 BF layer allocation과 global update magnitude를 같은 coefficient로 구현했다.
그 결과 Llama의 QP round는 MEMIT/Alpha 모두 `0/16` full-progress feasible이었고,
실제 path는 native C-distance의 `0.671819/0.686245`에 그쳤다. Overloaded layer는
두 family 모두 0개였지만 common frontier가 non-overloaded layer까지 잘라냈다.
따라서 c1의 efficacy harm은 layer routing 자체와 update-strength mismatch를 분리하지
못한다.

c2는 layer별 BF coefficient를 **상대 share**로만 사용하고, global hop magnitude
`h=D/4`를 독립적으로 적용한다. 네 hop의 총 C-distance는 정확히 native distance
`D`다. 이 실험은 deployable method 우위가 아니라 c1 negative가 구현 confound였는지
판별하는 Motivation 원인분리 diagnostic이다.

## 네 범주

### Proposal에서 온 내용

- 같은 snapshot에서 layer proposal과 rewrite-only utility를 비교한다.
- layer velocity는 capacity cost와 rewrite progress에 따라 재배분한다.
- 모든 layer를 simultaneous partial update하고 current state에서 다시 계산한다.
- MEMIT과 AlphaEdit projector/history에 같은 controller를 사용한다.

### Repo/protocol에서 확인한 사실

- EasyEdit는 read-only이며 ODE-edit-side hook만 수정한다.
- direct-z는 edit당 한 번, covariance/projector/Wikipedia artifact는 precomputed
  read-only로 재사용한다.
- c1 raw에서 Llama QP의 mean path/native는 MEMIT `0.671819`, Alpha
  `0.686245`; native rewrite utility fraction은 각각 `0.698943`, `0.834113`이다.
- Llama의 overloaded observation은 두 family 모두 0인데 round 2--4에서 거의 모든
  layer coefficient가 common-frontier cap에 걸렸다.
- server1 project cap은 GPU 4, host memory는 GPU당 `198117 MiB`다.

### GH 추정

- c1의 Llama harm은 direction relinearization보다 global update shrink의 영향이 크다.
- exact-distance c2에서 Llama efficacy가 회복되면 c1 negative는 implementation-confounded로
  확정할 수 있다.
- c2에서도 양 모델과 양 editor에서 efficacy가 나빠지면 그때부터 layer share 또는
  refreshed direction 자체를 scientific 원인으로 검토할 수 있다.

### 사용자 확인 필요

- 없음. 사용자가 구현 문제로 판정하고 재구현·실험·분석을 명시적으로 지시했다.

## c2 고정 구현

각 QP branch/edit에서:

1. ordered native proposal의 C-distance를 `D`로 측정한다.
2. 매 round current state에서 synchronous layer proposal, slope, cumulative capacity를
   다시 계산한다.
3. QP는 남은 native rewrite gap을 남은 round 수로 나눈 target으로 capacity-aware
   **상대 layer allocation**을 구한다. 이는 update 크기 자체를 정하지 않는다.
4. QP allocation을 per-layer cap 안에서 radial rescale하여 joint L2 share를 1로 만든다.
5. 별도 global step `h=D/4`를 적용한다. 매 hop C-energy는 정확히 `(D/4)^2`다.
6. 네 hop을 항상 완료하고 총 accepted path를 정확히 `D`로 맞춘다. First-hit,
   native-reference match, trust ratio는 기록만 하며 step shrink/early stop에 쓰지 않는다.
7. Common frontier는 overload detector다. Non-overloaded layer의 cap은 full hop envelope,
   truly overloaded layer만 current load 이상 증가하지 못하게 hard cap한다.
8. exact-distance share가 hard cap 안에서 구성 불가능하면 작은 endpoint를 내보내지 않고
   technical fail-closed한다.

Native branch는 ordered editor update를 그대로 한 번 적용한다. Llama/Qwen과
MEMIT/Alpha에는 byte-identical policy를 사용하며 모델별 threshold, K, rescue를 금지한다.

## Gate

### Technical gate

- 8 controller + 8 evaluator terminal/pass
- QP edit마다 exact 4 hops, 각 hop `C-distance=D/4`, total path `D`
- 모든 applied layer share의 L2 norm 1, cap/barrier violation 0
- 동일 policy/hash/case order/layer/seed, no model-specific branch
- action-before-evaluation, direct-z-once, exact rollback/replay/lineage
- EasyEdit/precomputed covariance/projector/history read-only

### Lenient Motivation gate

- 핵심 판정은 c2가 c1보다 Llama current rewrite utility를 양 family에서 회복하는지다.
- 다음으로 c2-minus-native current utility가 두 모델에서 방향상 크게 붕괴하지 않는지와
  capacity/KL proxy를 함께 본다.
- 4-edit signal이므로 작은 mixed result도 구현 confound 제거 신호로 허용하지만,
  lifelong·general superiority·preservation guarantee는 열지 않는다.

## Kill / next

- exact hop/path 또는 model-common identity가 깨지면 technical kill 및 재제출 금지
- c2가 c1과 거의 같거나 더 나쁘면 low-update confound 설명을 kill하고 layer share 또는
  refreshed direction을 다음 원인으로 올린다.
- Llama/Qwen 모두 efficacy가 회복되고 cost proxy가 유지되면 Motivation을
  `implementation-corrected positive/mixed signal`로 닫고 method-stage 설계로 넘긴다.

## 실행 envelope

- 목적: c1 update-strength mismatch 제거 및 BF layer share 원인분리
- 허용 write path:
  `local/results/raw/session01_motivation/caphist_*_c2_v1/`,
  `local/logs/slurm/session01_motivation/`,
  `local/state/slurm-submissions/session01_motivation/`, small Git reports/audits/runs
- Slurm: clean pushed main과 exact preflight PASS 뒤 GH one-shot만 allowed
- GPU / memory: server1 GPU 4, parent `260000M`, child별 GPU 1 / `65000M`
- red gate: exact-distance, model-common, firewall, read-only, resource cap; block이면 중단
- artifact broadcast: active peer clone이 없으면 no-peer exception 기록
- 완료 보고: `experiment-reports/global/`, `audits/global/`, `runs/`,
  `messages/server-heads/server1/`
- 금지: EasyEdit 수정, cache/projector 재계산, online download, model별 rescue,
  raw artifact/GPU log/credential Git commit, 다른 Codex session/repo 조작
- 예상 산출물: controller/evaluator 16개, checkpoint 32개, model analysis 2개,
  pair analysis 1개, 모델별 독립 분석 보고와 GH synthesis
- 중단 조건: session/CWD/repo/job/run/resource mismatch, output collision, dirty/unpushed Git,
  worker nonzero, exact C-distance 불일치, leakage/read-only 위반
