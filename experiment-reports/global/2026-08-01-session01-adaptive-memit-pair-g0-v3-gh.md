# Session 01 Motivation — adaptive MEMIT pair g0 v3 GH decision

날짜: 2026-08-01

방법명: **ODE-Edit**

상태: **pair technical-valid / adaptive selector kill / always-refresh candidate survives**

## Repo/protocol에서 확인한 사실

- Slurm job `15734`는 2026-08-01 12:22:08–19:32:34 KST에 실행되어
  `COMPLETED`됐다.
- Llama/Qwen 각각 exact 8 cases, 56 outcomes, 8 receipts, 8 direct-z artifacts다.
- 양 model에서 all-pass, exact counts/hash, rollback, firewall, lineage,
  matched-C, read-only precomputed cache gate가 모두 통과했다.
- 사전고정 model verdict:
  - Llama: `LENIENT_MODEL_COLLAPSE`
  - Qwen: `LENIENT_MODEL_PASS`
- 사전고정 pair verdict: `LENIENT_PAIR_PARTIAL_OR_FAIL`.

## Proposal에서 온 내용

- state-dependent direction refresh가 static direction보다 의미 있는 trajectory
  차이를 만들 때만 ODE motivation이 살아남는다.
- static policy가 충분하거나 controller가 안정적으로 선택하지 못하면 더 단순한
  policy로 pivot해야 한다.
- 양 model에 다른 method를 쓰는 것은 허용되지 않는다.

## 모델 간 비교

| Metric | Llama | Qwen |
|---|---:|---:|
| `A4-B4` mean | +0.828098 | +1.583319 |
| `A4-B4` positive cases | 8/8 | 7/8 |
| `B4-C4` mean | +0.385384 | +0.657322 |
| gate refresh rate | 2/24 | 23/24 |
| mean predicted refresh advantage | -0.165817 | +0.374526 |
| `G4-A4` mean | -0.772068 | +0.001426 |

## GH 판정

### Kill: current adaptive selector

같은 code/config인데 selector가 Llama에서는 거의 항상 fixed, Qwen에서는 거의 항상
refresh를 선택했다. Llama에서는 endpoint `A4`가 `B4`보다 8/8 우세했음에도 proxy가
이를 반대로 판단했다. model별 threshold/sign 보정은 금지되며, Qwen pass로 Llama
collapse를 평균내서 숨길 수도 없다. 따라서 current central-probe adaptive selector는
Motivation에서 kill한다.

### Survive: always direction refresh

selector를 제거하고 매 hop current state에서 direction을 다시 계산하는 `A4`는
양 model에서 `A4-B4` mean이 양수다. Llama 8/8, Qwen 7/8 cases가 practical floor
위 양수이므로 small-sample lenient common-direction signal로 본다.

같은 cases에서 `G4`를 `A4`로 치환하면 Llama utility는 평균 `+0.772068` 회복하고,
Qwen은 평균 `-0.001426`로 사실상 floor 수준 변화다. 이는 **in-sample policy
substitution forecast**이며 out-of-sample, preservation 또는 lifelong 기대효과가 아니다.

## 다음 실험 policy

1. AlphaEdit projected atomic pair는 기존 locked runner 그대로 실행한다. 그 안의
   `A4`, `B4`, `C4`, `G4`를 모두 공개하되 selector pass를 ODE 생존 조건으로 쓰지
   않고 always-refresh transfer를 별도로 본다.
2. 4-edit micro-sequential candidate는 양 model 공통 `always direction refresh`로
   precommit한다. model-specific branch/threshold는 없다.
3. controller와 evaluator를 process-level로 분리한다. 네 action/edit trajectory를
   모두 commit한 뒤에만 evaluation prompt를 load해 future-action leakage를 구조적으로
   차단한다.
4. micro-sequential 양성은 같은 retention/KL/capacity 축이 양 model에서 모두 양수이고
   current-edit mean degradation이 -0.10 이상일 때만 인정한다.

## Claim boundary

- 성립: direction relinearization이 두 model에서 coefficient/fixed alternatives와 다른
  작은 atomic signal을 보였다.
- 미성립: adaptive gate/controller가 양 model에서 작동한다.
- 미검증: edit/model preservation, long-horizon retention, downstream capability,
  AlphaEdit synergy, NAS/ENCORE/BetaEdit/CrispEdit 대비 우위.

## 사용자 확인 필요

- 없음. 이 판정은 사용자가 허용한 lenient Motivation gate 안의 단순화이며,
  model별 method 분기를 만들지 않는다.
