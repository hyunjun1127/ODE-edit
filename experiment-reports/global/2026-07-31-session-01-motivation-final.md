# Session 01 Motivation Validation — GH 최종 보고서

> **Historical report — superseded by C3.** 이 문서는 2026-07-31 `NO MV3`
> decision을 보존하며 현재 Session 01 verdict가 아니다. 현재 canonical 상태는
> `CLOSED_DIRECTIONAL_POSITIVE; STRONG_METHOD_GATE_FAIL`이다. 최신 종결은
> [`2026-08-03-session01-motivation-final-closure-and-method-handoff.md`](2026-08-03-session01-motivation-final-closure-and-method-handoff.md),
> 최종 causal evidence는
> [`2026-08-02-session01-caphist-pair-c3-v1-synthesis.md`](2026-08-02-session01-caphist-pair-c3-v1-synthesis.md)를
> 따른다.

- 보고일: 2026-07-31
- repo/method name: `ODE-Edit`
- 최종 판정: **Motivation not supported cross-model**
- 실행 gate: **`NO MV3`; no retune; no further Motivation job**
- proposal rationale:
  `project/proposals/sections/01-motivation-validation.md`
- canonical plan:
  `plans/global/2026-07-30-session-01-motivation-validation.md`

## 1. 결론

ODE-Edit proposal의 전부가 실패한 것은 아니다. Native EasyEdit hook의
정확성과 same-snapshot layer allocation signal은 두 고정 모델에서
살아남았다. 그러나 ODE라는 이름을 정당화해야 하는 핵심 primary,
즉 partial update 후 proposal direction을 다시 계산하는
state-dependent refresh advantage는 두 모델에서 재현되지 않았다.

Llama에서는 refreshed direction이 fixed direction보다 모든 12 case에서
나빴고, Qwen에서만 양의 신호가 관측됐다. 이 architecture-dependent sign
reversal은 작은 metric 하나의 실패가 아니라 primary/total contrast의
일관된 반대 방향이다. 사전등록 pair gate를 그대로 적용한 결과는
`NO MV3`이며 추가 fold나 retuning으로 구제하지 않는다.

## 2. Proposal에서 온 내용

원 proposal은 direct-z를 구현하는 layer actuator가 여러 개이고,
same-snapshot joint allocation과 state-dependent relinearization이 더 나은
parameter path를 고를 수 있다고 제안했다. 이는 연구 출발점일 뿐 최종
paper plan이나 검증된 claim이 아니었다. Session 01은 다음 최소 chain으로
이를 검사했다.

1. native EasyEdit MEMIT 보존
2. held-out allocation signal
3. refreshed-direction advantage
4. 통과 시에만 matched-progress/capacity/retention diagnostic

3에서 cross-model gate가 실패했으므로 4를 실행하지 않았다.

## 3. Repo/protocol에서 확인한 사실

### 핵심 결과

| 단계/contrast | Llama3-8B-Instruct | Qwen2.5-7B-Instruct |
| --- | ---: | ---: |
| MV-0 fidelity | `3/3/3`, exact | `3/3/3`, exact |
| MV-1 allocation mean | `+0.0102628`, `20/20` | `+0.0483206`, `20/20` |
| MV-2 direction mean | `-0.0044081`, `0/12` | `+0.0068874`, `7/12` |
| MV-2 coefficient mean | `+0.0001037`, `12/12` | `+0.0016649`, `11/12` |
| MV-2 total mean | `-0.0043044`, `0/12` | `+0.0085523`, `7/12` |
| model verdict | fixed-direction coefficient pivot | direction refresh clear |

Llama direction CI는 `[-0.0073945, -0.0021643]`, total CI는
`[-0.0071797, -0.0021300]`로 모두 음수다. Qwen direction/total CI는 0을
포함하지만 사전등록 balanced gate의 mean·trimmed mean·median·sign 조건을
충족했다. Pair gate는 CI endpoint 하나로 뒤집지 않고 model-level rule을
first-match로 적용했다.

### 실행과 기술 무결성

- MV-2 job `15610`은 Llama와 Qwen을 순차 제출하지 않고 같은 parent에서
  동시에 실행했다.
- 두 child는 각각 GPU 1개, memory `65000M`을 사용했고 모두 exit `0:0`이다.
- 최초 analyzer 차단은 raw scientific result가 아니라 JSON object key-order
  parser artifact였다. Raw hashes를 유지하고 mapping key-set만 허용하는
  analyzer repair를 독립 RCA/red review 후 적용했다.
- repair 후 focused `19/19`, 전체 Motivation `172/172` test가 통과했다.
- model report는 서로의 결과를 보지 않는 별도 Terra Ultra agent가 작성했고,
  pair red는 aggregate projection만 읽었다.
- GH/primary session은 Sol Ultra, subagent는 Terra Ultra라는 session
  boundary를 지켰다.

## 4. 기대효과와 claim boundary

MV-1 forecast는 양의 방향을 맞혔지만 Qwen magnitude를 약 `56.86%`
과대평가했다. MV-2가 model별 반대 부호이므로 full method의 공통 개선폭을
한 숫자로 예측할 근거가 없다.

현재 보고 가능한 수치:

- MV-1 adaptive-minus-static teacher-forced progress
- MV-2 direction/coefficient/total의 model별 absolute rewrite-utility proxy
- model별 descriptive bootstrap CI와 controlled compute

현재 보고할 수 없는 수치 또는 claim:

- EasyEdit MEMIT/AlphaEdit 대비 accuracy·efficacy 개선률
- locality, retention, downstream capability 또는 long-horizon horizon gain
- pooled two-model ODE effect
- oracle을 실현 가능한 method gain으로 해석
- ODE-Edit의 novelty, superiority, paper readiness

따라서 “방법이 얼마나 개선할 수 있는가”에 대한 현재 답은
**cross-model full-method improvement는 예측 불가**다. Qwen에 한정된
diagnostic proxy `+0.0085523`을 일반 성능 개선으로 번역해서는 안 되며,
Llama의 대응 proxy는 `-0.0043044`다.

## 5. GH 추정: kill과 pivot 경계

### Kill

- refreshed-direction ODE/relinearization을 두 backbone에 공통 적용하는 현재
  방법 가설
- 이를 전제로 한 MV-3 matched-progress, MV-4 retention 및 capacity-aware
  long-horizon narrative
- “first ODE/adaptive/history/capacity-aware editor” 같은 넓은 claim

### 별도 proposal로만 남는 후보

- fixed-direction dynamic coefficient controller
- static/same-snapshot allocation controller
- Qwen-specific refreshed-direction controller

이 후보는 현재 ODE-Edit 성공의 연장이 아니다. 새 estimand, baseline,
information budget, kill criteria와 이름을 갖는 독립 연구 방향으로 다시
시작해야 한다.

## 6. Related work와 baseline 영향

ROME/MEMIT/PMET/AlphaEdit/EMMET은 locate-and-edit의 target, distribution,
preservation geometry를 이미 다루며, WilKE·NSE·HiEdit은 동적 layer/neuron
selection의 넓은 최초성을 차단한다. MPES+norm, NAS, BetaEdit, CrispEdit은
long-horizon regularization·norm·null-space·capability constraint 최초성을
차단한다. ODESteer와 ODE-M은 ODE/trajectory/feedback 최초성을 차단하고,
`ODEdit`은 이름과 trajectory 서술까지 충돌한다.

EasyEdit checkout에는 ROME, MEMIT, PMET, AlphaEdit, EMMET, WISE, GRACE 코드가
있다. 코드 존재는 고정 backbone에서 run-ready 또는 canonical reproduction을
뜻하지 않는다. Motivation이 종료됐으므로 외부 baseline을 새로 이식하거나
큰 benchmark를 실행하지 않았다.

상세 문헌 및 implementation boundary:
`project/proposals/sections/02-related-work-and-novelty-boundary.md`.

## 7. 사용자 확인 필요

Session 01을 닫는 데 추가 확인은 필요 없다. 사용자가 새 연구를 원하면
fixed-direction coefficient, static allocation, Qwen-specific track 중 어느
하나를 새 proposal로 선택해야 한다. Repo 이름은 현재 `ODE-Edit`으로
유지하되, paper method name은 `ODEdit`·`ODESteer`·`ODE-M` 충돌 때문에
후속 가설이 생존한 뒤 다시 정해야 한다.

## 8. Artifact와 Git 경계

- raw logs, generations, action receipts, model artifacts, checkpoints와 failed
  retries: ignored `local/`
- Git: compact metadata, analyzer/test, Korean model report, pair audit,
  reproducibility note만 포함
- EasyEdit: read-only import, 변경 없음
- credential/SSH key/token/private dataset secret: Git 기록 없음
- 최종 pair audit:
  `audits/global/2026-07-31-mv2refresh-pair-v2.postrun.md`
