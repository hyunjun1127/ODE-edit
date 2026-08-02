# Session 01 Motivation — ODE-Edit final GH closure (c1 supersession)

날짜: 2026-08-02

상태: **Motivation closed — atomic possibility only; c1 sequential controller cross-model negative; Method Session 진입 blocked**

## 결론 먼저

ODE-Edit의 **current-state direction relinearization**은 atomic/local possibility
signal을 남겼다. 이후 full remaining-native-progress, trust ratio, rollback,
capacity QP와 canonical Alpha history를 포함한 c1 controller까지 구현해 두 모델에서
같은 policy로 재실험했다. 구현은 기술적으로 통과했고 c0보다 크게 회복됐지만,
Llama의 MEMIT/Alpha와 Qwen MEMIT이 current non-collapse를 실패했다.

따라서 Motivation은 다음 두 결론으로 최종 종료한다.

1. state-dependent path의 atomic possibility는 보존한다.
2. 현재 fixed-`K=4` constrained sequential controller와 그 method-superiority claim은
   cross-model negative로 kill한다. 추가 Motivation rescue는 열지 않는다.

## 2026-08-02 c1 최종 supersession

- canonical c1 report:
  `experiment-reports/global/2026-08-02-session01-caphist-pair-c1-v3-synthesis.md`.
- technical: controller/evaluator `16/16` terminal/pass, pair technical pass.
- QP-minus-native current utility: Llama MEMIT `-4.032392`, Llama Alpha
  `-2.495123`, Qwen MEMIT `-0.397837`, Qwen Alpha `+0.029707`.
- passing family: `[]`; pair verdict `CAPACITY_HISTORY_HARM_SIGNAL`.
- c0 fixed-fraction/weak-terminal 결과는 superseded지만, c1은 유효한 과학적 음성
  evidence다.
- Direct-z 별도 session의 atomic/local positive boundary는 이 sequential verdict와
  합치지 않는다.

아래 본문은 c1 이전 motivation ladder와 full-controller 진입 조건의 역사적 기록이다.
위 supersession과 충돌하는 “Method Session 조건부 진입” 문장은 더 이상 현재 판정이
아니다.

## 네 범주

### Proposal에서 온 내용

- finite one-shot edit를 state-dependent parameter trajectory로 재정의한다.
- direct-z는 유일한 parameter endpoint가 아니며 layer actuator 사이에서 capacity
  cost를 재배분할 수 있다고 가정한다.
- H1 utility heterogeneity, H2 non-stationarity, H3 capacity concentration,
  H4 routing benefit, H5 long-horizon preservation을 검증해야 한다.
- static/fixed policy가 충분하거나 relinearization이 이득을 주지 않으면 ODE를
  제거하거나 단순화해야 한다.

### Repo/protocol에서 확인한 사실

- EasyEdit source는 수정하지 않았고 ODE-Edit-side reusable hook만 사용했다.
- model은 `llama3-8b-inst`, `qwen2.5-7b-inst`로 고정하고 동일 code/config를 썼다.
- Wikipedia covariance와 Alpha projector는 pinned read-only로 재사용했으며
  재계산하지 않았다.
- MEMIT atomic `A4-B4`: Llama `+0.828098`, Qwen `+1.583319`.
- projected Alpha atomic `A4-B4`: Llama `+0.826986`, Qwen `+1.668561`;
  양 model 모두 8/8 positive와 noncollapse `0.0`.
- central-probe selector는 Llama/Qwen에서 크게 다른 선택률과 endpoint 손실을
  보여 kill됐다.
- 4-edit microseq always-refresh는 공통 capacity/neighborhood KL reduction을
  만들었지만 current utility와 retention을 양 model 모두 악화시켰다.
- 모든 terminal run은 technical-valid이며 failed pre-action run은 별도 local
  archive에 보존하고 scientific evidence에서 제외했다.

### GH 추정

- H2의 작은 atomic signal은 살아 있다. 방향이 실제로 state에 따라 달라지고,
  fixed/transported direction과 다른 utility를 만든다.
- projected Alpha에서도 같은 신호가 남아 actuator-family transfer 가능성이 있다.
- microseq trade-off는 low-capacity path가 존재할 가능성을 지지하지만, capacity
  minimization만으로 efficacy가 보장되지 않음을 보여준다.
- proposal의 QP progress constraint, first-hitting terminal event와 trust-region
  acceptance는 장식이 아니라 harm를 막기 위한 필수 method다.

### 사용자 확인 필요

- Session 01 Motivation은 여기서 닫는다.
- 다음 Method Session에서 full controller(first-hit + trust ratio + capacity QP)를
  구현할지 사용자 확인이 필요하다. 확인 전 100+ edit/lifelong scale은 제출하지
  않는다.

## 왜 proposal이 최종 paper plan이 아닌가

proposal은 direction refresh와 capacity-aware flow가 함께 좋은 frontier를 만들
것이라 예상했지만, 실험은 direction refresh만 항상 켜면 sequential harm가 생김을
보였다. H4 controller와 H5 long-horizon benefit은 구현·검증되지 않았다. 따라서
proposal의 abstract/contribution 문장을 paper claim으로 사용할 수 없으며,
다음 method가 efficacy를 회복한 뒤에만 다시 평가해야 한다.

## 실험 ledger

| 실험 | 최소 질문 | 결과 |
|---|---|---|
| exact split control | 작은 step 자체가 효과인가 | native와 practical floor; 음성 대조 통과 |
| adaptive MEMIT atomic | selector와 refresh가 양 model 공통인가 | selector kill; always-refresh atomic signal |
| projected Alpha atomic | projection 뒤에도 refresh signal이 남는가 | 양 model transfer signal |
| 4-edit microseq | capacity/preservation specialization이 보이는가 | capacity/locality proxy 양성, efficacy/retention harm |

별도 task/session이 소유한 evidence는 본 GH closure 수치에 합치지 않았다.

## 가설별 판정

| 가설 | Motivation 판정 |
|---|---|
| H1 layer utility heterogeneity | 직접적인 layer-QP 검증 전; open |
| H2 state non-stationarity | atomic small signal로 survive |
| H3 capacity concentration | capacity path 차이는 관측; concentration은 model-common 아님 |
| H4 capacity-aware routing benefit | 실제 QP 미구현; 미검증 |
| H5 long-horizon preservation | 4-edit skeleton harm; claim closed-negative |

## Kill criteria 적용 결과

- **Kill:** central-probe adaptive selector.
- **Kill:** unconditional always-refresh full-distance sequential execution.
- **Kill:** current implementation의 preservation/lifelong/method-superiority claim.
- **유지 금지:** Llama/Qwen별 threshold, sign, branch 또는 policy rescue.
- **Survive:** common direction relinearization mechanism과 capacity/utility trade-off.

## 다음 단계 진입 기준

다음 구현은 양 model에 byte-identical해야 하며 아래를 모두 만족해야 한다.

1. rewrite-only first-hit terminal condition.
2. actual/predicted progress trust ratio와 악화 step rollback.
3. 동일 required progress에서 layer별 covariance-capacity를 최소화하는 small QP.
4. held-out 4-edit pilot에서 양 model current mean delta `>= -0.10`.
5. retention 또는 capacity/locality 중 같은 축이 양 model에서 양수.
6. 평균 accepted macro-round `<=2` 또는 compute frontier 이득.
7. 위 조건 전 large sequential/lifelong 실험 금지.

## Blue team 실행 checklist

- first-hit, trust ratio, QP를 독립 reusable module로 구현
- MEMIT/Alpha actuator interface와 evaluator를 그대로 재사용
- same case/order/model policy와 no-evaluation controller firewall 유지
- native, static/global scaling, coefficient-only, NAS/ENCORE 대비 compute budget 명시
- NFE, proposal build, wall, rejected step, QP slack을 모두 보고

## Red team kill-test checklist

- evaluation prompt/outcome가 controller로 들어가는지
- first-hit가 evaluation metric이 아니라 rewrite-authorized context만 쓰는지
- model별 threshold/hidden branch가 없는지
- capacity만 줄이고 efficacy를 under-edit하는지
- rejected step rollback과 W0/state lineage가 exact인지
- covariance/projector를 재계산하거나 EasyEdit를 수정하는지
- NAS/ENCORE보다 큰 compute를 작은 proxy 이득으로 정당화하는지
- 4-edit signal을 lifelong/downstream claim으로 과장하는지

## 기대효과 예측

현재 수치가 허용하는 기대는 제한적이다. 다음 controller가 성공한다면 목표는
microseq에서 관측된 공통 `capacity_reduction`과 `neighborhood_kl_reduction`을
유지하면서 current/retention 손실을 first-hit/rollback으로 제거하는 것이다.
구체적인 개선폭은 기존 결과로 식별할 수 없어 숫자 forecast를 만들지 않는다.

## Related work 경계

상세 보고서는
`experiment-reports/global/2026-08-01-session01-related-work-v0-gh.md`에 있다.
NAS/ENCORE는 훨씬 저렴한 강한 sequential baseline이고, AlphaEdit/BetaEdit은
projection/history 축, LyapLock은 cumulative constraint, NSE는 dynamic layer
selection의 직접 overlap 위험이다. 다음 method가 efficacy–retention–capacity–
compute frontier를 개선하지 못하면 ODE paper story는 유지할 수 없다.

## Protocol/artifact/agent

- GH direct submit은 SH 부재와 사용자 time-critical 지시의 기록된 예외였다.
- raw/log/checkpoint는 ignored `local/`에만 있고 Git에는 compact hash/report만 둔다.
- server4는 clone/SH session이 없어 artifact broadcast 대상이 아니다. no-peer
  exception을 적용하고 임의 SSH/rsync를 하지 않았다.
- Terra Ultra subagents는 runtime metadata를 확인할 수 없어 매번 무열람 종료했다.
  GH report를 독립 agent review로 표시하지 않는다.

## 최종 판정

**Motivation final closure.** State-dependent relinearization의 atomic possibility는
남지만 실제 constrained controller c1이 Llama/Qwen 공통 gate를 실패했다. 현재
sequential ODE-Edit controller는 closed-negative이며, 추가 Motivation retune이나
Method/Lifelong 확대를 진행하지 않는다.
