# Session 01 Motivation — capacity/history related work v1

- 날짜: 2026-08-02
- 방법명: **ODE-Edit**
- 상태: **GH primary-source update; capacity/history 결과 반영 전**
- 조사 범위: cumulative constraint, history-aware locate-and-edit, dynamic layer
  selection, sequential norm/preservation만 최소 확인

Terra Ultra 격리 related-work agent를 시작했으나 runtime metadata에서
`gpt-5.6-terra / ultra`를 증명할 수 없어 repo와 웹을 전혀 읽지 않고 종료했다.
따라서 이 문서는 독립 agent review가 아니라 GH의 primary-source 대조다. 아래
문헌 metadata와 abstract/full-text는 2026-08-02에 arXiv 또는 공식 proceedings에서
확인했다.

## 네 범주

### Proposal에서 온 내용

- ODE-Edit은 current model state의 editor proposal을 layer actuator로 보고,
  rewrite progress를 누적 capacity marginal cost가 낮은 layer로 재배분하려 한다.
- MEMIT을 첫 actuator, AlphaEdit projected proposal을 확장 actuator로 두며,
  persistent sequential history와 long-horizon preservation을 후속 검증 대상으로 둔다.
- Proposal의 novelty와 lifelong 문장은 출발 가설이지 검증된 claim이 아니다.

### Repo/protocol에서 확인한 사실

- 현재 실험은 `llama3-8b-inst`와 `qwen2.5-7b-inst`에 같은 controller를 적용하는
  4-edit directional diagnostic이다.
- Alpha native/QP 모두 accepted edit 뒤 current key를 한 번 append하고 다음 edit의
  solve에 historical Gram term으로 넣는다. Projector와 Wikipedia covariance는
  기존 artifact를 read-only로 재사용한다.
- Capacity QP는 per-layer cumulative displacement와 새 proposal의 covariance
  cross term을 포함하며, 현재 frontier를 넘긴 positive-load layer의 추가 write를
  막고 feasible progress를 다른 layer로 넘기도록 설계됐다.
- 이 실험은 4 edits이므로 lifelong superiority나 collapse horizon을 검증하지 않는다.

### GH 추정

- 현재 가장 가까운 novelty 후보는 dynamic layer selection 자체가 아니라,
  **동일 fixed actuator set에서 current cumulative capacity의 exact marginal cost와
  rewrite-only progress를 함께 사용해 multi-layer share를 재배분하는 것**이다.
- Canonical history-on native가 이미 accumulated constraints를 반영하므로,
  ODE-Edit은 그 baseline보다 나은 capacity/preservation frontier를 보여야 추가
  controller 복잡성을 정당화할 수 있다.

### 사용자 확인 필요

- 없음. 현재는 사용자가 지시한 Motivation 규모와 lenient directional gate만
  적용하며, paper-level novelty 확정이나 long-horizon baseline sweep로 확장하지
  않는다.

## 1. AlphaEdit과 historical constraint

AlphaEdit은 preserved-key covariance의 approximate null space로 perturbation을
제한한다. Sequential objective에는 이전 edit key `K_p`를 포함하고, 현재 update는
개념적으로 `K_p K_p^T`가 들어간 normal system을 푼다. 따라서 historical term은
ODE-Edit이 새로 제안하는 요소가 아니며, 현재 실험의 Alpha native/QP 양쪽에 같은
history를 넣은 것이 공정한 비교다.

- AlphaEdit, ICLR 2025 Oral: <https://arxiv.org/abs/2410.02355>
- Full text의 sequential objective와 Eq. 14:
  <https://arxiv.org/html/2410.02355v4>

현재 실험이 추가로 묻는 것은 history 유무가 아니라, **같은 history-constrained
proposal family 안에서 cumulative per-layer capacity routing이 부가 신호를 주는가**다.

## 2. BetaEdit: history와 approximate null-space leakage

BetaEdit은 approximate/pseudo null space에서 작은 leakage가 edit마다 누적된다고
지적하고, history-aware update가 과거 edit interference와 cumulative weight drift를
줄일 수 있다고 분석한다. 또한 preserved-knowledge penalty와 history-aware projector를
함께 사용한다. 이는 ODE-Edit Alpha track에 두 가지 경계를 준다.

1. Static precomputed projector를 재사용한 4-edit 실험은 BetaEdit의 projector refresh나
   leakage-control method를 재현하지 않는다.
2. History term만으로 생기는 안정화와 capacity QP 효과를 분리하려면 history-on native가
   반드시 comparator여야 한다.

- BetaEdit: <https://arxiv.org/abs/2605.09285>
- Full text의 history-aware update와 cumulative perturbation 분석:
  <https://arxiv.org/html/2605.09285v1>

따라서 `Alpha capacity-QP > Alpha native-history`가 양성이어도 BetaEdit 우위나 exact
null-space preservation을 주장할 수 없다.

## 3. OTE–SE alignment: 가장 강한 단순화 반론

`The Labyrinth and the Thread`는 past editing constraints를 정확히 누적하면 sequential
update가 corresponding one-time editing solution을 재구성할 수 있고, 안정성의 핵심이
복잡한 regularizer보다 이 OTE–SE alignment에 있다고 주장한다. 이 관점은 ODE-Edit에
직접적인 단순화 반론이다.

- <https://arxiv.org/abs/2605.26670> — ICML 2026 표기 확인
- Full text: <https://arxiv.org/html/2605.26670v1>

ODE-Edit은 기존 OTE endpoint 복제가 목표가 아니라, 동일 edit acquisition을 유지하면서
더 낮은 cumulative layer-capacity realization을 선택하려 한다. 그러므로 다음 stage에서
history-on/OTE-aligned native보다 efficacy를 희생하지 않고 capacity 또는 preservation을
개선하지 못하면 ODE routing을 제거하고 accumulated-constraint solver로 단순화해야 한다.

## 4. Dynamic layer selection은 선행이 존재한다

### WilKE

WilKE는 knowledge별 pattern matching degree를 이용해 editing layer를 선택하고 1,024-edit
lifelong setting을 평가했다. 따라서 ODE-Edit은 “knowledge에 따라 layer를 동적으로 고른
최초 방법”이라고 주장할 수 없다.

- <https://arxiv.org/abs/2402.10987>
- Full text: <https://arxiv.org/html/2402.10987v2>

현재 잠정 차이는 WilKE가 edit별 wise layer를 고르는 selection 문제인 반면,
ODE-Edit은 MEMIT/AlphaEdit의 fixed editable set 안에서 multiple concurrent actuator의
share를 current state와 cumulative capacity로 반복 재배분한다는 점이다. 이 차이는
algorithmic claim이며 장기 성능 우위를 뜻하지 않는다.

### NSE

NSE도 original weights로 target hidden state를 계산하고 activation에 따라 여러 layer의
neurons를 반복 선택한다. Dynamic multi-layer intervention의 직접 선행이므로 novelty
문장에서 반드시 분리해야 한다.

- <https://arxiv.org/abs/2410.04045>
- ACL 2025 paper: <https://aclanthology.org/2025.acl-long.815/>

### DMLE

DMLE는 rule-level knowledge의 form-specific layer 분포를 근거로 early/middle layer에
서로 다른 update를 적용한다. Fact-level sequential capacity routing과 질문은 다르지만,
multi-layer allocation 일반 claim을 넓게 쓰지 못하게 하는 인접 선행이다.

- <https://arxiv.org/abs/2604.08284>

## 5. Cumulative preservation과 norm-control baseline

- ENCORE는 over-optimized target과 edited-matrix norm growth를 분석하고 early stopping과
  Frobenius constraint를 결합한다: <https://arxiv.org/abs/2502.01636>.
- NAS는 solved value와 edited weight norm의 positive feedback loop를 끊는 original-norm
  anchor를 제안한다. 낮은 overhead의 강한 baseline이다:
  <https://arxiv.org/abs/2602.02543>.
- LyapLock은 long-term preservation constraint를 queue/Lyapunov 기반 stepwise problem으로
  분해한다: <https://arxiv.org/abs/2505.15702>.
- AlphaEdit reproducibility study는 newer architecture와 더 긴 horizon에서 보호가
  균일하거나 무조건적이지 않다고 보고한다: <https://arxiv.org/abs/2606.26783>.

현재 QP의 covariance load 감소는 위 방법들의 norm, preservation constraint 또는
downstream capability 보장을 대체하지 않는다. 4-edit 양성도 method-stage에서 NAS,
ENCORE, history-on Alpha/BetaEdit 계열과 matched-efficacy frontier를 비교하기 전에는
short-chain mechanism signal로만 남긴다.

## 6. 결과 반영 전 positioning lock

현재 허용 가능한 잠정 문장은 다음과 같다.

> ODE-Edit은 existing MEMIT/AlphaEdit low-rank proposals와 precomputed geometry를
> 유지하면서, current rewrite progress와 exact cumulative per-layer covariance cost를
> 이용해 fixed editable-layer set 안에서 write share를 재배분하는 controller를
> 연구한다.

금지 문장은 다음과 같다.

- 최초의 dynamic layer selection 또는 최초의 history-aware sequential editor.
- Null-space, historical constraint, norm control, cumulative preservation constraint
  자체가 ODE-Edit의 novelty라는 주장.
- 4-edit signal로 WilKE, NSE, BetaEdit, ENCORE, NAS, LyapLock보다 우수하다는 주장.
- Covariance capacity 감소가 downstream preservation을 보장한다는 주장.

Capacity/history terminal 결과가 도착하면 model/family별 gate와 routing observation만
별도 절에 추가하며, 결과에 맞춰 위 novelty boundary를 사후 확장하지 않는다.
