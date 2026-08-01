# Session 01 Motivation — ODE-Edit related work v0

날짜: 2026-08-01

상태: **GH primary-source survey; 2026-08-02 Motivation closure 반영; novelty 확정 문서 아님**

분석 주체: GH. Terra Ultra subagent는 runtime model/reasoning metadata를 검증할
수 없어 파일 열람 없이 종료했으므로, 이 문서는 독립 agent review를 대체하지
않는다. arXiv abstract/current metadata를 중심으로 최소 범위의 primary-source
검증만 수행했다.

## Proposal에서 온 내용

- ODE-Edit은 locate-and-edit가 만든 finite low-rank write를 parameter-space의
  state-dependent trajectory로 재정의하려는 연구 handoff다.
- MEMIT을 1차 editor, AlphaEdit projected proposal을 2차 editor로 둔다.
- sequential failure, null-space preservation, norm control, breadth-first scheduling,
  ODE steering/merging을 관련 축으로 제시한다.
- proposal의 bibliography와 novelty 문장은 검증된 claim이 아니다.

## Repo/protocol에서 확인한 사실

- 현재 Motivation 구현은 Llama3-8B-Instruct와 Qwen2.5-7B-Instruct에 같은
  code/config를 사용한다.
- direct-z는 case/branch 안에서 고정하고, current weight state에서 layer proposal
  direction을 다시 계산하는 atomic diagnostic을 수행한다.
- evaluation prompt는 controller에서 차단하고 EasyEdit 및 precomputed
  covariance/projector를 read-only로 사용한다.

## 1. Locate-and-edit 기반

### ROME / MEMIT

ROME은 transformer MLP를 key-value memory로 보고 한 factual association을
low-rank weight update로 바꾸는 기반을 제공한다. MEMIT은 이를 여러 association과
여러 layer로 확장한다. ODE-Edit은 factual target 생성 자체를 새로 제안하는 것이
아니라 MEMIT이 제공하는 target/proposal의 **실행 trajectory**를 연구한다.

- ROME: <https://arxiv.org/abs/2202.05262>
- MEMIT: <https://arxiv.org/abs/2210.07229>

### 직접적인 주의점

`Does Localization Inform Editing?`은 causal localization 결과와 edit success가
강하게 연결되지 않을 수 있음을 보였다. 따라서 ODE-Edit의 layer routing을
“knowledge가 실제 저장된 layer를 더 잘 찾는다”로 설명하면 안 된다. 현재 주장은
동일 editor actuator 사이의 utility/capacity allocation으로 제한해야 한다.

- <https://arxiv.org/abs/2301.04213>

## 2. Sequential degradation과 안정화

### Gradual/catastrophic forgetting

Gupta et al.은 ROME/MEMIT sequential editing에서 과거 edit와 downstream 능력이
점진적으로 저하된 뒤 급격히 붕괴할 수 있음을 보고했다. 이는 long-horizon 문제의
동기이지만, 현재 4–8 case Motivation run으로 그 현상을 재현했다고 말할 수 없다.

- <https://arxiv.org/abs/2401.07453>

### ENCORE

ENCORE는 edit target overfitting과 edited-matrix norm growth를 원인 축으로 보고,
early stopping과 norm constraint를 결합한다. 10K sequential edits와 빠른 runtime을
주장하므로 차후 ODE-Edit의 강한 compute/quality baseline이다. atomic direction
refresh 결과만으로 ENCORE 대비 우위를 말할 수 없다.

- <https://arxiv.org/abs/2502.01636>

### Norm-Anchor Scaling (NAS)

arXiv `2602.02543`의 현재 v3 제목은 proposal의 초기 제목이 아니라
**Norm Anchors Make Model Edits Last**다. 논문은 solved value와 edited weight norm
사이의 positive feedback를 분석하고, original-model reference norm으로 value를
rescale하는 NAS를 제안한다. one-line/negligible-overhead baseline이므로 ODE의 추가
NFE는 NAS보다 뚜렷한 frontier 이득을 보여야 한다.

- <https://arxiv.org/abs/2602.02543>

### LyapLock

LyapLock은 sequential editing을 cumulative preservation constraint가 있는 online
optimization으로 보고 Lyapunov/queue 기반 stepwise subproblem을 푼다. ODE-Edit의
cumulative load state와 관련 있지만, LyapLock은 long-term constraint management가
핵심이고 ODE-Edit은 within-edit proposal trajectory가 핵심이라는 차이가 있다.

- <https://arxiv.org/abs/2505.15702>

### NSE

Neuron-Level Sequential Editing은 original weights를 사용해 target hidden state를
최적화하고, activation에 따라 여러 layer의 neuron을 반복 선택한다. 동적 multi-layer
selection이라는 점에서 가까운 직접 선행이다. ODE-Edit이 단순히 “여러 layer를
동적으로 고른 최초 방법”이라고 주장해서는 안 된다.

- <https://arxiv.org/abs/2410.04045>

## 3. Projection과 preserved behavior

### AlphaEdit

AlphaEdit은 update를 preserved-knowledge representation의 approximate null space에
projection한다. 이는 허용 direction을 제한하는 hard geometric layer이고,
ODE-Edit의 current-state relinearization/routing과 원칙적으로 직교할 수 있다.
그러나 approximate projector의 empirical protection을 exact global preservation으로
과장하면 안 된다.

- <https://arxiv.org/abs/2410.02355>

### BetaEdit

BetaEdit은 approximate null-space leakage와 sequential degradation을 지적하고,
leakage control과 history-aware update를 결합한다. ODE-Edit Alpha track이 history-free
projector만 사용한다면 BetaEdit보다 강한 lifelong baseline이라고 볼 수 없다.
proposal의 `IJCAI 2026` 표기는 현재 확인한 arXiv metadata만으로는 검증하지 않았다.

- <https://arxiv.org/abs/2605.09285>

### CrispEdit

CrispEdit은 capability-loss의 low-curvature subspace에 update를 projection하는
second-order constrained optimization이며 K-FAC/matrix-free projector를 사용한다.
AlphaEdit의 preserved-key null space와 다른 capability geometry다. ODE-Edit의
evaluation firewall 때문에 capability gradient를 controller에 넣지 않는다면 직접
결합이 아니라 external capability-preservation baseline으로 다뤄야 한다.

- <https://arxiv.org/abs/2602.15823>

### AlphaEdit reproducibility study

2026 reproducibility study는 original-scope 결과를 대체로 재현했지만, newer
architectures와 더 긴 horizon에서 AlphaEdit 이득이 균일하지 않고 보호가 bounded할
수 있다고 보고한다. 따라서 Llama와 Qwen 양쪽에서 같은 projected ODE signal을
요구하는 현재 gate는 정당하다.

- <https://arxiv.org/abs/2606.26783>

## 4. Scheduling과 ODE analogy

### Fine-tuning Done Right

이 연구는 edit sample 하나를 수렴시킨 뒤 다음 sample로 가는 depth-first pipeline과
epoch/minibatch 단위 breadth-first scheduling을 구분한다. ODE-Edit의 breadth-first는
한 request 내부의 layer actuator 축이므로 동일 알고리즘이 아니다. 다만 “execution
schedule 자체가 over-optimization/interference를 만든다”는 대안가설을 강화하며,
향후 LocFT-BF를 강한 baseline에 포함해야 한다.

- <https://arxiv.org/abs/2509.22072>

### ODESteer

ODESteer는 activation steering을 ODE로 해석하고 barrier/log-density-ratio에 따라
multi-step adaptive activation flow를 만든다. parameter editing이나 factual direct-z
realization은 아니므로 직접 선행이라기보다 state-dependent vector field의 인접
문헌이다.

- <https://arxiv.org/abs/2602.17560>

### ODE-M

ODE-M은 continual model merging에서 low-loss connecting path를 따라 velocity를
적분하고 loss-increasing component를 barrier로 제어한다. model merge endpoint와
calibration loss를 사용하는 점이 edit-authorized context만 쓰는 ODE-Edit과 다르다.

- <https://arxiv.org/abs/2605.19409>

## 5. GH 추정: 현재 가능한 positioning

현재 확인한 문헌 범위에서 ODE-Edit의 잠정 구분점은 다음의 결합이다.

1. MEMIT/AlphaEdit가 만든 low-rank actuator와 fixed direct-z를 유지한다.
2. 같은 atomic edit 안에서 current weight state마다 proposal direction을 다시 만든다.
3. C-distance를 match해 단순 update-size 효과와 direction-refresh 효과를 분리한다.
4. evaluation prompt 없이 rewrite-authorized context와 editor-native geometry만 쓴다.
5. 양 model에 동일 policy를 요구한다.

이는 novelty 확정이 아니다. 특히 NSE, LyapLock, BetaEdit와의 algorithm-level 비교,
2026년 이후 동시 연구, conference proceedings를 추가 조사해야 한다.

## 6. Motivation 결과가 관련 문헌에 주는 현재 해석

- Llama atomic run에서 always-refresh arm은 fixed/coefficient arms보다 높았지만,
  central-probe adaptive selector는 그 방향을 거의 선택하지 못했다.
- 이는 state dependence의 가능성은 남기되 “현재 gate/controller가 작동한다”는
  claim을 닫는다.
- AlphaEdit projected run에서 always-refresh `A4-B4`는 Llama `+0.826986`,
  Qwen `+1.668561`이고 양 model 모두 8/8 positive였다. 이는 projected actuator의
  atomic transfer signal이지 null-space preservation이나 lifelong superiority가
  아니다.
- 4-edit micro-sequential always-refresh는 두 model에서 capacity와 neighborhood
  KL을 줄였지만 current utility와 retention을 악화시켜 `MICROSEQ_HARM_SIGNAL`로
  닫혔다. 따라서 relinearization만 항상 켜는 skeleton은 ENCORE/NAS보다 강한
  baseline이 아니며, first-hit·rewrite constraint·capacity routing을 실제 구현한
  뒤에만 비교할 수 있다.
- ENCORE, NAS, LyapLock, BetaEdit, CrispEdit, LocFT-BF 대비 우위는 미성립이다.
  특히 현재 약 6.4--6.8배 measured wall과 49배 controlled NFE는 harm 결과로
  정당화되지 않는다.

## 7. 사용자 확인 필요

- paper-level novelty search 범위를 Semantic Scholar/OpenReview/conference proceedings와
  코드 repository까지 확장할지는 Motivation 종료 후 결정해야 한다.
- 현재는 실험 속도를 우선해 arXiv primary-source 최소 검증만 수행했다.
