# ODE-BF Proposal

## Cold-start Target ODE와 Fixed-Euler Soft-Preservation Routing을 결합한 Transactional Knowledge Editing

- **Project:** ODE Edit
- **Method name:** **ODE-BF** — *ODE with Barrier-Filtered Dynamic Layer Routing*
- **현재 설계 버전:** **Cold-FR-E8** — cold \(z_{\mathrm{base}}\), full-residual arms, fixed Euler \(K=8\)
- **Base editor:** AlphaEdit 계열 locate-then-edit editor
- **보존 범위:** historical edits와 pretrained/locality knowledge
- **배포 원칙:** inner ODE trajectory는 전부 virtual state에서 수행하고, 실제 model parameter는 terminal endpoint에서 정확히 한 번만 갱신

---

## 0. Proposal 요약

ODE-BF의 현재 연구 방향은 기존의

1. Native direct-\(z\)를 독립적으로 먼저 산출하고,
2. 고정된 layer 순서와 residual 분배 규칙으로 write하는 구조

를 다음의 closed-loop flow로 대체한다.

```text
z_0 = z_base에서 cold target trajectory 시작
                         ↓
고정 Euler grid: K=8, h=1/8, tau=0→1
                         ↓
각 grid state에서 현재 key와 layer-local full-residual arm 재계산
                         ↓
target-new suffix NLL progress를 확보하면서
structural H/P와 functional H/P를 soft routing signal로 사용
                         ↓
각 grid를 정확히 한 번 전진; scientific retry/backtracking 없음
                         ↓
first-hit은 기록만 하고 tau=1까지 전체 trajectory 관찰
                         ↓
W64 solve + BF16-authoritative virtual endpoint를 freeze
                         ↓
배포 시 terminal update를 정확히 한 번 atomic commit하고
post-commit 검증 뒤 history를 정확히 한 번 append
```

최종 메커니즘은 다음 여섯 부분으로 구성된다.

### 1. Cold-start target ODE

Native direct-\(z\)를 초기값으로 사용하지 않고 현재 lookup activation \(z_{\mathrm{base}}\)에서 시작한다. Target velocity의 primary objective는 Native와 같은 방향의 **target-new suffix NLL**이며, layer-local virtual write를 통과한 gradient로 target와 weight state를 같은 Euler clock에서 전진시킨다.

### 2. State-dependent layer proposals

각 Euler grid state에서 현재 virtual model의 key, residual, rewrite efficiency와 Alpha proposal을 다시 계산한다. 따라서 layer direction 자체가 trajectory를 따라 변한다.

### 3. Structural/functional H/P soft routing

각 layer coefficient는 작은 convex QCQP로 구한다. 새 edit progress와 technical trust는 hard contract로 유지하되,

- active historical key subspace를 많이 건드리는 layer,
- pretrained key distribution에 큰 local disturbance를 주는 layer,
- cumulative load가 큰 layer

의 비중은 **soft risk objective**로 줄인다. 현재 evidence 없이 정한 H/P budget을 scientific hard veto로 사용하지 않는다.

### 4. Functional soft-risk basis와 audit

Covariance risk는 local structural proxy이므로 실제 보존을 보장하지 않는다. 각 field에서 baseline endpoint와 layer probe를 공통 schedule로 평가해

- historical target margin damage,
- pretrained teacher에 대한 incremental KL

의 layer별 functional sensitivity를 만든다. 이 값은 routing을 부드럽게 바꾸고 모든 step에서 audit되지만, 현재 main design에서는 transition을 거부하거나 clock을 되감는 threshold가 아니다.

### 5. Transactional one-shot commit

중간 Euler state도 deployed model과 persistent history를 변경하지 않는다. Terminal endpoint가 확정된 뒤 누적 update를 한 번만 적용하고, post-commit 검증이 성공한 뒤에만 history를 한 번 append한다.

### 6. W64 low-rank execution과 BF16-authoritative transaction

Dense historical matrix와 \(d\times d\) Alpha solve를 반복하지 않는다. Active history를 projected-key factor로 보관하고 mixed-FP64 small-system Woodbury path(`W64`)로 canonical Alpha linear system을 푼다. Scientific verdict는 FP32 overlay가 아니라 \(Q_{\mathrm{BF16}}(W_0+\Delta)\)의 virtual/committed identity로 판정한다.

핵심 claim은 단순히 ODE가 더 많은 early-stop 지점을 제공한다는 것이 아니다.

> **현재 model state가 사용 가능한 layer direction을 결정하고, H/P soft-risk가 각 direction에 실릴 rewrite progress를 재분배하는 fixed-Euler target–write flow가 static direct-\(z\)와 fixed layer sharing보다 lifelong sequential collapse를 늦출 수 있다.**

이 claim은 세 선행 기여를 포함한다고 주장하지 않는다. Downstream-aware target refinement는 MetaKE, causal-score 기반 residual allocation은 CAKE, evolving null-space와 low-rank sequential solve는 EvoEdit가 각각 직접 다룬다. ODE-BF가 검증해야 할 남은 기여는 이들을 다시 이름 붙이는 것이 아니라, **fixed Euler clock에서 갱신되는 multi-layer target–write field, H/P soft-risk joint routing, BF16-authoritative virtual transition, terminal atomic commit을 하나의 controller로 결합했을 때의 추가 가치**다.

## 0.1 현재 evidence와 pivot의 범위

현재까지 확인된 사실은 proposal과 완료된 method를 구분하게 한다.

| 관찰 | 현재 evidence | 설계에 미친 영향 |
|---|---|---|
| W64/BF16 transaction | Llama/Qwen B10에서 W64 certificate, virtual-vs-commit bytes/logits/event, rollback과 최종 \(W_0\) restore가 통과했다. N32와 W64는 bit-exact하지 않다. | W64를 production candidate로 유지하되 Native-equivalence claim은 금지한다. |
| Hard functional-P | Full-residual trajectory가 efficacy를 올리거나 Qwen에서 exact hit에 도달해도 임의의 \(10^{-3}\) P threshold가 terminal을 거부했다. | functional-P를 hard veto가 아니라 soft signal과 audit로 이동한다. |
| No-budget full-\(\tau\) | 현재 가장 강한 완료 결과인 ALLOFF full-\(\tau\)에서 Llama는 Eff 10/10, Gen 20/20, Loc 80/100으로 Native Loc 79/100을 유지·상회했고, Qwen은 10/10, 20/20, 80/100으로 Native와 동일했다. | 임의 H/P budget 통과 여부가 아니라 same-seal Eff/Gen/Loc와 absolute NLL을 primary scientific evidence로 사용한다. ALLOFF를 final method로 즉시 승격하지 않고 Neutral/Soft 비교의 실증적 기준점으로 둔다. |
| Adaptive retry | 동일 state retry와 \(\Delta\tau\) 축소가 trial 수와 replay 비용을 크게 늘렸고, cap/termination 해석도 복잡해졌다. | main design을 \(K=8,h=1/8\) fixed Euler로 단순화한다. |
| Warm start | Native-\(z\) displacement는 특히 Qwen에서 매우 컸고 initial field scale에 직접 들어갔다. Target-only state 자체는 functional-P의 입력이 아니다. | main은 \(z_{\mathrm{base}}\) cold start, Native-\(z\)는 control/ablation으로만 둔다. |
| Full residual과 routing | Legacy full-residual pilot의 online efficacy는 Llama 최대 9/10, Qwen 10/10까지 갔지만 terminal hard-P가 모두 막았다. 기록된 Llama initial routing은 5개 layer에 분산되어 collapse 증거가 없었다. | residual pre-sharing을 제거하고 stepwise routing vector를 필수 기록하되 scientific success로 승격하지 않는다. |
| Historical H | 현재 B10-1 실험의 active history는 0이라 structural/functional H가 vacuous했다. | H benefit은 B10-2 이후와 long stream에서만 검증한다. |
| Fixed-E8 solver | 최신 P1R7은 scientific endpoint 전에 수치 경계에서 멈췄다. Llama Neutral은 \(\tau=0.75\)에서 적용하지 않는 Soft shadow certificate 때문에 중단됐고, Qwen Neutral은 \(\tau=0.125\)에서 SLSQP status 8로 중단됐다. E8-SOFT와 terminal Native/Gen/Loc panel은 실행되지 않았다. | R7을 scientific result로 사용하지 않는다. Neutral authority와 diagnostic Soft shadow를 분리하고, backend status가 아니라 독립 certificate를 authoritative하게 만든 뒤 새 namespace에서만 재검증한다. |
| Target-field \(h\)-coupling | Proposal은 target field를 \(W_k+h\sum_l v_lB_l(z)\)에서 계산한다. 감사한 R7 구현은 \(W_k+\sum_l v_lB_l(z)\)에서 target field를 계산하고 실제 weight transition에만 \(h\)를 적용했다. | 동일 Euler candidate co-evolution claim 전 target probe와 physical write가 같은 \(h\)를 사용하는 source/receipt identity gate가 필요하다. |

Legacy structural/functional-P budget은 관찰값으로 calibration된 손상 경계가 아니므로 `budget 초과 = model damage`로 해석하지 않는다. ALLOFF에서 기록된 P 값 Llama 0.08233, Qwen 0.00250도 calibrated damage가 아니라 **raw observable**이다. Zero-write floor, 반복 측정 분산, Native-relative teacher-KL과 실제 Eff/Gen/Loc 관계를 함께 보기 전에는 절대 threshold로 성공·실패를 판정하지 않는다.

따라서 아래 문서에서 **현재 pilot**은 구현·기계적 증거를 뜻하고, **proposed final method**는 아직 solver gate와 sequential scientific gate를 통과해야 하는 설계를 뜻한다. MetaKE/CAKE/EvoEdit 비교와 대규모 sequential 결과 전에는 `ODE-BF Full`이라는 이름을 사용하지 않는다.

---

# 1. 문제 정의

Sequential edit index를 \(t\)라 하자.

- 최초 pretrained model: \(W^0\)
- edit \(t\) 직전 deployed model: \(W_{t-1}\)
- 새 edit request: \(e_t=(x_t,o_t^*)\)
- candidate rewrite layers: \(\mathcal L=\{l_1,\ldots,l_m\}\)
- 현재 유효한 committed historical edits: \(\mathcal H_t^{\mathrm{act}}\)

기존 multi-layer AlphaEdit 실행은 direct target을 먼저 구한 뒤, 정해진 layer order에 따라 residual을 나누어 write한다. 이 구조에는 네 가지 문제가 있다.

## 1.1 Target–write disconnect

Activation intervention에서는 효과적인 direct-\(z\)라도 실제 rewrite layer들이 이를 구현할 때는 큰 update norm, historical interference, pretrained drift를 요구할 수 있다.

즉,

$$
\text{semantic target feasibility}
\neq
\text{safe write feasibility}.
$$

## 1.2 Static layer allocation

동일한 residual share를 모든 edit에 적용하면 다음 차이를 반영하지 못한다.

- 현재 edit에 대한 layer별 rewrite efficiency
- 이전 edit key와의 충돌 정도
- pretrained key distribution disturbance
- layer별 누적 edit load

## 1.3 Proposal의 state dependence

한 Euler transition이 virtual하게 적용되면 이후 hidden state, key, residual, layer proposal이 달라진다. 초기 model에서 만든 proposal bank를 끝까지 재사용하면 현재 trajectory의 geometry를 반영하지 못한다.

## 1.4 Dense solve와 mutable history 문제

Stock AlphaEdit를 Euler grid마다 다시 호출하면 다음 비용과 오류가 발생한다.

- 거대한 \(d\times d\) matrix construction
- dense LU/solve 반복
- dense `cache_c` 유지
- rejected trial에서도 `cache_c += KK^\top`가 실행될 가능성

ODE-BF는 위 네 문제를 동시에 다루되, 실제 deployed parameter는 terminal에서 한 번만 변경한다.

## 1.5 Closest related work와 novelty boundary

현재 proposal과 가장 가까운 선행연구는 MetaKE, EvoEdit, CAKE다. 세 방법은 서로 다른 축을 이미 선점하므로, ODE-BF의 claim과 baseline은 이 축들을 분리해야 한다.

### MetaKE: downstream-aware target refinement

[MetaKE](https://arxiv.org/abs/2603.12677)는 target–write disconnect를 bi-level optimization으로 직접 다룬다. Inner problem은 AlphaEdit-style constrained write이고, outer problem은 실제 downstream write의 피드백을 받아 target representation을 갱신한다. Full multi-layer differentiation 대신 final edited layer의 frozen Structural Gradient Proxy를 사용해 virtual look-ahead를 수행한 뒤, 정제된 target을 standard multi-layer editor로 한 번 write한다.

따라서 ODE-BF는 다음을 novelty로 주장하지 않는다.

- downstream constraint를 target optimization에 되먹임하는 최초의 방법
- virtual write look-ahead 또는 write-aware target refinement 자체
- target–write disconnect의 최초 정식화

차이는 target-side feedback의 존재가 아니라 controller의 범위다. MetaKE는 final-layer Structural Gradient Proxy로 target을 먼저 정제하고 multi-layer solver를 최종 실행한다. ODE-BF는 fixed Euler state마다 모든 candidate layer의 key, layer-local full-residual arm과 routing geometry를 다시 만들고 target와 cumulative write를 함께 전이시키며, H/P structural·functional signal로 layer share를 바꾼다. 이 차이는 설계상 distinction이며, empirical superiority를 뜻하지 않는다.

### EvoEdit: evolving null-space와 lifelong sequential editing

[EvoEdit](https://arxiv.org/abs/2510.13851)는 fixed AlphaEdit projector의 null-space drift를 직접 다룬다. 새 committed key가 들어올 때 projected key direction을 SVD로 추출하여 projector를 deflation하고, prior key span을 점진적으로 null하도록 geometry를 갱신한다. 또한 hidden dimension의 dense solve 대신 edit-rank space의 low-rank/Woodbury-style solve를 사용하며, 2K–10K sequential edits에서 early-edit retention과 runtime을 보고한다.

ODE-BF의 현재 \(\Pi_l\)는 read-only static projector다. Historical projected-key registry와 risk factor가 증가해도 \(\Pi_l\) 자체를 EvoEdit처럼 진화시키지 않는다. 따라서 다음 claim은 금지한다.

- 최초의 evolving/dynamic null-space alignment
- online projector deflation 또는 prior-key output invariance
- sequential AlphaEdit의 최초 low-rank/Woodbury acceleration
- 10–100 edit pilot만으로 EvoEdit보다 강한 lifelong robustness를 입증했다는 주장

ODE-BF의 구분점은 projector novelty가 아니라 fixed pretrained geometry 위에서의 **dynamic safety control over candidate writes**다. 향후 EvoEdit projector를 채택하면 이를 ODE-BF의 원래 기여가 아니라 `EvoEdit-projector + ODE-BF-controller` hybrid로 명시하고, solver exactness와 H/P risk contract를 다시 검증한다.

### CAKE: causal-score adaptive residual allocation

[CAKE](https://aclanthology.org/2026.acl-long.918/)는 causal tracing으로 얻은 layer importance를 softmax weight로 바꾸어 residual allocation을 비균일화한다. Idealized constrained quadratic allocation을 제시하고, 실용 구현에서는 shallow-to-deep edit 중 현재 remaining residual을 다시 계산하며 아직 편집하지 않은 layer의 causal weight를 동적으로 재정규화한다. 즉 CAKE를 단순 fixed-residual method로 기술하면 안 된다. 다만 causal attribution weight 자체와 sequential coordinate update가 중심이며, ODE-BF처럼 매 virtual Euler state의 full joint field를 H/P risk로 다시 푸는 구조는 아니다. CAKE는 최대 5K sequential-edit 평가도 보고하므로 long-stream 비교에서 제외할 수 없다.

따라서 ODE-BF는 adaptive/non-uniform layer allocation, causal importance를 이용한 routing, current residual 재계산, 또는 residual reallocation 자체를 최초 claim으로 두지 않는다. 검증 대상은 **동일 full-residual arms에서 target-new-NLL progress와 structural/functional H/P soft risk로 joint routing을 다시 푸는가**다.

### 비교 요약

| 축 | MetaKE | EvoEdit | CAKE | ODE-BF가 검증할 추가 축 |
|---|---|---|---|---|
| Target–write feedback | final-layer structural proxy를 통한 bi-level target refinement | 주기여 아님 | 주기여 아님 | fixed-grid full multi-layer field와 target/write 공동 전이 |
| Layer allocation | final target을 standard multi-layer solver로 실행 | within-edit dynamic routing 없음 | causal weight + current residual 재계산 기반 sequential allocation | full-residual arms의 target-new-NLL/H/P joint routing |
| Historical geometry | AlphaEdit projector와 prior-key matrix | committed edit마다 evolving projector | AlphaEdit-style structural backend | static projector 아래 solve/risk registry와 functional history를 분리 |
| Pretrained covariance | core routing barrier가 아님 | initial projector geometry | AlphaEdit projector/statistics를 상속 | Wikipedia second moment를 explicit structural P soft-risk로 사용 |
| Candidate preservation | outer edit/locality objective | geometric null-space preservation | 별도 functional H/P routing 없음 | structural/functional H/P를 soft routing signal과 audit로 사용 |
| State/commit | target refinement 후 standard write | edit마다 model/projector 갱신 | sequential coordinate write | fixed-E8 virtual trajectory 후 terminal atomic commit |

따라서 방어 가능한 contribution은 다음 결합에 한정한다.

1. fixed Euler state마다 recompute되는 full multi-layer target–write field,
2. 모든 layer에 동일 full residual arm을 제공한 뒤 결정되는 constrained joint routing,
3. pretrained covariance를 projector가 아니라 explicit structural P-risk로 사용하는 H/P separation,
4. structural proxy와 별개인 functional replay,
5. BF16-authoritative virtual trajectory와 terminal transaction.

현재 CPU/P0/P1 pilot이 이 다섯 축을 모두 통과하기 전에는 구현을 **fixed-E8 soft-routing pilot**로 부르고 `ODE-BF Full` 또는 formal continuous-time CBF guarantee로 승격하지 않는다.

---

# 2. Scope와 claim 경계

## 2.1 보존 대상

### Historical knowledge \(H\)

- 성공적으로 commit된 과거 edit
- 과거 target margin
- active historical key subspace

### Pretrained knowledge \(P\)

- fixed Wikipedia second moment가 표현하는 pretrained key distribution
- unrelated/locality calibration population에서의 pretrained-teacher incremental drift

## 2.2 의도적으로 제외하는 범위

본 proposal의 method/controller는 historical edit와 pretrained/locality preservation만 다룬다. Claim은 다음으로 제한한다.

$$
\boxed{
\text{new-edit efficacy}
+
\text{historical retention}
+
\text{pretrained/locality preservation}
}
$$

따라서 성능 향상의 원인을 외부 anchor design이 아니라 cold target–write coupling과 dynamic layer routing에 귀속시킬 수 있다.

## 2.3 여기서 ODE가 의미하는 것

실제 model tensor를 매 step 갱신하는 것이 아니다. Continuous state는 virtual state다.

$$
S(s)=\bigl(z(s),\Delta W(s)\bigr),
$$

여기서

- \(z(s)\): direct-target state
- \(\Delta W(s)\): \(W_{t-1}\) 대비 누적 virtual low-rank update

이다.

Vector field는 다음처럼 state dependent하다.

$$
\dot z=F_z(z,\Delta W),
$$

$$
\dot W
=
\sum_{l\in\mathcal L}
 v_l^*(z,\Delta W)B_l(z,\Delta W).
$$

Proposed final method의 주 grid는

$$
\tau_k=\frac{k}{8},
\qquad
h=\frac18,
\qquad
k=0,\ldots,8
$$

로 고정한다. 실제 구현은 continuous solver가 아니라 **state-dependent vector field를 fixed-step explicit Euler로 이산화한 constrained flow**다. H/P는 soft routing signal이고 first-hit은 관찰 event이므로, 현재 design-validation trajectory에는 scientific accept/reject나 adaptive time-step이 없다.

따라서 현재 이름의 `Barrier-Filtered`는 H/P risk로 velocity를 shaping한다는 설계 계보를 나타내며, \(\dot h+\alpha(h)\ge0\) 형태의 formal continuous-time CBF safety guarantee나 H/P set invariance를 뜻하지 않는다. 논문 framing은 evidence가 쌓이기 전까지 `soft preservation-aware routing`을 우선한다.

ODE claim을 위해 다음 네 값을 분리해 기록한다.

- numerical time \(\tau_k\)
- fixed Euler step \(h\)
- target displacement와 weight displacement
- technical solver/backend invocation 수

Optimizer continuation 횟수를 Euler step 수로 세거나, first-hit index를 실제 조기 종료 시간으로 부르면 안 된다.

---

# 3. State와 transaction contract

## 3.1 Virtual Euler state

Euler grid \(k\)에서

$$
W_k=W_{t-1}+\Delta_k,
\qquad
S_k=(W_k,z_k),
\qquad
\Delta_0=0
$$

로 둔다.

\(W_k\)는 full model copy가 아니라 base model과 누적 low-rank hook의 합으로 평가한다.

## 3.2 Read-only static artifacts

Layer \(l\)마다 다음 artifact를 process 시작 시 한 번 load한다.

- Alpha projector \(\Pi_l\)
- pretrained non-centered second moment
  \[
  C_l^0=\mathbb E_{k\sim\mathcal D_0}[kk^\top]
  \]
- tokenizer, context template, rewrite-module metadata

Online edit path에서는 Wikipedia covariance, SVD, projector를 다시 계산하지 않는다.

## 3.3 Persistent mutable state

Persistent state는 다음을 포함한다.

- active historical solve factors
- weighted historical risk factors
- functional history metadata
- cumulative layer load
- subject–relation version metadata
- transaction ID와 history version

이 state는 terminal commit과 post-commit verification이 성공한 뒤에만 변경한다.

## 3.4 Transaction table

| Event | Virtual state | Deployed weights | Persistent history | Current field cache |
|---|---:|---:|---:|---:|
| E8 field/probe evaluation | 임시 생성 | 불변 | 불변 | 같은 grid에서 재사용 |
| E8 Euler transition | 다음 virtual state로 이동 | 불변 | 불변 | 다음 grid에서 무효화 |
| observation-only first hit | trajectory 계속 | 불변 | 불변 | 영향 없음 |
| \(\tau=1\) terminal audit | endpoint freeze | 불변 | 불변 | freeze |
| terminal commit | endpoint 고정 | 정확히 1회 변경 | 아직 불변 | freeze |
| post-commit verification 성공 | 종료 | 유지 | 정확히 1회 갱신 | 삭제 |
| post-commit verification 실패 | rollback | 원복 | 불변 | 삭제 |

이 transaction 경계가 무너지면 probe나 중간 Euler state가 deployed model/history를 바꾸므로 알고리즘 자체가 달라진다.

---

# 4. Cold-start target ODE

## 4.1 Outer edit당 하나의 cold target trajectory

Target layer의 outer-entry lookup activation을 \(z_{\mathrm{base}}\)라 하면 main method는

$$
\boxed{z_0=z_{\mathrm{base}}}
$$

에서 시작한다. Native `compute_z` endpoint는 main controller의 초기값, scale, radius, fallback에 들어가지 않는다. Native direct-\(z\)는 matched reference와 warm-start ablation으로만 별도 계산한다.

하나의 persistent \(z_k\)는 fixed Euler grid 전체에서 유지된다. Layer마다 또는 step마다 독립 direct-\(z\) solve를 다시 실행하지 않는다.

## 4.2 Target-new-NLL objective

Genuine batch \(\mathcal B_t\)의 target-new suffix token 집합을 \(\mathcal T_i\)라 하자. Controller objective는

$$
\boxed{
\Phi_{\mathrm{new}}(W)
=
-\frac1{|\mathcal B_t|}
\sum_{i\in\mathcal B_t}
\frac1{|\mathcal T_i|}
\sum_{q\in\mathcal T_i}
\log p_W(o^*_{i,q}\mid x_i,o^*_{i,<q})
}
$$

이다. CounterFact의 `new NLL < old NLL` success와 zsRE의 exact teacher-forcing bits는 **evaluation/event metric**이지 routing loss가 아니다. Target-old NLL, success margin, paraphrase, locality, H/P score를 \(\Phi_{\mathrm{new}}\)에 섞지 않는다.

이는 Native의 target likelihood 방향과 맞추면서, old-target term과의 cancellation 때문에 continuous progress는 생기지만 target-new likelihood가 충분히 오르지 않던 pilot 문제를 분리하기 위한 선택이다.

## 4.3 Zero-write target recovery

\(z_0=z_{\mathrm{base}}\)에서 모든 write arm이 0이면 target-only recovery를 수행한다. 다만 별도 adaptive bootstrap loop를 만들지 않는다.

$$
z_{k+1}=z_k+hF_z^{\mathrm{target-only}}(z_k),
\qquad h=\frac18.
$$

이 transition은 E8 grid 한 칸을 소비하며,

- deployed weight, history, sampler와 RNG를 변경하지 않고,
- first-hit 또는 scientific endpoint로 계산하지 않으며,
- retry/backtracking을 만들지 않는다.

다음 grid에서 nonzero field를 다시 만든다. Recovery 뒤에도 technical field contract를 만들 수 없으면 typed fail-close한다.

## 4.4 Layer-local write-aware target field

현재 state \((W_k,z_k)\)에서 layer \(l\)의 stored full residual을 \(R_{k,l}(z_k)\)라 한다. Target differentiation 중에는

$$
\boxed{
R_{k,l}(z)
=R_{k,l}(z_k)+(z-z_k)
}
$$

를 사용한다. 모든 layer가 자기 residual anchor를 유지해야 하며 terminal-layer `current_z` 하나를 공통 재사용하지 않는다.

Routing coefficient \(v_k^*\)를 stop-gradient로 둔 one-step overlay를

$$
\widehat W_k(z)
=W_k+h\sum_l v_{k,l}^*B_{k,l}(z)
$$

라 하면 target field는

$$
F_z(S_k)
=-G_z^{-1}\nabla_z
\Phi_{\mathrm{new}}\bigl(\widehat W_k(z_k)\bigr)
$$

이다. Pilot에서 \(G_z\)는 outer-entry에 고정된 preconditioner이며, target와 weight는 같은 \(h\)를 사용한다. H/P는 routing share를 정하는 soft signal이지만 target-new-NLL primary objective를 대체하지 않는다.

### 4.4.1 Target-field Euler-coupling conformance gate

Proposal의 authoritative semantics는 위 식처럼 target field와 physical write가 **동일한 Euler displacement**를 보는 것이다. 다음 구현은 허용하지 않는다.

$$
\widehat W_k^{\mathrm{wrong}}(z)
=W_k+\sum_l v_{k,l}^*B_{k,l}(z),
\qquad
W_{k+1}=W_k+h\sum_l v_{k,l}B_{k,l}.
$$

이 경우 \(h=1/8\)에서 target field가 관찰하는 candidate displacement는 실제 weight step보다 8배 크며, target와 weight의 same-step co-evolution claim이 깨진다. 감사한 R7 implementation은 이 불일치를 포함했으므로 scientific evidence로 사용하지 않는다. 다음 run 전 source gate와 runtime receipt는 다음 identity를 증명해야 한다.

$$
\boxed{
\Delta W_k^{\mathrm{target\ probe}}
=
\Delta W_k^{\mathrm{physical\ trial}}
=
h\sum_l v_{k,l}B_{k,l}
}
$$

Target probe의 \(h\)를 바꾸는 동시에 routing coefficient, target gradient 또는 actual write에서 다시 보상하여 이중 scaling하는 것도 금지한다.

## 4.5 Fixed Euler transition

Main trajectory는

$$
\tau_k=\frac{k}{8},
\qquad
k=0,\ldots,8
$$

이고

$$
z_{k+1}=z_k+hF_z(S_k),
$$

$$
W_{k+1}^{\mathrm{virt}}
=W_k^{\mathrm{virt}}
+h\sum_l v_{k,l}B_{k,l}
$$

로 전진한다. Target displacement와 write displacement는 같은 grid step을 공유한다. Scientific metric이 나쁘다는 이유로 \(h\)를 줄이거나 같은 state에서 다시 시도하지 않는다.

## 4.6 First-hit은 observation-only

Design validation에서는 benchmark별 exact event가 처음 성립한

$$
k_{\mathrm{hit}}
=\min\{k:e_k=1\}
$$

을 기록하되, 그 지점에서 멈추지 않고 \(k=8\)까지 진행한다. First-hit prefix, persistence, hit 이후 efficacy/generalization/locality/H/P 변화와 terminal endpoint를 모두 비교한다.

향후 evidence가 쌓인 뒤 `immediate first-hit stop`을 compute-saving ablation으로 다시 평가할 수 있다. 현재 main의 first-hit은 step, routing, commit, target velocity에 영향을 주지 않는다.

---

# 5. State-dependent layer proposals

## 5.1 각 Euler grid state에서 field 재계산

Grid state \(S_k\)에서 모든 candidate layer에 대해 다음을 다시 계산한다.

- current key \(K_{k,l}\)
- current residual \(R_{k,l}(z_k)\)
- Alpha input-side factor \(Q_{k,l}\)
- raw low-rank proposal \(B_{k,l}\)
- rewrite efficiency \(a_{k,l}\)
- structural H/P coefficients

Functional probe와 optimizer backend continuation에서는 위 field 전체를 재사용한다. 다음 field는 Euler transition이 완료된 뒤에만 만든다.

## 5.2 Raw full-residual Alpha arm

Weight orientation을

$$
W_l\in\mathbb R^{d_{\mathrm{out}}\times d_{\mathrm{in}}}
$$

로 두고

$$
K_{k,l}\in\mathbb R^{d_{\mathrm{in}}\times b},
\qquad
R_{k,l}\in\mathbb R^{d_{\mathrm{out}}\times b}
$$

라 하자.

Layer \(l\)의 Alpha input-side system은

$$
\left[
\lambda I
+
\Pi_l
\left(
K_{k,l}K_{k,l}^{\top}
+C_{t,l}^{H,\mathrm{solve}}
\right)
\right]Q_{k,l}
=
\Pi_lK_{k,l}
$$

이다.

Raw proposal은

$$
B_{k,l}
=
R_{k,l}Q_{k,l}^{\top}
$$

이다.

Single request에서는 rank 1이고, batch size가 \(b\)이면 rank는 최대 \(b\)다.

Native처럼 residual을 \(1/(m-i)\)로 미리 나누지 않는다. 각 layer는 자기 layer-local current full residual에 대한 arm을 만들고, 실제 share는 routing coefficient \(v_l\)가 결정한다.

## 5.3 Rewrite efficiency

Target-new-NLL loss를 \(\Phi_{\mathrm{new}}\)라 하면 layer \(l\)의 signed first-order progress는

$$
a_{k,l}
=
-D\Phi_{\mathrm{new}}(W_k)[B_{k,l}]
$$

이다.

JVP/VJP 또는 low-rank virtual forward로 계산할 수 있다. \(a_{k,l}\le0\)인 arm은 outcome을 본 뒤 clipping해서 숨기지 않고 active set에서 명시적으로 제외하며 mask를 기록한다.

## 5.4 Joint routing

모든 layer arm은 같은 grid state에서 계산하고 동시에 최적화한다. Shallow-to-deep, deep-to-shallow coordinate routing은 ablation으로만 둔다.

이를 통해 coefficient search가 다시 layer-order path dependence를 만드는 것을 방지한다.

---

# 6. Low-rank Woodbury Alpha solve와 W64 contract

## 6.1 Historical factorization

Active historical key matrix를

$$
H_{t,l}
=
[K_{1,l}^{H},\ldots,K_{r,l}^{H}]
$$

라 하면

$$
C_{t,l}^{H,\mathrm{solve}}
=
H_{t,l}H_{t,l}^{\top}
$$

이다.

Current key와 history를 합쳐

$$
X_{k,l}=[K_{k,l},H_{t,l}]
$$

로 두면 Alpha matrix는

$$
A_{k,l}
=
\lambda I
+
\Pi_lX_{k,l}X_{k,l}^{\top}
$$

가 된다.

다음을 정의하자.

$$
Z_{k,l}=
\Pi_lX_{k,l},
\qquad
G_{k,l}=
\Pi_lK_{k,l}.
$$

\(\Pi_l\)가 symmetric idempotent projector이면

$$
X_{k,l}^{\top}\Pi_lX_{k,l}
=
Z_{k,l}^{\top}Z_{k,l}.
$$

Woodbury identity에 의해

$$
\boxed{
Q_{k,l}
=
\lambda^{-1}
\left[
G_{k,l}
-
Z_{k,l}
\left(
\lambda I+Z_{k,l}^{\top}Z_{k,l}
\right)^{-1}
Z_{k,l}^{\top}G_{k,l}
\right]
}
$$

이다.

따라서 다음이 필요 없다.

- dense \(C_l^H\)
- dense \(\Pi_lC_l^H\)
- \(d\times d\) LU factorization
- wide RHS dense solve

## 6.2 Projected-key registry

Commit된 active edit마다

$$
Z_{i,l}^{H}=
\Pi_lK_{i,l}^{H}
$$

를 terminal commit 이후 정확히 한 번 계산해 저장한다.

Alpha RHS와 small Gram은 projected historical key만으로 계산할 수 있다. Main implementation은 dense `cache_c` 대신 이 factor registry를 사용한다.

Projector exactness를 확인하기 위해

$$
\epsilon_{\mathrm{sym},l}
=
\frac{
\|\Pi_l-\Pi_l^{\top}\|_F
}{
\|\Pi_l\|_F
},
$$

$$
\epsilon_{\mathrm{idemp},l}
=
\frac{
\|\Pi_l^2-\Pi_l\|_F
}{
\|\Pi_l\|_F
}
$$

를 기록한다.

Tolerance를 넘으면 raw key도 저장하고 \(Z^{\top}Z\) 대신 정확한 \(X^{\top}Z\)를 사용하는 fallback을 둔다.

## 6.3 Cached history Cholesky

한 outer edit 동안 committed history는 immutable하다. 따라서

$$
G_{H,l}
=
\lambda I
+
(Z_{t,l}^{H})^{\top}Z_{t,l}^{H}
$$

의 Cholesky factor를 edit 시작 시 한 번 cache할 수 있다.

Euler grid에서 바뀌는 것은 current key block뿐이므로 block Cholesky 또는 Schur complement로 small system을 갱신한다.

## 6.4 Full proposal materialization 제거

$$
B_{k,l}=R_{k,l}Q_{k,l}^{\top}
$$

이므로 hidden vector \(h\)에 대한 action은

$$
B_{k,l}h
=
R_{k,l}
(Q_{k,l}^{\top}h)
$$

이다.

Virtual hook은 \((R_{k,l},Q_{k,l},v_{k,l})\)만 저장한다. Probe/transition 중 full \(d_{\mathrm{out}}\times d_{\mathrm{in}}\) update matrix를 만들지 않는다.

여러 Euler update가 누적되면

$$
W_{k,l}^{\mathrm{virt}}h
=
W_{t-1,l}h
+
\sum_{j<k}
 hv_{j,l}R_{j,l}
(Q_{j,l}^{\top}h)
$$

로 평가한다.

## 6.5 Exactness claim의 범위

Woodbury solve는 동일한

- projector
- current/history key
- residual
- regularization
- numerical precision

을 사용할 때 proposal이 정의한 per-layer dense Alpha system과 수학적으로 동일하다. Production candidate는 reduced system을 FP64로 풀고 동일한 assembler를 거쳐 BF16 endpoint를 만드는 `W64` path다.

다만 전체 ODE-BF trajectory는 stock AlphaEdit와 동일하지 않다. ODE-BF는 의도적으로

- fixed grid state마다 proposal을 다시 계산하고,
- dynamic routing을 사용하며,
- obsolete active history를 제거할 수 있고,
- target와 write를 coupled하게 진화시킨다.

따라서 **exactness는 수학적 per-layer system과 W64 virtual/commit transaction에 대한 claim**이다. Association order가 다른 N32/D32/W32/W64가 같은 BF16 bytes를 낸다는 claim도, W64가 Native endpoint와 bit-exact하다는 claim도 하지 않는다.


---

# 7. Historical structural preservation

## 7.1 Solve factor와 risk factor 분리

Alpha solve history와 routing history는 목적이 다르므로 별도로 유지한다.

### Solve factor

$$
Z_{t,l}^{H,\mathrm{solve}}
=
[\Pi_lK_{i,l}]_{i\in\mathcal H_t^{\mathrm{act}}}
$$

- active committed key를 unweighted로 보관
- exact Alpha linear system 정의
- obsolete version은 active set에서 제거

### Risk factor

$$
Z_{t,l}^{H,\mathrm{risk}}
=
\left[
\sqrt{\omega_i/Z_t}
\,\Pi_lK_{i,l}
\right]_{i\in\mathcal H_t^{\mathrm{act}}}
$$

- recency, reservoir, 중요도 weighting 적용 가능
- dynamic routing의 historical interference 계산에 사용

두 factor를 하나로 합치면

- weighted solve를 사용해 native-equivalent linear system을 잃거나,
- unweighted risk로 인해 recent/reservoir policy가 사라지는

문제가 생긴다.

## 7.2 Per-layer historical risk

Raw proposal \(B_{k,l}\)의 historical disturbance는

$$
c_{H,k,l}
=
\operatorname{tr}
\left(
B_{k,l}
C_{t,l}^{H,\mathrm{risk}}
B_{k,l}^{\top}
\right)
$$

이다.

Projected factor를 사용하면

$$
c_{H,k,l}
=
\left\|
B_{k,l}
Z_{t,l}^{H,\mathrm{risk}}
\right\|_F^2
$$

이다. Alpha input factor \(Q_{k,l}\)는 \(\Pi_l\)의 range에 있으므로

$$
Q_{k,l}^{\top}K_{i,l}
=
Q_{k,l}^{\top}\Pi_lK_{i,l}
$$

가 성립하며, historical risk 역시 projected key만으로 정확히 계산할 수 있다.

$$
B_{k,l}=R_{k,l}Q_{k,l}^{\top}
$$

이므로 실제 계산은

$$
c_{H,k,l}
=
\left\|
R_{k,l}
\left(
Q_{k,l}^{\top}
Z_{t,l}^{H,\mathrm{risk}}
\right)
\right\|_F^2
$$

로 수행한다.

Full \(B_l\) 또는 dense \(C_l^H\)를 만들 필요가 없다.

## 7.3 Cumulative historical risk

현재 step의 \(h^2v_l^2c_{H,l}\)만 보면 이전 Euler update와의 cross term을 놓친다.

Outer edit 시작점 대비 현재 cumulative update를 \(\Delta_{k,l}\)라 두고

$$
S_{H,k,l}
=
\Delta_{k,l}
Z_{t,l}^{H,\mathrm{risk}},
$$

$$
T_{H,k,l}
=
B_{k,l}
Z_{t,l}^{H,\mathrm{risk}}
$$

를 정의한다.

Applied coefficient \(\theta_l=hv_l\)에 대한 cumulative structural risk는

$$
\begin{aligned}
\mathcal R_H^{\mathrm{str}}(\theta)
&=
\sum_l
\|S_{H,k,l}+\theta_lT_{H,k,l}\|_F^2
\\
&=
 d_{H,k}
+2g_{H,k}^{\top}\theta
+\theta^{\top}M_{H,k}\theta
\end{aligned}
$$

이다.

여기서

$$
d_{H,k}
=
\sum_l\|S_{H,k,l}\|_F^2,
$$

$$
(g_{H,k})_l
=
\langle
S_{H,k,l},T_{H,k,l}
\rangle_F,
$$

$$
M_{H,k}
=
\operatorname{diag}
\left(
\|T_{H,k,1}\|_F^2,
\ldots,
\|T_{H,k,m}\|_F^2
\right).
$$

Main routing은 이 cumulative form의 **positive incremental risk**를 normalization한 soft score로 사용한다. 관찰로 calibration되지 않은 \(B_H^{\mathrm{str}}\)를 scientific hard veto로 두지 않는다.

---

# 8. Pretrained structural preservation

## 8.1 Fixed pretrained geometry

Pretrained key distribution의 non-centered second moment를

$$
C_l^0
=
\mathbb E_{k\sim\mathcal D_0}[kk^{\top}]
$$

라 한다.

Proposal \(B_{k,l}\)의 평균 local disturbance는

$$
c_{P,k,l}
=
\operatorname{tr}
\left(
B_{k,l}C_l^0B_{k,l}^{\top}
\right)
$$

이고

$$
c_{P,k,l}
=
\mathbb E_{k\sim\mathcal D_0}
\|B_{k,l}k\|_2^2
$$

이다.

이는 임의의 소수 anchor가 아니라 pretrained key distribution 전체에 대한 평균 local disturbance approximation이다.

## 8.2 Low-rank covariance cost

$$
B_{k,l}=U_{k,l}V_{k,l}^{\top}
$$

이면

$$
\boxed{
\operatorname{tr}
(B_{k,l}C_l^0B_{k,l}^{\top})
=
\operatorname{tr}
\left[
(U_{k,l}^{\top}U_{k,l})
(V_{k,l}^{\top}C_l^0V_{k,l})
\right]
}
$$

이다.

Rank 1이면 \(C_l^0v_l\) matvec 한 번으로 계산된다. Current field에서 얻은 scalar risk는

- QCQP coefficient 변경
- neutral/soft routing comparison
- numerical backend continuation
- functional probe comparison

동안 그대로 재사용한다.

## 8.3 Cumulative pretrained risk

Historical risk와 동일하게

$$
\mathcal R_P^{\mathrm{str}}(\theta)
=
 d_{P,k}
+2g_{P,k}^{\top}\theta
+\theta^{\top}M_{P,k}\theta
$$

로 둔다.

Main low-compute controller에서는 layer-local covariance risk를 사용하므로 \(M_{P,k}\)는 diagonal이다. Cumulative update \(\Delta_{k,l}\)도 Euler-step low-rank factor list로 유지하므로 \(d_{P,k}\)와 \(g_{P,k}\)는 factor Gram과 cached \(C_l^0v\) product만으로 계산하고 full \(\Delta_{k,l}\)를 materialize하지 않는다. 이 값도 heuristic \(B_P^{\mathrm{str}}\) hard veto가 아니라 normalized soft score다.

Stronger ablation에서는 candidate layer가 \(m\)개일 때 actuator-space Fisher/JVP Gram

$$
(M_{P,k}^{F})_{lr}
=b_{k,l}^{\top}F_0b_{k,r}
$$

을 사용할 수 있다. 이는 \(m\times m\) matrix만 필요하며 layer 간 상쇄와 증폭을 반영한다. 다만 초기 main table에는 필수로 넣지 않는다.

## 8.4 Normalization geometry 분리

Main method는 raw Alpha arm을 사용한다.

동일한 \(C_l^0\)로

$$
\widehat B_l
=
\frac{B_l}
{\sqrt{
\operatorname{tr}(B_lC_l^0B_l^{\top})
}}
$$

처럼 normalize한 뒤 다시 같은 geometry를 risk로 사용하면

$$
\operatorname{tr}
(\widehat B_lC_l^0\widehat B_l^{\top})
=1
$$

이 되어 layer별 pretrained routing signal이 사라진다.

따라서 다음 계약을 둔다.

> **Proposal normalization metric과 preservation-risk metric은 동일하게 두지 않는다.**

Numerical reparameterization이 필요하다면 \(v_l\), progress coefficient, structural coefficient를 모두 일관되게 변환해야 한다.

---

# 9. Dynamic layer routing

## 9.1 Capacity objective

Layer \(l\)의 outer edit 시작 전 committed cumulative load를 \(\Omega_{t-1,l}\), 현재 virtual trajectory에서 이미 누적된 temporary load를 \(\omega_{k,l}\)라 한다.

$$
\omega_{k,l}
=
\sum_{j<k}
 h^2v_{j,l}^2\|B_{j,l}\|_F^2.
$$

Routing cost를

$$
Q_{k,ll}
=
(1+\Omega_{t-1,l}+\omega_{k,l})
\left(
\|B_{k,l}\|_F^2+\epsilon_Q
\right)
$$

로 둔다.

Low-rank arm의 Frobenius norm은 full matrix를 만들지 않고

$$
\|B_{k,l}\|_F^2
=
\operatorname{tr}
\left[
(R_{k,l}^{\top}R_{k,l})
(Q_{k,l}^{\top}Q_{k,l})
\right]
$$

로 계산한다.

Progress constraint에 이미 \(a_{k,l}\)가 들어가므로 rewrite efficiency가 높은 layer는 같은 progress를 더 작은 \(v_l\)로 제공할 수 있다. Capacity objective는 low-norm이며 committed/virtual load가 낮은 layer 조합을 선호한다.

## 9.2 Soft structural/functional risk

Structural component \(j\in\{H_{\mathrm{str}},P_{\mathrm{str}}\}\)의 positive incremental risk를

$$
r_{j,k}^{\mathrm{str}}(v)
=
\frac{
\left[
2h g_{j,k}^{\top}v
+h^2v^{\top}M_{j,k}v
\right]_+
}{s_{j,k}^{\mathrm{str}}+\epsilon}
$$

로 둔다. \(s_{j,k}^{\mathrm{str}}\)는 locked trace/Gram scale이며 outcome을 보고 바꾸지 않는다.

Functional component \(j\in\{H_{\mathrm{func}},P_{\mathrm{func}}\}\)는 current baseline endpoint와 각 layer 단독 probe를 이용한다.

$$
D_{j,k}^{0}=D_j(W_k),
\qquad
D_{j,k,l}^{1}=D_j(W_k+hB_{k,l}),
$$

$$
\delta_{j,k,l}
=
\left[D_{j,k,l}^{1}-D_{j,k}^{0}\right]_+.
$$

Pilot의 additive functional surrogate는

$$
\widehat r_{j,k}^{\mathrm{func}}(v)
=
\frac{\sum_l v_l\delta_{j,k,l}}
{s_{j,k}^{\mathrm{func}}+\epsilon}
$$

이다. Baseline endpoint는 별도 baseline method가 아니라 **현재 state 대비 incremental damage를 centering하기 위한 endpoint**다. Historical set이 비어 있으면 H component는 inactive이며 H probe를 추가하지 않는다.

Functional interaction은 이 basis가 정확히 표현한다고 주장하지 않는다. Candidate와 terminal audit에서 실제 joint damage를 별도로 기록하고, basis–actual calibration이 실패하면 functional slope를 routing signal로 쓰는 claim을 제거한다.

## 9.3 Main lexicographic routing problem

현재 active normalized soft-risk 집합을 \(\mathcal J_k\)라 하고

$$
\xi_k(v)
=
\max_{j\in\mathcal J_k}r_{j,k}(v)
$$

라 한다. 먼저 box와 technical trust 아래의 maximum feasible progress \(p_k^{\max}\)를 구한다. Main soft arm은

$$
\begin{aligned}
\min_{v,\xi}\quad&\xi\\
\text{s.t.}\quad&
a_k^{\top}v\ge\kappa p_k^{\max},
\qquad \kappa=0.25,
\\
&r_{j,k}(v)\le\xi,
\quad j\in\mathcal J_k,
\\
&h^2\sum_l v_l^2\|B_{k,l}\|_F^2
\le R_{W,k}^2,
\\
&0\le v_l\le1
\end{aligned}
$$

을 푼 뒤, \(\xi\le\xi_k^*+\epsilon_\xi\) 안에서

$$
\min_v\frac12v^{\top}Q_kv
$$

를 푼다. H/P risk에는 hard budget이 없고, progress와 physical write trust만 hard constraint다. `E8-NEUTRAL` control은 같은 **arm-local current field construction**, progress와 trust 아래에서 H/P soft objective를 제거하고 capacity만 최소화한다. Neutral과 Soft는 동일 initial state와 seal에서 시작하지만 첫 transition 뒤 state가 갈라질 수 있으므로, 전체 trajectory가 계속 같은 field를 공유한다고 표현하지 않는다.

Main method에서 \(v_l\ge0\)를 사용하는 이유는 active raw arm이 current residual을 줄이는 방향으로 정의되기 때문이다. Negative coefficient를 허용하는 signed routing은 별도 ablation이다.

## 9.4 Hard contract와 soft signal의 경계

### Hard technical/method contract

- target-new-NLL predicted progress fraction
- \(0\le v_l\le1\)
- physical BF16 write trust
- finite W64 factors와 solver certificate
- exact virtual transaction/rollback/restore

### Soft scientific signal

- historical structural disturbance
- pretrained covariance disturbance
- historical functional margin damage
- pretrained teacher-KL increment

이 네 값은 routing priority와 audit를 바꾸지만 candidate를 reject하거나 \(\tau\)를 소비하지 않게 만들지 않는다. 외부 deployment safety policy를 별도로 둘 수 있으나 이를 ODE-BF mechanism efficacy로 해석하지 않는다.

## 9.5 Numerical solver policy — unresolved

각 field의 mathematical QCQP 수와 backend invocation 수를 구분한다. Numerical polish/continuation은 같은 objective·constraint·tolerance를 유지하는 backend operation일 뿐 새로운 Euler step, scientific retry 또는 후보가 아니다.

현재 SLSQP pilot은 최종 solver policy를 확정하지 못했다.

- Llama fixed-E8 step 6: 같은-QP continuation 4회 후에도 stationarity \(1.71\times10^{-4}\)
- Qwen fixed-E8 step 1: 세 pass 모두 SLSQP status 8이었다. 당시 lock이 SciPy `success`를 요구했으므로 R7은 실패로 유지한다.

따라서 tolerance 완화, status 무시, backend continuation 증가를 조용히 main method로 채택하지 않는다. Final policy 후보는 finite/primal/stationarity/complementarity로 구성된 **독립 certificate**를 authoritative하게 두고 backend status를 diagnostic으로 남기는 것이다. `E8-NEUTRAL`의 certified solve는 observation-only soft shadow와 typed하게 분리하여 shadow failure가 neutral candidate를 막지 않게 한다. Independent certificate가 실제로 실패할 때만 다른 backend를 한 번 fallback으로 호출한다. Backend·version·option·constraint order·analytic derivative를 잠그고 dry-repeat identity가 통과한 경우에만 이를 deterministic이라고 부른다. 다음 raw-free receipt에는 작은 coefficient vector, ordered slacks와 KKT multipliers를 포함하여 certificate를 사후 재구성할 수 있게 한다. 이 정책은 새 lock과 regression을 거친 뒤에만 적용하며 R7을 소급 승격하지 않는다. 현 numerical lock이 허용하는 최대 104 backend invocations/arm은 low-overhead final contract가 아니라 진단 상한이다.

Solver가 locked certificate를 내지 못하면 해당 run은 technical fail-close한다. \(p\), \(h\), H/P score, tolerance를 outcome에 맞춰 줄이거나 같은 state에서 scientific retry하지 않는다.

---

# 10. Functional H/P soft routing과 audit

Structural H/P는 local proxy이며 실제 behavior 보존의 충분조건이 아니다. Functional H/P는 proxy를 보완하지만, 현재 main에서는 heuristic threshold로 scientific trajectory를 중단하지 않는다.

## 10.1 Active historical replay set

Functional history에는 다음 edit만 포함한다.

- outer edit가 성공적으로 terminal commit됨
- 현재 subject–relation의 active version임
- 현재 outer edit 시작점 \(W_{t-1}\)에서도 success 상태임

이미 실패한 fact는 controller hard gate가 아니라 repair/audit queue로 보낸다. Benchmark held-out paraphrase는 controller에 넣지 않고 action freeze 이후 evaluation에만 사용한다.

## 10.2 Historical functional damage

Historical edit \(i\)의 target margin을

$$
m_i(W)
=
\log p_W(o_i^*\mid x_i)
-
\max_{o\in\mathcal N_i}\log p_W(o\mid x_i)
$$

라 하고 outer-entry 기준 positive damage를

$$
d_{H,i}(W)
=
\left[m_i(W_{t-1})-m_i(W)\right]_+
$$

로 둔다. Recent/reservoir의 mean, smooth-max, raw-max와 충분한 표본에서의 CVaR를 모두 기록한다. 작은 history에서 tail count가 2 미만이면 CVaR를 decision-like summary로 사용하지 않는다.

Current soft score는 fixed-size recent/reservoir sample에서 계산한 damage sensitivity다. Full active history는 terminal/post-commit audit에만 사용할 수 있으며, 그 비용은 별도로 보고한다.

## 10.3 Obsolete target 처리

동일 subject–relation에 새 value가 commit되면 obsolete target은 다음 세 곳에서 함께 제거한다.

1. functional active history
2. historical solve factor
3. historical risk factor

단, multi-valued relation은 benchmark의 overwrite/version semantics를 따른다. Rejected probe나 virtual Euler state는 history를 append/remove하지 않는다.

## 10.4 Pretrained functional drift

고정된 소수 사례 자체가 아니라 population, seed, strata, edit/step별 sample order를 seal한다. Outer edit \(t\)의 incremental drift는 fixed outer-entry를 기준으로

$$
\begin{aligned}
D_P^{\mathrm{func}}(W)
=
\mathbb E_x
\Bigg[
&\operatorname{KL}
\left(p_{W^0}(\cdot\mid x)\Vert p_W(\cdot\mid x)\right)
\\
&-
\operatorname{KL}
\left(p_{W^0}(\cdot\mid x)\Vert p_{W_{t-1}}(\cdot\mid x)\right)
\Bigg]_+
\end{aligned}
$$

로 둔다. Waypoint마다 baseline을 현재 \(W_k\)로 이동시키지 않는다. Teacher logits와 outer-entry baseline KL은 cache할 수 있다.

Scientific run 전에 동일한 outer-entry weight를 write 없이 반복 평가하여 functional-P의 numerical/no-op floor를 측정한다. 각 step receipt는 최소한 다음을 함께 저장한다.

- raw functional-P,
- repeated raw functional-P,
- zero-write floor,
- floor-corrected signed value,
- floor-corrected positive-part value.

이 값들은 measurement calibration과 Native-relative teacher-KL 분석을 위한 observation이다. 이번 main에서 routing, candidate acceptance, clock, first-hit 또는 endpoint를 바꾸는 hard threshold로 사용하지 않는다.

## 10.5 Pilot functional basis와 48 endpoint의 의미

Candidate layer가 5개일 때 한 field의 diagnostic basis는

- current-state baseline 1개
- single-layer endpoint 5개

로 총 6개다. E8 전체에서는

$$
(1+5)\times8=48
$$

functional basis endpoints/arm이 된다. 이 48은 final endpoint 수나 48번의 edit가 아니라 layer별 functional slope를 분리하기 위한 **diagnostic pilot cost**다.

Baseline endpoint는 \(D_j(W_k)\)를 재어 probe의 incremental damage \(D_j(W_k+hB_l)-D_j(W_k)\)를 center한다. 같은 state, sample order와 teacher cache를 공유하면 field당 baseline은 한 번만 계산한다.

Historical token cost는 active history 전체를 매 endpoint에서 열면 edit 수와 함께 증가한다. Main scalable contract는 controller history를 fixed-size recent \(R_H\)+reservoir \(S_H\)로 제한한다. Full-history audit는 terminal에서만 수행하고 별도 lifelong cost로 보고한다.

## 10.6 Bounded final replay contract

48-endpoint pilot은 functional slope가 routing에 유효한지 확인하기 위한 상한이다. Production candidate는 다음 중 하나가 matched pilot과 충분히 일치할 때만 채택한다.

1. 5개 layer probes를 하나의 batched functional forward 또는 JVP/VJP로 계산,
2. field마다 structural soft score만 사용하고 selected joint candidate 1개에만 functional feedback,
3. functional basis를 presealed periodic grid에서만 갱신하고 중간 step에서는 cache.

고정 \(R_H,S_H,n_P\)와 cached baseline을 전제로 final per-arm functional endpoint ceiling은

$$
N_{\mathrm{func-endpoint}}
\le
8\,(1+N_{\mathrm{joint-feedback}})+N_{\mathrm{terminal-audit}},
$$

로 잠근다. Candidate-only contract이면 \(N_{\mathrm{joint-feedback}}=1\)이므로 E8 online endpoint는 최대 16개이고, terminal audit를 더한 값이 전체 상한이다. Batched/JVP contract는 endpoint-equivalent 수와 실제 model forward 수를 모두 기록한다.

어느 approximation도 48-endpoint diagnostic과 layer ordering, efficacy, H/P frontier가 맞는지 확인하기 전에는 조용히 main으로 승격하지 않는다.

---

# 11. Joint waypoint transition

Grid state \(S_k=(W_k,z_k)\)에서 다음 순서로 진행한다.

1. current layer-local full-residual field 계산
2. common functional basis와 structural score 계산
3. routing QCQP로 \(v_k\) 산출
4. routed virtual write를 통과해 target field \(F_z(S_k)\) 계산
5. \(h=1/8\) joint Euler transition을 정확히 한 번 적용
6. 모든 stepwise metric과 routing telemetry 기록

Write step은

$$
\Delta W_k^{\mathrm{step}}
=
h\sum_l v_{k,l}B_{k,l}
$$

이다.

Explicit Euler이므로 현재 \(z_k\)에서 계산한 field가 current write step을 만든다. \(W\)와 \(z\)가 함께 다음 state로 이동한 뒤 field를 다시 계산한다. \(\beta\) backtracking, same-state scientific retry와 adaptive \(\Delta\tau\)는 main에 없다.

## 11.1 Step contract

각 grid는 다음 중 하나로 정확히 한 번 종료된다.

1. certified routed Euler transition,
2. zero-write target recovery transition,
3. technical fail-close.

Efficacy, generalization, locality, H/P damage가 나쁘다는 이유로 동일 grid를 다시 풀거나 clock을 되돌리지 않는다. H/P soft score가 routing을 바꾸는 효과는 initial-state same-field coefficient 비교와, 이후 각 arm의 own-state field를 사용하는 same-seal matched-schedule trajectory 비교로 분리한다.

## 11.2 Terminal endpoint와 observation events

Design-validation terminal은

$$
\boxed{\tau_8=1}
$$

의 endpoint다. Exact first-hit, Native floor, structural/functional H/P damage, gen/loc 변화는 각 step의 observation으로 기록한다. 이들 observation은 현재 trajectory 선택이나 terminal index를 바꾸지 않는다.

---

# 12. Transactional one-shot commit

## 12.1 Commit protocol

\(\tau=1\) endpoint와 action-freeze 이후 다음 순서로 commit한다.

1. terminal low-rank endpoint freeze
2. layer별 accepted factor aggregate
3. affected deployed weight만 snapshot
4. cumulative update를 실제 model에 정확히 한 번 적용
5. exact post-commit parameter bytes/logits/event와 audit metric 재측정
6. technical identity가 실패하면 weight rollback, history 불변
7. technical identity가 성공하면 persistent state를 정확히 한 번 갱신

Efficacy/H/P audit가 나쁘다는 사실은 scientific result로 남기되, heuristic threshold로 virtual endpoint를 바꾸지 않는다. 실제 deployment에서 별도 release policy를 적용할 수 있으나 method trajectory와 분리한다.

## 12.2 History append protocol

Post-commit verification이 성공한 뒤에만 다음을 수행한다.

- final committed model에서 key 재계산
- dataset semantics에 따라 obsolete version 제거
- projected key를 `H_solve`에 append
- weighted factor와 metadata를 `H_risk`에 append
- canonical fact를 functional history에 append
- cumulative layer load \(\Omega_l\) 갱신
- transaction ID 저장

Functional probe와 virtual Euler waypoint는 stock

```python
cache_c += K @ K.T
```

경로를 절대 호출하면 안 된다.

## 12.3 Exactly-once semantics

각 outer edit에는 unique transaction ID를 부여한다. Finalization retry가 발생해도 동일 ID가 이미 commit되었다면 history를 다시 append하지 않는다.

History update는 pre-edit history version에 대한 compare-and-swap 형태로 처리한다.

---

# 13. 전체 알고리즘

```text
INPUT
  deployed model W_{t-1}
  genuine B10 edit request batch E_t
  candidate layers L
  read-only projector Π_l and pretrained second moment C_l^0
  active projected-key registry H_solve / H_risk
  functional history
  pretrained sampling contract

BEGIN TRANSACTION
  deployed weights와 persistent-history version freeze
  Δ_0 = 0
  z_0 = current lookup activation
  virtual low-rank hooks 초기화
  h = 1/8, tau_0 = 0

FOR k = 0, ..., 7

  # 현재 Euler state에서 field 생성
  current virtual key K_{k,l} 계산
  layer-local full residual R_{k,l}(z_k) 계산

  FOR each candidate layer jointly
      G_{k,l} = Π_l K_{k,l}
      W64 Woodbury/Cholesky로 Q_{k,l} 계산
      B_{k,l} = R_{k,l} Q_{k,l}^T를 low-rank factor로 표현
      target-new-NLL signed progress a_{k,l} 계산
      cumulative historical/pretrained structural soft score 계산
  END

  IF 모든 write arm이 zero이면
      target-new-NLL target-only recovery를 한 grid step 수행
      tau_{k+1} = tau_k + h
      telemetry 기록 후 CONTINUE

  current-state baseline + layer probes로 functional H/P soft basis 계산
  해당 arm의 current field에서 Neutral 또는 Soft routing 문제 계산
  selected arm의 v_k와 solver certificate 기록

  layer-local overlay를 통한 target field F_z(S_k) 계산
  BF16-authoritative joint Euler transition을 정확히 한 번 수행
      Δ_{k+1} = Δ_k + h Σ_l v_{k,l} B_{k,l}
      z_{k+1} = z_k + h F_z(S_k)
      tau_{k+1} = tau_k + h

  stepwise eff/gen/loc, new/old NLL, H/P audit와 routing vector 기록
  exact first-hit이면 index만 기록하고 trajectory 계속
END

TERMINAL AT tau=1
  action freeze 후 Native reference와 held-out evaluation open
  first-hit-to-terminal persistence와 all-step trajectory 분석

OPTIONAL DEPLOYMENT COMMIT
  accumulated low-rank update를 실제 model에 한 번 materialize
  W64 virtual vs BF16 committed bytes/logits/event 검증

  technical identity 실패:
      deployed weight rollback
      history append 없음

  technical identity 성공:
      obsolete active version 제거
      final projected key 정확히 한 번 append
      functional history 정확히 한 번 append
      layer load 정확히 한 번 갱신
      transaction finalize
```


---

# 14. 계산량과 메모리 설계

## 14.1 Naive grid-wise AlphaEdit가 무거운 이유

현재 Alpha solve의 down-projection input dimension은 대략 다음과 같다.

- Llama3-8B: \(d=14{,}336\)
- Qwen2.5-7B: \(d=18{,}944\)

Dense 구현을 grid마다 반복하면 layer마다 다음 비용을 다시 지불한다.

- \(\Pi_l(KK^{\top}+C_H)\) dense construction
- \(d\times d\) factorization
- wide RHS solve
- dense mutable historical matrix update

Euler grid가 증가할수록 비용이 거의 선형으로 누적되고, 각 solve 자체가 매우 크다.

## 14.2 Factorized historical memory

FP32, single-request edit당 context-averaged key 1개, active historical key 100개, candidate layer 5개를 기준으로 projected-key factor memory는 다음과 같다.

| Model | Factorized history | Dense \(C_H\) |
|---|---:|---:|
| Llama3-8B | 약 27.3 MiB | 약 3.83 GiB |
| Qwen2.5-7B | 약 36.1 MiB | 약 6.68 GiB |

주의할 점은 이 수치가 **historical state memory만** 의미한다는 것이다.

- dense static projector \(\Pi_l\)
- dense pretrained moment \(C_l^0\)

를 GPU에 상주시킬 경우 해당 memory는 별도다.

Projected key만 저장하면 위 factor memory를 유지할 수 있다. Projector numerical error 때문에 raw key fallback까지 저장하면 historical factor memory는 대략 두 배가 된다.

## 14.3 Per-grid-field complexity

다음을 두자.

- current request rank: \(b\)
- active history rank: \(r\)
- input dimension: \(d\)
- candidate layer 수: \(m\)

Dense \(\Pi_l\)를 사용할 때 grid field마다 current projection은

$$
O(d^2b)
$$

이다.

History Cholesky를 cache하면 Woodbury small solve와 reconstruction은 대략

$$
O(drb+r^2b+b^3)
$$

이다.

즉 dense \(O(d^3)\) factorization은 제거되지만, dense projector matvec는 남는다.

Pretrained structural risk는 low-rank proposal factor마다 \(C_l^0v\) matvec를 추가한다. 이 값은 field에서 한 번만 계산하고 neutral/soft routing 및 numerical continuation이 공유한다.

## 14.4 Fixed-E8 pilot의 exact operation ceiling

Candidate layer 5개, arm 1개 기준 P1R7 diagnostic ceiling은 다음과 같다.

| Operation | Per arm ceiling | 의미 |
|---|---:|---|
| Euler fields / target backward | 8 / 8 | \(K=8\) 고정 |
| joint Euler candidate | 8 | grid당 정확히 1개 |
| functional basis endpoints | 48 | baseline 1 + layer probe 5, 8 fields |
| candidate functional audit | 8 | selected joint candidate |
| terminal/audit endpoint-equivalent | 8 | stepwise audit schedule의 상한 |
| rewrite evaluation | 9 | \(k=0\) 포함 trajectory |
| mathematical QCQP | 32 | field당 정의된 문제 수 |
| nominal backend invocation | 40 | stage-2 polish 8회 포함 |
| scientific retry/backtracking | 0 | method contract |

두 live arms를 동시에 돌리면 위 값은 정확히 두 배다. 이 표는 **현재 diagnostic pilot**의 ceiling이지 production budget이 아니다.

최신 numerical lock이 continuation을 포함해 허용한 104 optimizer backend invocations/arm은 low-overhead 목표와 맞지 않으며 final method로 승격하지 않는다. Backend policy가 해결되기 전에는 32 mathematical problems와 실제 invocation을 별도로 공개한다.

## 14.5 남는 실제 병목

Exact low-rank conversion 이후 예상되는 online bottleneck은 다음이다.

1. 각 grid에서의 dense \(\Pi_lK_{k,l}\) matvec
2. 각 grid에서의 \(C_l^0v\) matvec
3. write-aware target backward
4. baseline+layer-probe functional basis
5. history가 커질 때의 replay token 수

따라서 “Wikipedia covariance 사용 자체”가 병목인 것은 아니다. 병목은 static artifact matvec와 functional evaluation이다. Historical controller sample을 fixed recent/reservoir 크기로 묶지 않으면 lifelong stream에서 replay 비용이 선형 증가한다.

## 14.6 Compute contract

Main implementation은 다음 upper bound를 명시해야 한다.

- Euler field와 candidate 정확히 8회
- target backward grid당 최대 1회
- scientific retry/backtracking 0회
- controller history recent/reservoir fixed size
- pretrained sample size와 grid schedule 고정
- mathematical QCQP와 backend invocation 별도 ceiling
- diagnostic 48-probe path와 bounded final path를 별도 명칭으로 관리
- dense \(C_H\), full \(B_l\), full trial-model copy 생성 금지

Bounded final path는 fixed E8을 유지하고 cached baseline + fixed-size H/P sample을 사용하며, batched/JVP layer slope 또는 selected candidate-only functional feedback으로 online endpoint-equivalent ceiling을 16+terminal audit 이하로 낮추는 것을 목표로 한다.

## 14.7 필수 compute logging

단순히 “ODE step 수”만 기록하면 실제 비용을 비교할 수 없다. Edit마다 다음을 기록해야 한다.

$$
\begin{aligned}
&N_{\mathrm{target\_backward}},\\
&N_{\mathrm{Euler\_field}},\\
&N_{\mathrm{joint\_candidate}},\\
&N_{\mathrm{functional\_baseline}},\\
&N_{\mathrm{functional\_probe}},\\
&N_{\Pi K},\\
&N_{C^0v},\\
&N_{\mathrm{small\_chol}},\\
&N_{H\text{-replay}},\\
&N_{P\text{-replay}},\\
&N_{\mathrm{mathematical\_QP}},\\
&N_{\mathrm{backend\_invocation}},\\
&N_{\mathrm{numerical\_continuation}},\\
&\text{wall-clock/edit},\\
&\text{peak GPU/CPU memory},\\
&r_{\mathrm{active}}.
\end{aligned}
$$

Endpoint-equivalent 수, 실제 forward/backward NFE, processed tokens와 wall time을 함께 보고한다. Numerical continuation을 scientific retry 또는 Euler step으로 합산하지 않는다.

---

# 15. 실험 설계

## 15.1 다음 immediate same-seal panel

다음 scientific run은 outcome에 따라 method 조합을 선택하지 않고 정확히 세 arm을 동일 seal, case order, evaluator와 compute ledger에서 비교한다.

| Immediate arm | 역할 | H/P 사용 | Hard scientific budget/veto |
|---|---|---|---|
| Native AlphaEdit | same-seal editor baseline | 없음 | 없음 |
| **COLD-FR-E8-NEUTRAL** | cold fixed-E8 자체와 full-residual refreshed field 검증 | observation/audit only | 없음 |
| **COLD-FR-E8-SOFT** | 동일 initial seal과 matched schedule에서 H/P soft routing의 추가 가치 검증 | routing objective + audit | 없음 |

Primary 판정은 budget 통과 여부가 아니라 Eff/Gen/Loc, target-new/old absolute NLL, update capacity, Native-relative teacher-KL, target-to-write realization fidelity와 historical retention으로 한다. Soft가 Native 수준의 primary metric을 유지하면서 Neutral보다 preservation 이득을 보이지 못하면 **Neutral을 final 후보**로 둔다.

ALLOFF full-\(\tau\) 결과는 이 panel을 설계하게 한 strongest completed evidence이지만, different run/implementation의 수치를 새 panel outcome으로 재사용하지 않는다. `ALLOFF`라는 legacy label은 H/P scientific budget/veto가 꺼졌다는 사실만 뜻한다. Physical write-trust를 포함한 모든 technical axis가 current Neutral과 동일하다는 receipt가 없으면 두 방법을 동치라고 부르지 않는다.

## 15.2 장기 Core main table

| Method | Target | Proposal/history geometry | Layer routing | Functional use | Clock/commit |
|---|---|---|---|---|---|
| Native AlphaEdit | fixed native direct-\(z\) | fixed projector, native sequential | fixed remaining-residual sharing | 없음 | native |
| Native AlphaEdit-WB | fixed native direct-\(z\) | native와 동일, low-rank solve | native와 동일 | 없음 | native |
| MetaKE + AlphaEdit | downstream-aware refined target | frozen final-layer proxy + standard AlphaEdit | standard remaining-residual sharing | outer edit/locality objective | standard final write |
| CAKE | fixed/native target | causal attribution profile + AlphaEdit backend | causal-weight sequential allocation | 별도 H/P replay 없음 | native-style sequential write |
| EvoEdit | fixed/native target | committed edit마다 evolving null-space projector | within-edit joint routing 없음 | 별도 H/P replay 없음 | per-edit model/projector update |
| Warm-FR-Adaptive legacy pilot | native direct-\(z\) warm start | accepted-state full-residual arms | H/P hard-budget QCQP | hard replay veto | adaptive/retry; no promotion |
| **COLD-FR-E8-NEUTRAL** | cold target-new-NLL | layer-local full-residual arms, static projector | progress + capacity | observation only | fixed E8; virtual endpoint |
| **COLD-FR-E8-SOFT** | cold target-new-NLL | layer-local full-residual arms, static projector | structural/functional H/P soft + capacity | soft routing + audit | fixed E8; transactional one-shot |
| EvoEdit-projector + COLD-FR-E8-SOFT | 동일 target/controller | evolving projector | 동일 soft routing | 동일 | fixed E8; transactional one-shot |

`Native AlphaEdit-WB`는 solver replacement의 계산량 효과만 분리하기 위한 필수 baseline이다. Target, residual sharing, layer order, history를 native와 동일하게 고정했을 때 numerical tolerance 내에서 같은 endpoint를 재현해야 한다.

MetaKE, CAKE, EvoEdit는 선택적 appendix baseline이 아니라 claim attribution을 위한 필수 external baseline이다. 특히 다음 두 factorial comparison을 분리한다.

1. `standard target` 대 `MetaKE target` 대 `ODE-BF target/write flow`,
2. `static projector` 대 `EvoEdit evolving projector`와 `native controller` 대 `ODE-BF controller`의 \(2\times2\) 비교.

CAKE와의 비교에서는 동일 layer set, target construction, batch, edit order와 compute accounting 아래 causal-weight/current-residual sequential allocation과 fixed-E8 H/P joint routing을 비교한다. 서로 다른 target/editor를 동시에 바꾸어 allocation 효과로 해석하지 않는다.

## 15.3 Mechanism ablations

### Target 관련

- cold \(z_{\mathrm{base}}\) vs native direct-\(z\) warm start
- fixed direct-\(z\) vs evolving direct-\(z\)
- intervention-only target gradient vs write-aware target gradient
- target-new-NLL vs legacy new-minus-old margin routing
- MetaKE Structural Gradient Proxy target vs full fixed-grid multi-layer feedback
- MetaKE target + standard sharing vs MetaKE target + ODE-BF routing
- routing stop-gradient vs implicit differentiation
- first-hit stopping vs observation-only full E8

### Proposal/routing 관련

- fixed grid state마다 proposal 재계산 vs initial proposal 고정
- refreshed field vs frozen initial field의 same-seal ODE attribution
- fixed-grid refinement \(K\in\{4,8,16\}\)와 common-\(\tau\) endpoint 비교
- joint routing vs shallow-to-deep coordinate routing
- joint routing vs deep-to-shallow coordinate routing
- CAKE causal-weight/current-residual allocation vs ODE-BF full-residual H/P soft routing
- cumulative structural risk vs step-only risk
- cumulative layer load penalty 제거

### Preservation 관련

- historical structural soft score 제거
- pretrained structural soft score 제거
- historical functional soft score 제거
- pretrained functional soft score 제거
- diagonal covariance risk vs small Fisher/JVP Gram
- static AlphaEdit projector vs EvoEdit evolving projector
- EvoEdit evolving projector 단독 vs evolving projector + ODE-BF controller

### Transaction/compute 관련

- dense Alpha solve vs exact Woodbury
- 48-endpoint diagnostic basis vs batched/JVP or candidate-only bounded feedback
- virtual low-rank hook vs trial full materialization
- numerical backend/polish policy

P-only, trust-only, P+trust, neither 같은 넓은 factorial은 method core와 solver policy가 안정된 뒤 ablation에서 수행한다. Main pilot에서 outcome을 보고 조합을 고르지 않는다.

## 15.4 Primary metrics

### New edit

- canonical efficacy
- target-new NLL와 target-old NLL
- benchmark별 exact success bit/count
- held-out paraphrase generalization
- exact first-hit index와 hit persistence
- \(k=0,\ldots,8\) 전 step trajectory

### Historical retention

- historical success rate
- recent history success
- reservoir history success
- CVaR margin deficit
- worst-decile margin drop
- obsolete version 처리 정확도

### Pretrained/locality preservation

- incremental teacher KL
- unrelated factual/locality accuracy
- neighborhood/locality change rate
- structural pretrained covariance risk

### Update geometry

- layer별 \(v_l\)
- layer별 normalized routing share
- layer별 raw proposal norm
- layer별 rewrite efficiency \(a_l\)
- layer별 \(c_{H,l}\), \(c_{P,l}\)
- cumulative layer load
- top-1/top-2 share, HHI, normalized entropy, Gini, effective layer count
- raw-to-soft routing L1/cosine와 active mask
- target displacement \(\|z_k-z_{\mathrm{base}}\|_{G_z}\)
- target-to-write realization fidelity: intended intervention displacement와 virtual weight-only lookup activation displacement의 residual ratio, cosine과 norm gain

### Compute

- target backward NFE
- Euler field/candidate count
- functional baseline/probe endpoint count
- projector/covariance matvec count
- historical/pretrained replay NFE
- mathematical QP/backend invocation/continuation count
- wall-clock/edit
- peak GPU/CPU memory
- active key rank

Target-to-write realization fidelity는 request \(i\), step \(k\)에서 intervention이 의도한 lookup displacement

$$
\Delta z_{k,i}^{\mathrm{intent}}=z_{k+1,i}-z_{k,i}
$$

와 intervention hook 없이 virtual weight \(W_{k+1}^{\mathrm{virt}}\)가 실제로 만든 displacement

$$
\Delta z_{k,i}^{\mathrm{write}}
=z_i(W_{k+1}^{\mathrm{virt}})-z_i(W_k^{\mathrm{virt}})
$$

를 비교한다. 최소한

$$
r_{k,i}^{\mathrm{fid}}
=
\frac{\|\Delta z_{k,i}^{\mathrm{write}}-\Delta z_{k,i}^{\mathrm{intent}}\|_2}
{\|\Delta z_{k,i}^{\mathrm{intent}}\|_2+\epsilon}
$$

와 cosine, norm-gain \(\|\Delta z^{\mathrm{write}}\|/(\|\Delta z^{\mathrm{intent}}\|+\epsilon)\)을 기록한다. 이 metric은 observation-only이며 target hook이 좋아졌다는 이유만으로 successful weight edit을 주장하지 못하게 한다.

## 15.5 Matched-compute comparison

ODE-BF와 generic penalty optimizer를 동일한 coefficient state와 동일한 field 위에서 비교한다.

최소한 다음 두 기준을 모두 맞춰야 한다.

- Euler field와 functional endpoint-equivalent 수
- total forward/backward NFE 또는 wall-clock

Generic optimizer가 같은 compute에서 동일한 efficacy–preservation frontier를 보이면 contribution은 ODE 자체보다 constrained dynamic routing으로 재해석해야 한다.

## 15.6 권장 pilot configuration

현재 fixed-E8 pilot은 다음 범위로 잠근다.

- candidate layer: 4–6개
- genuine edit batch: 10 requests
- \(K=8,h=1/8,\tau_{\mathrm{final}}=1\)
- scientific retry/backtracking: 0
- target backward: grid field당 최대 1회
- historical replay: fixed recent + reservoir, exact size preseal
- CVaR tail: \(\rho\in\{0.1,0.2\}\)
- pretrained replay: sealed rotating batch; diagnostic basis와 bounded path 분리
- W64 small solve + BF16-authoritative endpoint
- held-out paraphrase: evaluation only

Pretrained replay sample 수는 별도 no-op/native profiling에서 정한 뒤 scientific outcome 전에 seal한다.

10/50/100-edit pilot은 implementation 및 mechanism motivation gate다. Lifelong large-scale claim에는 충분하지 않다. 그 claim을 위해서는 EvoEdit와 비교 가능한 최소 2K sequential edits와 가능하면 10K까지 확장하고, early-edit retention curve, recent/reservoir retention, abort/violation rate, cumulative teacher-KL, SVD/projector cost, replay cost, wall time과 memory를 함께 보고한다. Small pilot에서의 성공을 lifelong superiority로 외삽하지 않는다.

---

# 16. 필수 implementation invariants

## 16.1 Woodbury–dense equivalence

작은 model 또는 축소 dimension에서

$$
\frac{
\|Q_l^{\mathrm{WB}}-Q_l^{\mathrm{dense}}\|_F
}{
\|Q_l^{\mathrm{dense}}\|_F
}
$$

를 측정한다.

Full model에서는 항상 linear residual을 기록한다.

$$
\frac{
\|A_lQ_l^{\mathrm{WB}}-
\Pi_lK_l\|_F
}{
\|\Pi_lK_l\|_F
}.
$$

## 16.2 Native-Woodbury endpoint equivalence

Target, residual sharing, layer order, history를 native와 동일하게 두었을 때 `Native AlphaEdit-WB`가 native endpoint를 numerical tolerance 내에서 재현해야 한다.

이것은 attribution control이다. Main W64 path가 N32와 bit-exact해야 한다는 조건으로 바꾸지 않는다. 검증 전에는 “Native exact replacement” claim을 사용하지 않는다.

## 16.3 Dense materialization 금지

Probe/Euler transition 중 다음 tensor가 생성되지 않는지 assertion을 둔다.

- dense \(C_l^H\)
- dense \(\Pi_lC_l^H\)
- full \(B_l\)
- full copied trial model

## 16.4 Probe와 observation purity

Functional basis probe와 observation-only evaluation 전후 다음 hash/version이 동일해야 한다.

- deployed weights
- solve/risk history factors
- functional history
- cumulative layer load
- transaction state
- sampling schedule state

## 16.5 Exactly-once history update

Successful outer edit 하나에 대해 다음이 성립해야 한다.

- transaction ID 1개
- active version append 1회
- obsolete version removal 1회 이하
- projected factor append 1회
- finalization retry 시 중복 append 0회

## 16.6 Field cache identity

Cache key에는 최소한 다음이 포함되어야 한다.

```text
outer_edit_id
euler_step_index
tau_numerator_denominator
virtual_state_version
target_state_version
layer_id
proposal_version
history_version
```

Probe와 numerical continuation은 cache key를 바꾸지 않는다. Fixed Euler transition만 새로운 field version을 만든다.

## 16.7 Virtual endpoint와 committed endpoint 일치

Commit 전 W64/BF16 virtual prediction과 commit 후 actual model output의 차이를 기록한다.

- new-edit logits
- historical replay margins
- pretrained replay KL
- affected layer weight delta

BF16 parameter bytes, logits와 canonical event가 exact하지 않으면 low-rank hook 또는 materialization bug로 처리하고 history를 append하지 않는다. FP32 overlay 결과를 authoritative verdict로 사용하지 않는다.

## 16.8 Structural proxy calibration

다음을 edit/grid별로 log한다.

- predicted historical risk
- actual historical margin damage
- predicted pretrained risk
- actual incremental teacher KL

ODE-BF의 mechanism claim을 위해서는 최소한 다음 경향이 보여야 한다.

$$
\text{high }c_H
\Rightarrow
\text{larger historical damage tendency},
$$

$$
\text{high }c_P
\Rightarrow
\text{larger pretrained drift tendency}.
$$

Structural risk가 functional damage와 무관하면 routing signal로서 의미가 없다.

## 16.9 Stepwise routing telemetry

Ordered layer \([l_1,\ldots,l_m]\)에 대해 모든 grid에서 다음을 raw-free receipt로 남긴다.

- signed progress vector와 active mask
- neutral/raw, soft, applied coefficient vector
- normalized share vector
- top-1/top-2 share와 layer index
- HHI, normalized entropy, Gini, effective layer count
- raw-to-soft L1와 cosine
- structural/functional H/P component별 contribution

Telemetry on/off가 model state, routing decision, RNG, pointer/version/grad와 terminal bytes를 바꾸지 않는 회귀 테스트가 필수다.

## 16.10 Fixed-grid와 accounting identity

정상 run은 정확히 \(K=8\), \(h=1/8\), \(\tau_8=1\)을 만족한다. First-hit, metric degradation, H/P score는 grid count를 바꾸지 않는다. Mathematical QP, backend invocation, continuation, functional endpoint-equivalent와 actual NFE의 cross-file 합계가 일치해야 한다.

---

# 17. 예상 실패 모드와 대응

## 17.1 Initial target field가 형성되지 않음

**증상:** cold \(z_{\mathrm{base}}\)에서 모든 layer arm의 residual 또는 rewrite efficiency가 0에 가까움.

**대응:**

- zero-write target recovery의 target-new-NLL gradient 점검
- target token loss gradient 점검
- target-layer intervention 위치 점검
- recovery 한 grid 뒤에도 field가 없으면 technical fail-close
- native direct-\(z\) warm-start는 fallback이 아니라 명시적 ablation으로만 사용

## 17.2 Structural risk가 functional damage를 예측하지 못함

**증상:** covariance risk는 낮지만 historical replay나 pretrained KL가 크게 악화됨.

**대응:**

- key token 위치와 context aggregation 일치 여부 확인
- diagnostic functional basis와 joint audit 비교
- small Fisher/JVP Gram ablation
- 상관이 없으면 해당 structural soft score를 제거

## 17.3 Target ODE가 write-infeasible region으로 이동

**증상:** target step 이후 progress/trust QCQP가 certificate를 만들지 못함.

**대응:**

- target-new-NLL field와 coefficient/update unit 점검
- fixed \(h=1/8\) refinement ablation을 별도 실행
- 독립 solver certificate와 backend status 분리
- write-aware gradient와 intervention-only gradient 비교

Main run 안에서 \(h\), progress fraction, tolerance를 adaptive하게 바꾸지 않는다.

## 17.4 Layer collapse

**증상:** 거의 모든 edit/step에서 한 layer가 \(v_l\approx1\)을 차지함.

**대응 순서:**

1. raw proposal scale와 \(a_l\) 계산 검증
2. same-geometry normalization 여부 확인
3. cumulative load가 실제로 반영되는지 확인
4. 그 뒤에만 layer cap 또는 약한 diversification 추가

실제 edit별 layer specialization일 수 있으므로 entropy regularization을 처음부터 넣지 않는다.

## 17.5 Active history rank 증가

**증상:** long stream에서 small Gram과 factor memory가 커짐.

**대응:**

Main exact experiment에서는 benchmark horizon 전체 active factor를 유지한다. Rank compression, sketching, reservoir solve는 별도 long-horizon approximation으로 분리하고 exact라고 부르지 않는다.

## 17.6 Dense pretrained matvec가 병목

**증상:** \(C_l^0v\) 시간이 grid field의 대부분을 차지함.

**대응:**

- layer별 streaming
- repeated factor cache
- CPU/GPU overlap
- fixed low-rank eigenspace approximation을 명시적 ablation으로 비교

Main exact covariance result를 조용히 approximation으로 바꾸지 않는다.

## 17.7 Functional basis가 총비용을 지배

**증상:** replay forward가 field construction보다 많음.

**대응:**

- baseline cache와 common random numbers 사용
- recent/reservoir controller 크기를 고정
- batched/JVP slope 또는 candidate-only feedback 검증
- full active history는 terminal audit로 제한
- endpoint-equivalent와 actual forward를 함께 기록

## 17.8 Woodbury numerical instability

**증상:** small Gram condition number 증가, Cholesky failure, linear residual 증가.

**대응:**

- W64 reduced Gram/solve
- predeclared jitter
- condition number logging
- certificate miss에서만 QR/SVD small solve fallback

Fallback도 동일 linear system을 풀어야 한다.

## 17.9 H/P soft routing이 무의미함

**증상:** same-field E8-NEUTRAL과 E8-SOFT의 coefficient와 efficacy–preservation trajectory가 실질적으로 동일함.

**해석:** H/P soft signal이 layer routing을 바꾸지 않으며 48-endpoint basis 비용을 정당화하지 못한다.

**필수 비교:** initial-state same-field neutral/soft coefficient와 이후 arm-local own-state trajectory, equal sharing, candidate-only feedback을 matched compute로 비교한다.

## 17.10 Routing solver가 scientific run을 가로막음

**증상:** primal/dual residual 또는 backend status 때문에 full E8 trajectory 전에 fail-close.

**대응:** authoritative arm certificate와 diagnostic Soft shadow를 분리하고, finite/primal/stationarity/complementarity를 backend status와 독립적으로 재구성한다. Backend fallback은 true certificate miss에서 동일 QP에 최대 한 번만 허용한다. Tolerance 완화나 무제한 continuation으로 우회하지 않는다. Solver가 최대 8 Euler step보다 훨씬 많은 backend call을 요구하면 method practicality failure로 판정한다.

## 17.11 Context degeneration 또는 target-scale outlier

**증상:** controller context에 비정상적인 연속 중복 token·구두점 run이 생기거나, 일부 request의 \(\|z_{\mathrm{base}}\|_2^2\)가 같은 batch의 robust scale에서 크게 벗어남.

**대응:** model action 전 context generator의 source, seed, order와 canonical hashes를 고정하고, duplicate-token run, punctuation run, token length, per-request \(z_{\mathrm{base}}\) norm quantile/max-ratio를 raw-free receipt로 남긴다. Nonfinite, empty 또는 schema violation만 technical fail-close한다. Outcome을 본 뒤 case를 제거·재생성하거나 target norm을 clip하는 model-specific rescue는 금지한다. 공통 generator bug가 증명되면 양 모델에 동일한 semantic-neutral repair와 fresh seal을 적용하고, 그렇지 않으면 sensitivity caveat로 보고한다.

---

# 18. Go/No-Go 판정 기준

## 18.1 Technical Go

다음이 모두 성립해야 한다.

1. Woodbury와 dense per-layer solve equivalence 확인
2. probe/observation의 persistent mutation 0건
3. W64/BF16 virtual endpoint와 committed endpoint exact
4. history append exactly once
5. 정확히 8 Euler fields, \(h=1/8\), \(\tau=1\)
6. stepwise routing/evaluator schedule 재현 가능
7. mathematical QP/backend invocation/cost accounting 일치
8. independent solver certificate policy 통과
9. Neutral authoritative solve와 diagnostic Soft shadow fail-close 분리; shadow failure 전후 Neutral state/clock/endpoint byte identity
10. zero-write functional-P raw/repeat/floor/floor-corrected telemetry와 telemetry on/off state identity
11. Qwen context 반복·구두점과 \(z_{\mathrm{base}}\) norm outlier의 source/seed/order RCA; outcome 기반 case 제거·재생성 0건
12. target-field probe와 actual BF16 trial의 \(h\)-scaled displacement identity
13. stepwise target-to-write realization fidelity receipt 완전성

하나라도 실패하면 scientific comparison 전에 구현을 수정한다.

## 18.2 Scientific Go

Small-batch gate에서 COLD-FR-E8-SOFT는 same-seal Native에 대해 다음 세 row를 각각 비열등하게 유지해야 한다.

- official efficacy
- official generalization
- official locality/preservation

Count metric은 동일 denominator에서 integer loss 0을 기본으로 한다. 한 row의 gain으로 다른 row의 loss를 상쇄하지 않는다. 이 floor를 통과한 뒤 다음 mechanism evidence를 본다.

- Native AlphaEdit보다 높은 historical retention
- Native AlphaEdit보다 낮은 pretrained/locality incremental drift
- COLD-FR-E8-NEUTRAL보다 soft routing의 개선
- warm-start control보다 cold trajectory의 명확한 이점
- Fixed proposal bank보다 state-dependent proposal recomputation의 이점
- matched-NFE generic coefficient optimizer보다 우수하거나 더 낮은 violation rate
- MetaKE의 downstream-aware target refinement만으로 설명되지 않는 controller 이점
- CAKE의 causal allocation 대비 fixed-E8 H/P soft routing의 추가 이점
- EvoEdit의 evolving projector와 분리된 controller 이점

이 판정은 단계별로 분리한다.

- **Pilot scientific go:** same-seal Native 수준의 efficacy/generalization/locality를 유지하면서 mechanism이 동작하고 치명적 miss가 없다.
- **Related-work attribution go:** MetaKE/CAKE/EvoEdit와 matched target, geometry, controller, compute comparison에서 어느 축이 gain을 만드는지 분리된다.
- **Lifelong claim go:** 2K 이상 sequential stream에서 early-edit retention과 pretrained/locality drift가 강한 sequential baseline보다 개선된다.

Pilot go만 통과한 결과에는 `ODE-BF Full`, formal CBF safety, lifelong superiority를 사용하지 않는다.

## 18.2.1 이후 attribution과 sequential claim 순서

Immediate three-arm panel이 끝난 뒤에만 다음 순서로 확장한다.

1. refreshed field와 frozen initial field를 같은 seal에서 비교하여 state-dependent ODE contribution을 분리한다.
2. \(K=4,8,16\)을 같은 \(\tau=1\)에서 비교하고 endpoint refinement와 compute scaling을 기록한다.
3. B10-2 이상 ordered sequential edit부터 historical H가 실제로 active한 retention을 평가한다.
4. 장기 stream에서 Eff/Gen/Loc, early-edit retention, teacher-KL, capacity와 compute의 누적 곡선을 비교한다.

B10-1의 empty history 결과로 historical barrier 이득을 주장하지 않고, single-edit full-\(\tau\) 결과로 lifelong collapse 방지를 주장하지 않는다.

## 18.3 결과별 해석

### Woodbury만 성공

`Native AlphaEdit-WB`는 빠르고 exact하지만 full method가 보존 성능을 개선하지 못하면, solver optimization은 유효하되 ODE-BF mechanism은 지지되지 않는다.

### Warm-start control과 동일

Cold target trajectory의 독립 기여가 지지되지 않고 dynamic routing만 남는다.

### E8-NEUTRAL과 동일

H/P soft routing이 불필요하고 cold target/full-residual field가 핵심이다.

### Functional soft score가 있을 때만 보존 개선

Structural proxy만으로 부족하고 functional feedback이 의미 있다. 48-endpoint diagnostic을 bounded feedback으로 줄여도 효과가 유지되는지 추가 검증한다.

### Generic penalty optimizer와 동일

ODE-specific claim을 줄이고 constrained state-dependent routing으로 framing한다.

### State-dependent recomputation 이점 없음

Initial proposal bank를 재사용하는 저비용 ODE-Alloc 계열이 더 적절할 수 있다.

### MetaKE target과 동일

Target–write coupling의 독립 contribution은 지지되지 않는다. ODE-BF claim을 routing/soft audit/transaction으로 축소한다.

### CAKE allocation과 동일

State-dependent H/P routing의 추가 가치가 지지되지 않는다. CAKE-style causal allocation과 더 단순한 audit 조합을 대안으로 둔다.

### Solver 때문에 full E8 미완료

Scientific outcome이 아니다. 현재 P1R7처럼 Llama/Qwen이 numerical boundary에서 멈추면 efficacy나 H/P 결론을 내리지 않고 solver policy를 technical gate로 되돌린다.

### EvoEdit 또는 hybrid만 개선

Gain이 evolving null-space에서 발생한 것이다. Dynamic-projector component는 EvoEdit attribution으로 두고, ODE-BF controller의 독립 novelty를 주장하지 않는다.

---

# 19. 최종 proposal statement

Proposed ODE-BF는 한 번의 knowledge edit을 Native direct target과 정적 residual sharing으로 보지 않는다. 대신 다음 fixed-clock closed loop로 정의한다.

$$
\boxed{
z_0=z_{\mathrm{base}}
\rightarrow
\text{target-new-NLL field}
\rightarrow
\text{layer-local full-residual arms}
\rightarrow
\text{structural/functional H/P soft routing}
\rightarrow
\text{fixed Euler }(K=8,h=1/8,\tau=1)
}
$$

각 Euler grid에서 model state, target state, key, residual과 proposal을 갱신한다. Functional probe와 numerical continuation은 deployed model/history를 바꾸지 않는다. Scientific retry/backtracking은 없고 first-hit은 observation-only다.

High-dimensional Alpha system은 dense하게 풀지 않는다. Active historical knowledge를 projected-key factor로 유지하고 W64 Woodbury/Cholesky solve로 per-layer arm을 계산한다. Historical/pretrained structural risk는 low-rank factor에서 계산하고, functional H/P는 fixed-size sample의 soft slope와 terminal audit로 사용한다.

Design-validation endpoint는 \(\tau=1\)이다. 각 step에서

- efficacy/generalization/locality,
- target-new/old NLL,
- historical/pretrained structural·functional damage,
- full layer routing vector와 concentration,
- exact first-hit와 persistence,
- NFE/token/wall/memory/backend invocation

를 기록한다. Heuristic H/P threshold는 endpoint를 거부하지 않는다. Terminal endpoint만 배포 후보이며, W64/BF16 technical identity가 통과한 경우에만 실제 model에 정확히 한 번 commit할 수 있다.

최종 방법은 다음으로 요약된다.

$$
\boxed{
\text{Cold target-new-NLL ODE}
+
\text{Layer-local full-residual proposals}
+
\text{Structural/functional H/P soft routing}
+
\text{Fixed E8 full-}\tau\text{ rollout}
+
\text{W64/BF16 transactional commit}
+
\text{Stepwise telemetry and exact accounting}
}
$$

본 proposal을 main research direction으로 두고, 기존 fixed-target ODE-Alloc은 coefficient allocation contribution만 분리하는 component ablation으로 유지한다.

Related-work 이후의 claim boundary는 명확하다. ODE-BF는 MetaKE의 downstream-aware target refinement, CAKE의 adaptive residual allocation, EvoEdit의 evolving null-space/Woodbury solve를 최초로 제안한다고 주장하지 않는다. 최종 claim은 이 세 축과 matched comparison한 뒤에도 남는 **cold fixed-Euler target–write co-evolution + preservation-aware soft joint routing + BF16 transactional endpoint**에만 귀속한다.

현재 구현은 solver/backend policy와 bounded functional-feedback contract가 미완료인 fixed-E8 diagnostic pilot이다. 최종 목표는 single-edit 점수를 조금 높이는 것이 아니라, fixed compute 아래 수천 회의 ordered B10 sequential stream에서 new-edit efficacy/generalization/locality를 Native 수준으로 유지하면서 early-edit retention과 pretrained drift의 붕괴를 늦추는 것이다. 이 lifelong gate 전에는 `ODE-BF Full`, formal CBF safety 또는 lifelong superiority를 주장하지 않는다.
