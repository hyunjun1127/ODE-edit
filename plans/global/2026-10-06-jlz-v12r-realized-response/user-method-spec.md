# JLZ v12-R method 명세 — 실현 응답 주입과 요청 에너지 예산

Oct 6, 2026 · @Jang Hyunjun

## 1. 요약

v12-R은 v12의 공동 계획 골격과 native MEMIT ridge writer를 그대로 두고, 계획이 '요청'이 아니라 '실제로 실현되는 작용'을 보게 만든 방법이다. 예산은 writer 비용과 같은 단위인 요청 에너지로 다시 정의하고, 요청마다 필요한 만큼만 쓰도록 반경을 단계적으로 키운다.

풀려는 문제는 이렇다. 요청마다 정해진 요청 에너지 안에서, 실제 ridge write 아래의 편집 손실을 가장 낮추는 층별 요청을 찾는다. 에너지는 기본 수준에서 시작하고, 그 수준으로 기준을 못 넘는 요청에만 더 준다.

| 항목 | v12 | v12-R |
| --- | --- | --- |
| 최적화 변수 | 상대 좌표 u, 주입량 a·u | 절대 요청 R (층 l, 요청 r) |
| 계획에 주입하는 양 | 요청 그대로 (실현 100% 가정) | 후보 weight가 만든 실제 작용 v |
| gradient | 가상 subject 경로 | 같은 subject 경로 + 같은 층 응답 pullback |
| 요청 간 결합 | 없음 (요청별 독립) | 있음 (같은 batch write를 공유) |
| 예산 | 상대 norm 합 ≤ 0.75 | 요청 에너지 ≤ (0.75·a\*)², 층별 cap 0.75·a\_l |
| 총량 결정 | 항상 예산 포화 | 기본 반경에서 시작, 미달 요청만 확장 |
| writer | target tracking (잔차 = 가상 목표 − 실제) | 평가한 후보 weight를 그대로 commit |
| exact writer, divisor, tracking | tracking 사용 | 모두 사용하지 않음 |
| 예상 fit 비용 (B100) | 385초 | 약 500초 (추정, 미측정) |

유지하는 것은 다음과 같다. native rewrite·KL 문장과 lookup, NLL reduction, current‖entry KL과 계수, ridge 식과 C0·history, 모든 eligible 층의 후보 자격, 25회 평가 상한이다.

이 문서는 method와 구현 계약이다. 실모델 검증과 성능은 아직 없다.

## 2. 설계 결정과 근거

각 결정은 저장소에 있는 완료 실험의 수치에 묶여 있다. 모델은 Llama-3-8B-Instruct, 고정 CounterFact, L4–L8, BS100이다.

| 결정 | 근거 수치 | 출처 |
| --- | --- | --- |
| 요청이 아니라 실현 작용을 주입 | 같은 plan에서 ridge direct는 PS 73.5 (Q 143), tracking은 88.5 (Q 240). 실현 응답을 본 V14는 90.5 (Q 124) | V13 B1 5-branch, V14 B1 |
| tracking 제거 | L4 부족분 0.38이 L8에서 2.17 gap으로 증폭. tracking은 L5에서 계획의 6배를 기입. 2k 평균 L4 plan 53.6% → 실현 33.1% | V13 B1 realization, v12 2k 리뷰 |
| exact writer 배제 | 같은 plan에 MD Q 471, CD Q 581. batch Q가 CD 581→702, MD 471→835로 증가. W20 NS 71.0, 70.0 | V13 MD/CD 2k |
| 예산을 절대 요청 에너지로 | ridge 비용 Q/‖R‖²가 다섯 층 모두 0.229–0.242. anchor는 L4 3.06, L8 6.10. v12 plan에서 L7은 상대 예산 27%, 요청 에너지 47% | V13 B1 writer-summary |
| 기본 반경은 최하층 native cap, 필요할 때만 확장 | MEMIT-BLUE는 L4에 2.30(=0.75·a\_4)을 항상 요청. L8 잔차 평균은 B1 1.97, B10 0.43. W20 99.4/95.6/79.7 | BLUE 종합 리뷰 layer-updates |
| 조기 종료 대신 반경 게이트 | V14 B1에서 norm 항이 후보 1부터 0.0621로 일정. 2,400 update 전부 사영 적용. 평균 F는 후보 10에서 0.040, 후보 24에서 0.009 | V14 candidate-summary |
| K/P는 stop-gradient | V14 fit 2,038초 중 build 107초, reverse 722초, native F/B 962초, 정지 평가 234초. HJ에서 frozen-upper 95.7/79.0, full joint 95.25/79.5 | V14 R2 계측, HJ 1k 리뷰 |
| 배분이 history를 보게 함 | v9 ridge 실현율 B1→B20: L4 0.613→0.583, L7 0.634→0.420, L8 0.624→0.318 | v9 terminal-realization |
| 같은 에너지 대조를 claim 근거로 | HJ(BS10×100): joint 99.6/95.25/79.49, energy-matched divisor 99.6/94.4/75.93 | HJ 1k 리뷰 |

baseline 수치는 다른 runtime의 역사 기록이다. BLUE와 MEMIT-H의 요청 에너지는 직접 기록된 값이 없다.

이전 답변의 'F 기준 종료'는 여기서 '반경 확장 게이트'로 바꿨다. Adam의 첫 update가 반경을 바로 채우므로, 종료 시점으로는 에너지를 줄일 수 없기 때문이다.

## 3. 기호와 batch entry

batch entry에서 고정하는 것은 weight, history, metric, anchor, KL teacher다. fit 동안 이들은 바뀌지 않고, 변수는 요청 R 하나뿐이다.

| 기호 | 정의 |
| --- | --- |
| W^e, H^e\_l, C0\_l | batch entry의 선택 weight, 층 l의 history Gram, native covariance |
| A\_l | λ\_C·C0\_l + H^e\_l. native 조립 순서와 dtype 유지, 대칭화와 jitter 없음 |
| 𝓛, ℓ0, ℓ\* | 순서 있는 eligible 쓰기 층, 그중 첫 층, native anchor 층 (현재 L4–L8, L4, L8) |
| r, B | 요청과 논리 batch 크기. 고정 100이 아니다 |
| j, r(j), N | native 주입 행, 그 소유 요청, 총 행 수. 요청마다 rewrite 행과 KL 행 1개 |
| a\_lr | entry 모델의 canonical 행에서 층 l block 출력의 subject norm. 양의 유한값이어야 한다 |
| a\*\_r, a0\_r | a\_{ℓ\*,r} 와 a\_{ℓ0,r} |
| R\_lr, R\_l | 층 l, 요청 r의 요청 잔차 벡터와 그 d\_l×B 행렬. 0에서 시작 |
| K̄\_l, P\_l | 후보의 실제 rewrite 평균 key(n\_l×B)와 ridge 해 |
| v\_lj | 후보 weight가 행 j의 subject key에 낸 실제 작용 |
| M\_l | 행 응답 행렬 P\_l^T·K\_l^row (B×N). v\_l = R\_l·M\_l |
| Γ\_l | 에너지 형식 P\_l^T·A\_l·P\_l. Q\_l = tr(R\_l·Γ\_l·R\_l^T) |

R은 요청이고 v는 실현이다. 이 문서에서 두 양을 같은 이름으로 부르지 않는다.

요청별 반경과 층별 cap은 모두 native clamp 계수 c로 정한다.

```latex
\rho^{\mathrm{base}}_r=\min\{c\,a^{0}_r,\;c\,a^{\star}_r\},\qquad
\rho^{\max}_r=c\,a^{\star}_r,\qquad
\bar c_{lr}=c\,a_{lr}
```

기본 반경은 최하층에 native 요청 하나를 가득 넣는 크기다. 최대 반경은 anchor 층의 native z 요청 하나와 같다. 현재 profile의 평균은 2.30과 4.57이다.

| profile 상수 | 값 | 비고 |
| --- | --- | --- |
| c | 0.75 | native clamp 계수 |
| λ\_K, λ\_n, λ\_C | 0.0625, 0.5, 15000 | native KL, decay, ridge 계수 |
| η, ε, (b1, b2) | 0.1, 1e-8, (0.9, 0.999) | native Adam 값, 절대 좌표에 그대로 적용 |
| K\_eval | 25 | 최대 평가 수, update는 최대 24 |
| K\_grace | 12 | 기본 반경에서 머무는 update 수 |
| n\_exp | 4 | 기본에서 최대까지의 확장 단계 수 |
| τ\_F | 0.05 | native 정지 임계값을 task 손실에 적용 |

이 값들은 method 상수가 아니라 profile이다. K\_grace와 n\_exp만 이 방법이 새로 둔 값이다.

## 4. 후보 하나의 실제 write (BUILD)

후보 R 하나는 batch entry에서 출발해 층 순서대로 실제 MEMIT write를 적용한 모델이다. 계획이 평가하는 대상과 commit하는 대상이 이 모델 하나로 같다.

각 쓰기 층 l에 대해 오름차순으로 다음을 수행한다. gradient는 만들지 않는다.

1. 하층 후보 weight가 모든 token에 적용된 실제 상태에서, 모든 native 행의 subject key k\_lj를 읽는다.
2. rewrite 행만으로 native nested 평균 key K̄\_l을 만든다. KL 행은 평균에 넣지 않는다.
3. native ridge를 전체 논리 batch로 한 번 푼다.
4. 후보 weight를 FP32로 만들고, 그 weight가 각 행의 key에 낸 실제 작용 v\_lj를 기록한다.
5. 같은 후보 weight로 모든 token을 처리해 다음 쓰기 층의 경계로 간다.

```latex
P_l=(A_l+\bar K_l\bar K_l^{\top})^{-1}\bar K_l,\qquad
U_l=R_lP_l^{\top},\qquad
W_l^{\mathrm{cand}}=W_l^{e}+\operatorname{cast}_{32}(U_l)
```

```latex
v_{lj}=\bigl(W_l^{\mathrm{cand}}-W_l^{e}\bigr)k_{lj}\;\approx\;R_l\,M_l[:,j],\qquad
M_l=P_l^{\top}K_l^{\mathrm{row}}
```

v는 FP32 effective weight의 작용이고, R·M은 그 FP64 이상값이다. 손실은 v로 평가하고 gradient는 M으로 당긴다.

writer 에너지는 요청 에너지의 4분의 1을 넘지 못한다. 대칭 양의 정부호 A에서 성립하며, 이것이 6절의 예산이 writer 비용을 묶는 근거다.

```latex
Q_l=\operatorname{tr}(U_lA_lU_l^{\top})=\operatorname{tr}(R_l\Gamma_lR_l^{\top})\le\tfrac14\lVert R_l\rVert_F^{2},\qquad
\Gamma_l=P_l^{\top}A_lP_l
```

구조적 성질은 네 가지다.

- 첫 쓰기 층의 key와 P는 R에 의존하지 않는다. entry에서 한 번 계산해 모든 후보가 공유한다.
- 층 l의 key는 층 l 이상의 write에 의존하지 않는다. 순환 정의가 없고, commit 뒤 최종 모델의 key와 같다.
- R이 0인 요청의 열도 solve에 남는다. 다른 요청의 write가 그 key에 새는 양이 v에 그대로 나타난다.
- 물리 microbatch는 실행 단위다. 평균 key와 solve는 항상 전체 논리 batch로 한다.

## 5. 목적, 손실 경로, gradient

손실은 native 문장과 reduction 그대로 두고, 후보의 실제 작용 v를 subject 위치에 주입한 경로에서 계산한다. 주입 좌표는 v12와 같은 full block 출력이다.

```latex
h_{l,j}[s_j]\;\leftarrow\;h^{\mathrm{base}}_{l,j}[s_j]+v_{lj}
```

이 경로의 weight는 entry 그대로다. 하층 주입이 상층 hidden과 손실에 주는 영향은 graph에 남기고, weight와 anchor와 teacher에는 gradient를 주지 않는다.

```latex
F_r(R)=\mathrm{NLL}^{\mathrm{native}}_r\bigl(v(R)\bigr)+\lambda_K\,\mathrm{KL}_r\bigl(p^{\mathrm{cur}}\,\Vert\,p^{\mathrm{entry}}\bigr)
```

```latex
J(R)=\sum_r F_r(R)+\sum_r\beta_r\sum_{l\in\mathcal L}\lVert R_{lr}\rVert_2,\qquad
\beta_r=\frac{\lambda_n}{(a^{\star}_r)^{2}}
```

norm 항은 절대 요청 1단위에 층 공통 가격을 준다. anchor 층 하나만 쓰면 native decay와 같은 식이 된다.

### subject 경로를 쓰는 이유

subject 경로는 subject가 아닌 token에서의 후보 weight 작용을 포함하지 않는다. V14 B1의 terminal에서 두 경로의 차이는 NLL 0.00141 대 0.00132, KL 0.1261 대 0.1268였다.

all-token 손실로 fit한 two-arm과 v5는 PS가 67–73%였고, subject 경로를 쓴 V14는 B1에서 90.5%였다. 원인이 분리된 비교는 아니지만, 이 명세는 검증된 쪽을 쓴다. terminal에서 실제 all-token 손실은 별도로 기록한다.

### 같은 층 응답 pullback

요청 group마다 subject 경로를 한 번 backward해 행별 adjoint λ\_lj를 얻고, 행 응답 행렬로 요청 좌표에 당긴다.

```latex
\nabla_{R_l}\Bigl(\sum_r F_r\Bigr)\;\approx\;\Lambda_lM_l^{\top},\qquad
\Lambda_l=[\lambda_{lj}]_j\in\mathbb R^{d_l\times N},\quad
\lambda_{lj}=\frac{\partial\sum_rF_r}{\partial v_{lj}}
```

| 경로 | 처리 |
| --- | --- |
| R\_l → v\_l (같은 층) | 정확. P\_l과 k\_lj는 R\_l에 의존하지 않는다 |
| v\_l → 상층 hidden → 손실 | 정확. subject 경로 graph에 있다 |
| 다른 요청의 R → 내 행의 v (M의 비대각) | 포함. 요청별로 잘라내지 않는다 |
| 하층 R → 상층 key와 P → 상층 v | 생략 (stop-gradient) |

생략은 gradient에만 있다. 값은 매 후보의 BUILD가 실제 key와 P로 다시 계산한다. 마지막 쓰기 층과 단일 층 문제에서는 이 pullback이 완전한 gradient와 같다.

norm gradient는 해석식 β\_r·R\_lr/‖R\_lr‖로 한 번만 더한다. R\_lr이 0이면 0으로 둔다.

## 6. 요청 에너지 예산과 반경

요청마다 층별 요청의 제곱합을 반경 ρ\_r의 제곱으로 묶고, 각 층에는 그 층의 native cap을 둔다. 예산의 단위가 ridge writer의 비용 단위와 같아진다.

```latex
\mathcal C_r(\rho)=\Bigl\{R_{\cdot r}:\ \sum_{l\in\mathcal L}\lVert R_{lr}\rVert_2^{2}\le\rho^{2},\quad
\lVert R_{lr}\rVert_2\le\bar c_{lr}\ \ \forall l\Bigr\}
```

4절의 상한을 더하면 batch 전체의 writer 에너지가 묶인다. history가 쌓여도 이 상한은 커지지 않는다.

```latex
\sum_{l}Q_l\;\le\;\tfrac14\sum_r\sum_l\lVert R_{lr}\rVert_2^{2}\;\le\;\tfrac14\sum_r\rho_r^{2}
```

### 가능 영역이 포함하는 것

| 기존 방법의 요청 | 포함 여부 |
| --- | --- |
| anchor 층 단독 native z, ‖R‖ ≤ c·a\* | 최대 반경에서 정확히 포함. 단층이면 가능 영역이 native clamp ball과 같다 |
| 최하층 단독 native 요청 (BLUE의 L4 단계) | 기본 반경에서 정확히 포함 |
| 최하층 cap + anchor 층 보충 (BLUE의 L4+L8) | 최대 반경에서 anchor 층 요청 0.87·c·a\*까지 포함 (현재 profile) |
| v12의 상대 norm 합 0.75 | 다른 기하. 포함 관계가 아니다 |

### 반경은 필요할 때만 커진다

모든 요청은 기본 반경에서 시작한다. K\_grace번 update된 뒤에도 현재 후보에서 F\_r ≥ τ\_F인 요청만 한 단계씩 반경을 키운다.

```latex
\rho_r=\rho^{\mathrm{base}}_r\,g_r^{\,e_r},\qquad
g_r=\bigl(\rho^{\max}_r/\rho^{\mathrm{base}}_r\bigr)^{1/n_{\exp}},\qquad
e_r\in\{0,\dots,n_{\exp}\}
```

만족한 요청의 반경은 그대로 두고, 반경을 줄이는 규칙은 없다. 현재 profile에서 한 단계는 반경 1.19배, 에너지 1.41배이고 4단계에 에너지 3.96배다.

종료가 아니라 반경으로 총량을 정하는 이유는 optimizer의 보폭에 있다. Adam의 첫 update는 좌표마다 약 η만큼 움직여, 4,096차원 block의 norm이 6.4까지 나온다. 반경이 첫 update에서 차므로, 일찍 멈춰도 쓴 에너지는 같다.

같은 기준의 선례가 native에 있다. compute\_z는 첫 손실이 0.05 미만이면 δ=0으로 끝난다. BLUE의 L8 잔차 평균이 B10에서 0.43으로 줄어든 것은 이 규칙과 맞는다. BLUE 소스는 이 저장소에 없어 직접 확인하지는 못했다.

### 정확한 사영

제안값 x를 가능 영역에 Euclidean 사영한다. 해는 block마다 방향을 유지하고 크기만 바꾼다.

```latex
\Pi(x)_l=\min\{\bar c_{lr},\ \theta\lVert x_l\rVert\}\,\frac{x_l}{\lVert x_l\rVert},\qquad
\sum_l\min\{\bar c_{lr},\ \theta\lVert x_l\rVert\}^{2}=\rho^{2}\ \ (\theta\le1)
```

- cap만 적용해도 에너지가 반경 안이면 θ=1이다. 아니면 θ를 분기점 c̄\_lr/‖x\_l‖를 정렬해 닫힌 식으로 구한다.
- 계산은 FP64, 저장은 FP32다. 저장값의 위반이 1e-6을 넘으면 기술 실패로 남기고 숨은 축소를 하지 않는다.
- 사영 뒤 Adam moment는 유지한다.
- 이 사영은 block을 0으로 만들지 않는다. 모든 층이 끝까지 후보로 남고, 층 희소성은 강제하지 않는다.

## 7. Optimizer와 요청별 반경 제어

요청마다 독립된 EfficiencyAdam 상태를 두고, 절대 요청 좌표에서 native Adam 값으로 update한다. 반경은 요청별 controller가 정하고, 방향과 층 배분은 optimizer가 정한다.

```latex
m\leftarrow b_1m+(1-b_1)g,\qquad
v\leftarrow b_2v+(1-b_2)g^{\odot2},\qquad
s_l\leftarrow b_2s_l+(1-b_2)\operatorname{mean}_i\bigl(g_{l,i}^{2}\bigr)
```

```latex
\gamma_l=\sqrt{\hat s_l\big/\operatorname{mean}_{q\in\mathcal L}\hat s_q},\qquad
\tilde R_{lr}=R_{lr}-\eta\,\gamma_l\,\frac{\hat m_l}{\sqrt{\hat v_l}+\varepsilon},\qquad
R_{\cdot r}\leftarrow\Pi_{\mathcal C_r(\rho_r)}\bigl(\tilde R_{\cdot r}\bigr)
```

g는 5절의 pullback gradient에서 그 요청의 열과 norm gradient를 더한 것이다. hat은 그 요청의 실제 update 횟수로 한 bias correction이다. 모든 ŝ가 0이면 γ=1로 둔다.

### 층 배분이 정해지는 방식

반경이 찬 상태에서 사영은 cap에 걸리지 않은 block을 같은 비율로 줄인다. 따라서 층별 요청 크기의 비는 γ가 정하고, γ는 그 층 gradient의 RMS에 비례한다.

결과적으로 요청 에너지는 층별 gradient 제곱 크기에 가깝게 나뉜다. 이것은 선형화한 손실 감소를 에너지 예산 안에서 최대화하는 1차 배분과 같은 방향이다.

gradient가 행 응답 M을 거치므로 실현율이 낮은 층은 배분이 줄어든다. history가 쌓여 상층 실현율이 떨어지면 배분이 따라 움직인다. 좌표별 정규화와 유한 횟수 때문에 정확한 최적 배분은 보장하지 않는다.

층이 하나면 γ=1이고, update 식은 그 층의 native compute\_z Adam과 같다. v12에서는 이 대응이 anchor 층에서만 성립했다.

### 요청별 controller

매 후보에서 F\_r를 본 뒤, 그 요청의 다음 update를 아래처럼 정한다.

| 요청 상태 | 조건 | 동작 |
| --- | --- | --- |
| 무편집 | 후보 0에서 F\_r < τ\_F | R\_r = 0 유지. update와 backward 없음 |
| 무편집 해제 | 무편집 요청이 이후 F\_r ≥ τ\_F | 일반 요청으로 편입, 기본 반경에서 시작 |
| 기본 구간 | update 수 < K\_grace | 기본 반경에서 update |
| 만족 | update 수 ≥ K\_grace, F\_r < τ\_F | 현재 반경에서 update 계속 |
| 미달 | update 수 ≥ K\_grace, F\_r ≥ τ\_F | 반경 한 단계 확장 후 update (최대 n\_exp 단계) |

만족한 요청도 update를 멈추지 않는다. 같은 반경 안에서 방향과 배분을 다듬는 것은 에너지를 늘리지 않기 때문이다. V14 B1에서 norm이 일정한 동안 평균 F는 0.040에서 0.009로 내려갔다.

판정은 항상 그 후보의 실제 값으로 한다. 다른 요청의 write가 새어 들어와 F\_r가 다시 τ\_F를 넘으면 그때 확장한다.

### 종료

- 평가는 최대 25회, update는 최대 24회다. 마지막으로 평가한 후보가 terminal이고, terminal에서는 backward와 update를 하지 않는다.
- 모든 요청이 무편집이고 만족이면 그 후보에서 끝낸다. 그 밖의 조기 종료는 없다.
- terminal의 요청 상태는 ZERO\_STEP, SATISFIED\_BASE, SATISFIED\_EXPANDED, UNSATISFIED\_MAX, UNSATISFIED 중 하나로 기록한다.
- 미만족과 미수렴은 기술 실패가 아니다. 유한값, 예산 feasibility, 후보와 commit의 일치를 통과하면 commit한다.

## 8. Commit, history, transaction

commit은 terminal 후보의 BUILD가 만든 FP32 weight를 그대로 복사하는 것이다. 새 key로 다시 풀지 않고, 같은 update를 두 번 더하지 않는다.

계획이 평가한 작용과 commit의 작용이 같으므로 보정할 차이가 없다. 그래서 아래 네 가지를 쓰지 않는다.

- 가상 목표와 실제 hidden의 차이를 상층에 다시 쓰는 tracking 잔차
- 남은 층 수로 나누는 divisor
- 실현율의 역수를 곱하는 증폭
- mean-key 또는 context별 exact 제약

history는 v12와 같다. 모든 층을 commit한 뒤 최종 모델의 rewrite-only native 평균 key로 각 층 H에 정확히 한 번 append한다. KL key는 넣지 않고, 성공한 요청만 고르지 않으며, R이 0인 요청의 key도 포함한다.

terminal BUILD의 평균 key는 최종 모델의 key와 같아야 한다. parity를 검증한 profile에서는 이 key를 재사용하고, 아니면 최종 모델에서 다시 읽는다.

후보를 평가하는 동안 저장된 W와 H는 바뀌지 않는다. 기술 실패가 나면 batch entry의 W, H, RNG, cache를 복원하고 부분 commit을 남기지 않는다.

다음 batch는 자기 entry에서 anchor, teacher, metric factor, 첫 층 기하를 새로 만든다. optimizer 상태와 반경 단계는 batch를 넘기지 않는다.

## 9. 전체 알고리즘

batch 하나는 최대 25번의 후보 평가로 끝난다. 후보마다 실제 write를 만들고, 그 실현 작용으로 손실과 gradient를 얻고, 요청별로 반경을 정해 update한다.

&#91;embedded content: batch 하나의 fit 흐름 · 7단계, 분기 1개\]

왼쪽 열이 후보마다 도는 loop다. 마지막 평가에서만 오른쪽으로 빠져, 평가한 weight를 그대로 commit한다.

```text
입력: batch entry (W^e, H^e, A_l), native 행, anchor, KL teacher
상태: R = 0, 요청별 Adam 상태, update 수 t_r = 0, 확장 단계 e_r = 0

for k = 0 .. K_eval-1:
    built = BUILD(R)                        # 4절. 실제 key, P, 후보 weight, v, M. gradient 없음
    for 요청 group:                          # 5절. subject 경로에 v 주입
        F_r = NLL_r(v) + λ_K · KL_r(v)
        if k == 0: zero_r = (F_r < τ_F)
        if zero_r and F_r < τ_F: continue    # 무편집 요청은 forward만
        zero_r = False
        if k < K_eval-1: backward → λ_lj     # terminal 후보에서는 생략
    if k == K_eval-1 or 모든 요청이 무편집: break

    g_l = Λ_l · M_l^T  (모든 층)              # 같은 층 응답 pullback, FP64
    for 무편집이 아닌 요청 r:
        if t_r ≥ K_grace and F_r ≥ τ_F and e_r < n_exp: e_r += 1
        ρ_r = ρ_base_r · g_r^(e_r)
        R~ = EfficiencyAdam_r(R_r, g[:, r] + β_r · R_r/‖R_r‖)
        R_r = Π_{C_r(ρ_r)}(R~);  t_r += 1

commit:  W ← built의 후보 weight (terminal 후보 그대로)
history: H_l += K̄_l · K̄_l^T  (최종 모델의 rewrite 평균 key, 층마다 한 번)
```

논리 호출 수는 batch당 BUILD 최대 25회, subject 경로 forward 최대 25회, backward 최대 24회다. 추가 terminal forward와 backward는 없다.

## 10. 기존 방법과의 대응과 차이

v12-R은 V14의 '실제 후보를 평가하고 그대로 commit'을 가져오되, V14의 비용 원인인 builder reverse를 뺀 것이다. v12와는 주입량, 예산 기하, 좌표가 다르다.

| 항목 | v12 | V13 CD | V14 | v12-R |
| --- | --- | --- | --- | --- |
| 계획 시 주입량 | 요청 (가상) | 요청 (가상) | 실제 작용 v | 실제 작용 v |
| gradient | 가상 경로 | 가상 경로 | 완전 causal (K, P 미분) | 같은 층 pullback |
| writer | ridge, tracking 잔차 | context별 exact | ridge, 요청 직접 | ridge, 요청 직접 |
| 계획과 commit | 불일치, debt가 상층으로 | 직접 작용만 일치 | 일치 | 일치 |
| 예산 | 상대 norm 합 0.75 | 상대 norm 합 0.75, 실현 100% | 요청의 상대 norm 합 0.75 | 요청 에너지 반경, 층별 cap |
| 총량 | 항상 포화 | 항상 포화 | 항상 포화 | 기본 반경, 미달 요청만 확장 |
| history가 쌓일 때 | 상층이 debt를 덜 갚음 | 에너지 증가 | 실현량 감소 | 에너지 상한 고정, 배분 이동 |
| 요청 간 결합 | 없음 | writer에서만 | 있음, 공통 정지 | 있음, 요청별 반경 |
| B1 fit 시간 | 385초 | 385초 (같은 planner) | 2,038초 | 약 500초 (추정) |

### native와 BLUE에 대한 환원

- 쓰기 층을 anchor 층 하나로 제한하면 좌표, 보폭, clamp 집합, decay 식이 native compute\_z와 같다. 주입량이 실현 작용이라는 점과 writer가 요청을 직접 쓴다는 점은 다르다.
- 쓰기 층을 최하층 하나로 제한하고 n\_exp를 0으로 두면 요청 집합이 BLUE의 L4 단계와 같다.
- BLUE의 'L8은 필요할 때만'은 반경 확장 게이트에 대응한다. BLUE는 L4를 쓴 뒤 L8 목표를 다시 계산하는 순차 방식이고, v12-R은 한 번의 공동 fit 안에서 매 후보의 실제 값을 본다.
- MEMIT-H의 divisor 분산에 대응하는 설정은 없다.

이 환원들은 요청 집합과 optimizer 식의 대응이다. 같은 결과가 나온다는 주장이 아니다.

## 11. 구현 계약

새로 짜야 하는 것은 pullback, 절대 좌표 optimizer, 사영, controller 네 가지다. 나머지는 V14와 v12의 검증된 함수를 그대로 쓴다.

| 단계 | 함수 | v12-R에서의 변경 |
| --- | --- | --- |
| entry | `jlz_realized_subject.entry.prepare_entry`, `routes.annotate` | group을 요청 하나의 전체 native 행으로. 반경, cap, β를 추가 계산 |
| BUILD | `jlz_native_writer_aware.builder.build` | 그대로. 행 key `raw`, `P`, 행 작용 `v`, 후보 weight를 반환 |
| 손실 | `subject.evaluate`, `LlamaAdapter.masked` | group의 forward 직후 backward (v12 방식). 무편집 요청과 terminal은 forward만 |
| pullback | 신규 | 층마다 `adjoint[l].T @ (raw[l] @ P[l])`, FP64 축약 한 번 |
| optimizer | 신규 `EfficiencyAdamAbs` | v12의 anchor 스케일(lr/a\*, eps·a\*)을 제거. 요청별 상태 유지 |
| 사영 | 신규 `project_capped_energy` | 6절의 닫힌 식. FP64 계산, FP32 저장, 저장값 검사 |
| controller | 신규 | 7절의 표. 요청별 update 수와 확장 단계만 상태로 가진다 |
| commit, history | `telemetry.commit_measure`, `writer.commit`, `Transaction` | 그대로 |
| 쓰지 않음 | `builder.reverse`, `routes.cached_vjp`, R4a probe, v12 `writer.apply` | reverse는 qualification의 대조 gradient로만 쓴다 |

pullback 식은 V14 `physical.Linear.backward`가 같은 층 R에 대해 계산하던 항과 같다. V14의 reverse에서 key, P, 경계를 거치는 항만 빠진다.

### 계산량

| 구간 (B100, 후보 25) | 근거 | 초 |
| --- | --- | --- |
| BUILD 25회 | V14 B1 실측 (builder forward) | 107 |
| subject 경로 forward 25, backward 24 | v12 planner 실측 (V13 B1 fit 전체) | 386 |
| pullback, optimizer, 사영 | V14 optimizer 6.9초에서 추정 | 10 미만 |
| 합계 | 추정, 미측정 | 약 500 |

V14의 2,038초에서 빠지는 것은 builder reverse 722초, 정지 판정용 forward 234초, MB2 재생으로 늘어난 native F/B다. 2k 한 arm은 v12의 4.1 GPU시간보다 조금 긴 수준으로 예상한다. 실측 전에는 ETA로 쓰지 않는다.

### 정밀도와 실패 처리

- 모델, activation, 요청 R은 FP32다. ridge 기하, pullback 축약, 사영은 FP64다.
- ridge는 native 저장 A를 그대로 쓴다. solve 상대 잔차가 1e-8을 넘으면 기술 실패다.
- 비유한 값, anchor 실패, 사영 저장 위반, commit 불일치는 기술 실패로 끝낸다. jitter, 반경 축소, 후보 수 축소로 통과시키지 않는다.
- 품질, 특정 층 집중, 미만족 요청 수, 반경 확장 비율은 gate가 아니라 기록 대상이다.

### 범용성

쓰기 층과 차원, lookup, native reduction, context 수는 adapter와 pack에서 받는다. 논리 batch 크기와 요청당 행 수를 코드에 고정하지 않는다. 임의 batch 크기를 지원한다는 것이 batch를 나눠도 같은 결과라는 뜻은 아니다.

## 12. Telemetry

기록의 목적은 세 가지를 따로 볼 수 있게 하는 것이다. 얼마나 요청했는가, 얼마나 실현됐는가, writer가 얼마를 치렀는가.

| 단위 | 기록 | 해석 주의 |
| --- | --- | --- |
| 후보 × 요청 | F\_r, NLL, KL, 요청 에너지, 반경, 확장 단계, 상태, update 수 | F는 subject 경로의 값이다 |
| 후보 × 요청 × 층 | 요청 norm, 요청 에너지 share, cap 접촉, γ, 사영의 θ | share는 요청 기준이다. 실현 share와 구분한다 |
| 후보 × 층 | solve 잔차, ideal Q, update norm | Q는 그 후보 하나의 값이다 |
| terminal × 요청 × 층 | 실현 작용의 방향비 ⟨v,R⟩/‖R‖², norm 비, cosine. 평균 key, canonical, rewrite 행, KL 행 각각 | 요청이 0이면 비율은 null로 둔다 |
| terminal × 요청 × 층 | 다른 요청의 write에서 새어 들어온 작용의 norm | 0 요청에서는 이것이 전부다 |
| terminal × 층 | ideal Q, FP32 effective Q, update Frobenius norm, 요청별 용량 k̄^T·A^-1·k̄의 분포 | 용량 추이가 history 포화의 직접 지표다 |
| terminal × batch | subject 경로 손실과 실제 all-token 손실의 차, 요청 상태 분포, 요청 에너지 합, Q 합과 그 상한 | 차이는 기록 대상이고 gate가 아니다 |
| 평가 | RS, PS, NS 선호율과 TF strict, token-micro, prompt-macro, new와 true NLL, cohort 유지 | v12 2k와 같은 정의와 분모를 쓴다 |
| 비용 | BUILD, subject F/B, pullback, optimizer 시간, peak VRAM과 RSS, 논리와 물리 호출 수 | 중첩 timer를 더하지 않는다 |

평가 문항은 fit, 반경 판정, 계수 선택에 쓰지 않는다.

## 13. 검증

수식과 상태 규칙은 CPU reference로 확인했고, 실모델 qualification은 아직 하지 않았다. CPU 통과는 GPU 통과나 성능의 근거가 아니다.

### CPU reference (완료)

`v12r_reference.py`는 numpy와 autograd만 쓴다. toy는 모든 행이 subject token 하나라서 subject 경로와 실제 경로의 값이 같다. gradient 근사만 따로 볼 수 있게 한 구성이다.

| 검사 | 결과 |
| --- | --- |
| 사영의 최적성 | 무작위 300건에서 SLSQP 대비 목적값 차 8.5e-12 이하. 모든 해가 feasible |
| 단층 환원 | 층이 하나이고 반경이 c·a\*이면 사영이 native clamp와 같다 |
| ridge 항등식 | 응답, 에너지 식, Γ의 최대 고유값 ≤ 1/4를 기계 정밀도로 확인 |
| exact와 ridge의 관계 | mean-key exact는 RHS를 D(I+S^-1)로 둔 ridge와 같다 |
| pullback 식 | stop-gradient 목적의 유한차분과 1.9e-9 이내로 일치 |
| pullback 대 완전 gradient | 마지막 층 3e-16, 하층 2.6%와 1.7% (cosine 0.9997, 0.9999). 단일 층 5e-16 |
| 후보와 commit | terminal weight를 commit한 뒤 손실이 6e-16 이내로 재현 |
| controller | 반경 비감소, 기본 구간 중 확장 없음, 무편집 요청 불변, 해제 뒤 편입 |

toy planner에서는 기제만 확인했다. 요청 6개, 층 3개, 한 층의 metric을 25배로 둔 구성이다.

| toy 설정 | 만족 | 평균 요청 에너지 / 최대 | writer 에너지 Q |
| --- | --- | --- | --- |
| v12-R | 6/6 (3개는 기본 반경) | 0.51 | 2.04 |
| 처음부터 최대 반경 | 6/6 | 1.00 | 4.37 |
| 같은 반경 규칙, writer를 모르는 gradient | 6/6 | 0.54 | 2.93 |

writer를 모르는 gradient는 실현율이 낮은 층에 요청 에너지의 31%를 썼고, v12-R은 20%를 썼다. 이 수치는 언어 모델에 대한 증거가 아니다.

### 실모델 qualification (미수행)

2–4개 native 요청의 고정 후보에서 수행한다. fit, update, 물리 commit은 하지 않는다.

| 검사 | 기준 | 성격 |
| --- | --- | --- |
| BUILD가 명시적 순차 write, 그리고 V14 MB2 경로와 일치 | key, P, 후보 weight, v에서 atol 2e-5 + rtol 2e-4 | gate |
| 요청 단위 group의 손실이 기준 경로와 일치 | 요청별 loss atol 1e-5, rtol 1e-4 | gate |
| pullback이 V14 reverse의 같은 층 항과 일치 | RMS 차 ≤ 1e-6 + 1e-3 × 기준 RMS | gate |
| 마지막 쓰기 층에서 pullback이 완전 gradient와 일치 | 같은 기준 | gate |
| 하층에서 pullback 대 완전 causal gradient | 층별 상대오차와 cosine | 기록 |
| subject 경로 손실 대 실제 all-token 손실 | 요청별 차 | 기록 |
| 사영 저장값 feasibility | 위반 ≤ 1e-6 | gate |
| commit 정합 | 평가 weight와 commit weight의 SHA 일치, history 층당 1회, terminal key와 최종 key 일치 | gate |
| rollback | 실패 주입 뒤 W, H, RNG 복원 | gate |

허용오차는 V14 계약의 값을 그대로 쓴다. 결과를 보고 넓히지 않는다.

## 14. 실험 설계

실험은 하나의 연속 trajectory다. qualification 뒤 cold W0/H0에서 BS100×20을 이어 가고, B1은 그 첫 batch의 중간 보고 지점이다.

공통 조건은 v12 2k와 같다. Llama-3-8B-Instruct, 고정 CounterFact 앞 2,000개, L4–L8, FP32 모델과 FP64 기하, W0와 매 batch 전후 평가, W5·W10·W15·W20 누적 평가다.

| arm | 설정 | 분리하는 요인 | 우선 |
| --- | --- | --- | --- |
| MAIN | 전 층, 기본 반경 c·a\_L4, 최대 c·a\*, n\_exp 4 | 방법 전체 | 1 |
| BLIND | MAIN과 같되 gradient만 writer를 모름 (자기 행의 adjoint 합) | 실현 응답 gradient의 기여 | 2 |
| L4-ONLY | 쓰기 층을 L4 하나로, n\_exp 0 | 공동 배분이 최하층 단독보다 나은가 | 2 |
| NO-EXPAND | MAIN에서 n\_exp 0 | 필요 시 확장의 기여 | 3 |
| BASE-2X | MAIN에서 기본 반경의 에너지 2배 | 편집 강도와 보존의 교환비 | 2 |

대조 arm은 모두 같은 코드의 설정만 바꾼 것이다. baseline은 새로 돌리지 않고 기존 W20 수치를 역사 참고로 둔다.

| 참고 (W20, 역사 수치) | RS | PS | NS |
| --- | --- | --- | --- |
| MEMIT-BLUE (L4+L8) | 99.4 | 95.6 | 79.7 |
| MEMIT-H | 99.4 | 91.0 | 79.0 |
| v12 (tracking) | 98.5 | 90.3 | 76.1 |
| V13 CD | 98.6 | 93.6 | 70.0 |

### 결과를 읽는 법

아래는 사전에 정해 두는 해석이다. 진급 gate가 아니다.

| 관측 | 해석 |
| --- | --- |
| batch의 Q 합이 그 상한(요청 에너지 합의 1/4)을 넘는다 | 수치 오차를 넘으면 구현과 metric 대칭성을 점검한다 |
| W20 NS가 v12의 76.1 이하이고 PS도 비슷하다 | debt와 에너지 진단이 틀렸거나, subject 경로 근사가 누적에서 깨진다 |
| BLIND와 MAIN이 같다 | 효과는 값 일치와 예산에서 오고, 실현 응답 gradient의 기여는 없다 |
| L4-ONLY가 PS와 NS 모두 MAIN 이상이다 | 공동 배분 claim을 지지하지 않는다. 방법은 최하층 단독 편집의 변형이 된다 |
| 확장된 요청의 비율이 batch를 따라 늘어난다 | history 포화의 신호다. 층별 용량 추이와 함께 본다 |
| first500의 W5→W20 PS 하락 | v12의 −6.3, CD의 −7.5와 같은 분모로 비교한다 |

### 에너지 눈금 (B1)

기본 반경은 V14와 같은 에너지 수준이고, 최대 반경은 MD 수준이다. CD는 최대 반경보다 위에 있다.

| 방법 (B1, 같은 100요청) | Q 합 | 요청당 요청 에너지 (환산) | PS |
| --- | --- | --- | --- |
| V14 | 124 | 5.3 | 90.5 |
| v12 tracking | 240 | 10.2 | 88.5 |
| MEMIT-BLUE (다른 runtime, update norm에서 추정) | 약 350 | 약 14.7 | 92.5 |
| V13 MD | 471 | 20.0 | 94.0 |
| V13 CD | 581 | 24.6 | 95.0 |
| v12-R 기본 반경 | 132 이하 | 5.3 | 미측정 |
| v12-R 최대 반경 | 522 이하 | 20.9 | 미측정 |

요청 에너지 환산은 Q를 ridge 비율 0.236으로 나눈 값이다. exact writer에는 같은 update를 내는 ridge 요청의 크기로 읽는다. v12-R의 두 행은 평균 anchor로 계산한 상한이다.

BLUE는 B10 이후 batch Q가 약 180(추정)으로 내려가는데 누적 PS는 95대였다. 초기 PS는 에너지와 함께 오르지만, 누적 구간의 PS는 에너지만으로 정해지지 않는다.

### 비용과 순서

MAIN 한 arm은 약 4.7 GPU시간으로 추정한다. v12 2k의 4.1시간에 BUILD 20 batch분을 더한 값이고, 실측이 아니다.

구현 전에 할 수 있는 확인이 하나 있다. 기존 V13 B1 5-branch harness에 'L4만 exact, 상층은 ridge 직접' 가지를 더하면, debt의 원천이 L4인지 fit 1회로 볼 수 있다.

## 15. Claim 범위와 한계

이 방법이 구조로 보장하는 것은 네 가지이고, 성능은 그중 어느 것에도 들어 있지 않다.

1. 계획이 평가한 실제 작용을 그대로 commit한다. 계획과 실현의 차이가 없고, 그 차이를 상층에 넘기는 debt도 없다.
2. 모든 eligible 층의 요청을 한 fit에서 공동으로 정한다. 층별 배분은 실현 응답을 거친 gradient와 요청 에너지 예산이 정한다.
3. batch의 writer 에너지는 요청 에너지 합의 4분의 1로 묶인다. history가 쌓여도 이 상한은 커지지 않는다.
4. 요청별 총량은 기본 수준에서 시작하고, 실제 후보에서 미달인 요청만 늘린다.

### 주장하지 않는 것

- locality와 retention의 개선, baseline 대비 우위. 14절의 실험 대상이다.
- 최소 에너지나 최적 배분. Adam은 유한 횟수이고 좌표별로 정규화하며, gradient는 stop-gradient 근사다.
- 에너지가 의미적 손상의 상한이라는 것. C0와 history metric의 proxy다. MEMIT-H는 B1 update가 v12보다 큰데도 W20 NS가 높았다.
- τ\_F 만족이 paraphrase 일반화를 뜻한다는 것. 같은 plan에서 V14는 Q 124에 PS 90.5, CD는 Q 581에 PS 95.0이었다. 기본 반경이 PS를 좌우하는 주된 dial이다.
- subject 경로 손실과 실제 손실의 일치. B1에서 차이가 작았을 뿐, 누적 구간에서는 확인되지 않았다.
- native compute\_z나 BLUE와의 동일성.

### 미해결

- 기본 반경의 수준. 지금은 최하층 native cap이고, PS가 부족하면 BASE-2X arm이 그 답을 준다.
- K\_grace 12와 n\_exp 4. V14 B1의 손실 궤적 하나에 맞춘 값이라 다른 모델과 누적 구간에서 다시 봐야 한다.
- 요청 단위 group의 BUILD parity. CD-Q r1이 물리 regrouping의 gradient parity에서 실패한 전례가 있다.
- 누적 구간의 근사 오차. pullback과 subject 경로의 기록을 B5, B10, B20의 고정 후보에서도 남기는 것을 권한다.
- CD-Q와 CD-C 2k, V14 budget 1.5 2k, v12-AlphaEdit 2k의 결과. 저장소에 없어 이 명세에 반영하지 못했다.
