# V11 CPU 수식·계산그래프 검산

2026-10-04. 선택한 **entry geometry 고정 + full-batch allocation coupling + local increment writer** 계약을 작은 FP64 행렬과 비선형 causal surrogate로 검산했다. 결과: **PASS**. Production 모델·코드 parity나 성능 검증은 아니다.

재현:

```bash
/mnt/raid5/janghj/EasyEdit/.venv/bin/python plans/global/2026-10-04-jlz-native-increment-v11/math/validate_contract.py
```

스크립트는 표준 라이브러리와 CPU PyTorch만 사용하며, 결과를 같은 디렉터리의 `results.json`에 기록한다. Repository runtime을 import하거나 GPU/모델을 실행하지 않는다.

## 수치적으로 안정적인 비용

Entry geometry에서 \(S=\widehat K^T A^{-1}\widehat K\), \(T=I+S=L_TL_T^T\)를 고정한다. 비용과 arm A 결합은 다음과 같다.

\[
C_l=\|L_{T,l}^{-1}D_l^T\|_F^2
=\operatorname{tr}[D_l(I-\widehat M_l)D_l^T]=G_l+E_l,
\]

\[
c_l=\frac{\|L_{T,l}^{-1}D_l^T\|_F}{\sqrt{B\sigma_l^2}},
\qquad \sigma_l^2=\frac1B\sum_r a_{lr}^2,
\qquad \mathcal R_A=\sum_lc_l.
\]

구현 검산은 `I-M`의 차를 직접 만들거나 큰 두 값을 빼서 C를 얻지 않는다. Cholesky solve와 norm을 사용한다. **D=0에서 norm은 고전적 의미로 미분 가능하지 않다.** 최소 norm subgradient0을 선택하며 NaN 없이 전체gradient0인지 확인했다. `sqrt(sum(square))`의0점 backward를 그대로 사용하는 방식과 구분한다.

## 확인한 계약

- SPD Cholesky solve와 dense ridge solve 일치, ridge stationarity, stable cost와 실제 \(\operatorname{tr}(UAU^T)+\|U\widehat K-D\|_F^2\) 일치.
- B=1/2/3, layer수1/2/3, 서로 다른 write input 차원, hidden 차원3/4. Partial batch는 actual B와 actual anchor 평균으로 정규화한다.
- 같은 nonlinearity/causal token mixing 경로에서 모든층 direct subject delta를 함께 fit하고, NLL·current‖entry KL·비제곱 native-form norm·allocation gradient를 유한차분으로 검산한다.
- \(D_{lr}=s_{lr}q_{lr}\), \(s=a/\sqrt{d_lm}\). **\(B L_{mean}\)의 D-gradient를 s로 한 번 옮긴 q-gradient**와 직접q 역전파가 일치한다.
- Microbatch별 request loss의 합과 **whole-B allocation항을 한 번만 더한** gradient가 full 계산과 일치한다. Request loss의 SUM 계약에서는 allocation도 \(B\lambda_{alloc}\mathcal R_A\)다.
- B=1,m=1의 비제곱 delta norm은 native 형태와 정확히 같다. Added allocation, q Adam 및 다층확장까지 native와 동일하다는 주장은 하지 않는다.
- 다른 요청의 delta를 바꾸면 해당 요청의 gradient도 바뀐다. Upper delta를 바꾸면 lower task gradient도 바뀐다. Request/layer coupling을 실제 작은 계산그래프에서 확인했다.
- Commit은 아래층 actual write 후 upper key를 다시 측정한다. **전체층 D=0이면 해당층 U=0**이다. 다른 요청과의 cross-talk 때문에 개별 D열0은 그 요청의 실제 mean-key 변화0을 보장하지 않는다.
- Entry proxy와 actual commit 비용이 다르고, increment writer에서도 virtual/actual gap이 남는 반례를 보존한다. 이 차이는 검증 실패가 아니다.
- 고정예산25평가/24갱신, terminal 평가 이후 갱신0, 총joint-mean 목적함수<.05에서 전체조기종료를 검산한다. Per-request 종료/동결을 복제하지 않는다.
- 물리delta의 층·요청별 .75 clamp를 q에 적용하고 **Adam moment와 step state를 변경하지 않는** 계약을 확인한다.

## 결과의 범위

FP64 대수 허용오차1e−10, 유한차분 abs1e−7/relative1e−5를 사용했다. 네 dimensional case의 full joint-gradient 최대 절대오차는 **1.46e−9 미만**이다.

Increment writer 예에서 upper key가 entry에서 달라지고, 전체zero-delta층 U는0인 반면 해당층 hidden의 virtual/actual 차이는 남았다. Entry proxy/actual 비용 최대차0.00858, stale/refreshed key 사용 시 logits 최대차0.00603을 관측했다. 이는 **정확한 geometry 상태·배분 책임 구분이 필요한 이유를 보여주는 synthetic 값**이며 실제 모델의 크기·성능·시간을 예측하지 않는다.

실제 모델에서 token 위치, native reduction, weight dtype/cast, hook 위치, K 평균과 history append, evaluator, 동일candidate commit을 확인하는 qualification은 SH4 구현 단계에 남는다. 이 CPU PASS로 대체하지 않는다.
