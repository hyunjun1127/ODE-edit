# JLZ v7 causal writer CPU 수학 검증

이 폴더는 **후보마다 하층 actual write가 만든 상층 key로 writer를 다시 계산하고, 그 key·solve의 미분을 하층 δ까지 전달하는 계산 그래프**를 검증한다. Python 표준 라이브러리만 사용한다. 언어 모델이나 GPU pilot 결과는 아니다.

```bash
python3 -B plans/global/2026-10-02-jlz-causal-writer-v7/math/validate_math.py
```

실행 결과는 [validation-results.json](validation-results.json), 명령·시각·소스 및 결과 SHA256은 [run-receipt.json](run-receipt.json)에 기록한다. 검증 스크립트를 수정하면 둘 다 다시 생성해야 한다.

## 검증 대상

작은 3층 모델에서 각 문맥은 두 token으로 이루어진다. 두 번째 token이 subject이며 그 입력은 첫 token의 상태에도 의존한다. Native surrogate는 고정 가중치에서 subject에만 공동 δ를 더한다. Actual branch는 각 층의 전체 current 문맥 key를 먼저 모아 shared writer를 풀고, 그 weight를 두 token 모두에 적용한 뒤 다음 층으로 진행한다. Actual 보조 loss는 일부 요청만 관측하지만 writer는 전체 요청으로 구성된다. Task loss는 미분 가능한 제곱 오차 surrogate이며 native LM NLL을 재현한 것은 아니다.

- B=1/3, 가변 문맥 수, 문맥 weight가 0인 경우의 dense/dual solve 일치.
- 전체 문맥 key의 causal layer 순서, 첫 writer cache, 상층 δ가 자기층·하층 key를 바꾸지 않는 성질.
- 하층 δ가 상층 writer를 바꾸는 효과와 이 경로의 total derivative.
- Native surrogate + dynamic G/E 비용 + selected actual surrogate의 전체 δ gradient를 중앙 차분과 비교.
- 같은 forward에서 P 미분만 끊었을 때 전체 gradient가 달라지는 반례.
- 전체 key를 모은 뒤 solve하는 microbatch 방식의 값·gradient 일치와 요청 permutation equivariance.
- 동일 후보 재구성 및 materialized weight를 고정한 commit replay의 key·출력 일치.
- 0 norm의 선택 subgradient 0. 일반 미분 가능성을 주장하지 않는다.

## Solve와 VJP

층과 후보 표기를 생략하면, 이번 batch 동안 고정된 SPD 행렬 A에 대해

\[
M=A+K\Omega K^\top,\quad \bar K=K\Omega Z^\top,\quad
P=M^{-1}\bar K,\quad U=DP^\top.
\]

K는 **같은 후보의 하층 actual write**로 얻는다. A, entry weight, clean anchor는 고정한다. 상층의 P 및 K를 detach하지 않는다.

\[
dP=M^{-1}\{dK\,\Omega(Z^\top-K^\top P)-K\Omega\,dK^\top P\}.
\]

Q=∂L/∂P, Y=M^(−T)Q이면 solve를 통한 key gradient는

\[
\left.\nabla_K L\right|_{P}
=Y(Z-P^\top K)\Omega-PY^\top K\Omega.
\]

E의 explicit K dependence나 모델 activation을 통한 다른 gradient는 여기에 추가한다. 하층→상층 K→P 경로를 끊으면 같은 forward 값을 얻더라도 이 total gradient와 달라진다.

Dual solve는 S=K√Ω, Q_A=A^(−1)S를 사용해

\[
P=Q_A(I+S^\top Q_A)^{-1}\sqrt\Omega Z^\top
\]

로 계산한다. 역행렬을 명시적으로 만들지 않고 SPD solve를 사용한다. Ω^(−1)이 없어 α=0에도 적용된다. Production에서는 고정 A의 분해를 재사용한다. 이 CPU 검증의 목적은 대수·미분 일치이며 구현 속도 측정이 아니다.

Row-major 선형층에서 X가 입력, G가 출력 gradient라면 direct VJP는

\[
\nabla_D L=G^\top(XP),\quad
\nabla_P L=X^\top(GD),\quad
\nabla_X L=G(W_{entry}+DP^\top).
\]

현재 v6 형태에서 P gradient를 반환하지 않는 backward를 재사용하면 이 계약을 만족하지 않는다. 위 식의 binary64 검증과 실제 FP32 materialization/cast 경계에서의 연산 재결합 검증은 구분한다.

## Geometry 항등식과 context weight

\[
G=P^\top AP,\quad
E=(P^\top K-Z)\Omega(P^\top K-Z)^\top,
\]

\[
G+E=Z\Omega Z^\top-\bar K^\top P.
\]

ZΩZᵀ=I인 이상적 정규화에서만 오른쪽 첫 항을 I로 바꿀 수 있다. FP32로 저장된 `.1`을 FP64로 읽으면 `.5 + 5 × .1`의 합이 정확한 1이 아닐 수 있다. 이 검증은 해당 경우를 포함하고 **기존 weight를 다시 정규화하지 않는다**. 실제 stored ZΩZᵀ를 사용한 항등식을 확인한다.

G/E는 후보에 따라 움직인다. 따라서 두 arm은 layer 비용을 합하는 방식의 차이이며, 전체 D에 대해 고정된 convex norm을 최적화한다는 주장은 하지 않는다. `1/sqrt(layer 수)` 분할 예시는 geometry와 효과를 인위적으로 고정한 경우로 한정해야 한다.

## 남는 한계

Local write의 목표는 계속 incremental D Z다. 절대 virtual z를 actual pre-write hidden에서 빼서 context별 residual로 바꾸는 새 writer를 도입하지 않는다. 따라서

\[
h^a-h^v=(h^{a,pre}-h^{v,pre})+(UK-DZ)
\]

의 첫 항, ridge shrinkage, 모든 token에 대한 weight 효과는 여전히 남는다. Dynamic key refresh가 없애는 것은 **하층 write 이전의 key를 고정해서 쓰는 불일치**다. 완전한 virtual/actual 일치나 locality 보존을 보장하지 않는다.

이 suite는 native LM 목적함수, Adam/clamp, teacher detach와 pulse 일정, FP32/FP64 경계, 실제 모델의 module 좌표, GPU 메모리·시간·성능을 검증하지 않는다. 이 항목은 SH4의 구현 및 model qualification에서 확인해야 한다. 작은 pilot에서 key 또는 P를 detach하거나 full current writer를 subset writer로 바꾸는 근사는 이 테스트가 검증한 method와 다르다.
