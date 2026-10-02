# v8 CPU 수학 검증

이 디렉터리는 v8의 평균 key writer, causal gradient, 정규화 좌표 Adam에 대한 작은 CPU 검증이다. **실제 언어 모델, production runner, GPU 성능, 편집 품질을 검증한 결과가 아니다.** 다른 실험의 코드나 작업 상태를 변경하지 않는다.

## 실행 결과

- `validate_math.py`: 재현 가능한 독립 스크립트. Production 모듈을 import하지 않는다.
- `results.json`: **404/404 PASS**. 기존 207개 검사의 이름·순서·관측값을 그대로 보존하고 FP32 optimizer/bridge 검사 197개를 추가했다.
- `receipt.json`: 실행 환경·시간·스크립트 SHA256·결과 SHA256.
- Python 3.12.0 / PyTorch 2.9.1+cu128, CPU, `CUDA_VISIBLE_DEVICES=''`, `torch.cuda.is_available()==False`.
- Full causal finite difference의 최대 절대 오차: `1.2929639997549813e-11`.
- FP64 causal `D` leaf adjoint → `q` adjoint와 full autograd의 최대 절대 오차: `1.3877787807814457e-17`.
- 추가한 FP32 fixed-D bridge와 full autograd의 최대 절대 오차: `0.0`.
- Implicit solve VJP와 autograd의 최대 절대 오차: `8.326672684688674e-17`.

```bash
CUDA_VISIBLE_DEVICES='' /mnt/raid5/janghj/EasyEdit/.venv/bin/python -B \
  /mnt/raid5/janghj/.codex/worktrees/odeedit-jlz-v5-a-w5-review-20261002-v1/plans/global/2026-10-03-jlz-native-writer-v8/math/validate_math.py
```

재실행하면 이 디렉터리의 `results.json`, `receipt.json`을 갱신한다. 실행 시간은 작은 CPU 검증에 걸린 시간이며 method 가속 수치가 아니다.

## 검증한 정의

### Native 평균 key와 history

각 요청의 context를 기존 native group으로 유지한다. 각 group 내부의 FP32 평균을 계산하고, group 평균들의 FP32 평균을 계산한 뒤에 FP64로 변환한다.

\[
\kappa_r=\operatorname{FP64}\left[
  \operatorname{mean}_{g}^{\rm FP32}
  \left(\operatorname{mean}_{c\in g}^{\rm FP32} k_{r,c}\right)
\right].
\]

Group 크기 `[1,5]`에서 이를 실행한 값의 정확한 일치, 균일한 전체-context 평균과의 차이, FP64 변환을 평균 앞으로 옮겼을 때 생기는 차이를 검사했다. Context adjoint는 group 수와 해당 group 크기로 나누어 전달된다. Solve와 직접 E adjoint까지 합친 뒤 이 FP32 평균을 통과하는 gradient도 검사했다.

History는 CPU FP32에서 평균 key당 한 번의 Gram을 더하는 식이다.

\[
H_{t+1}=H_t+\kappa_{\rm FP32}\kappa_{\rm FP32}^{\top}.
\]

Shape·dtype·device, 한 번 더한 차이, full-context covariance와의 차이를 확인했다. **실제 commit 경로가 history를 정확히 한 번만 append하는지는 production 검증이 필요하다.** 여기서 테스트한 append 횟수는 toy 코드에서 한 번이다.

### 평균 key writer와 adjoint

\[
M=A+\kappa\kappa^\top,\quad P=M^{-1}\kappa,\quad U=DP^\top,
\]
\[
G=P^\top AP,\quad E=(P^\top\kappa-I)(P^\top\kappa-I)^\top,
\quad G+E=I-\kappa^\top P.
\]

작은 SPD 문제에서 dense/dual solve, residual, G/E identity를 확인했다. 같은 평균 key를 사용하더라도 full-context covariance를 남긴 writer는 다른 해를 낸다는 반례를 포함한다.

Upstream adjoint가 \(Q=\partial L/\partial P\), \(Y=M^{-\top}Q\)일 때:

\[
\bar\kappa_{\rm solve}
=Y(I-P^\top\kappa)-P(Y^\top\kappa).
\]

직접 E 의존성의 gradient를 추가해야 한다. 이 항을 포함한 전체 geometry gradient와 autograd를 비교했다. 실제 선형 연산의 D/P/input VJP도 별도로 검사했다.

3개 layer의 작은 nonlinear toy에서 각 upper key를 실제 lower writer가 변경한 activation으로 계산한다. `B=1,3` × `A/B`에 대해 모든 D 성분의 central finite difference를 검사했다. Upper key 또는 P를 detach하면 **동일한 forward 값인데도 전체 gradient가 달라지는 negative control**을 포함한다.

초기 D가 모두 0일 때는 unsmoothed geometry norm의 지정된 zero subgradient를 사용한다. 두 arm의 policy gradient가 0이고 전체 gradient가 유한한지도 검사했다. 단순 `sqrt`의 원점 미분을 그대로 사용하는 구현은 이 정의와 다르다.

이 finite difference는 **FP64 평균을 사용하는 매끄러운 algebra surrogate**이다. FP32 반올림을 미분한 실험이 아니며, 실제 모델의 전체 mixed-precision 경로가 검증되었다고 해석하지 않는다. FP32 native 평균의 실행 의미와 adjoint는 앞의 별도 검사로 확인했다. Toy loss는 native subject 방식의 가상 경로, 실제 writer 경로, norm과 geometry를 포함하는 수학적 surrogate다. 이 toy의 task loss와 편의상 사용한 계수는 실험 method의 목적함수나 하이퍼파라미터를 변경하지 않는다.

### 정규화 좌표와 D leaf bridge

\[
D_{\ell r}=s_{\ell r}q_{\ell r},\qquad
s_{\ell r}=\frac{a_{\ell r}}{\sqrt{d_\ell m}}.
\]

고정된 양의 anchor와 고정된 eligible-layer 수 \(m\)를 사용한다. 같은 physical D에서 native norm과 feasible set은 유지된다. 전체 branch의 D adjoint를 먼저 합산한 후 한 번만

\[
\bar q_{\ell r}=s_{\ell r}\bar D_{\ell r}
\]

를 전달하는 방식과 full autograd가 일치함을 확인했다. Scale을 누락하면 일치하지 않는 negative control을 포함한다. 이 검사는 fixed-D loss/gradient의 등가성이며, 기존 D 좌표 Adam과 q 좌표 Adam의 trajectory가 같다는 주장이 아니다.

`m=1,5`, 서로 다른 dimension `[1,3,7,11,4096]`, `B=1,3,4`, 서로 다른 anchor와 zero/tiny gradient를 포함한다. q의 반경 `.75*sqrt(d*m)`과 실제 D의 반경 `.75*a`가 같은 constraint임을 확인했다.

q 초기값이 0인 Adam의 첫 갱신에서 각 좌표 변화의 절댓값은 학습률 \(\eta\) 이하이므로:

\[
\frac{\|D_{\ell r}^{(1)}\|}{a_{\ell r}}\le\frac{\eta}{\sqrt m},
\qquad
\sum_\ell\frac{\|D_{\ell r}^{(1)}\|^2}{a_{\ell r}^2}\le\eta^2=.01.
\]

Adam은 LR `.1`, betas `(.9,.999)`, epsilon `1e-8`, weight decay `0`, warm-up 없음으로 실행했다. 이 첫 갱신 bound는 epsilon을 포함한다.

추가 dtype 검사는 **q/s/D, D gradient, q gradient, Adam의 first/second moment를 모두 FP32**로 둔다. 고정 scale은 `s = (a.double() / sqrt(d*m)).float()`로 생성한다. `d=4096,m=5,B=1/3`, `d=4096,m=1,B=1/3`, 가변 dimension `[1,7,19,128,4096],m=5,B=1/3`을 직접 검사했다.

- 실제 FP32 Adam의 첫 갱신에서 여섯 case 모두 clipping 0.
- FP32 physical D의 joint normalized squared norm을 FP64로 reduction하여 보고했다. 최대 관측값은 `.009999990952019377`.
- 실수 연산의 이론 bound는 `.01`이다. FP32 저장 scale·Adam·D 곱셈의 반올림을 고려한 검사 기준은 **`measured <= eta^2 + 1e-8 + 1e-5*eta^2`**이며, `eta=.1`에서 추가 허용오차는 `1.1e-7`이다. FP32 결과에 실수 bound의 비트 단위 엄밀 일치를 요구하지 않는다.
- `B * mean_loss`를 정확히 한 번 적용한 SUM 목적함수에서, 실제 FP32 D leaf gradient에 고정 scale을 한 번 곱한 q gradient와 full autograd를 비교했다. B를 누락하면 달라지는 negative control, 두 gradient로 실행한 실제 FP32 Adam 한 번의 결과도 비교했다.
- 이 추가 검사의 task/norm은 FP32, 작은 geometry surrogate는 FP64다. 실제 모델의 모든 writer 경로를 FP32로 재현한 검사는 아니다. 전체 causal finite difference는 앞서 설명한 FP64 검사를 유지한다.

`q` 반경 `.75*sqrt(d*m)`과 physical D 반경 `.75*a`의 등가는 실수 연산 정의다. 저장 scale을 FP32로 반올림하면 비트 단위 등가는 아니므로 **clamp의 정본은 실제 D norm으로 계산한 scale을 q에 적용하는 경로**다.

## 의도적으로 남긴 반례와 해석 한계

- 기존 D 좌표 Adam에서 dimension 4096, LR `.1`이면 첫 sign 방향 step norm이 약 `6.4`가 되어 작은 native 반경을 즉시 넘는 반례를 확인했다.
- 같은 상황에서 새 q 좌표는 첫 갱신에 clipping되지 않는다. 그러나 `m=5`의 일정한 정렬 gradient가 계속되면 **17번째 갱신에 모든 층이 clamp에 도달한다.** 정규화가 최종 포화나 모든 층의 경계 해를 금지하지 않는다.
- 별도의 1차원 볼록 예제에서 최적점 `.65`가 clamp `.75` 안에 있어도, 누적 Adam momentum 때문에 gradient가 안쪽으로 바뀐 뒤 계속 경계에 머무를 수 있음을 확인했다. 실제 실험의 momentum 방향을 측정했다는 뜻은 아니다.
- 두 층이 모두 eligible인 비대칭 toy에서 24갱신 후 A는 약 `[.577,.171]`, B는 `[.578,.177]`의 서로 다른 진폭을 얻었다. 이는 불균등 진폭이 **가능하다**는 예일 뿐, 실제 layer 배분이나 locality 개선을 보장하지 않는다.
- Gradient 성분별 dominance, 실제 모델의 성공률·locality·수렴, mixed-precision commit parity, batch 전체 요청 barrier, production memory·속도는 이 검증 범위에 없다.

수치 parity의 기본 허용오차는 `atol=1e-10, rtol=1e-9`, full causal finite difference는 `atol=2e-7, rtol=5e-5`, solve residual은 상대 norm `1e-12` 미만이다. FP32 native nested-mean 및 단순 context adjoint 검사는 dtype과 tensor 값의 정확한 일치를 사용했다. 합성 writer→context gradient는 FP32 tolerance `atol=1e-7, rtol=1e-6`으로 검사했다.
