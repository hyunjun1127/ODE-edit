# v9 realization / merged-cost CPU 수학 검증

**181/181 PASS.** 이 디렉터리는 **ridge main + exact-realization pilot only** 설계의 수학을 작은 CPU tensor로 검사한다. Production 코드, 모델, benchmark, GPU 실험은 실행하지 않는다. Pseudoinverse는 rank-deficient 반례 분석에만 쓰며 **실행 fallback을 구현하거나 승인하지 않는다.** v8의 q scale·Adam·native norm·pulse/replay 조건은 변경하지 않는다.

- `validate_math.py`: 독립 재현 스크립트. Production import 없음.
- `results.json`: 검사별 오차·허용오차·반례 수치.
- `receipt.json`: 실행 환경과 script/results SHA256.
- Python 3.12.0, PyTorch 2.9.1+cu128, CPU only, `CUDA_VISIBLE_DEVICES=''`.
- 잘 조건화된 3층 causal toy의 finite difference 최대 절대 오차: `1.7513038762240196e-11`.
- Exact-P implicit VJP 최대 절대 오차: `4.440892098500626e-16`.
- Merged policy의 G/E 경로와 독립 inverse-system 경로의 D/K gradient 최대 절대 오차: `2.0816681711721685e-17`.

```bash
CUDA_VISIBLE_DEVICES='' /mnt/raid5/janghj/EasyEdit/.venv/bin/python -B \
  /mnt/raid5/janghj/.codex/worktrees/odeedit-jlz-v5-a-w5-review-20261002-v1/plans/global/2026-10-03-jlz-realization-v9/math/validate_math.py
```

재실행은 이 디렉터리의 results/receipt만 갱신한다. 실행 시간은 작은 CPU 검사의 시간이며 method 성능 측정이 아니다.

## Ridge와 exact realization

\(A\succ0\), 평균 key 행렬 \(K\in\mathbb R^{n\times B}\), 요구 변위 \(D\in\mathbb R^{d\times B}\)에서

\[
X=K^\top A^{-1}K,\quad
P_{\rm ridge}=(A+KK^\top)^{-1}K,\quad
M=X(I+X)^{-1}
\]

이면 \(U_{\rm ridge}=DP_{\rm ridge}^\top\)의 평균-key 실현값은

\[
U_{\rm ridge}K=DM.
\]

Full-column-rank K에서 ridge의 M eigenvalue는 0과 1 사이이며 감쇠한다. 이 표현의 **diag(M)는 해당 요청 열의 계수**다. 실제 요청 출력 \((DM)_r\)에는 다른 요청의 off-diagonal 항도 들어가므로 diag(M)만을 실제 edit 성공률로 읽으면 안 된다.

X가 가역일 때 exact constraint \(UK=D\)를 만족하는 최소 A-energy writer는

\[
U_* = D X^{-1}K^\top A^{-1},\qquad
\operatorname{tr}(U_*AU_*^\top)=\operatorname{tr}(DX^{-1}D^\top).
\]

직접 solve, constraint, energy identity와 feasible nullspace perturbation에 대한 최소성을 검사했다. 임의의 \(ZK=0\)에 대해 \(U_*+Z\)의 energy 증가는 \(\operatorname{tr}(ZAZ^\top)\)이다. Exact-P의 K adjoint도 autograd와 비교했다.

이 결과는 exact를 main writer로 권고하는 근거가 아니다. Exact는 감쇠를 없애는 대신 작은 capacity eigenvalue 방향의 비용을 크게 증가시킬 수 있다.

## Rank-deficient compatibility

Rank가 부족하면 정확한 실현의 필요충분조건은

\[
D=D X^\dagger X.
\]

같은 key를 가진 두 요청이 다른 D를 요구하는 toy를 검사했다. 모든 선형 writer는 duplicate key에 같은 출력을 주므로 두 요구를 동시에 정확히 실현할 수 없다. Pseudoinverse writer는 \(D X^\dagger X\)를 실현하며, incompatible D 자체를 복원하지 못한다.

이 검사에서 `pinv(..., rtol=1e-12, atol=0, hermitian=True)`는 수학적 projector 확인용이다. 실제 pilot에서 rank·conditioning·compatibility 처리 기준은 별도 contract에 따르며, 이 파일은 pseudoinverse fallback을 제공하지 않는다.

## 평균 key의 exactness와 context exactness는 다르다

Native `[1,5]` context group에서 FP32 group 내부 평균 → FP32 group 평균 → FP64 변환을 적용했다. 평균 key는 \([1,0]^\top\), 요구 변위는 1이다.

- 최소 mean-exact writer \(U=[1,0]\)의 context별 출력: `[1, 3, -1, 1, 1, 1]`.
- Native group 평균 출력은 정확히 1이지만 개별 context 오차는 `[0, 2, -2, 0, 0, 0]`이다.
- 이 예제에서는 더 강한 context-exact 해 \(U=[1,-1]\)도 존재한다. 모든 context에 1을 출력하지만 A-energy는 1에서 2로 증가한다.

따라서 평균-key 실현을 1로 만드는 것만으로 모든 학습 context, paraphrase 또는 locality의 개선을 보장하지 않는다.

## 같은 g/e에서도 요청별 실현 행렬은 다를 수 있다

\(D=I_2\)인 full-rank 변위에서 M의 eigenvalue를 `.2,.8`로 고정하고 eigenvector를 회전했다. 두 행렬 모두 off-diagonal cross effect를 가진다.

| 회전 | diag(M) | raw \(g^2\) | raw \(e^2\) |
|---|---|---:|---:|
| \(\pi/12\) | `.240192, .759808` | `.32` | `.68` |
| \(\pi/4\) | `.5, .5` | `.32` | `.68` |

여기서 \(g^2=\operatorname{tr}(DGD^\top)\), \(e^2=\operatorname{tr}(DED^\top)\)이며 normalization 전 값이다. `B=2, sigma=1`로 정규화해도 두 예제 모두 `g=.4, e≈.583095`로 같다. 그런데 \(DM\)과 diag(M)는 다르다. 이는 aggregate g/e 두 숫자만으로 요청별 realization을 식별할 수 없다는 반례다.

서로 다른 target 열 \(D=[1,2]\)을 둔 보완 반례도 결과 JSON에 포함했다.

## Main의 merged policy

Main ridge writer의 기존 geometry를 사용하되 층별 비용을 합친다.

\[
c_\ell=\sqrt{g_\ell^2+e_\ell^2}
=\sqrt{\frac{\operatorname{tr}(D_\ell(I-M_\ell)D_\ell^\top)}{B\sigma_\ell^2}}.
\]

- Arm A: \(.1\sum_\ell c_\ell\).
- Arm B: \(.1\sqrt{\sum_\ell c_\ell^2}\).
- 원점은 smoothing 없이 지정된 zero subgradient를 사용한다.

두 arm 모두 같은 coefficient에서

\[
L_{\rm merged}\le L_{\rm split}\le\sqrt2 L_{\rm merged}
\]

이다. 이것은 기존 split cost와 동일한 목적함수라는 주장이 아니다. 동일 coefficient를 유지한 새로운 aggregation이다.

Fixed D, fixed normalization인 scalar capacity \(x\ge0\)에서 새 비용은 \(1/\sqrt{1+x}\)에 비례하여 단조 감소한다. 기존 split 비용은 \((\sqrt{x}+1)/(1+x)\)에 비례하며 `x=0→.1`에서 오히려 증가하는 구간이 있다. Fixed D에서 matrix capacity를 PSD 순서로 증가시킨 검사도 통과했다. **D·K·여러 층·task loss가 동시에 변하는 실제 학습 전체에 대한 단조성은 아니다.**

G+E로 계산한 값/전체 D·K gradient와 독립적인 inverse-system 식을 비교했다. 후자는

\[
\operatorname{tr}\bigl[D(I+X)^{-1}D^\top\bigr]
\]

이다. `I-M`은 수학적 identity이며, M이 I에 가까울 때 이를 직접 빼면 cancellation이 생길 수 있다. 실제 계산은 G+E 또는 SPD solve 형태를 사용하여 검증하는 편이 적절하다.

## Causal gradient와 conditioning

작은 3층 nonlinear 실제-weight toy에서 lower D → lower writer → upper K → upper exact P 경로를 구성했다. `B=1,2`에서 모든 D 좌표를 central finite difference로 검사했다. P나 upper K를 detach하면 forward 값은 같지만 gradient는 달라졌다. 관측된 최대 gradient 차이는 각각 약 `.00351` 및 `.00750`이었다.

이 FD는 잘 조건화된 **FP64 toy**다. 실제 FP32 materialization, native loss, q optimizer, GPU graph 또는 commit parity 검증이 아니다.

Near-collinear 예제는

\[
A=I,\quad K=\begin{bmatrix}1&1\\0&\epsilon\end{bmatrix},\quad D=[1,-1]
\]

로 두었다. Exact 해는 \(U_*=[1,-2/\epsilon]\), energy는 \(1+4/\epsilon^2\)이다.

| epsilon | exact energy 이론값 | condition(X), 근사 |
|---:|---:|---:|
| `.1` | `401` | `402` |
| `.01` | `40,001` | `40,002` |
| `.001` | `4,000,001` | `4,000,002` |
| `.0001` | `400,000,001` | `400,000,006` |

조건수가 커지면 FP64 solve의 오차도 증가했다. 마지막 행의 실제 constraint 오차 norm은 약 `1.22e-8`이며 상세값을 JSON에 남겼다. Ridge는 이 예제의 충돌 방향을 강하게 감쇠한다. **A-energy 증가가 실제 locality 손실의 측정값이라는 뜻은 아니다.**

## M inverse 보정과 prior scale

Full-rank일 때 ridge target을 \(D M^{-1}\)로 보정하면

\[
D M^{-1}P_{\rm ridge}^\top
=D X^{-1}K^\top A^{-1}=U_*.
\]

즉 exact writer의 conditioning/energy 문제를 그대로 가진다. \(A=\alpha C\)를 균일하게 scale하고 D/K를 고정하면 exact writer에서 alpha가 상쇄된다. `.1,1,10,15000`에 대해 확인했다. Ridge writer에서는 scale 효과가 남는다.

다만 \(A=\alpha C+H\)에서 H가 고정되고 C에 비례하지 않으면, alpha 변경은 A의 균일한 scaling이 아니다. 이 경우 exact writer도 달라지는 반례를 포함했다. 이 조건을 빼고 “lambda-C가 항상 소거된다”고 일반화하면 안 된다.

## 검증 범위

기본 비교는 `atol=1e-10, rtol=1e-9`, causal FD는 `atol=2e-7, rtol=5e-5`이다. Near-collinear 예제의 energy 비교는 `atol=1e-7, rtol=2e-8`, realization 비교는 `atol=2e-8, rtol=2e-8`로 별도 지정했다. 본 검사는 수학적 identity와 반례를 확인한다. 실제 학습 품질, 수렴, locality, 속도, 모델별 안정성이나 pilot 통과를 주장하지 않는다.
