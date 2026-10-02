# CPU 수학 검증

`validate_math.py`는 표준 라이브러리만 사용하는 작은 합성 행렬 검증이다. 모델 로딩, GPU 실행, production 코드 변경 없이 다음을 확인한다.

## 고정 entry geometry

요청 수를 실제 batch 크기 \(B\), context 수를 \(C\)라 하고, \(K\in\mathbb R^{d_{in}\times C}\), 요청과 context의 대응을 나타내는 \(Z\in\mathbb R^{B\times C}\), 대각 context 가중치를 \(\alpha\)라 한다. 요청별 가중치 합은 1이므로 \(Z\alpha Z^\top=I_B\)이다.

\[
\bar K=K\alpha Z^\top,\quad M=A+K\alpha K^\top,\quad
P=M^{-1}\bar K,\quad U=DP^\top.
\]

\[
G=P^\top AP,\qquad
E=(P^\top K-Z)\alpha(P^\top K-Z)^\top.
\]

검증한 동치식은 다음과 같다.

\[
\operatorname{tr}(DGD^\top)=\operatorname{tr}(UAU^\top),
\]

\[
\operatorname{tr}(DED^\top)
=\operatorname{tr}\big[(UK-DZ)\alpha(UK-DZ)^\top\big],
\]

\[
E+G=I_B-\bar K^\top P.
\]

마지막 식은 \(MP=\bar K\)로부터 얻어진다. \(G,E\)의 off-diagonal을 제거하면 요청 간 cross term이 달라진다는 점도 합성 행렬로 확인한다. 이 비용은 고정 entry key에서의 geometry이므로, 아래층 write로 key가 바뀐 뒤의 실현오차와 동일하다고 주장하지 않는다.

## 두 arm의 비용과 gradient

\(Q_\ell\)을 \(G_\ell\) 또는 \(E_\ell\)라 하고, \(q_\ell=\operatorname{tr}(D_\ell Q_\ell D_\ell^\top)\), \(\sigma_\ell>0\)는 고정 entry scale이라 한다.

\[
R_A=\sum_\ell \frac{\sqrt{q_\ell}}{\sqrt B\sigma_\ell},\qquad
R_B=\sqrt{\sum_\ell \frac{q_\ell}{B\sigma_\ell^2}}.
\]

각 root가 양수인 경우 analytic gradient는 다음과 같다.

\[
\nabla_{D_\ell}R_A
=\frac{D_\ell Q_\ell}{\sqrt B\sigma_\ell\sqrt{q_\ell}},\qquad
\nabla_{D_\ell}R_B
=\frac{D_\ell Q_\ell}{B\sigma_\ell^2R_B}.
\]

이 gradient를 모든 좌표에 대해 중앙 유한차분으로 검증한다. Root가 0인 지점에서는 미분 가능하다고 주장하지 않으며, 선택하는 subgradient를 0으로 명시한다. 실제 구현은 `sqrt`의 원점에서 NaN이 발생하지 않도록 이 convention을 구현해야 한다.

동일한 \(Q\), \(\sigma\), 동일한 기능적 경로를 갖는 \(L\)개 층에 \(D/L\)씩 나누는 통제된 예에서는 \(R_A\)는 유지되고 \(R_B\)는 \(1/\sqrt L\)배가 된다. 서로 다른 실제 층 사이에서 이 등식이 보장되는 것은 아니다.

## 실행 및 범위

검증 사례는 실제 \(B\in\{1,3,7\}\), 층별 서로 다른 입력·출력 차원, 과거 feedback 요청 수 \(M_{past}\in\{0,1,5,10\}\)을 포함한다. Native column별 norm ball projection도 확인한다.

4개 feedback partition은 native **후보 번호 5/10/15/20에서 해당 Adam update 전에** 사용한다. 이때 완료된 Adam update 수는 각각 4/9/14/19회이다. 현재 요청과 선택된 과거 요청은 각각 정확히 한 번 포함되며, 작은 batch에서 비어 있는 partition은 건너뛴다. 후보 평가 25회는 초기 상태 평가 1회와 Adam update 후 평가 24회이다.

요청 행 기준 native 계산은 forward \(25B\), backward \(24B\)이다. Feedback은 forward와 backward 각각 \(B+M_{past}\)이고, 마지막 전체 현재 batch의 physical forward \(B\)에는 backward나 추가 Adam update가 없다. 따라서 최적화 request-row 비용은 총 forward \(27B+M_{past}\), backward \(25B+M_{past}\)임을 검증한다. 이 수는 geometry·teacher 준비와 benchmark 평가를 제외하며, 토큰 수 또는 실제 GPU forward 호출 횟수와도 구분해야 한다.

```bash
python3 plans/global/2026-10-02-jlz-subject-adam-v6/math/validate_math.py
```

정확한 인터프리터·명령·SHA는 `run-receipt.json`, 개별 오차와 cross term은 `validation-results.json`에 기록된다. 이 결과는 대수적 정확성을 확인하며 학습 성능, convergence, locality 보존 또는 가속 배율을 입증하지 않는다.
