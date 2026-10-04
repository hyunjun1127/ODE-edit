# JLZ v12 CPU 수치 계약

이 디렉터리는 v12 method의 **수치 계약을 확인하는 NumPy 참조**다. 모델 편집기의 production 구현, GPU 실험, 성능 검증 또는 dispatch가 아니다. 첨부 원본을 변경하지 않았다. 원본 7개 검사는 `test_original.py`에 수치 내용을 유지하고 import 경로와 범위 설명만 조정했다.

## 파일과 적용 범위

| 파일 | 확인하는 계약 |
|---|---|
| `budget.py` | 유한 입력·반경·양의 가중치 검증, 가중 group-L1 ball의 Euclidean 사영, `tau_proj` 반환 |
| `optimizer.py` | 요청 하나의 Adam moment와 층 RMS 계수, 좌표변환 epsilon, smooth gradient + norm subgradient |
| `kkt.py` | smooth F의 gradient로 계산하는 완전한 KKT 잔차; 관측용이며 solver가 아님 |
| `witnesses.py` | 실제 EfficiencyAdam의 비-KKT 고정점과 sparse-gradient 첫 보폭 반례 |
| `test_original.py` | 첨부 원본 검사 7개; 마지막 fixed-point 검사는 ordinary projected gradient |
| `test_contract.py` | 추가 입력·좌표변환·재진입·요청 독립성·예상 한계 검사 |
| `test_kkt.py` | 활성 방향, 비활성 subgradient, 내부점·경계·반경 0의 진단 검사 |
| `audit.py`, `audit.json` | 전체 검사 결과, 예상 한계의 원시 수치, 버전과 입력 파일 hash |

## Optimizer 계약

요청마다 별도의 `EfficiencyAdam` 인스턴스를 만든다. 같은 요청의 모든 층은 동일한 local step counter를 공유하며, 다른 요청의 평가·종료 일정은 이 counter와 moment를 바꾸지 않는다. 비활성 층도 gradient와 moment를 갱신한다. 사영으로 0이 된 층의 moment를 지우거나 그 층을 영구 제외하지 않는다.

`objective_gradients(u, smooth_grads, decay)`에서 `decay = lambda_norm / a_star`다. 모델의 기존 backward 한 번으로 얻은 `smooth_grads = grad_u F`에 `decay * u / ||u||`를 더한다. 정확히 0인 norm의 subgradient는 0을 선택한다. 작은 양수 norm에도 diagnostic tolerance를 적용하지 않는다. 이 helper 자체는 모델을 호출하지 않는다.

Adam은 이 `g_J`를 받아 native betas `(0.9, 0.999)`의 좌표별 moment와 `s_l = EMA(mean(g_J,l**2))`를 갱신한다. bias 보정 뒤 `gamma_l = sqrt(s_hat_l / mean(s_hat))`다. 모든 `s_hat`가 0이면 gamma는 모두 1이다. 각 요청의 초기화와 step state가 독립임을 interleaved schedule로 검증한다.

`EfficiencyAdam.from_native(shapes, anchor_star, ...)`는 `lr_u = lr_native / a_star`, `eps_u = a_star * eps_native`를 사용한다. **선언된 변수 집합 자체가 native z층 하나이고 `a_layer = a_star`일 때** delta 좌표 native Adam 및 clamp와 수치적으로 일치한다. anchor가 1이 아닌 경우와 epsilon보다 작은 gradient까지 검사한다. 여러 변수 중 결과적으로 한 층만 활성인 경우, 임의 다른 층 하나만 선택한 경우는 이 환원 범위에 포함하지 않는다. 여기서 확인하는 native Adam은 독립된 NumPy 수식이며, 실제 모델 프레임워크의 전 과정 parity를 주장하지 않는다.

이 optimizer는 유한 step의 **heuristic**이다. 좌표별 Adam 뒤의 Euclidean 사영은 원래 목적의 KKT 정지점을 보장하지 않는다. 원본의 dense-gradient 첫 step RMS 비율 검사는 해당 toy 설정에서만 성립한다. sparse-gradient 반례도 함께 기록한다.

## 사영과 KKT 진단의 구분

`project`는 `sum_l w_l ||u_l|| <= radius`의 Euclidean projection을 계산한다. 층마다 다른 shape를 허용하고, 상대 변화 예산에서는 모든 가중치를 1로 둔다. 반환하는 `tau_proj`는 **사영 부분문제의 multiplier**다. objective의 multiplier `mu`와 동일한 값으로 해석하지 않는다. 반경 0은 모두 0으로 사영하는 수치 경계 케이스이며, method의 표준 예산은 양수다.

`kkt_diagnostics`에는 norm penalty를 제외한 **smooth F gradient**를 넣는다. `beta_norm = lambda_norm / a_star`, `B = sum ||u_l||`일 때 다음 원시 잔차를 반환한다.

- 활성 층: `grad F_l + (beta_norm + mu) * u_l / ||u_l||`의 벡터와 norm.
- 비활성 층: `max(0, ||grad F_l|| - beta_norm - mu)`.
- primal violation `max(0, B-c)`, complementarity `abs(mu*(B-c))`, dual violation `max(0,-mu)`.
- budget, 원래 층 norm, 진단용 활성 mask, signed radial price, norm-only 잔차. norm-only 값은 방향 오류를 놓치는 약한 진단임을 명시한다.

경계에서 활성 층이 있으면 `mu = max(0, mean_active(-<grad F_l,e_l> - beta_norm))`를 사용하고, 내부 또는 경계 밖에서는 0을 사용한다. 반경 0이고 모든 block이 정확히 0인 특수 경우에는 `mu = max(0, max_l ||grad F_l|| - beta_norm)`를 선택한다. 이것은 `zero_radius_degenerate`라는 별도 convention으로 기록한다.

CPU Float64 진단의 기본값은 `active_tol = boundary_tol = 1e-12`다. method의 저정밀 caller는 명시적으로 `active_tol=1e-10`, `boundary_tol=1e-6*max(1,c)`를 전달한다. 작은 양수 block은 `near_zero_nonzero`로 따로 표시하고 원래 norm 및 전체 예산에서 제외하지 않는다. tolerance는 진단 분류에만 쓰며, **pruning·budget 재보정·과학적 통과 gate로 쓰지 않는다**. `max_residual`도 단위 보정 없는 관측값일 뿐이다.

실제 method의 terminal 후보는 stop-before-backward를 지킨다. 그 후보의 KKT는 `null`, 사유는 `NO_BACKWARD_TERMINAL`이며, KKT 수집을 위한 추가 backward나 model call은 없다. 이 참조에는 모델 평가 루프가 없으므로 production 종료 규칙을 실행 검증했다고 간주하지 않는다.

## 재현

저장소 루트에서 Python 환경에 `requirements.txt`의 NumPy/SciPy를 설치한 뒤 실행한다. 검증 환경은 Python 3.12.0, NumPy 2.5.3, SciPy 1.18.1이다.

```bash
python -B -m unittest discover -s plans/global/2026-10-04-jlz-v12-marginal-allocation/reference -p 'test_*.py' -v
python -B plans/global/2026-10-04-jlz-v12-marginal-allocation/reference/audit.py
```

두 번째 명령은 `audit.json`을 재생성한다. `--output /tmp/v12-audit.json`으로 다른 출력 위치를 선택할 수 있다. JSON은 검사별 결과와 source SHA256를 담으며 wall-clock 시간이나 임시 경로를 기록하지 않아 같은 버전의 같은 입력으로 비교 가능하다. 파일을 고친 뒤에는 감사를 재생성한다.

`EXPECTED_LIMITATION`은 구현 실패를 숨기는 표지가 아니다. 잘못된 강한 수학적 claim을 배제하기 위해 **그 한계가 실제로 재현됨을 검사**한다. 양의 norm penalty를 포함한 convex toy 목적에서 비-KKT 고정점을 구성하고, zero moment의 실제 EfficiencyAdam으로 24회 갱신해 고정 상태와 0이 아닌 full KKT 잔차를 함께 기록한다. 이 toy는 모델 trajectory나 24회 모델 최적화의 성능을 대표하지 않는다.
