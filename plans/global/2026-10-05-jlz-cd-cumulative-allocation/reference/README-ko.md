# CD 누적 배분 수학 참조

이 코드는 synthetic FP64 수식·gradient·microbatch 누적 계약만 검사한다. production 구현, 모델 parity, 정확도·locality 개선, 수렴을 검증하는 코드는 아니다. 원 데이터나 모델 weight를 읽거나 저장하지 않는다.

`geometry.py`는 batch entry에서 고정한 weighted retained SVD로 `Bmap`, `F`, `J`, `S`를 만든다. `U=D Bmap`, `Y=D S`, `Q=tr(D F D^T)`, `cross=<D,J>`이며 두 비용은 `Q`와 `Q+2 lambda_cov abs(cross)`다. retained singular value와 threshold를 함께 보관한다. 실현 불가능한 target은 숨기지 않고 discarded-target 값으로 기록한다.

`A=lambda_cov C0+H`는 SPD여야 하고 `C0,H`는 PSD여야 한다. 다른 precision, hidden jitter, rank 재시도, 성공 요청만 남기기 같은 정책은 없다. 참조 검사의 threshold와 계수는 작은 수치 문제용 값이며 실제 실험 설정이 아니다. allocation 가격은 호출자가 명시하는 parameter다.

실행:

```bash
/mnt/raid5/janghj/EasyEdit/.venv/bin/python run_audit.py
```

다른 환경에서는 NumPy가 설치된 Python으로 실행하면 된다. `audit.json`에는 통과 여부, runtime, seed, 작은 오차 요약, source hash만 기록된다.

검증 범위:

- Full-rank exact write와 rank-deficient compatible/incompatible target 구분.
- Retained response와 비용의 같은 nullspace, rank-zero 경계.
- Dense weight-space 비용/gradient와 축약 계산의 일치, 유한차분.
- 누적 C0 에너지 변화 절댓값 upper bound, update 취소에 음의 비용을 주지 않는 성질.
- 누적 변위가 0인 B1의 두 arm 동일성과 abs 0점의 0 subgradient 선택.
- 요청별 context 개수 차이, logical batch 1/2/4 및 작은 tail.
- Microbatch 1/2/4와 마지막 작은 묶음에서 task pullback 합산, regularizer 단 한 번 적용.
- 동일 retained 공간에서 global weight scale 및 중복 context의 불변성.
- 모든 weighted contrast를 보존하는 직교 재표현의 동일성.
- 잘못된 shape/weight/geometry 입력의 명시적 실패.

각 요청의 context loss는 `S` 때문에 다른 요청의 D에도 gradient를 전달할 수 있다. Microbatch마다 자기 요청 변수에만 gradient를 더하면 안 된다. 참조 코어에는 batch 평균을 숨겨 넣지 않으며, task와 regularizer를 모두 합산한 뒤 같은 상수로 보고한다. Logical batch 크기가 달라지면 geometry 자체가 바뀌므로 batch 불변성을 주장하지 않는다.

Production에서 매 층 다시 구한 actual key는 이 고정 entry geometry와 다를 수 있다. 이 코드는 그 차이를 제거하지 않으며, abs-cross는 의미적 지식 보존의 보장이나 평가 gate가 아니다.
