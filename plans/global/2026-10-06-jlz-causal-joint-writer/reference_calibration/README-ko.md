# Radial calibration·prox 수학 검산

이 디렉터리는 합성 FP64 수식과 반례만 검사한다. 기존 `reference/`를 변경하거나 production method, reference fit, optimizer, 계수를 구현하지 않는다. 모델·GPU·benchmark·PS/NS/ACC를 읽지 않는다.

## 확인한 식과 범위

- 고정 key의 scalar ridge 보존 에너지 `Q=R^2*kappa/(1+kappa)^2`는 history 증가에 대해 비단조다. `dQ/dkappa=R^2*(1-kappa)/(1+kappa)^3`이며 history 증가 시 kappa는 감소한다. Exact 에너지 `R^2/kappa`와 ridge 최적 목적 `E+Q=R^2/(1+kappa)`의 단조성과 혼동하면 안 된다.
- 고정 B1 L8-only reference에서 실제 후보 loss의 gradient를 사용하면 `a=-<gF,R>`, `b=<gN,R>`, `d=<gQ,R>=2Q`이고 radial break-even price는 `(a-b)/d`다. Native norm이 degree one이면 `b=N`이다. 이 값은 radial condition 하나만 맞추며 native-clamp equivalence나 전체 KKT를 보장하지 않는다.
- Native subject-injection의 reference를 actual candidate loss로 평가하면 양의 가격이 나오지 않을 수 있다. 0·음수·분모 0을 숨기지 않는다. Batch 합계 조건은 요청별로 반대 방향의 잔차가 상쇄될 수 있다.
- `R=a*u`이면 native norm `.5*||R||/a^2`는 `.5*||u||/a`다. 공통-step proximal threshold는 `eta*.5/a`이며, 각 block의 `.75` ball에 별도로 사영한다.
- 상층 의존 gradient를 routing으로 끊으면 일반적으로 기존 전체 목적 gradient가 아니다. Routed stationarity가 실제 KKT가 아니고 routed field에 scalar potential이 없는 작은 반례를 포함한다.

## 권하는 calibration 계약 표현

`native-clamp equivalent coefficient`보다는 **사전 고정된 B1 L8 reference의 actual-objective radial break-even price**라고 명시한다.

1. Reference 생성 규칙·모델 상태·writer·key 평균·anchor·후보 번호를 성능 평가 전에 고정한다. Reference 생성에 별도 native fit이 필요하면 그 연산을 숨기지 않는다.
2. 동일 physical R에 대해 실제 후보 NLL/KL, requested native norm, actual Q의 full gradient를 평가한다. 고정 L8-only 방향의 `d=2Q`를 검산한다.
3. `d>0`과 `a>b`가 양의 가격의 조건이다. 0 reference, zero denominator, interior/zero price, negative price, nonfinite는 별도 상태로 보고한다. 임의 epsilon, clip, default coefficient를 넣지 않는다.
4. 한 batch aggregate 가격을 정하면 요청별 radial 잔차, reference의 clamp 활성 상태, tangent·비활성 층 gradient 조건을 진단으로 함께 기록한다. 이것들을 새로운 성능 gate로 만들지는 않는다.
5. 기준 R가 finite-step native subject optimization의 반환값이면 actual objective의 정상점이라는 전제를 두지 않는다. `a,b,d`의 단위 규칙과 성능 최적값을 구분한다.

위 규칙의 유효성 실패 시 어떤 scientific 정책을 택할지는 이 수학 참조가 임의로 결정하지 않는다. λ를 정한 뒤 고정 objective로 backtracking해야 하며, accepted candidate와 λ를 서로 정의하는 순환은 피해야 한다.

실행:

```bash
/mnt/raid5/janghj/EasyEdit/.venv/bin/python run_audit.py
```

NumPy가 있는 다른 Python으로도 가능하다. `audit.json`에는 작은 검사 결과와 source hash만 저장된다.
