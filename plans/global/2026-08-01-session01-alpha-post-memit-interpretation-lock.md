# Session 01 Motivation — Alpha post-MEMIT interpretation lock

- 날짜/고정 시점: 2026-08-01 20:32 KST
- Alpha job: `15772`, 양 model `events=0`, `outcomes=0` 확인 뒤 작성
- 방법명: **ODE-Edit**
- 지위: 실행 contract 불변, post-MEMIT scientific interpretation만 사전 고정

## 네 범주

- proposal에서 온 내용: AlphaEdit null-space projector와 current-state
  relinearization이 상보적인지 RQ4의 작은 transfer signal로 확인한다.
- repo/protocol에서 확인한 사실: MEMIT pair에서 central-probe selector `G`는
  Llama를 잘못 fixed로 보내 pair gate에 실패했다. 반면 always-refresh `A4-B4`는
  Llama `+0.828098`, Qwen `+1.583319`로 두 model에서 같은 양의 방향이었다.
- GH 추정: projection 뒤에도 stale-direction 비용이 남는다면 `A4-B4`가 양
  model에서 양수일 수 있다. projector가 허용 subspace를 제한하므로 효과 크기는
  MEMIT보다 줄거나 달라질 수 있어 숫자 forecast는 하지 않는다.
- 사용자 확인 필요: 없음. 사용자는 두 model 공통 method와 Motivation 규모의
  lenient signal을 요구했고 model별 rescue를 금지했다.

## 실행 contract 불변

- 기존 job의 model, 8 cases, seed, layers, `K=4`, `D/4`, `D/64`, arm
  `G4/A4/B4/C4/native/split4/no-op`, direct-z, projector, covariance와 code는
  변경하지 않는다.
- EasyEdit/projector/covariance는 pinned read-only이며 재계산하지 않는다.
- `G`를 삭제하거나 결과를 숨기지 않는다. 단, 이미 MEMIT에서 kill된 selector를
  Alpha ODE 생존 gate로 다시 사용하지 않는다.

## 사전 고정 Alpha transfer gate

technical gate가 모두 통과한 뒤 다음을 model별로 계산한다.

1. `direction_transfer = mean(A4-B4)`
2. `coefficient_support = mean(B4-C4)` — 보조축이며 성공에 필수 아님
3. `A4_noncollapse = mean(A4-max(A4,B4))` — casewise 상한 대비 손실
4. projector right-factor norm retention, finite/nonzero와 hash identity

양 model 모두 아래를 만족할 때만 `ALPHA_ALWAYS_REFRESH_TRANSFER_SIGNAL`이다.

- `direction_transfer > 1e-4`
- `A4_noncollapse >= -0.10`
- projector retention이 finite/nonzero이고 technical-valid

한 model이라도 실패하면 `ALPHA_COMMON_TRANSFER_NO_SIGNAL`로 AlphaEdit
generality claim을 닫는다. model별 threshold/sign/policy 변경, case 제외, `G`나
`B/C`를 사후 대체 성공축으로 승격하는 것은 금지한다. bootstrap CI는 기술하되
small Motivation gate의 자동 kill에는 쓰지 않는다.

## Claim boundary

양성이어도 projector와 relinearization의 atomic 가능성 신호일 뿐 native
AlphaEdit 재현, locality/model preservation, lifelong robustness 또는 method
superiority를 뜻하지 않는다. 그 주장은 별도 larger sequential baseline에서만
검증할 수 있다.
