# Session 01 Motivation — projected Alpha pair g0 v3 GH decision

날짜: 2026-08-02

방법명: **ODE-Edit**

상태: **pair technical-valid / selector kill 재확인 / always-refresh transfer signal**

## Proposal에서 온 내용

- null-space projection과 current-state relinearization이 상보적인지 작은 atomic
  transfer signal로 확인한다.
- state-dependent direction이 static/transported direction보다 공통 이득을 주지
  않으면 Alpha generality를 닫는다.
- proposal은 최종 paper plan이나 검증된 superiority claim이 아니다.

## Repo/protocol에서 확인한 사실

- Alpha job `15772`는 2026-08-01 19:55:26부터 2026-08-02 03:13:58 KST까지
  실행되어 `COMPLETED`됐다.
- 양 model 모두 exact 8 cases, 56 outcomes, 8 receipts, 8 target artifacts,
  all-pass, rollback/firewall/hash gate를 통과했다.
- 동일 case order, layers 4--8, `K=4`, `D/4`, `D/64`, seed, branch와 gate를 썼다.
- postpivot analysis SHA-256:
  `7991d63665dcd8f343f003eb419213f8069e835826858fd82e6813446a09464a`.

## 공통-policy 결과

| Metric | Llama | Qwen |
|---|---:|---:|
| `A4-B4` mean | +0.826986 | +1.668561 |
| `A4-B4` positive cases | 8/8 | 8/8 |
| `B4-C4` mean | +0.384075 | +0.540036 |
| A4 casewise noncollapse | 0.000000 | 0.000000 |
| projector mean/min retention | 0.999997 / 0.999971 | 0.987945 / 0.959304 |
| selector refresh | 2/24 | 21/24 |
| `G4-A4` mean | -0.770814 | -0.127810 |

locked pair verdict는 `ALPHA_ALWAYS_REFRESH_TRANSFER_SIGNAL`이다. 한 model의 큰
값으로 다른 model을 평균 상쇄하지 않았고 양쪽 model gate가 각각 통과했다.

## GH 판정

### Kill: central-probe selector G

MEMIT 때와 마찬가지로 Llama와 Qwen의 선택률이 크게 갈렸고, 양 model 모두
관측된 always-refresh endpoint보다 낮았다. model별 sign/margin/threshold 수정은
금지하므로 `G`를 재사용하지 않는다.

### Survive: projected always-refresh actuator

`A4-B4`가 양 model의 모든 case에서 양수이고 projection retention도 finite/nonzero다.
따라서 current-state direction refresh의 작은 atomic signal이 MEMIT뿐 아니라
AlphaEdit-projected actuator에도 전달된 것으로 판정한다. naive split과 native의
차이는 양쪽 모두 practical floor 안이어서 작은 step 자체는 설명이 아니다.

## 기대효과와 한계

- 관측된 atomic utility lift는 Llama `+0.827`, Qwen `+1.669`다.
- 이 값으로 4-edit retention, KL, capacity 개선폭을 환산할 근거는 없다.
- projector norm retention은 update가 허용 subspace에 남았다는 기술 신호이지
  preservation metric이 아니다.
- 따라서 microseq의 동일 축 양성 여부까지만 Motivation에서 추가로 본다.

## Claim boundary

- 성립: 동일 code/policy에서 projected direction relinearization이 두 model 모두
  transported direction보다 작은 teacher-forced signal을 보였다.
- 미성립: adaptive selector, native AlphaEdit reproduction, edit/model preservation,
  lifelong robustness, ODE-Edit deployable method superiority.

## 사용자 확인 필요

- 없음. 다음 실험도 양 model에 동일 always-refresh method만 사용한다.
