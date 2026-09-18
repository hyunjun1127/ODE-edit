# BG-TW 보존 한도 개정 — N4 사전 실행 의존성 제거

> 상태 갱신: 사전 손상 한도 자체를 제거하라는 최신 사용자 요청으로 이 고정 ρ/b 안은 채택 전 철회했다. 아래는 검토 이력이며 현재 실행 지시가 아니다. 무예산 논의안은 `plans/global/2026-09-15-bg-tw-no-budget-discussion.md`에서 다룬다.

작성: 2026-09-15. 상태: method·실행 계약 개정, CPU 수학 검산. 신규 LLM 실험·원격 전달은 수행하지 않았다.

## 1. 결정

**N4의 1000요청 경로에서 허용 손상을 산출하는 규칙을 폐기한다.** BG-1은 W0, 고정 reference, 현재 batch 요청, 사전에 선언한 보존 허용치만으로 시작한다. N4 checkpoint·새 C4 평가·N4 editing 실행은 method의 입력도 착수 조건도 아니다.

첫 설정은 다음과 같다.

\[
\rho_{\rm TV}=0.10,\qquad
b=2\rho_{\rm TV}^2=0.02\quad\text{nats/scored position}.
\]

이 값은 **선정한 reference에서 평균 조건부 분포의 total variation을 0.10 이내로 제한하는 충분조건**을 KL로 구현한 시작점이다. 최적값·전체 지식 보장·10% 정확도 손실 허용치가 아니다. 적절한 허용 손상은 원 모델만으로 유일하게 추론할 수 없으므로, 이 방법은 보존 허용치라는 hyperparameter를 명시적으로 가진다.

기계 판독 규칙은 [preservation-budget-contract.json](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-15-bg-tw-preservation-budget-contract.json)에 둔다. 데이터 ID는 C4-WebRef-v2를 유지하고 예산 정책은 `W0-KL-TV010-v1`, 첫 method config는 `BG1-W0-C4-TV010-v1`로 분리한다. 이 변경만으로 기존 C4 문서·token·동일 W0 teacher를 다시 만들 필요는 없다.

## 2. 이전 규칙이 method의 입력으로 부적절한 이유

이전의 “N4 terminal까지 관측한 최대 손상의 일정 비율”은 baseline과 같은 operating point를 사후 비교하는 진단에는 쓸 수 있다. 그러나 기본 method에 넣으면 세 문제가 생긴다.

1. **편집 시작 전 별도 순차 경로가 필요하다.** 기존 checkpoint가 있더라도 모든 endpoint의 새 reference 평가를 요구한다.
2. **우리의 허용 손상이 비교 대상에 종속된다.** N4가 더 많이 손상하면 ours도 더 많이 손상해도 되는 기준이 된다.
3. **미래 경로·stream 길이·order에 의존한다.** 첫 B100을 처리하기 위해 B10까지의 baseline 결과를 알아야 한다.

실제 전달문에도 N4 자료가 없으면 `CALIBRATION_MISSING`으로 scientific 실행을 막는 규칙이 있었다. 이는 구현 효율 문제가 아니라 method의 입력 계약 문제다. 해당 차단 조건과 N4 calibration 준비 job을 제거한다.

기존 BG의 **현재 batch native target/closed-form writer**는 유지한다. 이는 ours 안에서 지금 요청을 처리하기 위한 proposal이며, 별도로 W0→1000요청을 진행하는 N4 baseline chain과 구분된다.

## 3. 허용치의 수학적 의미

고정 reference의 scored prefix를 k, 비음수 가중치를 a_k라 하고 합을1로 둔다. 첫 S64는 문서64×position128에 균등 가중치를 준다.

\[
D(W)=\sum_k a_k\,\mathrm{KL}(p_{0,k}\|p_{W,k}),\qquad
\overline{\mathrm{TV}}(W)=\sum_k a_k\,\frac12\|p_{0,k}-p_{W,k}\|_1.
\]

자연로그 KL에서 Pinsker 부등식은 \(\mathrm{TV}(p,q)\le\sqrt{\mathrm{KL}(p\|q)/2}\)다. 이를 각 고정 prefix에 적용하고 Jensen 부등식을 사용하면

\[
\overline{\mathrm{TV}}(W)
\le\sum_k a_k\sqrt{\mathrm{KL}(p_{0,k}\|p_{W,k})/2}
\le\sqrt{D(W)/2}.
\]

따라서 \(D(W)\le2\rho^2\)를 만족하는 endpoint는 같은 finite reference에서 \(\overline{\mathrm{TV}}\le\rho\)를 만족한다. Pinsker의 정확한 상수와 자연로그 규약은 [Canonne의 부등식 정리, Lemma2](https://arxiv.org/pdf/2202.07198)에 확인했다. 평균으로의 적용은 여기의 직접 유도다.

**이는 충분조건이며 동치가 아니다.** KL 기준으로 탈락한 후보가 반드시 평균 TV 허용치를 넘었다고 말할 수 없다. KL 목적은 기존 method 그대로 유지하고 TV는 허용치에 의미를 주는 해석으로 쓴다. TV를 새로운 gradient objective나 별도 hard screen으로 추가하지 않는다.

허용치0.10은 각 prefix의 최대 token-집합 확률 차이를 reference에서 평균한 양에 대한 상한이다. 다음으로 확대하지 않는다.

- 모든 개별 prefix의 TV≤0.10;
- 각 정답 확률의 상대 변화≤10%;
- RS/PS/NS 손실≤10%p 또는 argmax 보존;
- 자유 생성 문장 분포의 TV≤0.10;
- Dev/Report, 전체 C4, 다른 corpus, 전체 old-edit의 보증.

평균이 작아도 소수 입력 손상은 클 수 있고, 거의 동률인 두 후보는 아주 작은 KL에서도 순위가 바뀐다. 따라서 기존 문서 p95/max·NLL·NS·accepted-old·coverage 평가는 계속 필요하다.

## 4. 왜 이 설정이 baseline 없이 가능한가

고정 KL radius를 사전 hyperparameter로 두는 방식에는 [TRPO(ICML2015), §4의 KL 제약](https://proceedings.mlr.press/v37/schulman15.pdf) 같은 선례가 있다. 다만 TRPO는 이전 policy와 다음 policy 사이의 제약이다. BG는 stream 전체의 **고정 original W0**를 기준으로 하며, 그 논문의 RL 개선 보증을 가져오지 않는다.

ρ=.10은 데이터에서 추정한 최적값이 아니다. 이것을 숨기지 않는 것이 N4 최대값에 기대는 것보다 method 정의상 명확하다. 사용자가 보존 수준을 정하거나 독립 개발에서 hyperparameter를 선택하는 일은 필요하지만, 다른 method를 먼저 완주할 필요는 없다.

첫 실행은 ρ=.10 한 설정만 사용한다. 나중에 허용치 민감도가 필요하면 다음 작은 범위를 쓴다.

| 용도 | ρ | mean KL 한도 b |
| --- | ---: | ---: |
| 더 엄격한 후속 설정 | .05 | .005 nats |
| **첫 실행** | **.10** | **.020 nats** |
| 더 넓은 후속 설정 | .20 | .080 nats |

이 sweep는 첫 실행의 선행조건도 이번 자동 제출 목록도 아니다. 각 설정의 과학적 비교는 W0부터 같은1000요청에서 한다. C4/S64→S128/Pile 첫 대조에서도 평균 규약과 ρ가 같으면 b=.02를 유지한다. Corpus별 efficacy가 같다는 주장은 결과로 검증한다.

## 5. Online controller는 남은 여유에 반응한다

허용 한도 b는 고정하되 barrier의 gradient 강도는 실제 candidate의 D64에 따라 달라진다.

\[
h=\frac{b-D}{b},\quad
F=E_{\rm cur}+\mu\,b_\tau(h),\quad
\lambda_{\rm eff}(D)=\frac\mu b\,\psi_\tau(h),
\]

\[
\psi_\tau(h)=
\begin{cases}
1/h,&h\ge\tau,\\
(2\tau-h)/\tau^2,&h<\tau.
\end{cases}
\]

τ=.1, μ=.01, b=.02일 때 유효 KL-gradient 가중치는 다음과 같다.

| candidate D64 | slack h | λ_eff |
| ---: | ---: | ---: |
| .000 | 1.0 | .5 |
| .010 | .5 | 1 |
| .018 | .1 | 5 |
| .020 | 0 | 10 |
| .040 | −1 | 60 |

보존 여유가 작아질수록 actual preservation gradient의 가중치가 커진다. 이것이 현재 상태 feedback이다. **허용 한도를 올리는 adaptation은 하지 않는다.** Relaxed barrier의 gradient만으로 feasibility가 보장되는 것은 아니므로 최종 actual-forward screen을 유지한다.

Gradient 구현에서는 **global mean D64의 barrier**와 **microbatch별 barrier의 평균**을 혼동하지 않는다. 한 번의 generic 순회로 D64와 \(G_D=\nabla_R D64\)를 모으고, 별도 \(G_E=\nabla_R E_{\rm cur}\)에 대해 마지막에

\[
G=G_E+\lambda_{\rm eff}(D64)\,G_D
\]

를 구성할 수 있다. 모든 component는 같은 고정 Rprop에서 계산한다. λ의 추가 미분 항은 필요하지 않으며, 이것이 합성 목적의 정확한 첫 미분이다. Global D64를 안 뒤 frozen slope로 다시 누적하는 구현도 가능하지만 실제 추가 forward 비용을 센다. 이 개정은 route-gradient update1회/B100 상한을 늘리지 않는다.

FixedPenalty의 초기 λ는 μ/b=.5로 둔다. 이는 D=0의 barrier slope를 맞춘 시작값이며 최적의 고정 penalty가 아니다. 나중에 두 방법에 같은 제한된 개발 예산을 주고, 초기 λ 하나가 실패했다고 barrier 우위를 선언하지 않는다.

## 6. Endpoint 한도와 수치 오차

전 batch에서 teacher p0와 b=.02를 유지한다. \(D(W_{t-1})+.02\)를 매번 허용하는 것이 아니며, 10 batches라고 .2, 100 batches라고2.0으로 늘리지 않는다. Original drift가 감소하면 그만큼 여유가 생길 수 있지만 per-batch allowance를 누적하지 않는다.

후보 raw1, corrected1/.5/.25 중 actual-forward D64≤.02를 만족한 후보에서 Ecur로 선택한다. 없으면 parent 유지, all-request 원분모·rejection·coverage에 기록한다. Rejection이 많아져도 실행 중 b를 높이거나 native fallback으로 우회하지 않는다. 성능이 나쁜 정상 실행과 기술 실패는 분리한다.

첫 screen epsilon은0으로 둔다. W0 self-KL 초기 절대 허용 오차는1e−6 nats로 두고 실제 dtype·mask·reduction receipt를 남긴다. 이를 넘으면 `REFERENCE_NUMERIC_INVALID`로 구현을 확인한다. **수치 noise로 scientific b를 키우는 positive floor를 제거한다.** 이후 명시적으로 epsilon을 사용하면 실제 screen ceiling b+epsilon과 \(\sqrt{(b+\epsilon)/2}\)를 함께 기록한다. 수학적 Pinsker 해석과 부동소수점 구현 보증을 같게 쓰지 않는다.

## 7. 배제한 우회안

| 대안 | 첫 방법에 사용하지 않는 이유 |
| --- | --- |
| 현재 batch native 손상의 일정 비율 | 독립 N4 chain은 없어지지만 허용 손상이 harmful proposal에 계속 종속됨 |
| 매 entry 손상 + 고정 allowance | 시간에 따라 original drift 허용치가 누적됨 |
| W0 self-KL/noise의 일정 배수 | 기기·kernel 오차가 과학적 보존 수준을 결정하게 됨 |
| KL/teacher entropy | reference의 confidence에 따라 의미가 달라지고 낮은 entropy에서 불안정; 첫 버전에 불필요 |
| λ만 두고 hard screen 제거 | 별도의 soft-penalty 방법이며 고정 preservation-budget claim이 달라짐 |
| adaptive dual/λ controller | 후속 후보는 가능하지만 허용 손상의 정의 자체를 없애지는 않음 |

## 8. 실행·재개 계약의 변경

새 의존성은 **W0/reference/teacher + 선언한 budget config + BG 기술 검사**다. N4 결과와 timing이 없어도 ours를 시작한다. 메모리·wall allocation은 BG 자체의 작은 기술 profile로 정하고, N4 대비 시간 배수는 사후 비교로 남긴다.

현재 신규 scientific 범위는 기존 ours-first 지시대로 **BG-1 한 chain, W0 B100×10=1000요청**이다. 다섯 지정 baseline과 REFIT4는 비교·재사용 대상이고 새 editing 경로를 자동 제출하지 않는다. Audit/MMLU·추가 corpus·ρ sweep도 새 선행조건이 아니다.

`preservation-budget-manifest.json`에 policy ID, ρ, b, KL reduction, reference/teacher hashes, numeric tolerance, μ/τ/ζ를 잠근다. 재개 시 이 hash가 다르면 같은 chain을 계속하지 않는다. 기존 N4 기반 정책에서 실제 write가 발생했다면 중간에 새 budget으로 바꾸어 같은 실험이라 부르지 않는다. 새 정책의 scientific chain은 W0에서 시작한다. 준비된 C4/teacher는 identity가 같으면 재사용한다.

G0에서 필요한 것은 fixed budget manifest와 실제 technical receipt다. N4 endpoint 누락은 `CALIBRATION_MISSING` 차단으로 처리하지 않는다. G0 이후 능동 모니터링 중단·정상 job 계속 실행·사용자 호출 대기의 기존 운영 규칙은 유지한다.

확인한 origin/main `28a75931f483aad80e6744b0d5d02d47c46e5050`에는 **V1 전달 ACCEPTED** 기록이 있다. 따라서 수정 전달문은 **V2 개정안**으로 식별하고, 이미 전달된 V1이 자동 변경됐다고 표기하지 않는다. 이번 작업은 로컬 canonical 설계·JSON·GH 전달문을 갱신한 것이며, 원격 실행에 개정안이 전달·적용됐다는 보고가 아니다.

## 9. 개정 범위와 검증

- 갱신: reference/method/budget 계약, method 설계와 survey의 budget 부분, ours-first dispatch/GH 지시문.
- 유지: C4 identity/token/teacher 규약, native target/writer, gradient quota, 후보 수, 과거 accepted-edit ledger, W0 SEQ1000, 비교 baseline 목록.
- 검산: 평균 Pinsker 해석, ρ→b 단위·상수, barrier slope, global vs microbatch barrier gradient, 평균 제약의 tail/argmax 반례, 문서 간 budget 일치.
- 미측정: b=.02의 실제 acceptance·PS/NS·old retention·wall, native FP32 gradient parity, full10k 효능.

이 개정의 결과는 **baseline에서 파생된 허용 손상을, 사전 선언한 original-response 보존 허용치로 교체한 것**이다. 이제 N4가 없다는 이유로 method 정의나 실행이 성립하지 않는 문제는 없다. 어느 허용치가 좋은 editing–preservation 균형을 주는지는1000요청 순차 결과로 판단한다.
