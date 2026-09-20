# Single-layer mechanism-first B1: 독립 상세 리뷰

작성: 2026-09-20 KST. 대상은 Server4 51058 / B1 완료 보고서와 해당 실행의 보존 raw/source다. 이번 작업은 **읽기 전용 자료 확인·CPU 재집계·작은 저장 Jacobian의 수학 검산**이다. 새로운 CPU cone LP24건과 제약군을 하나 제외한 작은 counterfactual 최적화2건을 수행했다. 새 GPU/model/forward/실험 제출, 원 production/source 수정은 없다.

사용자 지정 보고서의 [로컬 사본](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-20-slmf-b1-independent-review/source-report/report-ko.md), [입력 snapshot](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-20-slmf-b1-independent-review/source-snapshot.json). Published47개 member의 크기/SHA를 전부 대조했다. 약1.15MB의 공개 report package만 복사했으며 4.1GB raw 전체를 다시 전송·해시하지 않았다.

## 1. 판단

**B1에서 preservation 개선 방법은 확보되지 않았다. 그러나 DEC 실패 원인과 기존 제약 밖의 기능적 여유에 대해서는 기존 요약보다 구체적인 evidence가 있다.**

1. EN은 실제로 움직이고 train KL을5.55% 줄였지만 official neighborhood 성공 회복은0이다.
2. DEC는 이미 실패한424개 reference 문서 각각의 worst margin을 조금도 악화시키지 않도록 요구했다. 저장된1D/5D 후보 공간에서 이 조건들이 **수치적으로 feasible cone를 원점으로 축소**했다. Zero-gradient, phase2의 최소norm 선택, FP32 반올림이 근본 원인이라는 설명은 맞지 않는다.
3. Source-mode0 제거에서는 selected N4개가 회복됐다. 그러나 reference 신규 token flip6개와 Current NLL guard 위반96개가 함께 발생했다. **일부 기능적 성공을 유지하면서 다른 비용을 교환한 신호**이며 기존 DEC 조건을 충족한 보정이 아니다.
4. B1의100개 projected-key singular values는1.8115–5.0445다. 이 자료는 아직 누적 history 충돌이나 거의 특이한 key geometry를 보여주지 않는다.
5. 지금 S3로 연장하면 고장난 local proposal과 비싼 준비 계산을 반복할 가능성이 높다. 저장된 local 문제의 제약 구조와 실제 functional preservation 범위를 먼저 정리하는 것이 타당하다.

## 2. 독립 재집계한 기본 성능

네 arm의 raw target NLL과 token flags를 직접 다시 읽고, old/new prompt·target·token identity를 결속하여 집계했다. Raw 파일 hash는 published inventory와 모두 일치했다. [Raw 재집계](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-20-slmf-b1-independent-review/canonical-raw-audit.json).

|Arm|RS|PS|NS|Native 대비 성공 ID 변화|
|---|---:|---:|---:|---|
|N4|100/100|194/200|865/1000|기준|
|EN-KL-Q|100/100|194/200|865/1000|R/P/N 모두 loss0/gain0|
|DEC-LINE|100/100|194/200|865/1000|동일 native endpoint|
|DEC-MODES-CUM|100/100|194/200|865/1000|동일 native endpoint|

W0-correct N886개 중 native가24개를 잃고 기존 실패3개를 얻었다. EN도 이24개를 하나도 회복하지 않았다. 총점이 우연히 같으면서 성공 집합이 바뀐 경우가 아니다.

다만 PS97%는 pairwise NLL preference다. New-target TF-strict는125/200=62.5%, R와 두P 모두 strict인 요청은44/100이다. NS86.5%도 true target이 edit target보다 유리한 비율이며 true-target TF-strict는162/1000이다. 따라서 이 B1이 보여준 강한 결과는 **canonical rewrite100개 성공**이며, 모든 paraphrase의 정확한 재현이나 일반 모델 성능 보존까지 완성됐다는 뜻은 아니다.

## 3. EN: 실제 변화는 있지만 preservation 이득은 작고 불완전하게 관측됨

Train KL은 .00112039035013→.00105820016472, 약5.5508% 감소했다. Accepted correction norm은 .0084972로 native write norm7.61175의 약0.1116%다. 작은 크기 자체가 실패의 원인이라는 주장은 할 수 없지만, 실제 크기를 생략하고 full write 재설계와 같은 intervention으로 비교해서는 안 된다.

EN−N4 평균 safety-margin 변화:

|항목|평균 변화|해석|
|---|---:|---|
|R|−2.91e−7|거의 동일|
|P|−8.47e−5|작은 약화, 성공 ID는 동일|
|N|+2.75e−5|작은 개선, 성공 회복0|

W0→native의 N 평균 margin 감소는 .547026이다. EN의 평균 개선은 그 감소량의 약0.0050% 수준이다. 이 비율은 같은 scalar의 기술적 크기 비교이며 모델 성능 회복률이 아니다. 작은 systematic numerical 차이와 실제 일반화 이득을 bootstrap 구간만으로 분리할 수 없다.

Dev128에서는 total mismatch306개가 같지만 신규 flip2/회복2, Phi .2438134→.2440979로 증가했다. 따라서 평균 KL 개선과 choice-risk 개선은 분리해야 한다. **Selected EN의 train512 choice/Phi는 미측정**이므로 train 선택 회복이나 no-new-flips를 주장할 수 없다.

현재 자료만으로 확실한 것은 “이번 KL gradient/scale/Current 보호 조합이 official N 성공을 회복하지 못했다”다. C4 reference 전체가 무용하다는 일반화나 full allowed-space 최적해가 없다는 결론은 아니다.

## 4. DEC의 no-move는 제약 구조로 설명된다

### 4.1 Frozen local 문제

Native의 reference 문서별 worst margin을 μ_i, 후보 계수를 a, 저장 Jacobian을 J라 하면 목적은

\[
f(a)=\frac1{512}\sum_i[-(\mu_i+J_i a)]_+^2
\]

이고, solver는 각 문서에

\[
\mu_i+J_i a\ge\min(\mu_i,0)
\]

를 요구한다. 따라서 이미 unsafe한 μ_i<0 문서에는

\[
\boxed{J_i a\ge0}
\]

가 적용된다. 이미 깨진424개 문서의 worst margin을 모두 동시에 비악화시켜야 한다. Safe88개는 원점에서 positive slack을 가진다. Tie는0이다.

이 조건은 “아직 안전한 판단을 새로 깨지 않는다”보다 강하다. 이미 실패한 문서들 사이의 작은 악화/큰 회복 교환까지 금지한다. 원 설계에 있던 규칙이며 이번 frozen source가 임의로 추가한 조건이 아니다.

### 4.2 LINE은 부호만으로 정지 원인이 드러난다

Unsafe424개 중 line의 J는 positive378개, negative46개다. 같은 scalar a에 어떤 문서는 a≥0, 다른 문서는 a≤0을 요구하므로 **a=0만 가능**하다.

Saved J에서 계산한 objective gradient norm은12.011049882이고 controller 기록12.011049878과 상대오차3.64e−10으로 일치한다. Gradient가 사라져 멈춘 것이 아니다.

### 4.3 MODES도 저장5D 전체가 막혔다

Unsafe J는 rank5이고 최소 singular value는 .936624다. Row-normalized constraints와 box[-1,1]에서 각 ±coordinate를 최대화하는10개 독립 CPU LP가 모두0을 반환했다. Solver tolerance1e−9이며 실제 모델 평가가 아니다.

또한 unsafe C=J_unsafe에 대해 **w>0, Cᵀw≈0**인 positive-dependence witness와 full column rank가 확인됐다. 정확연산에서는

\[
Ca\ge0,\quad w>0,\quad C^Tw=0
\Rightarrow w^TCa=0\Rightarrow Ca=0\Rightarrow a=0.
\]

이번 floating-point witness residual과 LP tolerance를 receipt에 남겼다. Interval arithmetic으로 인증한 실수 정리는 아니지만, 저장 local model의 zero-cone 해석을 강하게 지지한다.

보고서의 주요 dual-active5개 row를 제거해도 나머지 unsafe rows가 cone를 원점으로 축소했다. 따라서 특정 reference 몇 개를 사후 삭제하는 것으로 원인을 설명할 수 없다.

### 4.4 무엇을 배제할 수 있는가

- Phase1부터0이었다. Phase2가 유의미한 개선해를 지운 것이 아니다.
- CUM ideal norm이 약3.86e−31이었다. 유의미한 update가 FP32 rounding으로 소실된 상황이 아니다.
- Radius는 .0135971이고 ball slack은1이었다. 더 큰 radius나 네 scale의 추가 halving으로 zero cone가 열리지 않는다.
- Covariance 방향은 실제로 추가됐고 rank5였다. 그러나 그5D 공간에서도 unsafe 비악화가 전부를 막았다.
- 전체 Q 허용 공간의 차원이나 capacity가0이라는 뜻은 아니다. 이번1D/5D span에 대한 결과다.

Unsafe 비악화만 제외하고 같은 objective·ball·safe88 제약을 유지한 작은 CPU 방향 검산에서는 양의 작은 step과 predicted risk 감소가 존재했다. LINE .163315→.159683, MODES→.160017이며 safe 최소 predicted margin은 약.00051>0이었다. **이 값은 저장 선형 모델의 counterfactual이고 actual full512/Current/PS/N 검증 결과가 아니다.**

같은 조건의 작은 convex 문제를 독립 CPU optimizer로 추가 계산한 결과도 일치한다.

|저장 선형 문제|원래 risk|Unsafe 비악화만 제외한 predicted risk|Margin이 악화된 unsafe 문서|
|---|---:|---:|---:|
|LINE|.163315|.121865:약25.38%감소|46|
|MODES|.163315|.105110:약35.64%감소|51|

두 해는 같은 radius ball 경계에 있고 safe88의 최소 predicted margin은 양수다. KKT stationarity residual은0과1.93e−8이다. 이는 제약군의 영향을 분리한 결과이며, unsafe 문서 내부의 나머지 native-safe token까지 보호했거나 실제 모델이 이만큼 좋아졌다는 뜻은 아니다. 이 CPU 실험을 새 B1 arm으로 세거나 제약 완화의 타당성으로 곧바로 해석하지 않는다.

원 실제 acceptance에는 worst-deficit1e−4 numerical tolerance가 있으나 local 제약은 exact nonregression이었다. 이 차이는 기록해야 하지만 수치 tolerance를 임의의 semantic damage budget으로 사용해서는 안 된다.

자세한 source/J/dual/LP 기록: [DEC solver 감사](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-20-slmf-b1-independent-review/decision-solver-audit.json).

## 5. Mode0 제거: 유망한 tradeoff와 명백한 보호 위반이 동시에 있음

### 5.1 실제 개입의 정의

Source mode0는 최대 target-loading mode, mode99는 그 외 mode 중 최대 raw-map weight cost로 선정됐다. Panel 성과를 본 뒤 label을 골랐다는 근거는 없다. 하지만 전체100mode의 optimal selection을 탐색한 것도 아니다.

Frozen 개입은 `W'=FP32(W_native−left@right.T)`이며 **해당 rank1 mode를100% 제거**한다. Q_E projection은 사용하지 않는다. Current는 canonical100개만 평가했다. PS와 full native-context Current guard는 평가하지 않았다.

|관측|Native|Mode0 제거|
|---|---:|---:|
|선정 N 성공|16/32|20/32:4회복/새손실0|
|선정 reference mismatch|46|36:16회복/신규flip6|
|선정 reference Phi|.0350905|.0112549:약67.93%감소|
|Canonical RS/TF-strict|100/100|100/100|
|Current desired-NLL 증가>1e−4|—|96/100|
|Current desired-NLL 평균 증가|—|약.0063034 nats|
|Current desired-NLL 최대 증가|—|약.3191714 nats|
|Reference worst deficit 악화|—|6문서|

Mode0는 기존의 Current NLL guard와 reference no-new-flips/nonregression에 불합격한다. **완전한 Pareto 개선**이 아니다. Mode99는 canonical Current strict도100→99로 감소했다.

Published CSV의 `N_new_strict`는 N에서 보존해야 할 true-target strict가 아니다. 이를 locality strict로 읽으면 해석이 뒤집힌다. Raw true-target strict는 native2→mode0의3으로 증가했다.

### 5.2 Random 대조군의 범위

Mode0 weight norm은 .759262, native write norm의 약9.975%다. Random control은 rank1와 Frobenius norm을 맞췄지만 current-response 작용량은 다르다:

\[
\|C_0K_{native}\|_F\approx3.83012,\qquad
\|C_{random,0}K_{native}\|_F\approx.17194.
\]

약22.3배 차이다. Mode99도 약9배 차이다. 따라서 mode0 대 random 비교에서 target-writing 작용량까지 통제했다고 할 수 없다.

Native mean keys가 Current의 locked TF-key span에 포함된다면 DQ_E=D인 보정은 이 keys에서0으로 작용해야 한다. Mode 제거는 모든100 native mean keys에 비영 반응을 만들었으므로 그 exact-current 보호 조건과 양립하지 않는다. 실제 큰 Q tensor를 새로 로드하여 CQ norm을 측정한 결과는 아니며, 이 해석의 가정과 직접 측정 DK를 구분한다.

### 5.3 연구적으로 남는 신호

Canonical 성공 지표에는 native response를 정확히 고정하는 조건보다 여유가 존재했다. 그 여유를 사용한 방향에서 일부 N이 회복됐다. 그러나 PS, full reference512, Dev, history에 대한 보존은 미확인이고 실제 reference 신규 손실도 있다.

따라서 다음 질문은 **편집 성공을 유지하는 기능적 허용 범위와 preservation 비용 사이의 교환을 어떻게 정량화할 것인가**다. “Mode0를 항상 삭제한다”거나 “exact-Q 안에서도 같은 이득이 가능하다”는 결론은 이 자료에서 나오지 않는다.

자세한 원시 전이·NLL·factor 검산은 [component 감사](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-20-slmf-b1-independent-review/component-mechanism-audit.json)에 기록했다. N32는24개 parent request에서 왔으므로32개의 독립 사실로 취급하지 않는다.

## 6. Key-capacity 가설에는 어떤 evidence인가

이번 B1은 M0=0이다. 이전 edit과의 구별성 저하를 측정할 과거 edit이 없다. 저장100mode의 ideal projected-key sigma는1.811519–5.044536, condition 약2.78470이다. 거의 singular한 key 때문에 write가 폭주한 사례라는 근거는 없다.

Native realization residual 상대값 .130086과 R/P 성공은 함께 해석해야 한다. Activation target fitting이 완벽하지 않아도 benchmark edit 성공은 가능했다. 반대로 N 손상은 이미 첫 batch에서 발생했다. 따라서 locality 손상에 누적 edit-capacity 고갈이 반드시 선행해야 하는 것은 아니다.

Mode0는 큰 key sigma/큰 target loading, mode99는 더 큰 weight cost를 가진다. 작은 selected panel에서 mode0 제거가 더 나은 N 결과를 보였다는 점도 norm 또는 낮은 sigma만으로 손상을 예측할 수 없음을 시사한다. Signed downstream sensitivity와 입력 overlap의 결합을 봐야 한다.

이후 Server2 checkpoint audit은 여전히 필요하다. 같은 probe의 history 변화, saved-z target demand, actual write, 같은 cohort의 margin 전이를 연결해야 한다. 다만 이번 B1과 Server2 archive B1은 서로 다른 실행이다. Server2의 R100/P190/N867을 이번100/194/865와 같은 endpoint로 합치지 않는다.

또한 이번 report의 mode index는 ideal key spectrum 기준이다. Server2 설계의 raw adjusted-key B-SVD와 index 의미가 다르므로 “mode0”라는 번호를 그대로 이식하지 않는다.

## 7. 계산량: 어디가 병목인지 다시 분리

비용 표와 raw counters/타이머의 수치 일치는 확인됐다. [비용·coverage 감사](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-20-slmf-b1-independent-review/cost-coverage-audit.json).

|Arm|Native 전액+필수 correction 회계|Native 대비|
|---|---:|---:|
|N4|272.84초|약1배|
|EN|1511.46초|5.56배|
|LINE|541.37초|1.99배|
|CUM|933.93초|3.44배|

이는 shared run의 비용 경계 회계이며 독립 walltime이 아니다. Setup/observer/commit/미분리 작업은 제외되어 있다. 이를 모두 합쳐 전체 연구 비용으로 다시 세면 중복이다.

- EN1179.52초 중 callback1166.70초(98.9%). R512 KL gradient1회+value4회이며 논리 teacher 통과량은 약334.07GB다. OS cache를 반영한 물리 disk read가 모두 이만큼이라는 뜻은 아니다.
- Postseal KL 한 sweep은145.89–156.50초, 그 안의 teacher read45.54–45.68초와 suffix84.73–93.24초가 모두 크다. 이 관측을 controller trial의 실제 분해로 대입하지 않는다.
- CUM441.58초 중 solver를 포함한 callback 전체는8.83초다. **최소432.75초는 callback 밖**이다. Covariance/J/basis 검사·identity/materialization 등의 개별 시간은 분리되지 않았다. “QP가 느려서441초”라는 해석은 틀리다.
- Empty history의 약469.76MB zero-gradient transfer와 네 no-move materialization은 제거할 수 있는 중복이다. 전자는 약.27초로 주된 병목은 아니다.
- Report가 제외한 writer timer7403588초는 start 변수 재사용 버그다. 유효한 upper timer와 분리하여 처리한 것은 타당하다.

같은 zero cone에서 iteration/scale 횟수만 늘리면 방법 실패는 해결되지 않는다. 먼저 작은 coefficient 문제에서 비영 이동 가능성을 판정하고, 그 이후에만 비싼 candidate 검증을 실행하는 것이 합리적이다. Frozen original source는 이번 리뷰에서 고치지 않았다.

## 8. 검증과 누락의 경계

- FD는 waiver이며 PASS가 아니다. Cached/direct gradient 일치와 stored KKT 일치는 전체 finite-step/model 수치검증과 다르다.
- Hook의 제한된 panel gradient 오차가 원 기준을 넘은 warning이 남아 있다. 새 GPU parity를 실행하지 않았다.
- `full512=true`는 native derivative/scan coverage다. DEC의 candidate full512 검사는 실제0회이며 candidate acceptance PASS가 아니다.
- B1에는 history가 없다. GSS, history retention, STEP/CUM 차이, 누적 상쇄 이득은 미측정이다.
- NoCP는 사용자 지시에 따른 것이며 실행 오류가 아니다. 저장 J/dual/factor/관측으로 CPU 분석은 가능하지만 selected EN W/D와 완전 상태가 없어 누락된 train choice를 현재 자료만으로 정확하게 채울 수 없다.
- 원 report는 주요 미측정을 대체로 정직하게 표시했다. 이번 리뷰의 추가 가치는 잘못된 총점 교정보다 **zero-cone 원인, component tradeoff, 대조군의 한계**를 더 구체적으로 밝힌 것이다.

## 9. 후속 판단

1. **현 DEC를 그대로 sequential 확대하지 않는다.** B1 gate 실패뿐 아니라 저장 local 문제의 정지 원인이 확인됐다.
2. Reference 전체512와 길이256을 줄이는 것이 이번 결과의 직접적 해결책이라는 근거는 없다. 선택 공간과 unsafe 문서별 비악화 조건의 조합을 먼저 검토한다.
3. Component 신호는 native의 과도한 응답 고정과 실제 편집 성공 사이의 여유를 조사할 이유다. 다음 실제 실험에서는 canonical뿐 아니라 PS·전체reference·Dev·N 신규손실을 함께 확인해야 한다. 이 자료로 이미 충족했다고 하지 않는다.
4. Server2 retrospective audit은 진행 가치가 있다. Target conflict를 분리한 lifelong 자료에서 같은 기전이 누적되는지 확인해야 한다. B1 healthy spectrum을 무시하고 key collision을 선결 원인으로 두지 않는다.
5. GSS는 history가 생긴 이후의 별도 문제다. 이번 B1 zero-cone나 reference 목적 불일치를 GSS만으로 해결할 수 있다고 연결하지 않는다.

판정: **방법의 preservation 성과는 미확보, DEC 제약 구조의 구체적 failure mode는 확보, task-level 여유를 활용할 수 있다는 제한된 component signal은 확보.**
