# Single-layer write 강도 직관과 Server2/Server4 기전의 연결

작성: 2026-09-20 KST. 새 실험 설계문이 아니라 기존 산출물의 추가 분석이다. Server4의 봉인 source·저장 factor·reference scan 9파일을 기존 inventory의 bytes/SHA와 대조하고 CPU에서 재계산했다. 새 모델 forward, z 최적화, GPU job은 모두 0이다.

재현: [recompute.py](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-20-single-layer-write-strength-analysis/recompute.py).
수치·입력 hash: [quantitative-analysis.json](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-20-single-layer-write-strength-analysis/quantitative-analysis.json).

## 1. 현재 판단

**Native L4 write의 강도를 낮출 때 preservation이 좋아질 여유가 있다는 직관은 일부 실제 후보와 저장 미분값에서 지지된다. 그러나 원인이 Adam의 overshoot라고 확인된 것은 아니며, 하나의 scalar를 줄여 RS·PS·NS를 모두 유의미하게 개선한 방법은 아직 확보되지 않았다.**

이번 분석에서 가장 중요한 구분은 다음 세 가지다.

1. EN의 약 0.112% 보정은 native response를 고정하는 null-space 보정이다. Native write를 줄이는 방향을 시험한 결과가 아니다.
2. Server4의 source-mode0 제거는 약 10% 크기의 보정으로 일부 N을 회복했지만, native write 전체 norm은 약 0.5%만 줄였다. 보정 에너지의 약 99%가 일괄 축소 방향과 직교한다. 이는 방향 선택의 신호다.
3. Server2는 높은 rewrite 선호와 큰 locality 누적 손실이 공존함을 보여준다. 하지만 history 충돌이 native write를 반드시 증폭한다거나, 특정 축소 배율이 이 손실을 해결한다는 인과 증거는 아니다.

따라서 분석의 초점은 **native가 요구한 hidden response 중, 실제 편집 성공을 유지하면서 줄이거나 바꿀 수 있는 부분과 그 preservation 비용**이다. 단일 레이어는 그 후보들의 activation 작용을 정확하게 계산하고 하나의 고정된 writer 공간에서 비교할 수 있게 한다.

## 2. 서로 다른 실행을 먼저 구분한다

|근거|시작점·범위|이번 질문에서의 역할|
|---|---|---|
|과거 C4/C45678 cold B1|W0에서 같은 native 후보를 공유한 scalar 분기|실제 배율 변화와 R/P/N의 관계|
|Server4 mechanism-first B1|별도 W0 실행; R100/P194/N865|EN, DEC, mode 제거, 저장 native 방향 미분|
|Server2 회수 checkpoint|L4-only 한 stream의 W0 및 B1–B100|누적 preservation 손실, history·writer·activation 기전|
|REFIT4 및 FROZEN2|W50/M50에서 B51–B60|부분 write 이후 재최적화의 과거 보조 근거|
|SL-ZFlow|별도 cold SEQ1000|고정 writer 좌표·actual-write 최적화만으로는 해결되지 않았던 반례|

Server2 첫100의 B1은 R100/P190/N867이다. Server4 B1과 동일한 endpoint로 합치지 않는다. C4의 R100/P194/N865도 Server4와 총점이 같을 뿐 actual weight와 reference 정의가 다르다. REFIT4는 warm 실험이므로 현재 cold 분석의 직접 대조군으로 삼지 않는다. 이번 작업은 warm checkpoint에서 새 비교를 실행하지 않았다.

## 3. 기존 scalar 실험은 무엇을 보여주는가

과거 C4 B1의 candidate.csv와 candidate-observer.csv를 arm/batch/candidate로 결합했다. 아래는 모두 같은 cold entry와 같은 최초 native target에서 실제로 실행한 L4-only 후보다. C45678의 마지막 행은 추가 layer 계수가 모두 0인 pruning 후 후보다.

|L4 배율 α|RS /100|PS /200|NS /1000|Canonical strict /100|Reference S64 KL|당시 controller|
|---:|---:|---:|---:|---:|---:|---|
|1|100|194|865|100|0.001720908|Native|
|0.75|100|182|876|99|0.000951336|E·strict 제약 위반|
|0.96875|100|193|866|100|0.001611670|E 제약 위반|
|0.995|100|193|865|100|0.001703174|E 제약 위반|
|0.998265282|100|194|865|100|0.001714740|Feasible|

이 자료는 다음을 보여준다.

- α=0.75에서 reference KL과 NS는 좋아지지만 PS는 12개 줄고 canonical strict도 1개 줄었다. Rewrite pairwise 성공 100만 보면 이 손실을 놓친다.
- α=0.96875는 canonical 성공을 유지하며 N 1개를 더 얻지만 P 1개를 잃는다.
- α≈0.998265는 R/P/N 총점과 strict를 유지하며 KL을 약 0.36% 줄였다. 의미 있는 N 성공 개선까지 나온 것은 아니다.
- C4의 E는 native-context target NLL이다. 원 controller의 E≤E_native+1e−4 조건은 binary 성공 유지보다 강했다. α=.995의 canonical strict가 100이어도 E 증가로 기각됐다.
- 그렇다고 E 제약만 풀면 PS가 유지된다는 결론도 나오지 않는다. E에서만 기각된 .96875와 .995의 PS가 이미 1개씩 낮다.

**강도를 낮출 여유는 존재하지만, 모든 요청과 문맥에 같은 여유가 있는 것은 아니다.** 이 관측은 고정 .75를 일반 정책으로 쓰기보다 현재 batch의 기능적 여유를 측정해야 한다는 근거다. PS는 사후 observer이며 이 표에서 가장 좋은 후보를 골라 독립 최종 성능으로 다시 보고하면 안 된다.

원자료: [candidate.csv](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-18-sequential-local-z-allocation-review/report-snapshot/candidate.csv), [candidate-observer.csv](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-18-sequential-local-z-allocation-review/report-snapshot/candidate-observer.csv).

## 4. Server4 EN의 보정은 작지만, 크기만으로 실패를 설명할 수 없다

실제 native와 accepted EN은 다음 크기다.

\[
\|\Delta_N\|_F=7.61174865,\quad
\|D_{EN}\|_F=0.00849721,\quad
\|W_0\|_F=77.422135.
\]

따라서 EN 보정은 native 대비 **0.1116%**, W0 대비 **0.0110%**다. 사용자 표현의 “0.1% 아래”와 달리 native 대비로는 조금 크고, W0 대비로는 훨씬 작다. 분모를 명시해야 한다.

EN의 동일 방향에 대한 실제 line search는 다음과 같다.

|EN 방향 scale|실제 보정 norm|Native 대비|Train KL|결과|
|---:|---:|---:|---:|---|
|1|0.06797765|0.8931%|0.003047223|기각|
|0.5|0.03398882|0.4465%|0.001655110|기각|
|0.25|0.01699441|0.2233%|0.001171781|기각|
|0.125|0.00849721|0.1116%|0.001058200|수용|

Native KL은 0.001120390이다. 더 큰 세 후보는 이 KL조차 악화시켰다. 현재 방향의 norm을 키우는 것만으로 해결된다는 근거가 없다.

더 근본적인 차이는 허용 공간이다. EN은 current의 고정 full-token keys를 K_E라고 할 때

\[
D=DQ_E,\qquad DK_E=0
\]

를 요구한다. 반면 native를 α배로 줄이는 보정은

\[
D_\alpha=(\alpha-1)\Delta_N,
\quad D_\alpha K_E=(\alpha-1)\Delta_NK_E.
\]

편집 반응이 비영이면 α≠1에서 이 값은 0이 아니다. **즉 native write를 25% 줄이는 실험은 EN의 norm을 크게 한 실험과 다른 공간의 실험이다.** Full-token 고정은 평균 key보다 강하며, QE가 지키는 key span에 native 평균 key가 포함된다는 조건에서 이 비교가 직접 성립한다. 이번 mode 개입은 모든 native 평균 key에서 비영 반응을 만들었다.

25% 축소의 보정 norm은 약 1.90294로 accepted EN의 약 224배다. 그러나 이 배율 차이를 적절한 EN learning rate로 읽으면 안 된다. 두 방향의 current 작용이 다르다.

원 리뷰: [Server4 B1 독립 감사](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-20-slmf-b1-independent-review/review-ko.md).

## 5. 새 재집계: native를 줄이는 방향의 reference 기울기는 유리하다

봉인 postselection.py는 각 문서의 native endpoint activation gradient를 실제 Δ_N K_i와 내적하여 signed_native_scalar_derivative를 저장했다. 과거부터 존재하던 이 값을 reference-native.json의 같은 document ID의 μ_i와 이번에 결합했다.

문서 i의 μ_i는 고정 base continuation의 답변 token 위치들 중 가장 작은 base-choice margin이다. 다음 경로를 생각한다.

\[
W(\alpha)=W_e+\alpha\Delta_N,
\quad d_i=\left.\frac{d\mu_i(W(\alpha))}{d\alpha}\right|_{\alpha=1}
=\langle\nabla_W\mu_i(W_N),\Delta_N\rangle_F.
\]

저장된 것은 native에서 노출된 worst position/competitor branch의 미분이다. 이를 합치면 다음과 같다.

|문서군|개수|d_i<0: 조금 축소하면 개선 방향|d_i>0: 조금 축소하면 악화 방향|
|---|---:|---:|---:|
|전체 reference|512|477|35|
|Native에서 μ_i<0|424|421|3|
|Native에서 μ_i≥0|88|56|32|

421/424=99.29%의 unsafe 문서는 축소 방향이 유리하다. 다만 이 비율은 gradient 부호의 기술통계이며 실제 회복률이 아니다.

문서별 risk를

\[
\Phi(\alpha)=\frac1{512}\sum_i[-\mu_i(W(\alpha))]_+^2
\]

로 두면 저장된 선형 모델에서

\[
\Phi(1)=0.163315118,\qquad
\Phi'(1)=\frac2{512}\sum_{\mu_i<0}\mu_i d_i
=1.361522626>0.
\]

따라서 α를 조금 낮추는 것이 risk descent 방향이다. 이는 기존 EN-Q 방향과 다른, 사용자 직관에 직접 대응하는 신호다.

동시에 이 결과는 기존 DEC 제약의 문제도 드러낸다. DEC는 이미 unsafe인 문서에조차

\[
\mu_i(\alpha)\ge\mu_i(1)
\]

를 요구했다. 선형화하면 (α−1)d_i≥0이다. Unsafe의 음수 기울기 421개는 α≤1, 양수 기울기 3개는 α≥1을 요구하므로

\[
\boxed{\text{이 scalar 선형 문제에서 feasible한 것은 }\alpha=1\text{뿐이다.}}
\]

**Native 방향을 허용 공간에 추가해도 “이미 실패한 모든 reference의 비악화”를 유지하면 축소가 다시 막힌다.** 안전한 문서를 새로 깨지 않는 조건과, 이미 실패한 문서 사이의 작은 악화/큰 개선 교환을 금지하는 조건은 다르다.

### 이 기울기가 아직 증명하지 않은 것

- α=.75 같은 큰 변화를 실제로 평가한 S4 결과가 아니다. 새 forward는 0회다.
- Gradient의 bound scalar는 현재 worst branch다. 이동하면 worst token과 경쟁 token이 바뀔 수 있다.
- 저장 선형식을 α=0까지 외삽하면 risk≈.0595가 남는다. Cold α=0이 본래 base W0라는 사실과 base가 선택한 token을 보호한다는 정의를 생각하면, 이를 실제 base risk로 읽을 수 없다. 같은 labels/prefix/decoding 조건에서 base-choice margin은 음수가 아니어야 한다. 이 외삽 자체가 큰 이동의 근사 오류를 보여준다.
- Native에서 safe였던 문서 88개만 확인하는 것으로 unsafe 문서 내부의 다른 안전한 token까지 보호한 것은 아니다.
- 원 실행의 FD waiver와 hook-gradient warning을 그대로 상속한다. 이번 CPU 산술 확인은 모델 미분의 새 수치 인증이 아니다.
- Reference risk 하락이 held-out NS 개선을 보장하지 않는다.

## 6. “축소하면 좋은 지점이 존재할 수 있다”의 정확한 수학적 범위

다음은 제한적인 존재 명제다. Capacity theorem이나 실용적인 배율 보장이 아니다.

**명제.** Native endpoint에서 보호할 유한 개의 실제 출력 margin f_j가 모두 양수이고, W(α)에 대해 연속이라고 하자. 또한 reference 목적 Φ의 native에서 축소 방향 미분이 음수라고 하자:

\[
f_j(W_N)>0\quad\forall j,\qquad
\lim_{h\downarrow0}\frac{\Phi(W_N-h\Delta_N)-\Phi(W_N)}{h}<0.
\]

그러면 어떤 ε>0가 존재하여 충분히 작은 0<h<ε에서 보호 margin을 모두 양수로 유지하면서 Φ를 줄인다.

**증명.** 각 f_j의 연속성과 f_j(W_N)>0로부터 양수를 유지하는 열린 구간이 존재한다. 유한 개 구간의 공통 반경도 양수다. 방향 미분이 음수이므로 충분히 작은 양의 h에서 Φ 차이는 음수다. 두 반경의 최솟값을 취하면 된다.

이 명제는 Adam을 가정하지 않는다. Native가 어떤 방법으로 만들어졌는지보다 **기능적 성공에 여유가 있고, 그 여유 안에 preservation descent 방향이 있는지**가 중요하다.

한계도 강하다. ε가 극히 작거나 FP32에서 구별하기 어려울 수 있고, 모든 성공 bit가 그대로라 NS 회복은 0일 수 있다. 현재 reference gradient의 수치 정확성·비미분 branch 문제 때문에 저장 d_i만으로 실제 신경망에 대한 명제를 인증한 것은 아니다. Canonical만 보호했다면 paraphrase와 history는 결론에 포함되지 않는다. “아주 작은 개선 지점의 존재”와 “논문에 충분한 Pareto 개선”을 구분해야 한다.

## 7. Adam overshoot와 closed-form realization을 구분해야 한다

z 최적화는 특정 위치에 activation을 주입하여 target loss·KL·anchor regularizer 등을 줄인다. Closed-form write는 그 target residual R을 key K와 history M 아래에서 실현한다. 최종 locality는 또 다른 입력에서 전체 downstream 함수의 출력을 평가한다.

따라서 세 목적은 같지 않다. Adam이 덜 수렴하거나 과하게 최적화되어야만 불일치가 생기는 것도 아니다.

이상적인 직교 projector P, PSD history M, λ>0에서 native는 다음 문제의 해로 볼 수 있다.

\[
\min_{\Delta=\Delta P}
\frac12\|\Delta K-R\|_F^2
+\frac12\operatorname{tr}(\Delta M\Delta^T)
+\frac\lambda2\|\Delta\|_F^2.
\]

Native 방향 Δ_N에 대한 scalar 미분을 α=1에서 0으로 놓으면

\[
\langle R,\Delta_NK\rangle
=\|\Delta_NK\|_F^2
+\operatorname{tr}(\Delta_NM\Delta_N^T)
+\lambda\|\Delta_N\|_F^2.
\]

그러므로 target fit 항만 최소화하는 배율은

\[
\alpha_{fit}
=\frac{\langle R,\Delta_NK\rangle}{\|\Delta_NK\|_F^2}>1
\]

이다. 비영 write·λ>0 및 비영 fitted response를 가정한다. 정규화가 target을 일부 덜 맞추는 해를 만든다는 뜻이다.

이번 Server4의 저장 FP32 residual R 및 adjusted-key map B로 Δ_alg=RB^T를 재구성하여 저차원 Gram 내적으로 계산하면:

|선형 target fitting 진단|값|
|---|---:|
|α_fit|1.140879917|
|α=1에서 상대 fitting error|0.130086354|
|α=.75에서 상대 fitting error|0.344795682|

실제 native update와 이 algebra factor의 상대 차이는 원 receipt에서 약 6.07e−6이다. 위 결과는 작은 factor 산술의 결과이며 새 모델 endpoint 평가가 아니다. K는 context 평균 key이고 R의 anchor 좌표도 native 정의를 따른다. 따라서 canonical 실제 출력 NLL과 동일한 손실로 읽지 않는다.

**현재 자료는 “closed form이 z를 너무 많이 실현했다”보다 “z를 덜 실현해도 실제 성공에는 충분할 수 있다”는 설명에 맞는다.** Adam의 기여를 분리하려면 optimizer·target objective·realization을 통제한 자료가 필요한데 이번 감사에는 없다.

## 8. Mode0 신호는 대부분 일괄 강도 조절이 아니다

저장된 source-mode0를 C_0라 하자. 개입은 Δ_N−C_0를 쓰는 것이다. Δ_alg와 C_0의 factor를 사용해 correction의 radial 성분을 계산했다.

\[
q=\frac{\langle C_0,\Delta_{alg}\rangle_F}{\|\Delta_{alg}\|_F^2}
=0.009949806.
\]

그러면

\[
-C_0=-q\Delta_{alg}+D_\perp,\quad
\langle D_\perp,\Delta_{alg}\rangle_F=0.
\]

|항목|추가 재계산|
|---|---:|
|보정 norm / native norm|약 9.975%|
|Native와 평행한 축소량 q|약 0.995%|
|보정 에너지 중 직교 성분|약 99.005%|
|Mode 제거 후 전체 write norm / native|약 99.501%|
|Mode가 native 평균 key에 만드는 반응 norm|3.83012|

따라서 이 결과를 “native 전체를 10% 덜 쓰니 좋아졌다”고 읽으면 틀리다. 총 norm은 약 0.5%만 줄었고 주된 변화는 방향이다. 이것은 방향 분해이며 q에 맞춘 scalar 후보를 실제로 비교한 causal control은 아니다.

해당 mode의 실제 신경망 관측은 제한적이지만 유용하다.

- Canonical RS/strict는 100→100.
- Selected N32는 16→20; 회복4/신규손실0.
- Selected reference32의 token mismatch는 46→36; 회복16/신규flip6.
- Current desired NLL 증가가 1e−4를 넘은 요청은 96/100.
- PS, 전체 N1000, 전체 reference512, history는 이 개입에서 미측정.

**즉 실제 성공을 유지하면서 어떤 hidden response를 덜 쓰는 여유는 관측됐지만, 이미 RS·PS·locality를 모두 보존하는 방향이 확보된 것은 아니다.** Mode99 제거는 canonical strict도 1개 잃었다. “최대 mode 하나 제거” 정책으로 일반화할 수 없다.

이 mode의 번호는 Server4의 ideal projected-key spectrum 기준이다. Server2 B-SVD mode 번호와 직접 대응하지 않는다.

## 9. Server2는 이 직관에 어떤 증거를 더하는가

### 9.1 강한 사실: editing preference와 preservation은 분리된다

동일 첫100을 B1→B100으로 추적하면 RS100→99, PS190→187/200, NS867→662/1000이다. N의 B1 성공 중 237/867=27.34%가 실패한다. 전체 at-write→B100 NS에서는 11,547/72,505=15.926%가 실패한다. Explicit prompt overwrite를 제외해도 큰 손실이 남는다.

이것은 native response와 history penalty를 그대로 유지하는 것이 실제 과거·주변 지식의 보존을 충분히 대리하지 못한다는 근거다. Single-layer 보존이 원리적으로 불가능하다는 증거는 아니다.

### 9.2 Weight norm과 preservation 비용은 다르다

저장 checkpoint 구간의 신규 weight 변화와 누적 weight 변화의 cosine 절댓값은 최대 약 .001577로 거의 직교한다. 하지만 고정 geometry512에서 누적 activation energy 평균은 B1 .258665→B100 8.486187이고, 11구간 평균 activation 교차항은 모두 양수다.

\[
E=W_a-W_0,\quad D=W_b-W_a,
\]
\[
\|EK+DK\|_F^2-\|EK\|_F^2
=2\langle EK,DK\rangle_F+\|DK\|_F^2.
\]

Weight 내적이 0에 가까워도 reference key에 작용시킨 내적은 양수일 수 있다. 따라서 W0 대비 Δ norm이나 누적 Frobenius norm 하나로 적절한 배율을 정할 수 없다.

또한 누적 energy 자체를 최소화하면 안 된다는 경고도 함께 있다. N64 panel에서 retained 사례를 포함해 모든 activation 교차항이 양수였다. Energy 증가와 N 손실은 동치가 아니다.

### 9.3 Key geometry가 “강도를 줄일 양”을 자동으로 주지는 않는다

관측 7개 native batch의 PK condition은 2.44–4.50으로 near-singularity 신호가 없다. 고정 key의 history-weighted response는 줄지만 key 자체는 고정이다.

B1→B91 write energy는 다음처럼 분해된다.

\[
\|\Delta_t\|_F^2=\|R_t\|_F^2g_{eff,t}^2,
\qquad 2.270\simeq1.688\times1.345.
\]

Demand 증가와 effective gain·target alignment 변화가 모두 관련된다. 다만 B1과 B91은 다른 요청이라 인과 분해는 아니다. B2의 K/R을 고정하고 M0→M1만 바꾼 counterfactual에서는 write energy가 3.09% 증가했지만 NS를 평가하지 않았다. 자신의 key가 history에 들어간 B1 control에서는 오히려 write가 줄고 target fit이 악화했다.

따라서 “history collision → 반드시 update 폭증 → scalar 축소가 해결”이라는 필연적 사슬은 성립하지 않는다. 이상적인 λ=1 regularized writer에는 singular gain≤.5 상한도 있다. 실제 관측 최댓값 .49793과 일관된다. 이것은 R·누적 norm·locality의 상한을 주지는 않는다.

### 9.4 Server2의 signed gradient는 scalar 최적 배율의 검증값이 아니다

N64에서 entry gradient의 finite-interval margin 예측과 실제 변화의 Spearman은 early .645, late .735였다. 그러나 전체 sign 적중49/64는 항상 감소를 예측한51/64보다 낮고, late retained의 MAE는 gradient2.657 대 변화0 예측1.191이다. Endpoint가 둘 다 성공하지만 중간 보간에서 실패하는 사례도 있다.

이 방향은 W10−W1 또는 W100−W50처럼 여러 batch를 합친 변화다. 현재 batch Δ_N의 배율을 조절한 시험이 아니며, 보간에서 그 구간에 들어온 모든 새 edit을 유지한 것도 아니다.

**Server2는 single-layer의 정확한 activation 계산을 downstream margin과 결합할 필요를 뒷받침한다. 그 margin이 비선형이므로 미분 한 번으로 큰 step의 안전성을 보증할 수 없다는 한계도 보여준다.**

원 분석: [Server2 checkpoint 독립 감사](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-20-sh2-checkpoint-mechanism-independent-review/review-ko.md).

## 10. .75 후 다시 최적화하는 방법의 의미

Single L4에서 같은 batch의 K, P, M 및 writer context를 고정하면 native map B가 고정된다. 첫 residual R1과 두 번째 residual R2를 쓰는 두 write는

\[
\Delta_1=R_1B^T,\qquad\Delta_2=R_2B^T,
\]
\[
\boxed{a\Delta_1+b\Delta_2=(aR_1+bR_2)B^T.}
\]

이는 실수 산술의 항등식이다. 같은 layer에 두 번 write해도 full residual-coordinate family {XB^T}의 input-side span을 넓히지는 않는다. 그러나 새로운 R2가 기존 R1과 다른 output 방향을 만들 수 있으므로, 단순 scalar나 기존 mode 계수만의 family와 동등하다는 뜻은 아니다. 실제 FP32에서 두 번 쓰는 rounding 차이도 별도다.

두 번째 target을 최초 Z1로 고정하고, native residual 기준 canonical key를 Kc, T=B^T Kc라 하면

\[
R_2=R_1-aR_1T,
\]
\[
\Delta_{total}=R_1[(a+b)I-abT]B^T.
\]

a=.75, b=1일 때 R1[1.75I−.75T]B^T다. 이는 단순 배율 .75나 1이 아니며, 반복 residual fitting으로 기존 regularization tradeoff를 바꾸는 효과다. Kc는 writer의 context 평균 K와 같다고 가정하지 않는다.

Fresh refit이면 Z2 자체가 달라진다. 과거 REFIT4의 두 번째 z는 1,000개 중 853개 zero-Adam, 146개24회, 1개21회였다. 이미 충분히 맞는 요청은 현재 anchor로 target을 재설정하고, 나머지에 최적화를 집중하는 효과와 일관된다. Target·teacher·clamp anchor·optimizer reset이 함께 바뀌므로 하나의 원인으로 분리되지는 않는다.

Warm suffix에서 N4→REFIT4는 PS1938→1950, NS7108→7175였지만 P strict1423→1405, 과거 active R/P에도 일부 손실이 있었다. 같은 첫 z를 반복 실현하는 FROZEN2는 PS1958로 높지만 NS7070으로 N4보다 낮았다. “한 번 더 write” 또는 “원 z를 더 정확히 맞춤”만으로 preservation을 설명할 수 없다.

부분 L4 뒤 L8에서는 zero-step이 다수여도 residual energy가 실제로는 최적화가 필요한 소수 요청에 집중됐다. Cold LD B1의 .75 이후 L8은 64개 zero-step/36개 active였고 active가 residual energy의 사실상 전부를 차지했다. 모든 target이 일찍 끝나서 의미 없는 write가 된 것은 아니다. 그러나 L4를 바꾸면 L8의 input key도 바뀌므로, 위 단일 L4의 고정 B 항등식을 다층 경로에 그대로 적용할 수 없다.

근거: [REFIT4 결과 리뷰](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-14-lowcost-seq10-review-ko.md), [REFIT 후속 기전 정리](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-16-refit-feedback-barrier-write-proposal.md), [L8 수렴 증거](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-17-sequential-local-z-allocation/convergence-evidence-ko.md).

## 11. Single-layer의 장점을 정량적으로 쓰는 방향

핵심 좌표는 layer별 weight가 아니라 **한 layer 안에서의 실제 target response**다. 가장 좁은 후보는 scalar α이고, 일반화하면 native component C_j의 계수 a_j다.

\[
W(a)=W_e+\sum_j a_j C_j.
\]

B=LΣV^T로 분해할 때 C_j=σ_j(Rv_j)ℓ_j^T로 두면 native는 모든 a_j=1이다. 이 좌표에서는 weight energy는 Σ_j a_j²σ_j²||Rv_j||²로 계산되지만, reference에서 C_jK_R 간 교차항은 일반적으로 0이 아니다. 비용을 mode별 norm만으로 더하면 reference 상호작용을 놓친다.

입력 token/prefix/mask를 고정하고 L4 down_proj만 바꾸면

\[
H_x(a)=H_{x,e}+\sum_j a_j C_jK_x,
\]
\[
H_x(a)-H_{x,0}=E K_x+\sum_j a_j C_jK_x.
\]

K_x와 L4 진입 이전 상태가 고정되어 이 response는 정확한 affine 함수다. Layernorm·attention 등을 포함한 downstream 함수의 파라미터도 고정되지만 그 함수가 선형이 되는 것은 아니다. Autoregressive free generation에서 token이 바뀌면 입력도 달라지므로, 정확성은 동일한 teacher-forced prefix에 대한 주장이다.

여기서 목표를 “native response를 같게 하라”에서 “필요한 실제 출력 품질을 유지하라”로 바꾸면 EN이 제외한 response 방향을 사용할 수 있다. 개념적으로는

\[
\min_a\ \Phi_R(W(a))
\quad\text{s.t. current와 유효 history의 실제 출력 보호 조건.}
\]

Reference는 전체 512개, base 최대256 token의 고정 답변을 사용한다. Φ_R를 선택 안정성으로 둘지 확률 약화까지 포함할지는 구별해야 하며, reference의 평균 KL 감소와 같은 결과라고 가정하면 안 된다. 이미 실패한 모든 문서의 개별 비악화를 강제하면 이번 scalar/DEC 정지 원인을 되풀이할 수 있다.

### Scalar 배율도 .75 대신 관측값으로 정할 수 있는 형태가 있다

Current의 실제 target margin을 c_j, native scalar 방향 derivative를 e_j라 하고 허용 floor를 τ_j로 두면

\[
c_j+(\alpha-1)e_j\ge\tau_j.
\]

축소에서 e_j>0인 항은

\[
\alpha\ge1-\frac{c_j-\tau_j}{e_j}
\]

라는 data-dependent 경계를 준다. Reference 선형 모델은

\[
\widehat\Phi(\alpha)=\frac1{512}\sum_i[-\mu_i-(\alpha-1)d_i]_+^2.
\]

활성 집합 S가 고정된 구간의 stationary 배율은

\[
\alpha^*=1-\frac{\sum_{i\in S}\mu_i d_i}{\sum_{i\in S}d_i^2}.
\]

활성 집합 전환점, 허용 구간, 경계 후보를 함께 고려하면 작은 piecewise quadratic 문제다. 다중 component는 동일한 coefficient-space 선형 제약 문제로 확장된다. 이것은 저장된 response와 현재 margin으로 강도를 정하는 방법의 수학적 형태이며, 이번 자료에서 실제 accepted 배율이나 성능으로 확인된 것은 아니다.

τ=0은 해당 token이 경쟁 token보다 뒤처지지 않는다는 해석이 있다. 확률 하락까지 제한하려면 별도의 해석 가능한 허용량이 필요하다. 수식이 그 선택을 없애 주지는 않는다. 동률·FP32 처리도 포함해야 한다. 실제 endpoint에서 전체 vocabulary와 전체 답변 위치를 재확인해야 하며, 단조성을 가정한 단순 이분법은 보장되지 않는다.

별도 paraphrase target set은 도입하지 않는다. 따라서 canonical 제약만으로 PS를 보장한다고 말할 수 없다. PS는 독립 observer로 남겨야 하고, 실제 PS 손실이 재현되면 이 제약이 충분하지 않은 것으로 판정해야 한다. History도 유효한 최신 target을 기준으로 관리해야 하며, base reference만 보호해서는 이전 edits를 지킨다는 결론을 낼 수 없다.

## 12. 계산량과 이전 실패까지 고려한 해석

이 방향의 계산상 가치는 z 최적화나 큰 writer solve를 각 후보마다 반복하지 않아도 된다는 것이다. K/P/M과 native R/B를 한 번 만들고, 여러 scalar 또는 component 후보의 L4 response를 작은 factor 곱으로 만들 수 있다. Full sequence의 C_jK를 후보마다 GPU에 따로 저장할 필요 없이 left factor와 projected keys로 표현할 수 있다.

512개의 문서별 coefficient Jacobian을 만들기 위해 512개의 full weight gradient를 각각 생성할 필요도 없다. 독립 sample을 처리하는 causal model의 한 microbatch에서 exposed margin 합을 backward하면 각 sample의 activation gradient가 분리되어 나온다. 그 gradient와 C_jK_i를 곧바로 내적하면 필요한 결과는 문서×coefficient의 작은 행렬이다. 이는 **microbatch당 backward 한 번**이라는 구조적 설명이며, 512개를 메모리에 한꺼번에 올리거나 전체 비용이 backward 한 번이 된다는 뜻은 아니다. 문서 내부의 여러 안전 token 제약까지 전부 미분하는 비용은 별도다.

Reference labels만 보호하는 경우 매번 base full-vocabulary teacher 분포를 읽을 필요는 없다. 그러나 현재 모델에서 전체 vocabulary의 경쟁 token을 찾는 forward는 여전히 필요하다. L4 뒤 suffix가 길다는 사실도 변하지 않는다. 따라서 “2step”만으로 계산량 문제가 해결되지는 않는다.

또한 **이 좌표 자체를 새로운 해결책으로 포장하면 안 된다.** SL-ZFlow는 이미 single-layer 고정 writer 좌표 {XB^T}와 actual full-token suffix feedback을 사용했다. 그 실행은 N4 대비 PS −2.95%p, NS −8.42%p였고 모든 batch에서 25 oracle를 사용했다. 고정 λ의 quadratic cost와 제한된 entry KL을 줄이는 것이 실제 문맥별 선택성 보존과 일치하지 않았다는 결과다.

따라서 가치가 있을 부분은 좌표를 다시 도입하는 데 있지 않다. **Exact hidden-response 고정이 실제 성공 유지보다 과도한 제약인 구간을 찾고, 그 기능적 여유를 reference·history의 실제 판단 비용으로 배분하는 선택 기준**에 있다. 이 기준이 unseen N과 PS까지 개선하는지는 아직 검증되지 않았다. 외부 선행연구와의 novelty 확정도 이번 artifact 분석의 범위를 넘는다.

원 반례: [SL-ZFlow 완료 리뷰](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-16-single-layer-zflow-seq1000-review-ko.md).

## 13. 지금 말할 수 있는 것과 남은 공백

|주장|판정|
|---|---|
|Native를 줄일 때 일부 preservation 지표가 좋아질 수 있다|실제 cold scalar 후보와 reference native gradient가 지지|
|EN이 작게 움직였으므로 더 큰 EN norm이면 해결된다|지지되지 않음; 더 큰 동일 방향 KL 악화, radial 방향은 QE 밖|
|Adam 때문에 closed form이 z를 과하게 실현했다|입증되지 않음; 해당 선형 target-fit 최적 배율은 오히려 1.141|
|Native response를 정확히 유지할 필요 없이 canonical 성공이 남는 여유가 있다|Scalar·mode 개입에서 관측|
|Scalar보다 component 선택이 유리한 기전일 수 있다|Mode correction의 약99%가 radial과 직교; 제한적 기능 개선 신호|
|Component 정책이 이미 PS·전체 N·reference·history를 개선했다|미확인; mode PS 미측정, reference 신규flip 존재|
|Server2가 single-layer capacity 고갈과 불가피한 locality 손실을 증명했다|아님; 누적 기능 손실은 강하지만 인과 사슬·불가능성은 미입증|
|Single-layer의 정확한 activation 계산으로 반복 z/solve 비용을 줄일 수 있다|구조적으로 가능; 새로운 end-to-end 속도·품질 측정은 없음|

현재 evidence가 지지하는 연구 질문은 **“어느 layer에 몇 % 쓸 것인가”보다 “이 L4 write의 어떤 반응이 실제 edit 성공에 필요하며, 나머지 반응을 어느 정도 바꾸면 reference와 history의 판단을 덜 훼손하는가”**다. EN의 작은 보정과 Server2의 누적 손상은 이 질문을 필요하게 만들고, scalar·mode·refit 결과는 서로 다른 제한적 신호를 제공한다. 아직 모두를 만족하는 최종 method가 확보됐다고 결론내리지는 않는다.
