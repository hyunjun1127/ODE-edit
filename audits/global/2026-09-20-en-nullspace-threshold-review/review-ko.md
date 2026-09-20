# EN의 approximate edit-null space 검토

현재 판단: margin을 새로운 최적화 제약으로 도입하는 대신, EN의 **exact edit-response 보존을 spectral approximation으로 완화**하는 것은 타당한 다음 분석 방향이다. 다만 작은 singular direction을 풀어주는 수치적 수정과, 실제 편집 반응을 바꿀 수 있도록 허용하는 방법적 완화는 구별해야 한다. 이번 저장 factor 재계산은 두 변화가 상당히 다름을 보여준다.

이 문서는 방법 도입 가능성에 대한 분석이다. 모델 forward/backward, GPU 실험, runtime 수정, checkpoint 변경은 수행하지 않았다. 원래 FWS draft의 margin controller를 채택한 결과로 해석하지 않는다.

## 1. AlphaEdit의 threshold는 무엇인가

공식 저장소 commit `b84624f44dfe8fc6cd9e41df916c44124a0c46dc`를 확인했다.

- `experiments/evaluate.py:get_project`: `get_cov`의 SVD를 구하고, covariance singular value가 `nullspace_threshold`보다 작은 방향으로 projector를 만든다.
- `hparams/AlphaEdit/Llama3-8B.json`: `nullspace_threshold=0.02`.
- `AlphaEdit/AlphaEdit_main.py:get_cov`와 `util/runningstats.py:SecondMoment.moment`: 입력 key의 **평균 non-centered second moment**를 사용한다. 중심화 covariance도 아니고 raw key matrix의 singular value에 0.02를 적용하는 것도 아니다.

[공식 projector 코드](https://github.com/jianghoucheng/AlphaEdit/blob/b84624f44dfe8fc6cd9e41df916c44124a0c46dc/experiments/evaluate.py), [공식 Llama3 설정](https://github.com/jianghoucheng/AlphaEdit/blob/b84624f44dfe8fc6cd9e41df916c44124a0c46dc/hparams/AlphaEdit/Llama3-8B.json), [평균 second moment 구현](https://github.com/jianghoucheng/AlphaEdit/blob/b84624f44dfe8fc6cd9e41df916c44124a0c46dc/util/runningstats.py).

AlphaEdit 논문의 정확한 null-space 대수는 eigenvalue가 0인 경우다. 실제 구현처럼 작은 양의 eigenvalue까지 포함하면 보존은 근사적이다. [논문 Appendix B](https://arxiv.org/html/2410.02355v4#A2).

EN에는 두 종류의 공간이 있다.

|공간|보호 대상|현재 처리|
|---|---|---|
|Native AlphaEdit의 허용 공간 P*|pretraining reference 통계|이미 threshold를 사용한 native projector의 허용 range|
|EN의 추가 edit-null 공간 Q_E|이번 batch의 편집 관련 full-token key|FP64 수치 rank 기준으로 모든 비영 방향을 차단|

지금 검토할 곳은 두 번째다. 첫 번째 threshold만 바꾸면 native baseline과 reference preservation geometry까지 동시에 바뀐다. `P* Q`를 단순 곱하는 방식도 일반적으로 올바른 교집합 projector가 아니다. 기존 EN처럼 P*의 직교 basis 안에서 edit-key SVD를 계산해야 한다.

## 2. 원 EN geometry를 재계산한 사실

대상은 Server4의 동일 cold B1, `single-layer-mechanism-first/20260919-v1/PROGRAM/b1-fd-waiver-r4/output/B1`이다. 이전 execution-reuse 보고서의 spectrum과 동일한 값을 사용하는지 확인했고, 아래 분석에는 해당 B1에서 직접 회수한 `space.json`을 사용했다.

- input dimension: 14,336.
- P* allowed dimension: 14,326.
- captured current-edit key: 4,596개.
- blocked numerical rank: 4,596.
- exact EN correction dimension: 9,730.
- numerical singular threshold: `1.0526021514e-10`.
- largest / smallest singular value: `33.0901763 / 2.06985534e-8`.
- 가장 큰 인접 spectrum gap: `0.0946616199 / 4.01088906e-6 = 23601.16`.
- 그 gap 아래의 방향 수: **281**.

추가로 key provenance와 실제 key tensor를 대조했다.

- distinct saved token-prefix hash: **4,315개**.
- 동일 prefix인데 FP32 bytes가 달라 별도로 저장된 추가 column: **281개**.
- 이런 차이가 있는 prefix group: 163개. 동일 prefix에서 position이 다른 group은 0개.
- 동일 prefix 대표 key와 변형 key의 최대 차이 norm: `6.08150764e-6`.
- 최대 상대 차이: `2.15372334e-6`.
- 변형 column들을 대표 key로 치환할 때 전체 차이 Frobenius norm: `6.02239828e-5`.

즉, 극소 singular tail의 차원 수가 동일-prefix 중복 변형 수와 정확히 일치한다. **동일 입력의 수치적 차이까지 독립적인 exact 보호 조건으로 승격했을 가능성이 강하다.** 다만 이번 작업에서 동일 mask·position-id·kernel 조건의 model replay를 다시 실행하지 않았으므로 수치 차이의 원인을 특정 kernel로 단정하지 않는다. 실제 다른 입력의 작은 semantic 방향이라고 단정해서도 안 된다.

원 exact EN은 저장된 FP32 key 행렬 자체를 정확히 보호한다는 계약을 충족한다. 그 계약이 원래 함수의 의미적 동일성보다 엄격할 수 있다는 것이 이번 발견이다. 과거 exact-space 구현 오류로 소급해서 부르지는 않는다.

## 3. 공간이 늘어난다는 것과 필요한 보정이 열리는 것은 다르다

저장된 SVD left vector의 순서를 실제 K에 작용시켜 검증했다. Native factor `R B^T`와 앞서 selected-neighbor 개선 신호를 보였던 `source_mode_0`를 각 완화 공간으로 직교 투영했다.

아래 c는 **raw reduced key `V^T K_E`의 singular cutoff**다. AlphaEdit의 mean-covariance threshold와 구별한다. `Q_c`는 sigma<c 방향과 기존 exact null directions를 허용한다.

|c|추가 허용 방향|전체 correction 차원|native 방향 투영 norm / native norm|source-mode0 투영 norm / 원 norm|
|---:|---:|---:|---:|---:|
|기존 numerical cutoff|0|9,730|0.000678%|0.001318%|
|1e-5|281|10,011|0.000689%|0.001341%|
|0.02|281|10,011|0.000689%|0.001341%|
|sqrt(0.02)=0.141421|322|10,052|0.363771%|0.122539%|
|0.5|1,632|11,362|3.524254%|1.653754%|
|1.0|2,863|12,593|6.225458%|3.190262%|
|2.0|3,724|13,454|8.851058%|5.368085%|
|sqrt(4596×0.02)=9.587492|4,542|14,272|96.884387%|38.145971%|

마지막 행은 `K_E K_E^T / 4596`의 eigenvalue에 AlphaEdit의 0.02를 숫자 그대로 대입했을 때다. **차단 방향이 4,596개에서 54개만 남는다.** 원 key에 0.02를 적용했을 때와 전혀 다른 방법이다. 또한 이 matrix는 dedup된 captured column의 평균이며, 실제 10,416개 token occurrence의 가중 평균이나 문서별 평균도 아니다.

핵심 해석:

1. 281개 near-zero 방향을 풀어주는 것은 데이터 spectrum에 근거한 수치적 정리다. cutoff를 1e-5부터 0.02까지 바꿔도 같은 공간이다.
2. 그 공간만으로는 이미 유망했던 source-mode0 변경을 거의 구현하지 못한다. native scalar shrink 방향도 거의 들어오지 않는다.
3. 그렇다고 새 reference-gradient 방향의 이득이 없다고 증명한 것은 아니다. 표는 **두 저장 방향의 표현 가능성**을 측정한 것이며, 임의의 새로운 보정 방향 전체를 평가한 것이 아니다.
4. native row는 저장 factor 재구성 `R B^T` 기준이다. actual native 대비 재구성 상대오차가 약 `6.07e-6`였으므로 exact/near-zero row의 극소 native 투영을 의미 있는 native 자유도로 해석하지 않는다.
5. source-mode0의 이전 결과는 canonical rewrite 100/100 유지 및 선정된 N32에서 4개 회복이었다. full PS, full N, full R512 및 history 개선의 증거는 아니다. 이번 투영된 방향에 대한 모델 품질도 측정하지 않았다.

## 4. Margin 없이 정의할 수 있는 approximate EN

Native endpoint를

\[
W_N=W_{entry}+\Delta_N
\]

로 두고, EN과 같이 reference KL을 줄이는 보정 D를 만든다. Local-z 및 native solve는 한 번만 수행한다.

P*=VV^T, V^T V=I이며, current protected key의 정규화된 reduced matrix와 covariance를

\[
X=V^T K_E/\sqrt{n_E},\qquad
C_E=K_EK_E^T/n_E
\]

로 정의한다. `X X^T=U Lambda U^T`라 하면

\[
Q_\tau
=V U_{\lambda\le\tau}U_{\lambda\le\tau}^T V^T.
\]

여기에는 X의 구조적 zero eigen-directions도 포함한다. 큰 d×d Q를 만들 필요 없이 기존 EN처럼 차단 basis를 빼는 형태로 적용할 수 있다.

방법의 최소 형태는

\[
\min_D\mathcal L_R(W_N+D)
\quad\mathrm{s.t.}\quad D=DQ_\tau,\quad\|D\|_F\le\rho.
\]

L_R은 기존 EN과 동일한 512개 reference의 base-teacher KL이다. Base 생성 길이 최대256/EOS, 고정 teacher-forced prefix를 그대로 사용한다. 별도 paraphrase target set을 도입하지 않는다. 이 형태에는 output-margin 제약이 없다.

### 정확하게 보장하는 것

Single L4 down-projection만 변경하고 입력 token/mask/position을 고정하면 해당 layer key는 고정이다. 따라서 **finite D에 대해서도** L4 output의 변화는 정확히 `D K_E`이며,

\[
\boxed{
\frac1{n_E}\|D K_E\|_F^2
=\operatorname{tr}(D C_E D^T)
\le\tau\|D\|_F^2
\le\tau\rho^2.
}
\]

증명: D의 right support가 고유값≤tau인 공간에 있으므로 해당 공간에 제한한 C_E의 최대 고유값이 tau 이하이다. 각 output row의 quadratic form을 더하면 된다. 비선형 downstream의 Taylor 근사를 사용하지 않는 activation-level 보장이다.

따라서 threshold와 correction norm을 함께 명시해야 한다. 작은 tau만 사용하고 D의 norm을 무제한 허용하면 edit 반응 변화는 작게 제한되지 않는다.

이 부등식은 RS/PS 보존 theorem이 아니다. downstream 함수는 고정이지만 비선형이고 민감도도 입력마다 다르다. 평균 activation 변화가 작아도 특정 요청이나 unseen paraphrase의 답변이 바뀔 수 있다. 별도의 downstream bound 없이 이를 성공률 보존으로 확대할 수 없다.

### Gradient와 연결

Native에서 한 번 얻은 `G=grad L_R(W_N)`에 대해

\[
D_{\tau,\eta}=-\eta GQ_\tau
\]

라 두면

\[
\left.\frac{d}{d\eta}\mathcal L_R(W_N-\eta GQ_\tau)\right|_{\eta=0}
=-\|GQ_\tau\|_F^2.
\]

공간 확대의 유효성은 dimension보다 다음 두 양으로 확인하는 것이 맞다.

\[
\underbrace{\|GQ_\tau\|_F^2-\|GQ_0\|_F^2}_{추가로\ 확보한\ reference\ descent},
\qquad
\underbrace{\frac1{n_E}\|(GQ_\tau)K_E\|_F^2}_{단위\ step의\ 편집\ 반응\ 변화}.
\]

Q_0는 exact EN projector다. 두 번째 양은 고정 key에서 정확히 계산할 수 있고, 첫 번째는 일차 목적함수 감소량이다. 추가 gradient가 열려도 finite step의 KL 감소 및 unseen locality 개선은 별도 확인해야 한다.

**현재 저장물에는 EN-KL의 raw gradient가 남아 있지 않아 이 두 양의 threshold 곡선은 재구성하지 못했다.** 저장된 512개 activation-gradient factor는 DEC의 exposed decision gradient용이며 EN-KL gradient가 아니다. 이를 대신 사용해 EN 개선을 주장하지 않았다.

### Threshold가 heuristic을 없애는가

완전히 없애지는 않는다. Margin budget을 activation-response budget으로 옮긴 것이다. 다만 보호할 행렬, 방향, leakage 상한을 명확하게 정의하고 실제 효과를 저렴하게 계측할 수 있다.

- spectrum의 23,601배 gap을 이용한 281개 numerical-tail 해제는 직접적인 데이터 근거가 있다. 그러나 같은 gap이 모든 batch에 존재한다고 가정하지 않는다.
- 그 이상을 허용하는 tau와 rho는 실제 method hyperparameter다. AlphaEdit가 0.02를 사용했다는 사실만으로 ours의 정당한 값이 되지 않는다.
- 수치 오차 cutoff와 의미적 편집 반응 허용량을 별도 개념으로 관리해야 한다.
- `n_E`는 4,596 unique byte-columns, 4,315 unique prefix, 10,416 occurrences 중 무엇을 쓰는지 명시해야 한다. exact span은 반복 가중치에 불변이지만 approximate covariance는 가중치에 민감하다. 방법용 covariance라면 요청별/토큰별 가중 평균을 먼저 정의해야 한다. 위 empirical 표는 원 saved matrix를 그대로 분석한 값이다.

Hard cutoff가 불안정하다면 같은 목적을 continuous filter로도 표현할 수 있다.

\[
\min_{D=DP^*}
\langle G,D\rangle+
\frac1{2\eta}\|D\|_F^2+
\frac\beta2\operatorname{tr}(D C_E D^T),
\]

\[
D^*=-\eta G V(I+\eta\beta V^T C_EV)^{-1}V^T.
\]

이는 표준 quadratic penalty의 해이며 새 이론이라고 주장하지 않는다. beta→infinity에서 exact EN, beta=0에서 P*만 사용하는 KL 보정으로 이어진다. 첫 원인 분리에서는 hard-threshold EN부터 확인하고, continuous filter를 동시에 바꾸지 않는 편이 해석하기 쉽다.

## 5. 이전 실패 원인과의 관계

S4 B1 EN의 수용 보정은 native norm의 0.1116%였고 reference KL은 5.55% 감소했다. 그보다 큰 scale에서는 **reference KL 자체가 악화되어** 거절됐다. 따라서 exact null-space만 풀면 보정이 커지고 locality가 좋아진다는 결론은 아직 나오지 않는다.

또한 현재 EN은 approximate 공간을 넣기만 하면 안 된다. `DK_E=0`, native logit 동일성, current NLL 변화1e-4 등의 기존 검사는 exact preservation 확인용이다. 그것을 근사 보호 arm에 그대로 유지하면 새로 허용한 변화를 다시 거절하게 된다. projector 계산 자체의 numerical correctness 검증은 유지하되, 보호 계약은 위 activation-response bound로 바뀌었음을 명시해야 한다. RS·PS는 별도로 사후 평가하며, PS/N을 threshold 선택에 쓰지 않는다.

DEC에서 확인한 reference별 무손실 제약으로 cone이 0이 되는 현상은 다른 문제다. 그 제약을 EN에 가져오면 threshold를 바꾸어도 다시 막힐 수 있다.

EN의 평균 KL과 neighborhood 사이의 목적함수 불일치도 남아 있다. 다음 결과가 나오면 해석은 구분해야 한다.

- extra reference-gradient descent가 거의 없다: 해당 threshold는 공간을 늘려도 원 EN을 실질적으로 바꾸지 못한다.
- extra descent 및 KL 개선은 크지만 NS 개선이 없다: geometry 외에 preservation objective/coverage 문제가 남아 있다.
- KL/NS가 개선되고 RS/PS가 유지된다: approximate edit-response 보존에 유효한 trade-off가 있다는 직접 근거다.
- RS/PS가 함께 하락한다: edit-response budget이 기능적 편집 유지에 충분하지 않거나 지나치게 느슨하다.

History가 있는 경우 current-key 공간만으로 과거 edit을 보호할 수 없다. GSS로 관리하는 유효 history key/target을 별도 포함해야 한다. stale overwrite target은 제거하고, current/history의 covariance 가중 및 response 변화는 각각 보고해야 한다. 이미 native에서 손상된 history를 native-response에 고정하는 것은 복원이 아니다. B1은 history가 없으므로 이번 계산은 이에 대한 근거를 제공하지 않는다.

## 6. 계산량과 현재 권고

기존 SVD를 재사용하면 여러 threshold는 blocked basis의 열을 다르게 선택하는 행렬 연산이다. 각 threshold마다 local-z, 모델 backward, SVD를 반복할 이유가 없다. Single-layer fixed key를 사용하면 후보 D의 current/reference/history activation 작용도 cache에서 계산할 수 있다. 다만 512×최대256 reference의 nonlinear suffix KL/gradient 비용은 threshold 도입만으로 사라지지 않는다.

현 단계에서는 **EN의 reference 목적함수를 유지하고 approximate edit-null 공간만 변화시키는 것**이 margin controller보다 작은 변경으로 핵심 가설을 확인할 수 있다. 첫 확인은 numerical tail만 해제한 경우와 실제 low-energy 방향까지 해제한 경우를 구분해야 한다. 보정 크기의 임의 확대나 AlphaEdit의 0.02 복사는 근거가 부족하다.

Threshold 자체는 AlphaEdit를 비롯한 기존 subspace 방법의 도구이며 독립적인 novelty로 보기 어렵다. 연구 기여 후보는 single-layer 고정 함수에서 편집 반응의 변화량을 정확히 계측하면서, exact response 고정이 locality 회복을 제한하는지 입증하는 부분이다. 이를 유효한 방법 기여로 주장하려면 실제 gradient overlap, RS·PS 유지, locality 개선 및 추가 계산량을 함께 확보해야 한다.

## 7. 재현 산출물과 검증 범위

- [spectral-audit.json](spectral-audit.json): spectrum, 동일-prefix 변형 수, source-mode/native projection, 입력 SHA.
- [spectral_audit.py](spectral_audit.py): remote CPU saved-factor 분석. 원 run은 읽기 전용.
- [math-checks.json](math-checks.json): 7개 작은 CPU 대수 검증.
- [check_math.py](check_math.py): projector, leakage bound, gradient 감소량, continuous filter 정상식 확인.
- [source/alphaedit-source.json](source/alphaedit-source.json): 공식 코드 commit 및 파일 SHA.
- [source/space.json](source/space.json), [source/en-geometry.py](source/en-geometry.py): frozen B1 geometry 및 구현.

Remote CPU 분석은 4 threads, 약5.22초였다. CUDA 미초기화를 확인했다. blocked/keys/key-provenance/native/writer/mode0/space의 SHA는 기존 B1 inventory와 일치했고 P* basis는 saved descriptor와 일치했다. 이는 저장 대수의 검증이며 새로운 threshold arm의 모델 품질 검증이 아니다.
