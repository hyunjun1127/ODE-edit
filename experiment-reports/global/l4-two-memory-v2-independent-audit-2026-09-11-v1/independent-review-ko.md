# L4 two-memory v2 최종 결과 독립 점검

작성: 2026-09-11. 대상은 server1 `l4-two-memory-conflict-routing-2026-09-11-v2/final-v1`이다. 정정 전 BF8/Frozen 결과를 최종 과학 비교에 섞지 않았다. 원보고서 SHA256: `bf4d1ac802a9f3d0cd835482bd84823cd7f406876663e6ea096f872dfe17fbc5`.

## 1. 판단

**이번 v2는 Current 편집을 대체로 유지하면서 구조 목적식을 낮췄지만, Native 대비 Base/Past의 기능적 보존 개선에는 실패했다.** BF8은 OS가 만든 controller-bank 손상을 줄였고 정해 둔 10% 목표도 달성했지만, Native와의 격차는 남고 독립 BaseAudit에 대한 추가 이득은 거의 없다. 이 결과는 OS의 보호 proxy·표본·경로 anchor를 우선 점검하도록 한다.

동시에 저장 산출물의 집계와 수정된 solver는 확인됐다. 이번에 원시 결과를 다시 집계하고 저장 tensor를 CPU로 읽어 검사했으며 모델 forward/edit/GPU 실행은 0회다. 새 설계나 실험을 실행한 검토는 아니다.

이 보고서가 원보고서에 추가하는 진단은 다음과 같다.

- Base 보호 위치는 B100에서 실제로 130/130/131개이며, 대부분 마지막 prompt token 한 위치다. Prefix 전체의 변화는 구조 비용에 포함되지 않는다.
- P*의 허용 차원은 14,326개이고 data factor column은 B100에서 1,088/1,088/1,089개다. 최소 13,237개 이상의 입력 방향에는 data 비용이 없고 ridge만 남는다.
- BF8 최종 누적 보정의 H 비용 중 99.916~99.920%가 ridge 항이다. 현재 표본 key가 거의 관측하지 못하는 방향을 이용한다는 수치다.
- 37개 node에서 진척 제약 적용 후 Past/Base gradient의 H-inverse cosine은 -0.00703~0.00565다. 관측한 두 channel 사이의 강한 국소 반대 방향 충돌은 나타나지 않는다.
- Native의 Base KL까지 내려오려면 OS 대비 40.99%/70.30%/53.14% 감소가 필요했다. OS 대비 10% 감소라는 operating point와 Native 대비 보존 우위는 상당히 다른 질문이었다.

## 2. 점검 범위와 결과

|대상|범위|결과|
|---|---:|---|
|최종 공개 package|54개, 95,680,525 bytes|파일·manifest·rooted receipt 대조 일치|
|CSV|29개, 합계 485,176행|전수 파싱; 아래 주요 raw/파생 수치 재집계|
|그림|5개|모두 열어 축·범례·방향과 생성 코드를 확인|
|봉인 분석/실행 source|41개|manifest source hash 전부 일치; 주요 구현과 정정 경로 검토|
|원시 manifest|422개, 58,239,664,973 bytes|328 JSON·94 tensor 파일 전수 SHA/size 일치|
|기존 준비 자산|5개|3 entry prepared·Praw·C0 SHA/size 일치|
|Endpoint prompt 관측|100,620행|NLL 성공 규칙·margin·identity·TF strict·token count 재확인|
|Endpoint 집계|783행|평균·중앙값·p90·최댓값·전이·request-cluster bootstrap CI 재계산|
|기능적 손상|50,688 context, 51,196 token 관측|Past ψ 및 Base KL의 token/context 가중 집계 재계산|
|Base/독립 BaseAudit NS|74,240 prompt-state 행|원시 결과와 58개 집계, ENTRY/W0 문항 전이 확인|
|Current context response|56,520행, 123개 요약|원시 관측·성분 분해·context 가중 요약 확인|
|Generation|852개 저장 관측|literal-prefix 집계와 분모 확인; native 재사용 관측 포함|
|새 endpoint tensor|16개|FP32 shape/finite, arm 사이 history/ledger 동일성 확인|
|BF step tensor|37개|누적 Z, step/net norm, anchor를 통한 최종 FP32 endpoint 복원 일치|
|OS anchor|5개|WN+Zos의 FP32 materialization, WOS−We anchor 일치|
|Teacher tensor|25개, 3,840 context|packing·예측 위치·선택 target NLL·가중치 확인|
|두 channel QP|37개|별도 NumPy active-set 풀이와 일치; 음수 dual/slack 0|
|실행 비용|11 allocation|scoped counters 및 14,024 GPU-seconds 합계 확인|

대조에서 불일치를 찾지 못했다. Endpoint/손상/전이 집계의 최대 부동소수 절대 차이는 4.62e−14였다. 같은 prompt를 여러 state/reference에서 평가한 행을 고유 문항이나 독립 실험 반복으로 세지 않는다.

이번 tensor 검증은 저장된 L4 weight/history/teacher와 경로의 CPU 검증이다. 모델 전체의 forward 재평가, GPU continuation replay, 미선택 전체 parameter의 독립적인 byte 검사까지 수행한 것은 아니다. 구조 action 표의 모든 dense quadratic을 새로 계산한 것도 아니다. 해당 생성 코드를 검토하고, 입력 파일 hash와 관련 scalar/tensor identity 및 목적식 성분 합을 대조했다.

## 3. 설계는 실제로 무엇을 했는가

주요 v2 요구와 구현은 일치한다.

- N은 같은 We에서 얻은 BLUE-L4 native endpoint다. B100의 N·z는 기존 준비 자산을 재사용했고, B1/B7에서는 해당 subset으로 native joint write를 별도 계산했다. z 자체는 고정된 기존 값을 재사용했다.
- Routing의 Cp는 active Past128의 보호 위치 keys다. Raw M을 Cp에 섞지 않았고 native writer의 M은 보존했다.
- Base teacher는 We다. W0 분포는 별도 평가에만 사용한다. OS 수식에도 W0 복구 cross term이 없다.
- Native는 Praw, routing은 sym(Praw)의 eigenvalue>.5인 전체 P*를 사용한다. V256/Ub100 절단은 없다.
- Current CE와 개별 context Cresp가 구분된다. OS는 scalar 진척 equality와 soft response-change 비용을 사용한다.
- BF는 We에서 시작해 저장된 WOS−We 방향을 진행한다. BF1도 functional barrier를 쓰며, OS는 barrier가 없다.
- BF의 1/h action·slack scaling, 누적 Z anchor, 공통 calibration과 Frozen 방향 고정이 구현됐다. Current gradient는 방향 계산에 들어가지 않는다.

따라서 이번 음성 결과를 W0 복구 목표가 섞였거나, raw M이 실수로 routing에 들어갔거나, rank100 제한을 그대로 유지한 실험으로 설명할 근거는 없다.

## 4. Native → OS: 구조 보존은 크게 좋아지고 실제 보호는 악화

세 entry는 각각 기존 1,000/5,000/9,000 edit 이후의 We에서 다음 B100을 한 번 편집한 비교다. 이 방법 자체의 새로운 10,000-edit 순차 chain은 아니다.

|entry|Current RS N→OS|PS N→OS|NS N→OS|Past harm N→OS|Base KL N→OS|BaseAudit KL N→OS|
|---|---:|---:|---:|---:|---:|---:|
|Early|100→100|97.5→98.0|81.3→78.4|.030312→.036893|.118147→.200220|.041017→.104291|
|Middle|100→100|98.5→98.5|71.1→70.1|.017502→.036850|.059035→.198794|.036628→.133508|
|Late|100→100|96.5→96.5|62.6→61.1|.023383→.037836|.051820→.110583|.035367→.051475|

성공률 단위는 %다. Base KL은 KL(p_We||p_W), Past는 context NLL 악화에 ψ를 적용한 평균이다.

|entry|J_pres N→OS|Base mapping energy N→OS|Past mapping energy N→OS|native M action N→OS|
|---|---:|---:|---:|---:|
|Early|.154806→.093715|.053078→8.42e−10|.051576→2.73e−9|5.713→156.396|
|Middle|.339480→.248944|.071267→6.63e−10|.066796→2.22e−9|21.226→786.000|
|Late|.509686→.410679|.076427→6.75e−10|.073579→2.07e−9|46.168→1233.466|

**보호한 bank 안에서 mapping energy와 실제 KL/NLL이 이미 어긋난다.** 독립 표본으로 일반화되지 않는 문제만으로는 이 현상을 다 설명할 수 없다. OS가 대수적으로 제대로 풀렸다는 사실과 proxy가 기능적 보존을 대변한다는 주장은 구분해야 한다.

Raw M action은 약 27.37/37.03/26.72배가 됐다. Raw M 제거와 표본 active-Cp 선택의 범위 차이가 실제 update에 크게 반영됐다는 진단이다. M action은 functional forgetting 그 자체가 아니고 퇴역 fact의 기여도 남아 있으므로, 이 수치만으로 raw M을 다시 넣어야 한다고 결론내리지 않는다.

## 5. Single-layer에서도 token 간 cross effect는 남는다

Single L4 down-projection만 바꾸면 고정 입력 sequence에서 그 module의 input key는 고정된다. 그러나 같은 weight는 sequence의 모든 위치에 적용된다.

예측 위치 t의 최종 logit 변화는 일차적으로도 다음과 같은 경로를 가진다.

\[
\delta\mathrm{logits}_t\approx\sum_{u\le t}J_{t,u}\,Dk_u.
\]

다음 layer들의 causal attention이 앞선 token의 변경된 표현을 읽기 때문이다. 예측 위치 자체의 Dk_t를 작게 만드는 것은 앞선 위치들의 Dk_u까지 고정하지 않는다. 이 식은 구조를 설명하는 일차 근사이며 이번에 전체 Jacobian을 계산한 식은 아니다.

저장 teacher packing에서 확인한 실제 범위는 다음과 같다.

|B100 bank|Early|Middle|Late|
|---|---:|---:|---:|
|Base 보호 위치 수|130|130|131|
|Base 입력 sequence 전체 위치 수|1134|1118|1136|
|Base 중 target 한 token인 context|126/128|126/128|125/128|
|Past 보호 위치 수|258|258|258|
|Past 입력 sequence 전체 위치 수|3703|3588|3589|
|Past 중 target 한 token인 context|254/256|254/256|254/256|

즉 Base의 구조 proxy는 전체 입력 위치의 약 11.5~11.6%, Past는 약 7.0~7.2%를 사용한다. 이 비율은 **input 위치의 관측 범위**이며 지식 보존율이 아니다. Base의 최대8-token 제한에 잘려 나간 긴 target이 주된 현상도 아니다. 실제 target 대부분이 한 token이고 최댓값도 두 token이다.

현재 결과와 source는 prefix를 통한 손상 경로가 남는다는 설명과 정합적이다. 하지만 어떤 prefix 위치가 손상의 얼마를 발생시켰는지 분리한 activation intervention은 없다. 매우 작은 직접 mapping 오차의 downstream 증폭도 별도 측정하지 않았으므로 prefix 경로 하나로 전부 인과 귀속하지 않는다.

**λb를 키우는 것만으로 이 결함이 해결된다고 예상하기 어렵다.** 현재 λb=1에서도 보호 위치의 mapping energy는 이미 거의0이다. 보호하는 입력 위치/반응과 실제 출력의 관계를 먼저 확인해야 한다.

## 6. 전체 P*가 있어도 표본 지식의 관측 공간은 작다

P* rank는14,326, 입력 차원은14,336이다. B100의 결합 data factor column은1,088/1,088/1,089개다. Column 사이 중복을 세기 전에도 허용 공간 중 최소13,238/13,238/13,237개 방향에는 data factor가 반응하지 않는다. 이 방향의 비용은 약2.3e−6인 ridge로만 정해진다. 이 수는 실제 data rank를 완전히 추정한 값이 아니라 안전한 하한이다.

BF 최종 누적 보정 Z의 비용을

\[
\|Z\|_H^2=\{\|ZU\|_F^2+\lambda_r\|Z\|_F^2\}/a_A
\]

로 분해했다. U는 Current·Response·Past·Base factors를 모두 연결한 행렬이다.

|BF8|‖Z‖F|‖Z‖F / ‖WOS−We‖F|H 비용의 ridge 비중|
|---|---:|---:|---:|
|Early|.012163|.1473%|99.9159%|
|Middle|.009869|.1156%|99.9204%|
|Late|.009837|.1179%|99.9168%|

이 분해는 저장 action과 독립 재계산한 Z norm을 이용한다. 보정은 표본 key에 거의 반응하지 않는 방향을 이용하고, 전체 OS write에 비해 Frobenius norm도 작다. 해당 방향이 bank 밖의 지식까지 보존한다는 뜻은 아니다.

또한 37개 node의 허용 metric 안에서 Past/Base gradient cosine은 -0.00703~0.00565, 중앙값 .00105다. 관측한 두 gradient는 거의 직교한다. 두 보호 목표가 강하게 반대 방향이라 공동 보정이 막혔다는 설명은 이 국소 관측에서 지지되지 않는다. Current의 실제 loss gradient나 전체 과거 지식의 충돌을 측정한 것은 아니다.

## 7. Barrier는 작동했지만 Native 대비 목표와 거리가 있었다

|entry|Base KL N|OS|BF1|BF8|OS→BF8 감소|BF8 / N|
|---|---:|---:|---:|---:|---:|---:|
|Early|.118147|.200220|.185198|.173574|13.31%|1.469배|
|Middle|.059035|.198794|.182794|.175177|11.88%|2.967배|
|Late|.051820|.110583|.102074|.095833|13.34%|1.849배|

Past도 OS 대비 BF8에서10.71/10.32/10.26% 감소한다. 따라서 BF8의 두 controller-bank 목표는 실제로 달성됐다. 이를 관측한 동시에 Native보다 Past/Base harm이 여전히 크다는 사실도 유지해야 한다.

Base KL이 N과 같아지려면 OS에서40.99/70.30/53.14%를 줄여야 했다. OS 대비10%라는 진단 operating point에 정확히 도달하더라도 Native 대비 보호 우위는 되지 않는다. Budget이 더 큰 개선을 금지한다는 의미는 아니지만, 최소 보정 비용의 현재 문제에서 요구한 개선 정도는 그 격차보다 훨씬 작았다.

BF1이 실제10%에 못 미친 것은 elastic 설계와도 일치한다. 첫 step에서 channel 간 결합을 무시하면 h=1, q=q_ref, ε≈.1q이므로 ξ/e≈.1/(.5+.1)=1/6이다. 선형 예측에서도 요구 감소량 일부를 slack으로 남긴다. 여기에 nonlinear prediction error가 추가된다. λ·ε의 숫자가 크거나 작다는 것만으로 강도나 solver 실패를 판단하지 않는다.

수정된 BF8은 세 entry 모두 처음 네 node에서 보정0이다. s=.625부터 보정이 생긴다. 당시 위험이 예산 내부였고 Z도0이므로 정당한 무보정 결과다. 다만 현재 구현은 이 node들에서도 gradient를 계산한다. 불필요한 backward를 줄일 여지는 있지만, 비용 절감만으로 현재의 held-out 결과가 좋아졌다고 간주할 수는 없다.

## 8. Held-out 전이와 Frozen 비교

|entry|BaseAudit KL OS→BF8|OS 대비 감소율|BaseAudit NS N→OS→BF8|
|---|---:|---:|---:|
|Early|.104291→.103784|.486%|1022→1013→1013 /1280|
|Middle|.133508→.133513|−.004%|930→931→932 /1280|
|Late|.051475→.051332|.277%|905→898→898 /1280|

Controller Base128에서는 OS→BF8 KL이 Early128개, Middle127개, Late127개 request에서 낮아졌다. 변화가 단 한 문항에만 생긴 결과는 아니다. 다만 감소량은 불균등해서 양의 개선량 중 상위5개 비중이38.7/57.9/56.2%다. 독립 BaseAudit에서는 개선량이 훨씬 작다.

Middle은 KL이 커졌어도 NS가 N보다2개 높다. Native 대비25개를 잃고27개를 얻은 결과다. Early는20개 lost/11개 gained, Late는17/10이다. **We KL의 증가를 모든 정답 지표의 악화와 동일시하거나 Native가 모든 지표에서 Pareto 지배한다고 쓰면 안 된다.** 다만 광범위한 보호 우위는 관측되지 않았다.

Middle Frozen-BF8→BF8:

- Past harm .03308555→.03304748, 약 .115% 감소.
- Base KL .1768535→.1751769, 약 .948% 감소.
- BaseAudit KL .1335018→.1335131, 소폭 증가.
- Current RS/PS/NS와 Fixed/Past 성공 수는 같다.

실제 방향 refresh가 controller-bank에 미세한 추가 효과는 냈지만, 이 한 entry에서 1,536번 추가 backward를 정당화할 held-out 이득은 입증되지 않았다. BF8과 BF1의 차이는 경로·step 수 전체의 차이이며, 방향 refresh의 별도 효과는 같은 h를 쓴 Frozen 비교로 읽는 것이 맞다.

Base/BaseAudit의 fact 및 canonical prompt overlap은0이다. 그러나 두 집합의 rewrite·rephrase·neighborhood 전체 prompt inventory에는 Early90/Middle75/Late40개의 정확한 문자열 중복이 있다. Candidate-disjoint/주 fact-disjoint가 모든 평가 문자열까지 독립이라는 뜻은 아니다. 이것이 이번 음성 결과를 뒤집는 증거는 아니며, 분모와 독립성의 범위로 기록한다.

## 9. Current가 유지됐다는 표현의 범위

Current rewrite 선호 성공은 모든 B100 arm에서100/100이다. Middle은 TF strict도 N99/100→OS100/100으로 개선됐다. 평균 NLL 개선도 실제 관측이다.

하지만 OS에서 N보다 rewrite NLL이 높아진 요청은 Early60/Middle64/Late64개, rephrase는111/126/126개다. Mean은 낮아지고 median delta는 양수이므로 일부 큰 실패의 개선과 다수의 작은 악화가 함께 있다. Scalar equality는 개별 요청의 NLL 비악화를 보장하지 않는다.

Middle Fixed100에서는 N100→OS99개의 RS 성공으로 한 요청을 잃었다. Current NS는 세 entry 모두 N보다1.0~2.9pp 낮다. 이들을 Current/Past 보존 완전 성공이라는 표현으로 덮지 않는다.

BF8 중간의 s=.25/.5 성공률을 N endpoint와 비교해 낮다는 이유만으로 편집 소실이라고 부르지 않는다. 당시 목표 write 자체가25%/50%만 진행된 상태다. 같은 s의 nominal OS 경로와 비교한 값이 있어야 보정 자체의 영향을 분리할 수 있다. 최종에서는 동일 write 진행량을 비교한다.

## 10. Batch size와 capacity

|Middle|N Base KL|OS Base KL|BF1 Base KL|OS−N|
|---|---:|---:|---:|---:|
|B1|.00030963|.00201406|.00185242|+.00170443|
|B7|.00931888|.02603034|.02420361|+.01671146|
|B100|.05903530|.19879436|.18279421|+.13975906|

B1/B7의 실행·분모·native solve는 확인됐다. 하지만 B별 요청과 bank 구성이 달라 위 표가 batch size만의 인과 효과를 의미하지 않는다. B1의 큰 상대 배율에는 작은 기준값도 있으므로 절대 차이를 함께 읽는다.

B1 OS는 native 대비 current context-response deviation이 약9.40e−14로 거의 같아도 Base KL이 커졌다. B100에서 Current 제약이 많아져서 생긴 capacity 문제 하나로 이번 현상을 설명하기는 어렵다. 적어도 적은 요청에서도 구조 proxy와 output의 차이가 존재한다.

반대로 허용 P* 차원이 크다는 이유로 공동 보존 capacity가 충분하다고 결론내리지 않는다. 표본 keys를 피할 수 있는 공간과 모든 주변 지식을 지키며 편집할 수 있는 공간은 다르다. 이번 결과는 input-space routing의 대수적 여력을 보여 주며, 실제 knowledge-preserving 방향의 존재/부재는 아직 판별하지 못했다.

## 11. 계산량과 기술 정정

전체 실제 allocation은14,024 GPU-seconds, 3.8956 GPUh다. 유효 경로의 functional backward는5,568회, 정정 전 제외 경로까지 포함한 기록 합은10,176회다. 취소된 두 관측 job의 중단 후 미기록 호출 수는 추정하지 않는다. GPU 상주 시간은 포함돼 있다.

BF8은 BF1과 공통으로 사용하는 OS calibration 외에 entry당1,536번 backward를 추가한다. Past256 context와 Base128 context를 microbatch2로 처리하므로 한 gradient stage당192 backward다. 논리적 stage8개를 backward8회라고 세면 크게 과소평가한다.

Full-space static factor/solve 구간은 B100당 약10초, 새 teacher/key capture는 약24초다. 나머지 capture·native 준비·평가·model load는 별도다. B100의 cold z/native 비용은 재사용했고 B1/B7도 z cache를 사용했으므로 이 campaign 시간으로 전체 one-shot 편집의 cold runtime을 주장할 수 없다. Peak GPU allocation은 B100 약41.0GiB다.

Node별 H inverse 단독 시간과 global model.forward hook 전체 계수는 없다고 원보고서가 명시한다. 따라서 정확한 per-arm 온라인 runtime이나 FLOPs를 복원했다고 주장하지 않는다. 작은2×2 QP 시간은 전체 observer/full-space 비용을 대표하지 않는다.

초기 문제는 Gram 행렬 크기에 비례한 residual tolerance를 dual 비음수 판정에도 쓴 것이었다. Dual은 약1e−11 수준인데 tolerance는 약1e−4 수준인 경우가 있어 음수 dual/slack을 허용했다. 정정은 수치 domain 위반을 고친 것이며, 나쁜 성능을 이유로 valid 경로를 골라 재실행한 근거는 없다. 교체 대상4개와 보존된 N/OS/BF1, canonical37 node가 일치한다.

**후속 구현에 옮길 때 남는 실제 주의점:** 일반 [runtime.py](/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-l4-two-memory-v2-final-integration/project/run_scripts/l4_two_memory_conflict_routing/runtime.py:12)는 아직 원래 [controller.py](/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-l4-two-memory-v2-final-integration/project/run_scripts/l4_two_memory_conflict_routing/controller.py:12)의 step을 import한다. 수정 solver는 [controller_repair.py](/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-l4-two-memory-v2-final-integration/project/run_scripts/l4_two_memory_conflict_routing/controller_repair.py:13)와 repair_runtime.py에 분리돼 있다. 최종 결과는 올바른 수정 경로지만, 일반 runtime만 복사해 새 실험을 만들면 알려진 domain 문제가 다시 들어갈 수 있다. 이번 리뷰에서 원 source를 변경하지 않았다.

## 12. 다음 논의의 우선순위

1. **보호 위치의 causal 범위를 먼저 확인한다.** 동일 bank에서 마지막 예측 위치 보존과 prefix 전체의 변화가 어떻게 연결되는지 확인해야 한다. 현재 mapping energy가 거의0이므로 단순 λb 확대가 첫 진단은 아니다.
2. **Native에 더 가까운 출발점에서 functional 보정의 가치를 분리한다.** N→OS가 손상 증가의 시작점이다. N을 anchor로 둔 한 번의 functional 보정과 현재 OS-anchor 보정을 같은 observer/Current 지표로 비교하는 것이 후보가 된다. 이 리뷰에서 실행하지 않았다.
3. **Native M과 표본 active-Past의 범위 차이를 분리한다.** Active bank의 정확성과 전체 과거 보호 coverage는 다른 장점이다. 퇴역 fact를 다시 보호하는 raw M 복귀를 자동 해법으로 삼지 않는다.

이들은 새로운 hard gate 제안이 아니다. We 기준 보존, 고정 z, Current 실제 지표와 N 대비 BaseAudit 전이를 유지하면서 원인을 좁히는 우선순위다. 현재 증거만으로 step 수·bank 크기·β·λ를 넓게 sweep하거나, single-layer capacity 부족을 확정할 이유는 없다.

## 13. 추가 산출물과 재현 위치

- [전체 독립 감사 receipt](/mnt/raid5/janghj/ODE-edit/experiment-reports/global/l4-two-memory-v2-independent-audit-2026-09-11-v1/audit-receipt.json)
- [원 package54개 목록](/mnt/raid5/janghj/ODE-edit/experiment-reports/global/l4-two-memory-v2-independent-audit-2026-09-11-v1/source-package-inventory.csv)
- [Native 대비 harm과 필요한 감소량](/mnt/raid5/janghj/ODE-edit/experiment-reports/global/l4-two-memory-v2-independent-audit-2026-09-11-v1/native-relative-harm-comparison.csv)
- [보호 token 위치와 전체 입력 길이](/mnt/raid5/janghj/ODE-edit/experiment-reports/global/l4-two-memory-v2-independent-audit-2026-09-11-v1/protection-position-coverage.csv)
- [BF metric 비용 분해](/mnt/raid5/janghj/ODE-edit/experiment-reports/global/l4-two-memory-v2-independent-audit-2026-09-11-v1/barrier-metric-decomposition.csv)
- [독립 QP replay와 gradient cosine](/mnt/raid5/janghj/ODE-edit/experiment-reports/global/l4-two-memory-v2-independent-audit-2026-09-11-v1/independent-joint-replay.csv)
- [Bank 개선의 request별 집중도](/mnt/raid5/janghj/ODE-edit/experiment-reports/global/l4-two-memory-v2-independent-audit-2026-09-11-v1/bank-change-concentration.csv)
- [Base/독립 BaseAudit의 문항 전이](/mnt/raid5/janghj/ODE-edit/experiment-reports/global/l4-two-memory-v2-independent-audit-2026-09-11-v1/base-locality-transitions.csv)
- [Active bank 선택·중복 확인](/mnt/raid5/janghj/ODE-edit/experiment-reports/global/l4-two-memory-v2-independent-audit-2026-09-11-v1/bank-selection-audit.csv)

원보고서는 final-integration worktree에 있고, 상대 raw 경로의 실제 기준은 `/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-l4-two-memory-routing-v2`다. Final-integration 폴더를 cwd로 삼으면 원시 상대 경로가 존재하지 않는다. 감사 코드와 원 package/source 복사본은 `/mnt/raid5/janghj/ODE-edit/local/reviews/l4-two-memory-v2-independent-audit-2026-09-11-v1/`에 보존했다. 원 모델·실험 파일과 원보고서는 변경하지 않았다.
