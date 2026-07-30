# MV-1 finite-action identifiability post-pilot red audit

- 작성일: 2026-07-30
- 대상:
  - `mv1_llama_c0p_v1`
  - `mv1_qwen_c0p_v1`
- 감사 성격: v1 post-pilot estimand/identifiability red
- 판정: **REVISE-before-full**
- 실행 허용: v2 `D0 calibration[3:8]`, model별 5 case 동시 pair만
- 실행 금지: 기존 v1 full C0, confirmatory, MV-2

## 1. 결론

v1 3-case pair는 실행·artifact·rollback fidelity에는 PASS했지만, 기존
six-arm action set은 continuous routing opportunity를 식별하지 못한다.
Five unit layer directions의 slope를 `s_l`이라 하면 uniform score는
`sum_l s_l/sqrt(5)`다. 여러 layer slope가 모두 양수이면 event별 크기
차이가 있어도 uniform이 각 single-layer arm보다 구조적으로 강할 수 있다.
따라서 양 model에서 `uniform`이 six-arm retrospective oracle까지 포화한
사실을 “routing heterogeneity가 없다” 또는 “continuous routing ceiling이
0이다”로 번역할 수 없다.

Full C0를 기존 설계 그대로 실행하면 더 많은 case에서 같은
non-identifiability를 반복할 위험이 있다. v1 full C0는 제출하지 않고,
`relu(s)/L2`로 만든 non-negative unit-`C` adaptive mixture와 uniform을
직접 비교하는 v2 D0로 교정한다.

## 2. 네 범주

### Proposal에서 온 내용

- fixed direct-z, same-snapshot proposal, allowed rewrite-side utility 아래
  event-specific allocation이 static allocation을 개선할 수 있는지가
  Motivation의 핵심이다.
- static policy로 충분하거나 outcome-blind signal이 actual progress를
  예측하지 못하면 routing/controller 방향을 빠르게 kill 또는 pivot한다.
- 작은 diagnostic이 생존한 뒤에만 refresh와 후속 capacity claim을 연다.

### Repo/protocol에서 확인한 사실

- Llama/Qwen v1 pair는 model별 planned/attempted/pass `3/3/3`, exact
  rollback, receipt chain, resource cap을 통과했다.
- v1 action set은 five single-layer unit directions와 uniform 하나였다.
- 양 model의 모든 `case × fraction`에서 controller, retrospective static,
  event-wise six-arm winner가 `uniform`이었다.
- Qwen은 sign concordance는 `1.0`이었지만 세 fraction 모두 p90 relative
  derivative error gate에 실패했다. Llama는 세 fraction gate를 통과했다.
- 기존 세 pilot case는 calibration split `[0:3]`이며, 모두 single-token
  target이었다.
- 기존 expected-effect audit은 oracle upper bound와 achievable held-out gain을
  분리하고, calibration 결과를 성능 claim에 쓰지 않도록 요구한다.

### GH의 추정 및 확정 운영 결정

- v1의 uniform 승리는 broad-positive slope geometry로 설명될 수 있으므로
  continuous routing null을 식별하지 않는다.
- 기존 six-arm panel은 sparse-vs-uniform sentinel로는 유효하지만 더 넓은
  routing kill estimand로는 불충분하다.
- v2 D0는 `q=1/256`, calibration `[3:8]` 다섯 case만 양 model에서 동시에
  실행한다. Pilot 세 case는 v2 estimand, selection, CI에서 제외한다.
- D0는 빠른 early kill만 허용하며 positive GO나 MV-2를 허용하지 않는다.

### 사용자 확인 필요

- 없음. 사용자는 claim-revealing Motivation 검증, 필수 최소 감사, 양 model
  동시 실행과 빠른 kill을 요구했다. v2 D0는 새 dataset이나 큰 stage를
  추가하지 않고 기존 calibration 안에서 non-identifiability만 교정한다.

## 3. 왜 v1 six-arm oracle이 broader kill estimand가 아닌가

각 layer의 unit-`C` direction을 `V_l`, local slope를:

```text
s_l = d U(W + t V_l) / dt at t=0
```

라 두면 v1 uniform direction은:

```text
V_uniform = sum_l V_l / sqrt(5)
score_uniform = sum_l s_l / sqrt(5)
```

다. 예를 들어 모든 `s_l=s>0`이면
`score_uniform=sqrt(5)s`이고 best single score는 `s`다. Layer별 slope가
서로 달라도 여러 양의 slope가 누적되면 같은 현상이 유지된다. 이때
six-arm oracle은 uniform을 고르지만, 그 결과는 continuous mixture 안에
uniform보다 좋은 회전 방향이 있는지 말해 주지 않는다.

Non-negative unit sphere에서 local-linear score를 최대화하는 방향은:

```text
w = relu(s) / ||relu(s)||_2
V_mix = sum_l w_l V_l
score_mix = sum_l w_l s_l
```

이다. 모든 slope가 non-positive이면 `argmax_l s_l` one-hot으로
fallback하고 exact tie는 낮은 layer를 택한다. 이 action은 outcome을 보지
않고 pre-outcome slopes만 사용한다. 따라서 같은 `C` energy의
`V_mix`와 `V_uniform` realized progress를 직접 비교해야 continuous routing
opportunity와 analytic selection을 한 번에 반증할 수 있다.

이 교정 때문에 기존 six arms는 sparse-vs-uniform sentinel로만 남긴다.
그 panel의 oracle gap, ordered arm contrast, native arm contrast를 합쳐
broader routing opportunity나 ODE-Edit expected gain을 만들지 않는다.

## 4. Pilot-only pure-rotation forecast

GH가 제공한 확정 post-pilot slope 계산은 같은 local slope에서 uniform을
`relu(s)/L2` mixture로 회전할 때 다음 정도의 analytic score 차이를
시사한다.

| model | casewise ratio 평균 | mean-score ratio |
| --- | ---: | ---: |
| Llama3-8B-Instruct | 약 `+0.71%` | 약 `+1.06%` |
| Qwen2.5-7B-Instruct | 약 `+1.22%` | 약 `+1.74%` |

이 값은 **pure-rotation structural forecast**일 뿐이다.

- actual operational branch를 새로 실행해 얻은 gain이 아니다.
- 세 pilot case의 descriptive score ratio이며 CI가 없다.
- v2 action, threshold, model selection 또는 final estimand에 넣지 않는다.
- ODE refresh, retention, displacement 또는 baseline 대비 총 개선률이 아니다.
- Qwen의 큰 raw slope-derived finite-step 값은 보고하지 않는다. Qwen
  p90 derivative failure 때문에 local score를 operational distance까지
  선형 외삽하면 nonadditivity/calibration artifact가 될 수 있다.

따라서 방어 가능한 기대는 “동일 `C` budget에서 방향 회전만으로 약
1% 안팎의 local analytic score headroom이 보였으나 actual gain은 아직
미식별”이다. 이 작은 headroom은 오히려 다섯 case D0로 빠르게 kill할
이유이며, 큰 full C0를 먼저 실행할 근거가 아니다.

## 5. v2 D0 precommit

### Action과 primary

```text
q = 1/256
cases = calibration[3:8]                    # model별 5
w_i = relu(s_i) / ||relu(s_i)||_2
       # all-nonpositive: max-slope one-hot
       # exact tie: lower layer
G_mix,i = P_i(V_mix,i) - P_i(V_uniform,i)
```

- `V_mix`와 `V_uniform`은 같은 snapshot, request, context, direct-z,
  covariance, RNG, operational distance와 `C` energy를 쓴다.
- Primary는 realized `G_mix`; `score_mix-score_uniform`은 prediction
  diagnostic이다.
- Existing six arms는 sparse-vs-uniform sentinel, ordered/native/replay는
  execution 및 contextual sentinel이다.
- Sentinel outcome을 primary gain에 더하거나 favorable denominator만
  고르지 않는다.

### Technical block

다음 중 하나라도 발생하면 scientific zero가 아니라 **technical BLOCK**이다.

- source/action/receipt hash mismatch
- teacher-forcing/tokenization contract failure
- covariance/projector recompute 또는 download 시도
- non-finite feature/outcome
- exact rollback, same-`C` budget, denominator 또는 firewall failure
- planned case/arm 누락

Technical BLOCK에서는 D0 effect를 해석하거나 D1을 자동 제출하지 않는다.

### Early kill / continue

Model별 replay/near-tie envelope는
`e_m=max(1e-12, max_i abs(P_i(no_op_replay)))`로 결과 전에 고정한다.
Replay exact-logits/hash와 모든 technical validity gate가 먼저 통과해야 하며
missing/non-finite replay는 scientific zero로 바꾸지 않는다. 양 model
각각에서 다섯 event가 모두:

```text
G_mix,mi <= e_m
```

이면 current continuous-routing direction을 early kill한다. D1,
confirmatory, MV-2는 제출하지 않는다. 한 model에서라도 한 event가
envelope를 넘으면 결론은 `D1 calibration continue`뿐이며, 양 model을
동시에 calibration `[8:20]` 12 case로 확장한다.

D0 positive는 effect claim, controller GO, expected improvement 수치,
confirmatory success 또는 MV-2 advance를 허용하지 않는다.

## 6. D1 뒤 selection과 confirmatory boundary

D0가 생존하면 D1을 완료해 model별 v2 calibration 17 case를 만든다.
Historical pilot `[0:3]`은 계속 제외한다. Confirmatory outcome을 열기 전에:

1. model별 slope feature `s_i`만 pooling한다.
2. `sbar_l=mean_i s_il`을 계산한다.
3. `w_static=relu(sbar)/||relu(sbar)||_2`를 만들고 all-nonpositive/tie
   fallback을 adaptive rule과 같게 둔다.
4. Code, feature manifest, weight와 hash를 freeze한다.

Confirmatory estimand는 adaptive `V_mix,i`와 frozen static pooled
`V_static`의 equal-`C` realized progress 차이다. Raw slopes 또는 outcomes를
두 model 사이에서 pooling하지 않는다. Adaptive probe의 추가 NFE,
wall-clock, memory를 숨기지 않고 static과 함께 보고하며 matched-NFE
sensitivity를 분리한다.

## 7. Deprecated path와 최소 audit

- 기존 20-case v1 full C0 wrapper/job은 **deprecated / 제출 금지**다.
- D0와 D1은 separate paired wave이며 Llama 후 Qwen 순차 제출을 금지한다.
- 각 wave의 필수 review는 execution preflight red 1건, model별 독립
  post-run analysis 각 1건, pair post-run red 1건으로 제한한다.
- D0 scientific null 때문에 추가 audit을 열지 않는다. Technical BLOCK,
  emergency triage, artifact mismatch일 때만 해당 영향 범위의 기록을
  추가한다.

## 8. 최종 판정

```text
v1 execution fidelity: PASS
v1 broader routing identifiability: FAIL
full-as-is: REVISE-before-full
next authorized experiment: v2 D0 [3:8], q=1/256, paired two-model
positive D0: D1만 CONTINUE
all-five null in both models: continuous routing EARLY KILL
confirmatory / MV-2: BLOCK
```

이 교정은 pilot의 negative signal을 지우는 것이 아니다. Uniform이
sparse single arms를 압도했다는 관측은 그대로 남긴다. 다만 그 관측이
대답하지 못한 질문—broad-positive slopes 안에서 outcome-blind continuous
rotation이 같은 budget의 uniform을 개선하는가—만 가장 작은 추가
diagnostic으로 직접 묻는다.
