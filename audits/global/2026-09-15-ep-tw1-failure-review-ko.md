# Server4 EP-TW-1 첫 실행 실패 진단 검토

작성: 2026-09-15 KST. 상태: 지정 보고·실행 source 읽기, 저장 scalar 재계산 및 후속 진단 설계. 새 모델 forward, GPU 작업, source 수정, scheduler 변경 또는 원격 지시 전달은 수행하지 않았다.

## 1. 판단

**보고서의 핵심 판단은 타당하다. 현재는 첫 current-loss gradient의 기술 검증이 해결되지 않은 상태이며, EP-TW-1의 성능 실패로 해석할 수 없다.** Native fit 뒤 `gE/gD` sweep을 거쳐 Current E의 AD–FD 검사에서 종료했다. Correction construction, 후보 screen, selected endpoint, history finalizer에는 도달하지 않았다.

특히 이번 결과는 이전 설계에서 설명한 **“projected correction이 finite quality 조건에서 탈락하는 현상”도 아니다.** 보정 방향 자체를 만들기 전 검사이므로 quality restoration이나 ODE 확장으로 넘어갈 근거가 아직 없다. 먼저 동일 Vp에서 derivative 검사와 실제 gradient 경로를 분리 확인하는 것이 맞다.

추가로 확인한 중요한 점은 **기존 두 probe에서 forward response의 구간 기울기가 매우 다르다**는 것이다. 따라서 “weight norm의 0.1%면 충분히 국소적”이라는 전제는 성립했다고 볼 수 없다. Probe locality를 우선 점검하되, 이 관측으로 AD가 옳다고 선언하지는 않는다.

## 2. 읽은 원문과 출처

사용자 지정 원문:

`server4:/data/janghj/ODE-edit/local/ep-tw1-c4/20260915-v1/attempt-v1/failure-diagnosis-r1/diagnostic-report-ko.md`

[원문 보존 사본](/mnt/raid5/janghj/ODE-edit/local/reviews/ep-tw1-failure-2026-09-15/source/failure-diagnosis-r1/diagnostic-report-ko.md), `failure.json`, `native-fit.json`, `cpu-evidence.json`, teacher reuse 검증, execution lock, 실제 `technical.py`·`model_adapter.py`·`runner.py`·native map을 읽었다. 총11개 작은 source/보고 파일을 exact SHA와 함께 로컬 보존했다. 대형 모델·teacher·native `.pt`는 이번 task에서 전송·로드하지 않았다.

[출처 manifest](/mnt/raid5/janghj/ODE-edit/local/reviews/ep-tw1-failure-2026-09-15/source-manifest.json). 실행 source 기준은 `3fb0bfb27773ec9d016e76fcdc978dc996f010a7`이며 핵심 세 source 파일의 로컬 사본 SHA가 execution lock의 SHA와 일치하는지 재확인했다. CPU tensor 검증 내용은 SH 보고의 증거로 읽었고, 이번 task에서 그 GPU/CPU tensor 검증 전체를 독립 재실행한 것으로 쓰지 않는다.

## 3. 실제로 어디까지 진행됐는가

| 단계 | 확인된 상태 |
| --- | --- |
| Teacher192 | `TEACHER192_FULLSHA_SCHEMA_FINITE_REUSED`; 기존 job47592 COMPLETED0:0, 할당98 GPU초 보고 |
| W0 S64 | 독립 실제 W0 forward, D64=0,64문서/8192 scored positions; 첫 self-KL 확인 |
| B001 entry 평가 | W0의 R100/P200/N1000 관측; 편집 후 성능 아님 |
| Native fit | target100, Adam2400, loss 평가2500, native solve1, actual Vp 저장 |
| Native map | direct native RHS와 Vp exact; factorized A와의 FP32 차이는 별도 기록 |
| Gradient | gE/gD sweep 후 E-FD 호출에 도달; raw gradient tensor는 미보존 |
| E-FD | AD 불일치·사전 수렴 기준 미충족으로 예외 |
| D-FD | E-FD 뒤에 있으므로 NOT_RUN |
| Policy/commit | correction·candidate screen·선택·finalizer·B2 미실행, commit0 |

Job47884는 473초 후 `FAILED/NonZeroExitCode/1:0`으로 종료했다. OOM·timeout·외부 signal 종료라는 증거는 없다. 할당 비용은0.1313889 GPUh이며 utilization·peak GPU memory를 측정한 값이 아니다. 이전 teacher98초를 이번473초에 중복 합산하지 않는다.

따라서 이전 리뷰의 “teacher 완료 미확인”은 이번 후속 증거에서 갱신된다. 반면 **teacher PASS가 model gradient PASS나 G0 PASS를 의미하지는 않는다.** 전체1000 중 최종 편집을 commit한 요청은0개다.

## 4. AD–FD 수치의 독립 재계산

방향은 `v=gE/||gE||`, `AD=<gE,v>`다. 검사 간격은

\[
h=10^{-3}\frac{\|V_p\|}{\|vA\|},\qquad
FD(h)=\frac{E(hv)-E(-hv)}{2h}.
\]

이 h는 method의 correction alpha나 backtrack scale이 아니다.

| 항목 | h | h/2 |
| --- | ---: | ---: |
| Residual step | 0.234695021 | 0.117347510 |
| E(+) | 0.121702178 | 0.053763338 |
| E(-) | 0.001603220 | 0.001706223 |
| AD | 0.026770690 | 0.026770690 |
| Central FD | 0.255861751 | 0.221807495 |
| FD / AD | **9.55753** | **8.28546** |
| Relative error | 855.753% | 728.546% |
| Nominal weight action norm | 0.077789047 | 0.038894523 |
| Native actual delta norm7.610187 대비 | 1.02217% | 0.511085% |

표의 nominal action은 `h||vA||`다. 실제 rounded candidate tensor가 저장되지 않았으므로 actual materialized perturbation norm이라고 표기하지 않는다.

보고된 `convergence_relative=1.27207`도 코드와 일치한다. 분모가 FD가 아니라 AD이기 때문이다. 두 FD의 차이는 큰 FD 대비13.31%지만 **AD 대비127.21%**다. 분모를 바꿔 통과로 바꾸는 것은 해결책이 아니며, derivative 자체의 큰 불일치는 그대로 남는다.

### 4.1 두 구간의 실제 secant가 크게 다르다

같은 네 loss 값에서

\[
s_+=\frac{E(h)-E(h/2)}{h/2}=0.578954250,
\qquad
s_-=\frac{E(-h/2)-E(-h)}{h/2}=0.000877763.
\]

를 얻는다. 비율은 약 **659.58배**다. 이는 관측된 forward 함수가 해당 구간 전체에서 동일한 기울기의 직선으로 근사되지 않음을 보여준다. Current100 평균이므로 일부 요청의 급격한 NLL 변화가 지배했을 가능성도 있지만, 개별 probe loss가 저장되지 않아 그 원인을 특정할 수 없다.

이 관측은 **더 작은 간격에서 locality를 확인할 필요성**을 뒷받침한다. 그러나 실제 E(0), 더 작은 h의 값, gradient tensor가 없으므로 AD correctness나 작은 h에서의 극한을 확정하지 못한다. 두 점으로 Richardson extrapolation을 만들어 정답처럼 사용하는 것도 피한다.

### 4.2 `signal_resolved`는 무엇을 뜻하는가

두 signal은 코드의 `8×FP32 epsilon×loss scale` 기준보다 각각 약103만·102만 배 크다. 따라서 보고서가 이를 단순한 작은 차분의 상쇄 문제로 설명하지 않은 것은 적절하다.

다만 이 기준은 **모델 전체의 실제 floating-point 오차 상한이 아니다.** Forward 반복 오차, 실제 rounded weight perturbation, 가중치별 ULP·loss 계산 누적을 측정한 값은 아니다. `signal_resolved=True`를 gradient PASS나 FP32 오차 완전 배제의 증명으로 확장하지 않는다.

## 5. Source 검토에서 확인된 것과 남은 것

[model_adapter.py:150](/mnt/raid5/janghj/ODE-edit/local/reviews/ep-tw1-failure-2026-09-15/source/source-v1/project/run_scripts/bg_tw_reference/ep_tw/model_adapter.py:150)의 custom node는 C=0에서 raw clone, C≠0에서 `Vp+C A`, backward에서 `gW Aᵀ`를 사용한다. 이는 의도한 실수 연산 map의 VJP다. Forward 일치만으로 custom backward가 실물에서 검증되는 것은 아니다.

[current loss:331](/mnt/raid5/janghj/ODE-edit/local/reviews/ep-tw1-failure-2026-09-15/source/source-v1/project/run_scripts/bg_tw_reference/ep_tw/model_adapter.py:331)는 요청별 target-token 평균을 만들고 AD에서는 group sum/전체100, 보고에서는100개 loss의 평균을 사용한다. Generic도 문서별 position 평균을 만들고 전체64로 나눈다. 읽은 코드에서 **명백한 ×100·×64 normalization 오류나 transpose 오류를 확인하지는 못했다.** 이것이 모델 AD 전체의 정당성 증명은 아니다.

Native의 direct solve와 factorized A 사이에는 delta 상대 L2 약`6.086e-6`의 차이가 있다. 그러나 이번 AD와 FD는 **같은 actual Vp와 같은 고정 A**를 사용한다. Native와 factorized endpoint의 비동일성만으로 AD–FD의8–10배 차이를 설명할 수 없다.

실행은 Torch2.9.1+cu128, Transformers4.44.2, FP32, eager attention, TF32 matmul=false다. 저장 근거 없이 TF32나 attention kernel이 원인이라고 단정하지 않는다.

[technical.py:175](/mnt/raid5/janghj/ODE-edit/local/reviews/ep-tw1-failure-2026-09-15/source/source-v1/project/run_scripts/bg_tw_reference/ep_tw/technical.py:175)의 C0 current/generic 일치와 zero/nonzero functional/materialized 검사를 통과하지 못했다면 E-FD까지 도달하지 못한다. 이것은 source control-flow 근거다. 함수가 마지막에 반환하기 전에 예외가 나 별도 receipt는 저장되지 않았다.

[runner.py:185](/mnt/raid5/janghj/ODE-edit/local/reviews/ep-tw1-failure-2026-09-15/source/source-v1/project/run_scripts/bg_tw_reference/ep_tw/runner.py:185)는 sweeps 계산 뒤 검사부터 호출하고, gradient tensor는 후보 선택 후229행에서야 저장한다. 따라서 실패 시 gE/gD·raw E/D 상세 자료가 사라지는 **진단 자료 저장 시점의 문제**는 확인할 수 있다. 각 기술 단계와 gradient/raw objectives를 검사 예외 이전에 보존하는 수정은 타당하다.

## 6. 다음에는 무엇을 얼마나 확인할 것인가

주 method의 품질 조건·보존 목적·ζ·후보 메뉴는 이번 실패를 이유로 바꾸지 않는다. 다음 범위는 **같은 저장 episode의 기술 진단**으로 한정하는 것이 적절하다.

### 6.1 먼저 저장과 derivative 경로를 분리한다

1. 봉인된 W0의 다른 weight, 저장된 Vp/A/Z/anchor/radius, 같은100요청·tokenizer·contexts·C4 teacher를 연결한다. 새 진단 namespace와 source/입력 hash를 기록한다.
2. E(0), D(0), 두 gradient, request/document별 loss, 기술 단계 receipt를 검사 직후 저장한다. Probe마다 h·C·실제 materialized delta norm/hash·E(+)/E(-)의 요청별 값도 저장한다. 기존 파일은 덮어쓰지 않는다.
3. 작은 CPU FP64 fixture에서 같은 custom affine node의 VJP를 확인한다. 전체 Llama를 FP64로 바꾸는 실험과 구분한다. PyTorch의 기본 gradcheck 허용치는 double 입력을 전제로 하므로 FP32 모델에 기본값을 그대로 적용하는 것은 적절한 해법이 아니다. [PyTorch2.9 공식 문서](https://docs.pytorch.org/docs/2.9/generated/torch.autograd.gradcheck.gradcheck.html)
4. 실제 Vp에서 직접 weight leaf에 대한 gW를 구하는 독립 경로와 `gC=gW Aᵀ`, `〈gC,v〉=〈gW,vA〉`를 비교한다. 이는 custom map 연결을 검사하며 동일 neural backward 자체의 correctness까지 독립 증명하지는 않는다.

### 6.2 Local FD는 단일 간격 조정이 아니라 수렴 구간을 확인한다

실행 전에 최대 probe 수와 선택 규칙을 고정한 diagnostic grid를 둔다. 구체적인 첫 안은 같은 normalized gE 방향에서 **기존 h를 시작으로 `h/2^k`, k=0…9**다. 기존 두 coarse point의 실패도 모두 보존한다. 새 작은 h를 정답이라고 미리 선언하는 것이 아니다.

- E0의 반복 측정과 실제 rounded action을 함께 기록한다.
- Central FD, AD 오차, `E(±h)-E0∓h AD`의 Taylor remainder, 요청별 loss 변화를 비교한다.
- 충분히 resolved인 연속한 작은 간격들에서 AD 일치와 안정성을 확인한다. 고립된 한 점이 우연히 맞았다는 이유로 PASS를 주지 않는다.
- 큰 h에서 truncation이 있는 구간과 너무 작은 h에서 resolution이 사라지는 구간을 구분한다. 전체 grid의 모든 coarse point가 AD와 맞아야 한다는 조건을 새 local 검증의 정의로 사용하지 않는다.
- 사전15% derivative tolerance를 이번 값에 맞춰 확대하지 않는다. 새 진단의 수렴-window 판정과 관측 상태는 실행 전에 명시하고, 이전 실패 기록은 그대로 둔다.
- 처음부터 매우 작은 간격·여러 kernel·새 objective를 동시에 바꾸지 않는다. 안정 구간이 없으면 `UNRESOLVED`로 남겨 원인을 분리한다.

Current microbatch16의100요청은 한 observation당7 forward groups다. 한 방향·10개 ± probe는 **140 groups**이며 E0·gradient·독립 경로 검사 비용은 추가다. 이 수는 계산 횟수 계약이지 latency 예측이 아니다.

E 문제가 해결되면 현재 미실행인 **D64 derivative도 별도로** 확인해야 한다. Self-gradient 한 방향의 통과를 전체 gradient 정확성으로 확대하지 않도록, 사전 고정한 독립 방향 검사도 포함한다. D는 한 ± scale당128문서 forward이므로 current 검사와 동일 비용으로 세지 않는다. 기존 모델·target 재최적화나 보호 token별 Jacobian 구축은 필요하지 않다.

### 6.3 결과에 따른 분기

| 새 진단 결과 | 판단·다음 조치 |
| --- | --- |
| Custom node/직접 gW 비교부터 불일치 | Map 또는 gradient 연결·누적 구현을 수정하고 같은 기술 입력으로 검증 |
| 독립 경로가 일치하고 작은 h들에서 FD가 AD로 안정 | 기존 두 coarse probe의 비국소성이 원인이었다는 근거; 검증 규약 수정 후 과학 실행 재개 판단 |
| 작은 h에도 resolved 상태에서 일정 배수 차이 유지 | Objective/scoring/reduction/custom backward 경로를 다시 분리; tolerance 확대 금지 |
| 작은 h에서 signal/action resolution 소실 | FP32 진단 미해결; gradient가 맞거나 틀리다고 단정하지 않음 |
| E만 통과, D 미검증/불일치 | 전체 model technical PASS로 승격하지 않음 |

## 7. 재사용과 1000 sequential 실험의 관계

약256 MB의 `native-targets-map.pt`에 Z·anchor·radius·K/H capture·A·Vp가 남아 있어 **다음 기술 진단을 위해 native target100/solve1을 다시 계산할 필요는 없다.** 원 native fit의 synchronized286.54초를 재사용할 수 있다. 이것은 그 파일을 완전한 scientific continuation checkpoint라고 부른다는 뜻이 아니다.

Post-fit RNG, original raw gradients, optimizer/local teacher 전체 payload, accepted ledger commit가 없고 commit0이다. 새 진단에서 재계산한 gradient를 기존 gradient tensor와 byte-identical이라고 주장할 수 없다. 기존 실패 run을 B2부터 이어가는 작업으로 해석하지 않는다.

기술 검증이 해결된 뒤 주 과학 평가는 여전히 **W0 B100×10의1000요청**이다. 이번 B001 기술 진단은 single-batch 성능 선별이 아니다. 기술 실패를 해결하는 동안 method efficacy·old retention·ODE·barrier에 대한 결론은 보류한다.

## 8. 이번 검토의 완료 범위

지정 보고를 실제 Server4에서 읽고, 관련 source·lock·기존 JSON을 대조했다. FD 수치·convergence 분모·secant·비용 산술과 source hash는 [재검산 기록](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-15-ep-tw1-failure-review-checks.json)에 남겼다.

현 단계의 권고는 **동일 Vp/A를 재사용한 bounded derivative 진단을 먼저 하고, 새 source에서는 실패 이전에 gradient와 기술 receipt를 보존하는 것**이다. Controller의 품질 조건 완화, N4 한도 복귀, quality restoration 추가,1000-chain 강제 재실행을 이번 실패의 해결책으로 삼을 근거는 없다. 이번 task에서는 그 후속 실행이나 source 수정을 수행하지 않았다.
