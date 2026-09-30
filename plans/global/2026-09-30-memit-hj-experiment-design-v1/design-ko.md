**MEMIT-HJ 실험 설계 v2**

2026-09-30. 첨부 방법 명세·NumPy 참조 코드·patch와 기존 MEMIT-H 완료 자료에 기반한 실험 설계다. 이번 산출물은 설계이며 runner 구현, patch 적용, GPU 실행, job 제출은 포함하지 않는다. 첨부 문서의 작업 지시는 검토 자료로 취급했다.

**주 비교는 BS100의 2k 요인 비교와 10k 장기 비교이며, 100건으로 끝나던 과학적 편집 시험은 모두 BS10 × 100 step = 1,000건으로 확대한다.** 배분 진단은 MEMIT-H와 HJ-v1 두 경로로 줄이고, SPG는 수치 교정 후에만 본 실험에 넣는다. R arm은 짝 non-R arm과 상태를 공유하다 최초 refresh 발동 시 RAM에서 분기한다. 아래의 실행량은 공유한 prefix를 한 번만 계산한다.

이번 개정은 사용자 리뷰의 필수 수정 1–3과 권장 4–7을 점검해 반영했다. 첨부 참조 구현 자체는 수정하지 않았다. 기존 BS1 과학적 trajectory는 폐지한다. CPU의 단일 요청 수식 검사, 16요청 loss-only 교정, native의 원래 BS100 한 batch를 재현하는 기술 parity는 1,000건 편집 시험과 구분한다.

**검증할 가설과 주장 범위**

|가설|직접 비교|확인할 결과|반증 또는 해석 제한|
|---|---|---|---|
|A 공동 배분이 divisor의 실현 손실을 줄인다|HJ-v1 대 MEMIT-H|실제 L8 잔차, 최종 PS, RS와 NS|OLS 목적만 감소하고 실제 잔차·PS가 개선되지 않으면 실제 모델의 이점은 미확인|
|Z 같은 z 목적을 더 낮게 푸는 것이 기능적 편집에도 유리하다|Z-only 대 MEMIT-H 및 HJ-v2 대 HJ-v1|동일 상태의 손실·NLL·KKT 잔차, 최종 편집 품질|손실 감소가 decay 감소에만 의존하고 PS가 나빠지면 좋은 solver가 좋은 editor라는 가설은 지지되지 않음|
|R 현재 key로 history를 갱신하면 이후 편집에서 과거 지식을 더 잘 보존한다|R-only 대 MEMIT-H 및 HJ-v3 대 HJ-v2|갱신 이후 lost/gained, age별 retention, 신규 편집과 NS|H를 갱신한 직후에는 W가 같아 출력도 같아야 함. 이후 쓰기를 거친 차이만 효과로 해석|

v1의 닫힌 해는 **고정 key와 가법 선형 surrogate**의 해다. 실제 모델에서 순차 재측정한 결과를 전체 신경망의 전역 공동 최적해라고 부르지 않는다. SPG는 비볼록 신경망 손실의 전역 정확해를 보장하지 않으므로 v2를 **SPG z solver**로 표기한다. 표준 SPG의 분석은 미분 가능한 목적을 다루며, 이 코드의 norm decay는 원점 처리도 필요하다. [SPG 원논문](https://www.ime.usp.br/~egbirgin/publications/S1052623497330963.pdf)

history를 포함한 sequential MEMIT의 근거는 논문 Remark A.1에서 확인된다. 다만 현재 key로 H만 다시 만드는 조작이 다층 신경망의 OTE–SE 등가를 복구한다는 주장은 이번 실험의 전제가 아니다. [논문과 Remark A.1](https://arxiv.org/html/2605.26670v1#A3)

**공통 조건과 기존 결과의 위치**

|항목|고정 조건|
|---|---|
|모델|Meta-Llama-3-8B-Instruct, revision `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`|
|데이터|`counterfact-fixed-10k-v1`의 기존 순서, 소규모 실험도 앞 N개|
|ordered root|`5b01356928ad2e2e46effad5a2c5621c87746b610992b7b92bf3ace16b6f5729`|
|dataset SHA256|`3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1`|
|층과 writer|L4–L8 down_proj, FP64 solve, FP32 weight materialization|
|보존|λ=15000, Wikipedia mom2 100k, history 계수 1|
|native z|L8, clamp .75, lr .1, 25회 평가 루프, decay .5, KL .0625, loss layer 31, subject_last|
|context|기존 여섯 context, SHA `33cec0eef9ec130f26c2f0e17f7c8be39e93c717ebb47eaa5e9f1c88b263524e`|
|history append|모든 층 write 완료 후 post-write key로 층마다 batch당 한 번|
|환경 기준|seed 20260907, torch 2.9.1+cu128, transformers 4.44.2, FP32/eager, autocast off; 세부 runtime lock 재사용|
|평가|기존 explicit-left evaluator와 tokenizer 규약, microbatch 16, tie 실패|

새 matched MEMIT-H 실행이 주 대조군이다. 과거 job54007은 RS/PS/NS **95.520/85.170/61.553%**, 12.439444 GPUh를 기록했다. 기존 AlphaEdit-BLUE는 **98.880/95.775/63.726%**이며 방법 전체의 외부 성능 기준으로 표시한다. BLUE는 L4+L8, 별도 target·projector·L2 정책을 사용하고 실행 GPU도 달랐으므로 그 차이를 배분만의 효과로 해석하지 않는다. 과거 MEMIT 및 AlphaEdit도 역사적 표에 포함하되 신규 matched 대조와 구분한다.

현재 기본 설계는 이 한 모델·데이터·순서에 대한 검증이다. 기존 고정 정책에 따라 새로운 순서 seed, 별도 holdout, CounterFact 재추출은 실행 목록에 넣지 않는다. 앞 2k는 10k의 prefix이므로 독립 validation/test가 아니다. 성능을 보고 λ·clamp·층·trigger를 조정하면 해당 결과는 개발 결과로 기록하고, 같은 10k 재실행을 독립 검증이라고 부르지 않는다.

**논리 실험군과 실제 RAM 분기**

세 자리 ID는 A=공동 배분, Z=SPG, R=history refresh 적용 여부다. 모든 arm이 기본 history를 사용한다. Z=0은 native Adam이며 R=0은 append-only다.

|ID|표시명|배분|z|history|종료 요청 수|실제 출발|
|---|---|---|---|---|---:|---|
|000|MEMIT-H|divisor|native|append-only|10000|W0/H0|
|100|HJ-v1|joint|native|append-only|10000|W0/H0|
|010|Z-only|divisor|SPG|append-only|2000|W0/H0|
|001|R-only|divisor|native|refresh|2000|000의 최초 발동 상태|
|110|HJ-v2|joint|SPG|append-only|10000|W0/H0|
|101|HJ-v1+R|joint|native|refresh|2000|100의 최초 발동 상태|
|011|Z+R|divisor|SPG|refresh|2000|010의 최초 발동 상태|
|111|HJ-v3|joint|SPG|refresh|10000|110의 최초 발동 상태|

부모 000/100/010/110에도 동일한 비개입 표본 기록과 trigger 검사를 둔다. 기록 전후 W/H, RNG, context, cache, ledger가 변하지 않는지 검증하고 관측 비용을 별도 표시한다. 부모를 편집하는 동작은 기존 방식 그대로다. 분기 경계는 **모든 층 write와 native H append가 끝난 뒤, refresh 직전**이다.

해당 자식의 종료 이전에 처음 발동한 시점 f에서 W, H, CPU/CUDA/Python/NumPy RNG, context, request ledger, 표본 reference와 dispersion, cache 및 다음 batch 위치를 독립적인 host RAM copy로 보관한다. 부모의 이후 상태를 참조하는 얕은 복사는 금지한다. 자식은 그 상태를 복원한 뒤 발동 층의 H를 갱신하고 다음 요청부터 진행한다. C0와 비편집 weight는 hash로 결속한 불변 자산을 공유할 수 있다. allocator·kernel 비결정성까지 포함한 재현성 gate를 통과해야 동일 trajectory 공유를 주장한다.

2k 자식 001/101/011은 1k 검사에서 발동하지 않으면 별도 편집을 실행하지 않고 `NOT_FIRED — paired parent와 동일`로 보고한다. 부모 평가 artifact를 hash와 alias로 참조하며 가짜 재실행 결과를 만들지 않는다. 111은 1k에서 미발동해도 2k…9k의 검사를 계속한다. **아직 발동하지 않았다는 이유로 장기 R arm을 1k에서 없애지 않는다.** 종료 시점의 refresh는 후속 write에 영향을 주지 않으므로 실행하지 않는다.

2k에서는 여전히 8개 논리 조합을 보고하고 `(100−000), (110−010), (101−001), (111−011)` 등을 계산할 수 있다. 다만 미발동 arm의 차이 0은 정책이 이 입력에서 개입하지 않았다는 결과이며, current-key refresh가 효과 없다는 증거가 아니다. 동일 artifact를 독립 반복으로 세지 않는다. R의 주 증거는 **10k의 111−110과 네 출발 시점의 강제 refresh 진단**이다. 10k의 주 비교는 100−000, 추가 요인의 비교는 110−100과 111−110이다. 모든 요인의 장기 상호작용을 식별했다고 주장하지 않는다.

**실행 순서와 규모**

|단계|실행 내용|규모와 고정 규칙|
|---|---|---|
|T0a|native parity, loss oracle, 수식, 상태 복원 검증|원래 BS100 한 batch의 기술 replay와 CPU 검사. 과학적 성공률 검정이 아님|
|T0b|SPG 수치 바닥·호출 예산 교정|W0 및 native 000의 BS100 누적1k RAM 상태에서 각각16요청. 000의 첫1k는 본체와 공유|
|E0와 T1|W0 writer 진단을 가장 먼저 보고|각 BS10×100 step. 000/100 두 경로가 T1이며 E0의 divisor/joint와 동일 실행을 재사용|
|E0의 보조 writer|upper-key 고정 근사와 energy-matched divisor|두 경로도 각 BS10×100 step. 예정 arm은 E0 성적에 따라 변경하지 않음|
|T2|BS100에서 2k의 8개 논리 조합|물리적으로 4개 부모와 발동한 R suffix만 실행|
|T3|000/100/110/111을 10k까지 진행|111은 최초 발동 전까지 110과 공유. 010 및 짧은 R arm은2k 종료|
|중간 진단|writer 1k·5k, history 1k·3k·5k·7k|각 분기는 모두 BS10×100 step, 상태 복원 뒤 본체 계속|

T0b의 1k 상태는 **BS100 native 부모**의 상태다. 새 BS10 진단의 1k 상태와 혼용하지 않는다. E0는 T0 후 다른 과학적 진단보다 먼저 결과를 제시한다. SPG gate가 실패해도 Z를 쓰지 않는 000/100 및 writer 진단은 진행할 수 있으며, Z 구현을 무기록 native 대체로 바꾸지 않는다.

본체의 논리 노출량은 기존과 같은 48,000건이지만 실제 편집량은 다음과 같다. 미발동 f는 무한대로 놓는다.

`N_main = 32000 + Σ(r=001,101,011) 1[f_r<2000](2000−f_r) + 1[f_111<10000](10000−f_111)`

따라서 본체는 **32,000–44,000건, BS100 write 320–440회**다. writer 진단은 3개 출발 상태×4개 방법×1,000건=12,000건, history 진단은4개 상태×2개 방법×1,000건=8,000건이다. T1의 두 경로는 이12,000건 안에 들어 있으므로 다시 더하지 않는다. 전체 물리 편집량은 **52,000–64,000건, write 2,320–2,440회**이며, 이 중 BS10 진단 write가2,000회다. 추가 에너지 shadow proposal, T0 replay, FP64/loss-only probe, 선택적 L4 분기, 실패 재시도는 별도다. 실제 예산은 요청 수만이 아니라 이 작업별 시간을 합산한다.

본체의 full all-seen R/P/N 평가 시점은 누적100/500/1000/2000/3000/4000/5000/6000/7000/8000/9000/10000으로 유지한다. 매batch current R/P/N과 all-seen RS도 유지한다. 공유 상태의 평가 결과는 한 번만 계산한다. W0 전체10k의 동일 evaluator 평가를 한 번 확보해 모든 경로의 W0-correct mask에 사용한다.

**BS10 진단의 분기와 결과 지표**

|진단|출발 상태|비교|공통 후속 입력|
|---|---|---|---|
|writer|BS100 000의 누적0/1k/5k|divisor, full joint, upper-key 고정 joint, energy-matched divisor|각 anchor 직후의 고정 순서 다음1,000건을10건씩100step|
|z solver 교정|W0/BS100 000의 누적1k|native Adam, SPG, 실측 oracle 예산을 맞춘 연장 Adam|다음 구간의 사전 고정16요청. loss-only이며 편집하지 않음|
|history|BS100 000의 누적1k/3k/5k/7k|stale H 대 L5–L8 전체 강제 current-key rebuild|각 anchor 직후의 다음1,000건을10건씩100step|

분기들은 처음의 W/H/context/RNG가 같다. 이후 100step 동안 각자의 W에서 z·key를 재계산한다. **동일 z를 공유할 수 있는 것은 동일 W에서 낸 proposal들뿐**이며, 갈라진 trajectory에서 부모의 z를 계속 쓰지 않는다. 첫 step의 차이는 공통 상태의 직접 반응,1,000건 결과는 후속 편집 피드백을 포함한 차이로 해석한다. history 진단의 강제 rebuild는 출발점에서 한 번만 수행하고, 이후 두 분기 모두 append-only로 유지해 초기 H 개입을 분리한다. 자동 trigger와 반복 refresh는 본체111에서 평가한다.

각 편집 분기에서 아래 항목을 빠짐없이 측정한다. step1의10건 결과를 전체 시험으로 끝내지 않고100step까지 관측한다.

|측정 대상|기준 시점과 분모|보고량|
|---|---|---|
|새 편집의 at-write|매step 직후10요청: R/P/N=10/20/100|성공 수·분모, desired NLL, TF strict. 100step 통합은1000/2000/10000|
|후속1k의 누적 보존|추가10/100/500/1000건 시점, 그동안 새로 편집한 전체|R/P/N, at-write→현재 lost/gained, TF strict|
|기존 prefix 보존|분기 진입과 추가10/100/500/1000건, 동일 past panel400|entry→현재 lost/gained, retained, R/P/N 및 TF strict, cohort별 값|
|이웃 보존|past panel의4,000 neighborhood와 후속1k의10,000 neighborhood를 별도 구분|W0 전체 mask와 entry/current의 paired 전이, true NLL·TF strict. W0 실패였던 이웃의 획득도 별도 보고|
|실현|매step 각층의 동일token/context L8 잔차|절대norm, 정규화q, α, cosine, 예측과 관측 차이|
|보호 비용|매step의Δ 및 분기 진입 공통A와 현재A|에너지 각각, 누적합, update norm; 국소 에너지와 누적 weight 차이의 에너지를 혼동하지 않음|
|계산 비용|분기 전체와 세부 phase|oracle 수, forward/backward·key·solve·history·평가 시간, GPU/host peak memory|

W0 writer에는 과거 편집이 없으므로 past retention은 `NA_NO_PAST`다. W0-correct neighborhood는 후속1,000건에 딸린 평가 prompt로 측정한다. W0-correct 집합 S0에서 유지율을 구하되, `entry→current lost`와 `entry→current gained`는 S0 안에서도 둘 다 셀 수 있다. `W0→current`의 S0 내 gained는 정의상0이므로 두 시간축을 섞지 않는다. 모든 방법에서 같은 prompt/target identity와 분모를 쓴다.

loss-only z 진단은 f의 NLL/KL/decay 분해, δ norm/r, stationarity, 종료 상태, 실제 호출 수·시간을 비교한다. weight write가 없으므로 그 표의 at-write·history·보호 에너지는 `NA_NO_WRITE`다. z의 기능적 효과는 본체010−000과110−100의 완전한 편집에서 평가한다. 교정용16요청의z를 그대로1,000건 전체에 적용하지 않는다.

**고정 past panel400**

패널은 이미 편집한 prefix 안의 평가 표본이며, 새로운 edit 순서나 독립 holdout을 만들지 않는다. anchor별로 ordinal을 시간4분위로 나누고 각 분위에서100 occurrence를 SHA 오름차순으로 선택한다. SHA 입력은 고정 namespace `memit-hj-v2-past400`, anchor, 원 ordered root, occurrence ordinal, request hash를 결속한 canonical JSON이다. ordinal은 고정 배열의 0-based 위치이고 분위는 `floor(4×ordinal/anchor)`로 정한다. JSON은 key 정렬, 공백 없는 separators, ensure_ascii=False의 UTF-8 바이트로 직렬화한다. 동률은 ordinal 순으로 처리한다. 현재 anchor는1k 이상이어서 각 분위에서100개를 확보할 수 있다. case 반복도 occurrence를 보존한다.

분기 전에 ID·stratum·prompt hash를 고정하고, 모든 방법과 후속 시점에서 같은400요청을 사용한다. 분모는 R/P/N=400/800/4000이다. 각 분위 평균과 같은 비중의 전체 평균을 보고한다. anchor가 달라지면 패널도 달라질 수 있으므로1k 패널과5k 패널을 같은 cohort처럼 빼지 않는다. conflict·superseded 여부는 보고용 층화로 남기고 성능에 따라 패널을 교체하지 않는다.

리뷰가 언급한 'E1의 SHA 층화400'과 정확히 같은 원본을 확인하기 전에는 위 규칙을 **이번 설계의 명시적인 패널 규칙**으로 취급한다. 확인된 다른 E1 구현의128개 패널을400개와 동일하다고 부르지 않는다.

**배분과 강도 대조의 구체적 정의**

energy-matched 분기는 매step 자신의 동일 출발W와 native z에서 divisor와 joint의5층 proposal을 각각 가상 실행한다. step 출발의 공통 `A_l=λC0_l+H_l`로 `E=Σ tr(Δ_l A_l Δ_lᵀ)`를 구하고, native proposal에 `s=sqrt(E_joint/E_divisor)`를 곱해 해당 출발W에 적용한다. 가상실행 뒤 W/H/RNG/context를 복원한다. E_divisor=0은NA이며 s 선택에R/P/N을 쓰지 않는다. 이 정책은 같은 상태에서 두 proposal의 에너지를 맞추며, 서로 갈라진 두 trajectory의 누적 에너지가 같다는 보장은 아니다. shadow solve와 lookahead 비용도 별도로 계수한다.

BS10에서는 request 간 coupling이 있으므로 예측은 `D_i=R_rem (I+Σ_{j≥i}G_j)^−1 G_i`를 사용한다. scalar κ 식은 CPU의B=1 수식 검사에만 남긴다. native BS1 runner를 새로 만들지 않는다. full Gram의 off-diagonal norm, trace-capacity 몫과 실제D norm을 함께 기록한다.

선택적인 'L4 자기 층 z' 분기는 기본 실행표에 넣지 않는다. 추가한다면 동일W0에서 L4 target과 L4 writer를 사용하는 별도 BS10×100step 진단으로 명명하며, L8 target 다층 writer와는 target 위치와 writer 층 수가 함께 달라진다는 제한을 명시한다. 이 분기 하나만으로 target 위치의 단독 인과효과를 주장하지 않는다.

**v1의 측정과 수치 조건**

각 batch·각 층에 대해 κ 대각, G의 rank와 고윳값 범위, `tr(G_l)/tr(ΣG)`, 실측 Δ norm, 보호 에너지, D norm, L8 잔차를 저장한다. 잔차는 동일 token/context/target을 사용해 `q_l=||R_after_l||F/max(||R_pre_L4||F, ε)`와 절대 norm을 함께 기록한다. 초기 R=0이면 상대값은 NA로 처리한다.

예측 변위 `D_l=Δ_l K_l`에 대해 `ΔR_obs=R_before−R_after`, 도달 gain `α=<ΔR_obs,D>/||D||²`, cosine, `||ΔR_obs−D||/||D||`를 기록한다. D=0이면 해당 비율은 NA로 처리한다. α만으로는 직교 방향의 오차를 포착할 수 없다. capacity share, weight energy share, 실제 기능적 기여는 서로 다른 양이다. W0에서 λ에 의존하지 않는 것은 trace-capacity share이며, 이를 BS100의 D norm share로 일반화하지 않는다.

|검사|사전에 요구하는 조건|
|---|---|
|native parity|동일 process·환경·입력에서 native와 adapter의 divisor 모드를 각각 실행해 z, key, Δ, commit 후 W/H, 평가 bit를 비교. 우선 bit 일치를 요구하고, 불일치하면 이유와 오차를 조사한 뒤 구현 버전을 고정|
|과거 자료와 대조|job54007의 z/key/W/H receipt를 대조. 기존 자료에 없는 Δ tensor SHA와의 비교는 요구하지 않음|
|PSD와 solve|`λ_min(G) >= −1e-10 max(1,‖G‖2)`, I+ΣG의 Cholesky 성공, adj identity의 상대 오차≤1e-8. G가 엄밀한 양의 정부호일 것은 요구하지 않음|
|adj로부터 G 복원|직접 구한 `Kᵀ solve(A,K)`와의 오차 및 `1−λ_max(F)`를 검증. 복원 오차>1e-8, PSD 불성립 또는 `1−λ_max(F)<1e-6`이면 직접 solve로 전환하고 횟수와 비용 기록|
|최상위 층|L8에서는 resid를 입력 R 그대로 반환하고 bit 일치를 확인. 일반 solve가 우연히 identity를 반환하는 데 의존하지 않음|
|상태|모든 값 유한, C0 고정, H append 한 번, observer 전후 W/H/context/RNG 불변, 분기 복원 일치|

직접 구한 G와의 비교는 T0 및 누적 0/1k/5k/9k 다음 batch에서 모든 층에 대해 수행하고, 일반 batch에서는 condition/PSD/solve residual을 감시한다. 위 표의 임계값은 이번에 제안하는 수치 판정값이며, 첨부 자료에서 이미 실측한 값은 아니다. A의 특이성을 가리기 위한 jitter를 추가하지 않으며, 필요하면 변경 버전으로 명시한다.

CPU 추가 검사에는 duplicate/near-collinear key, R=0, 매우 큰 capacity, 0이 아닌 history, 비가환 G, λ 변경 시 D 비율을 포함한다. 첨부된 16개 테스트는 기존 확인 항목으로 취급하며, 이번에 이미 실행했다고 간주하지 않는다.

**v2의 SPG 교정과 종료 계약**

고정 `tol=1e-8`과 production oracle cap 1000을 폐기한다. FP32에서 항상 도달 불가능하다거나 대부분 1000회에 이른다고 단정하지는 않지만, 정밀도 바닥을 무시한 상한 실행 위험은 타당하다. 실제 production tol·cap은 **PENDING_CALIBRATION**이다. 아래 산정 규칙을 먼저 고정하고 기능 성능표를 이용해 선택하지 않는다. 교정 데이터는 기존 prefix와 겹치므로 독립 검증이라고 부르지 않는다.

1. 교정 상태는 W0와 BS100 native 000의 누적 1k다. 각 상태에서 다음 구간의 첫 16요청을 사전 지정한다. token/context/radius/δ, loss 각 항과 계수를 동일하게 둔다. FP32 모델의 가중치와 입력 캐시를 FP64로 정확히 승격해 **실제 forward/backward를 FP64로 수행**한다. 계산한 gradient만 FP64로 cast하는 것은 비교가 아니다. prefix 캐시를 FP64로 다시 생성해 입력값까지 바꾸지 않는다. FP64 suffix 계산이 지원되지 않으면 교정 미완료로 기록한다.
2. 각 요청의 δ=0, native δ의 절반, native δ에서 g32/g64를 비교한다. 원점의 norm 항에는 아래 별도 조건을 적용한다. 동일 FP64 projector로 `e_PG=‖P_r(δ−g32)−P_r(δ−g64)‖`를 측정하고 production projector 자체의 반올림 오차도 보탠다. `max(1,‖g64(0)‖)`로 정규화한 오차를 상태별로 집계한다. production의 PG 정규화는 해당 요청의 `max(1,‖g32(0)‖)`를 사용한다.
3. 제안하는 고정 산식은 `tol=max(10ε32, 5×max_state Q95(normalized PG error))`다. ε32는 해당 dtype의 machine epsilon이다. 이 값이 1e-3을 초과하면 큰 오차를 느슨한 종료 기준으로 덮지 않고 oracle/정밀도를 재설계한다. 이 수치들은 이번 개정의 사전 설정이며 측정 결과가 아니다.
4. 위 tol로 32요청을 실행한다. **SPG 교정의 요청당 FP32 oracle 상한은 400, line search 최대 50회**이며, SPG 교정 전체 상한은 12,800회다. 최초·zero-step 확인·backtracking·반환점 검사를 모두 계수하고 반환점 검사 호출을 상한 안에 예약한다. native 진단·연장 Adam·정밀도 probe의 대응 FP32/FP64 호출·상태 생성은 이 상한과 별도로 계수한다. 따라서 12,800을 T0 전체 계산 상한으로 쓰지 않는다.
5. production cap은 `max(25, ceil(1.25×max_state Q95(total oracle calls to terminal rule)))`로 정한다. 32요청 모두 사전 종료 규칙에 도달한 경우에만 이 산식을 채택한다. 검열된 NOT_CONVERGED나 LINESEARCH_FAILED를 빼고 성공 사례만으로 P95를 계산하지 않는다. 산정값이 400을 넘거나 미도달 사례가 있으면 Z의 T2 진입을 보류하고 재설계한다. 산정값을 400으로 조용히 자르지 않는다. 32표본의 P95는 미래 최대 비용 보장이 아니다.
6. W0·1k 각 상태 및 전체에서 **호출 수 median>200이면 Z 재설계**로 고정한다. zero-step 포함·제외 집계를 모두 검사한다. 원래 loop 25회 대비 8배라는 값은 gate의 기준일 뿐 실제 oracle 시간 8배를 뜻하지 않는다. 호출당 시간도 별도로 측정한다.

|종료 상태|조건과 해석|
|---|---|
|POLICY_ZERO_STEP|native의 초기 loss<.05 규칙. 수렴 증명과 구분|
|CONVERGED|반환점의 정규화 projected-gradient가 교정 tol 이하이고 feasible. 비볼록 전역 최적해 주장은 하지 않음|
|STALLED_AT_PRECISION|Armijo를 통과한 수락점에서 연속 5회 loss 변화와 같은 구간 best-loss 개선이 모두 `10ε32×max_window abs(f)` 이하. underflow 방지에 dtype tiny를 floor로 사용. 운영상 정체 종료이며 CONVERGED와 합치지 않음|
|NOT_CONVERGED|교정한 전체 호출 cap 도달. 유한한 마지막 수락점과 최종 residual 반환|
|LINESEARCH_FAILED|최대 50시도 안에 Armijo 미충족. 작은 step만으로 강제 수락하거나 precision 정체로 재분류하지 않음|
|NONFINITE|loss·gradient·state의 NaN/Inf. 기술 중단|

연속 5회 변화는 6개의 수락점으로 계산하며 같은 구간의 running best 변화도 검사한다. SPG는 비단조 line search이므로 작은 loss 변화만으로 수렴을 선언하지 않는다. STALLED_AT_PRECISION에서도 최종 PG, 수락 step 크기, backtracking 수와 loss window를 남긴다. 이 명칭은 정밀도 수준의 운영상 정체 규칙이며, 정체 원인이 오직 FP32임을 입증하지 않는다. 목적함수의 실제 개선과 기능 평가는 별도다. 미수렴·정체 사례를 분모에서 제외하거나 무기록 native fallback을 하지 않는다.

첨부 코드의 두 오류는 구현 전에 수정해야 한다. `lam<1e-20`만으로 Armijo 불만족점을 수락하지 않고, 반환점에서 loss·gradient·PG를 다시 계산한다. `native_adam`의 마지막 호출과 `exact_z`의 최초 zero-step 호출이 evaluation count에서 빠지는 문제도 고친다. production에서는 memory 10, Armijo γ=1e-4, alpha 범위 [1e-12,1e12]를 유지한다. max_iter는 총 oracle cap보다 더 큰 우회 예산을 만들지 않는다.

norm decay의 원점에서는 매끄러운 NLL+KL 부분의 기울기 g와 `c=.5/‖h‖²`에 대해 `max(‖g‖−c,0)`으로 일차 정지 조건을 별도 진단한다. 유한 차분은 0이 아닌 δ에서, 원점은 한쪽 방향미분으로 검증한다. `‖δ‖≤r(1+1e-6)`를 feasibility 조건으로 두고, `g_total=∇(NLL+KL+decay)`로 구한 `ν=max(0,−<g_total,δ/r>)`는 KKT multiplier 추정치로 보고한다. 이 multiplier 식은 feasible한 0이 아닌 경계점에서만 적용한다. 경계는 `abs(‖δ‖/r−1)≤1e-6`으로 판정하고 내부점의 ν는 0으로 둔다. 원점은 별도 norm-subgradient 조건을 사용한다. stationarity와 complementarity만으로 국소 최소를 보장하지 않으며, clamp의 국소 가치함수 민감도 해석에는 local-minimum·regularity 조건이 필요하다.

교정 후 선정한 tol·cap, 입력 32요청과 probe hash, 정밀도 차이, 종료 상태별 수, 실제 호출 수 분포·시간, NLL/KL/decay와 native 대비 loss 차이를 기록한다. 이 lock이 생기기 전에는 기존 1000 cap으로 Z 본실험을 시작하지 않는다.

**v3의 refresh 규칙과 증거의 범위**

주 실험의 BS100에서는 누적 1000, 2000, …, 9000건의 write·history append 후 검사한다. 종료 시점에는 진단만 수행하고 이후 write가 없는 불필요한 rebuild는 하지 않는다. 각 batch에서 SHA 우선순위에 따라 5건을 저장하며, 과거 입력만 사용한다. 2k의 짧은 arm은 1k에서 실제 발동한 경우에만 갱신한 history를 다음1k에 사용한다. 미발동은 별도trajectory 없이 부모와 동일한 결과로 보고한다.

sample은 write-origin key와 해당 층의 history 구성을 대표하는 reference key를 구분한다. 최초에는 같은 값이다. **refresh 후에는 reference key와 context dispersion을 현재값으로 갱신하고, write-origin 값은 진단용으로 남긴다**. 이를 통해 동일한 오래된 기준으로 반복 발동하는 것을 피한다. 이 reference 갱신은 첨부 자료에 명시되어 있지 않으므로 본 설계에서 보완한 사양으로 취급한다.

층 L5–L8별로 reference 대비 relative drift 중앙값이 해당 reference 시점의 context dispersion 중앙값을 초과하면 발동한다. median 외에 p90/p95와 norm이 0인 건수도 기록한다. norm이 0이면 ratio를 NA로 처리하고, trigger 모집단에서 제외한 건수를 제시한다. 모든 건이 NA이면 TRIGGER_UNDEFINED로 처리한다. L4는 원리상 불변임을 T0에서 실측으로 확인하고, 일반 refresh 대상에 포함하지 않는다.

H는 **모든 committed occurrence의 현재 key**로부터 chunk 단위로 재구성한다. 중복·superseded·초기 편집 실패를 삭제하지 않는다. 현재 batch를 이중 append하지 않는다. request 집합과 multiplicity를 hash로 대조하고, FP64 chunk 누적 후 native H dtype으로 cast하는 방식을 고정한다. 소규모에서는 전체 Gram의 직접 일치를, 대규모에서는 고정 probe로 `uᵀHu=Σ(kᵀu)²`를 검증한다. FP32 cast 후 probe 상대 차이의 제안 허용값은 1e-5다. 표본 key 일치만으로 재구성 성공을 판정하지 않는다.

refresh 순간에는 W 불변과 동일 prompt 출력 parity를 확인한다. 이후 batch에서 lost/gained와 신규 획득을 측정한다. trigger가 한 번도 발동하지 않으면 발동 없음으로 보고하고, threshold를 사후에 낮추지 않는다. 위의 강제 refresh 분기로 개입 자체의 효과를 별도로 확인한다.

BS10의 배분 진단 T1에는 R을 넣지 않는다. BS10 history 진단은 출발 시 한 번의 강제 refresh만 비교한다. 본체의 자동 검사 주기는 1000 edit이며, 이를 BS10에서 10 batch=100 edit로 잘못 옮기지 않는다. 표본 밀도와 검사 비용도 edit 수 기준으로 명시한다.

**평가 지표와 통계 단위**

주 지표는 **최종 10k의 all-seen PS**다. RS·NS를 함께 제시한다. N건 시점의 분모는 RS=N, PS=2N, NS=10N이며, 최종 시점에는 10,000/20,000/100,000이다. RS/PS는 new의 target-token 평균 NLL이 true보다 작은 것을, NS는 그 반대를 성공으로 판정하며, tie는 실패로 처리한다. 이를 자유 생성 정답률이라고 부르지 않는다.

|평가|필요한 집계|
|---|---|
|쓰기 시점의 획득|각 batch commit 직후의 RS/PS/NS, desired NLL, TF strict|
|최종 보존|같은 case/prompt/target identity로 at-write→각 평가 시점을 대응시키고, lost, gained, retained, 초기 실패를 분리|
|초기 성공을 조건으로 한 보존|`retained / at-write successes`. 각 arm의 성공 집합 차이를 명시하고, 공통 성공 집합에서의 비교도 보조적으로 표시|
|age 효과|birth batch별 및 경과 edit 수별 성적. 첫 500건의 고정 cohort도 추적|
|국소성|NS 외에 W0-correct neighborhood의 lost/gained, true-target NLL, TF strict|
|conflict|공식 분모는 전체 occurrence. latest-active와 superseded를 별도로 집계하되 H에서는 제거하지 않음|
|절대적인 응답 품질|TF token-micro, prompt-macro, full-target strict를 분리. 추가 생성평가를 넣으면 고정 첫1,000건을 사용하며, 기존100건짜리 편집시험을 별도로 유지하지 않음|
|비용|GPUh, z/key/Gram 구성/factorization/solve/history/평가/I/O 분해, peak allocated/reserved GPU memory, host RSS|

BS100의 at-write는 '해당 100건을 한꺼번에 쓴 직후'이며, 100건을 개별적으로 썼을 때의 성공률이 아니다. all-seen 곡선은 시점에 따라 평가 집합이 커지므로 그 차이만을 forgetting으로 보지 않는다.

장기 arm의 각 비교에서는 동일 입력에 대한 paired difference를 산출한다. 보조적인 95% 구간이 필요하면 (subject, relation_id) 단위로 모든 paraphrase/neighborhood/occurrence를 묶어 paired bootstrap을 10,000회 수행한다. 이는 **고정된 모델과 trajectory를 조건으로 한 평가 항목 구성의 구간**이며, 독립적인 편집 순서·재학습·다른 데이터에 대한 불확실성이 아니다. 반복 prompt나 checkpoint를 독립 표본으로 취급해 n을 늘리지 않는다. bootstrap의 cluster 사이에도 의미적 의존성이 남을 수 있으므로 구간만으로 보편적인 유의차를 주장하지 않는다.

같은 순서·같은 난수로 재실행하는 것은 재현성 확인이며 독립 seed가 아니다. 현재 설계로 일반적인 순서 강건성을 주장하지 않는다. 다른 모델·다른 순서·독립 데이터에서의 확인은 본 설계 완료 후 별도의 연구 범위로 둔다.

**사전 판정과 결과별 후속 조치**

아래는 최적화용 threshold가 아니라 연구에서 제안하는 실용적 차이의 기준이다. 기존 성적에 맞춰 사후 변경하지 않는다.

|판정|기준과 처리|
|---|---|
|v1의 주 성공|새 matched MEMIT-H 대비 최종 PS +2pp 이상, RS와 NS 하락이 각각 1pp 이내. 실제 잔차·강도 대조군을 함께 설명|
|작은 개선|PS 차이가 0〜2pp이면 효과 크기를 그대로 보고하고 주 성공과 구분|
|편집과 보존의 상충|PS가 개선되어도 RS 또는 NS가 1pp를 초과해 하락하면 trade-off. 단순한 우월성으로 해석하지 않음|
|v2의 효과|사전에 탐색적 결과로 지정. 동일 상태의 loss 차이와10k의110−100을 모두 보고하되, 성적 확인 후 성공threshold를 만들지 않음. 수치·비용gate는과학적성공과별개|
|v3의 효과|사전에 탐색적 결과로 지정. 10k의111−110과1k/3k/5k/7k강제refresh 결과를전체보고. 발동없음과편집획득감소를보존개선으로해석하지않고 사후채택threshold를만들지않음|
|기전이 다른 경우|energy-matched divisor에서 개선이 재현되면 배분 고유의 효과에 대한 근거는 약함. α나 cosine이 나쁘면 가우스 소거 등 수식의 정밀도가 아니라 실제 전파 가정을 재검토|
|수치 불성립|parity, finite, state chain 등이 실패한 arm의 결과는 확정하지 않음. 수정 후 fresh W0/H0에서 다시 시작하고 실패 비용 기록|

PS +2pp는 과거 MEMIT-H와 BLUE 차이인 10.605pp의 약 19%에 해당하는 규모로 선택한 제안값이며, 달성 예측이 아니다. 비교 차이는 새로운 matched baseline에서 계산한다. BLUE를 넘는 것을 필수 조건으로 삼아 v1의 인과 검증과 섞지 않는다.

**비용 산정과 승인된 재개 정책**

기존 H200 MEMIT-H 10k는 총 12.439444 GPUh였고 z 5.309h, key 1.397h, 평가 5.177h였다. **이전 67–94 GPUh는 활성 예산에서 철회한다.** 산술은 맞지만 SPG 추가 시간을 1–10h로 놓은 가정이 검증되지 않았고, 이번에 BS10 진단도 확대했다.

리뷰의 '1000호출이면 10k Z arm당 150–200 GPUh'는 위험 규모를 보여주는 추정치다. 같은 호출 시간이라면 `5.309×1000/25≈212.4 GPUh`가 z 부분의 단순 환산값이다. prefix 재사용·forward/backward 구성·호출 계수 차이에 따라 달라지므로 150–200도 실측값은 아니다.

새 예산은 T0b에서 요청당 호출 수와 실제 oracle 시간을 측정하고, 상태별 분포와 Z 요청 수를 연결해 계산한다. 여기에 native z, key/Gram/Cholesky/solve, trigger 관측/rebuild, shadow proposal, RAM 복사·복원, 평가·checkpoint I/O를 합한다. zero-step/CONVERGED/precision 정체/상한 도달을 분리해 평균과 P95 시간을 제시한다. 본체의 고유 Z 요청은 12k–22k이며 첫 발동 시점에 따라 달라진다. 강제 refresh와 writer 진단은 native z를 쓴다.

각 anchor 진단은 batch 수가 많아 총 편집 요청에 비례하는 환산만으로 비용을 예측하지 않는다. shadow-only proposal과 실제 commit을 분리한다. 본체·진단·평가·FP64 probe·checkpoint·재시도 예산을 나누어 기록한다. 현재 총 GPUh는 `PENDING_CALIBRATION`이다.

**사용자는 이번 리뷰 후 1,000 edit 간격 임시 checkpoint를 허용하고 완료 후 삭제하는 방식을 선택했다.** 이 실험의 장기 경로 000/100/110/111에 적용하는 명시적 예외이며, 저장소 전체의 기본 방침을 바꾸는 것은 아니다. 물리 경로별 최신 2개만 rolling 유지한다. 공유 prefix는 하나의 payload를 alias로 참조할 수 있고, 아직 필요한 자식·재개 참조가 남아 있는 payload는 먼저 삭제하지 않는다.

1k 경계에서 commit·history 처리와 필요 refresh까지 완료한 재개 상태를 atomic write하고 hash 및 재로드 parity를 확인한다. 최초 R 분기 상태는 부모의 pre-refresh 상태와 자식의 post-refresh 상태를 구분해 기록한다. 해당 경로의 완료·평가·artifact 검증과 의존 분기 처리가 끝나면 임시 checkpoint를 삭제한다. 실패한 경로의 최신 상태는 재개가 끝날 때까지 유지한다. 기존 원본 모델과 연구 결과·로그는 이 삭제 대상이 아니다.

필수 저장 항목은 5개 편집 weight, H, CPU/CUDA/Python/NumPy RNG, context와 native cache, request/commit ledger, 표본 reference·dispersion, trigger 상태·다음 batch 위치, source/config/model/data hash다. 원래 모델/C0는 불변 참조로 결속한다. FP32 weight와 FP32 H만 약 **4.922 GiB/state**이며 표본·metadata가 추가된다. H가 FP64라면 약 8.75 GiB/state다. 최신 2개 공간은 약 두 배이며 동시에 보유하는 경로 수에 따라 증가한다. weight만 저장한 snapshot으로는 history 편집을 정확히 재개할 수 없다.

RAM 분기는 별도로 유지한다. CUDA 작업을 동기화한 뒤 CPU 독립 clone을 만들고, 부모와 자식의 실행 순서 및 RAM 최대 동시 보유 수를 고정한다. 이는 OS process fork가 아닌 명시적인 editor state 분기다. 임시 checkpoint가 정상적으로 생성되기 전의 실패는 여전히 필요한 prefix 재실행을 요구할 수 있다. 저장 기능의 실제 구현·재개 parity 검증은 실행 전 gate이며 이번 작업에서 checkpoint 파일을 만들지는 않았다.

저장 방식과 관계없이 source/config/input hash, batch ledger, 전후 W/H receipt, per-case 평가, solver 종료 상태, layer/refresh 진단, timer와 분모를 기록한다. raw는 ignored 실행 영역에 두고 CPU reducer로 최종 집계를 독립 검산한다.

**리뷰 점검 결과**

|항목|점검 결과와 반영|
|---|---|
|1 SPG 종료와 비용|문제 제기 타당. FP32에서 대부분 1000회라는 비율은 미측정. 수치 교정·상태 구분·200 median gate·관측 완료 P95 예산으로 수정|
|2 R 중복|비개입 관측과 결정론·state parity 조건 아래 타당. 최초 발동 RAM 분기와 NOT_FIRED artifact 공유. 장기 111은 후속 검사 계속|
|3 국소 지표 누락|타당. 모든 write 분기에 R/P/N·past400·W0-correct 이웃 전이·L8 잔차·에너지를 고정. z loss-only의 NA 항목도 명시|
|4 첫 batch PS 차이|MEMIT-H 87%와 AlphaEdit-BLUE 및 AlphaEdit-BLUE-L4-only 95%는 기존 자료와 일치. 일반적인 L4-only 전체가 95%인 것은 아님. W0 진단을 앞세우되 writer만의 원인으로 단정하지 않음|
|5 v2/v3 판정|둘 다 탐색적 결과로 사전 분류. v1의 주 판정과 분리|
|6 작은 진단 축소|Z의 BS1 경로 삭제. 사용자 최신 지시로 000/100을 BS10×100 step의 1,000건 진단으로 변경|
|7 재개 위험|사용자가 1k 간격 임시 checkpoint를 승인. 최신 2개 유지, 완료 검증 후 삭제로 확정|

리뷰 4의 95%는 기존 AlphaEdit-BLUE 계열의 수치다. 같은 자료의 MEMIT-BLUE-L4-only는 84%, MEMIT-BLUE는 92.5%이므로 방법명을 생략하지 않는다. 기존 비교는 target/projector/L2/층/GPU 차이가 있어 '누적 이전부터 차이가 있다'는 관측은 가능하지만 target 위치 또는 writer 배분 중 하나의 원인을 확정하지는 못한다.

G의 PSD 조건, adj에서 G 복원 불안정성, Armijo 강제 수락 오류, 반환점 PG 재계산, trace-capacity와 D norm 구별, refresh reference 재설정은 원래 설계의 수정을 유지한다. 참조 코드 patch 제작은 이번 점검에 포함하지 않는다.

**구현 전에 남은 구체적인 작업**

첨부 patch는 NumPy 참조 구현이며, 기존 runner에 적용하는 것만으로 GPU 실험이 되지는 않는다. torch/FP64 adapter, 추가 lookahead phase를 허용하는 observer, BS100 고정 runner의 BS10 및 RAM 분기 지원, native L8 oracle parity, SPG 종료 처리, context별 key 획득, refresh 재구성, 공통 상태에서의 국소 분기, W0 평가와 독립 reducer 구현이 필요하다. 본 설계는 이 작업들의 완료와 SPG 교정 lock을 해당 단계 진입 조건으로 삼는다. 과학적 실행은 아직 수행하지 않았다.

**참조 자료**

첫 batch 수치와 방법명은 [current-metrics.csv](/mnt/raid5/janghj/.codex/worktrees/odeedit-gh-native-delayed-write-e3-20260924-v1/experiment-reports/servers/server4/blue-native-lifelong-comprehensive-review-2026-09-11-v1/current-metrics.csv)와 [source-config-compatibility.csv](/mnt/raid5/janghj/.codex/worktrees/odeedit-gh-native-delayed-write-e3-20260924-v1/experiment-reports/servers/server4/blue-native-lifelong-comprehensive-review-2026-09-11-v1/source-config-compatibility.csv)를 대조했다. 확인한 E1 패널의 128개 규칙은 [panels.py](/mnt/raid5/janghj/.codex/worktrees/odeedit-gh-native-delayed-write-e3-20260924-v1/project/run_scripts/baseline_mechanism_first/panels.py:9)에 있다. 이는 리뷰가 지칭한 400개 원본과 같다고 확인된 자료가 아니다.

실험군별 실행표는 [cells.csv](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-30-memit-hj-experiment-design-v1/cells.csv), 주요 고정값과 첨부 SHA는 [contract.json](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-30-memit-hj-experiment-design-v1/contract.json)에 함께 기록했다. 이 파일들은 실행 runner가 아니다.

첨부된 [방법 명세](/mnt/raid5/janghj/.codex/attachments/89ff5bff-8739-4966-941f-e38e36dcfb26/2026-09-30-memit-hj-method-ko.md), [allocation.py](/mnt/raid5/janghj/.codex/attachments/89ff5bff-8739-4966-941f-e38e36dcfb26/allocation.py), [zsolve.py](/mnt/raid5/janghj/.codex/attachments/89ff5bff-8739-4966-941f-e38e36dcfb26/zsolve.py), [test_memit_hj.py](/mnt/raid5/janghj/.codex/attachments/89ff5bff-8739-4966-941f-e38e36dcfb26/test_memit_hj.py), [patch의 drift.py](/mnt/raid5/janghj/.codex/attachments/89ff5bff-8739-4966-941f-e38e36dcfb26/memit-hj-method-v1.patch:378)를 확인했다.

기존 결과는 [MEMIT-H job54007 완료 리뷰](/mnt/raid5/janghj/.codex/worktrees/odeedit-gh-native-delayed-write-e3-20260924-v1/experiment-reports/servers/server3/memit-history-fixed10k-20260928-v1/completion-review-r1/report-ko.md)와 [4개 방법 비교](/mnt/raid5/janghj/.codex/worktrees/odeedit-gh-native-delayed-write-e3-20260924-v1/experiment-reports/servers/server3/memit-history-fixed10k-20260928-v1/completion-review-r1/four-method-comparison-ko.md)에, 입력 방침은 [고정 10k 정책](/mnt/raid5/janghj/.codex/worktrees/odeedit-gh-native-delayed-write-e3-20260924-v1/plans/global/fixed-counterfact-10k-policy.md)과 [no-checkpoint 방침](/mnt/raid5/janghj/.codex/worktrees/odeedit-gh-native-delayed-write-e3-20260924-v1/plans/global/2026-09-19-default-no-experiment-checkpoints.md)에 근거한다.

참조한 기존 자료의 Git ref는 `59c96058bb0761f65d2baadcc420ed122d6c732e`이며, 이 작업 cwd의 HEAD는 `ddc178584ef14efd5d4e1271b3c324e3ebd3e443`이다. job54007의 실행 source는 또 다른 `3a904be9261d162239c6b780a62c52f3e59a9142`이므로, 자료 리뷰 시점과 실제 실행 버전을 구분한다.
