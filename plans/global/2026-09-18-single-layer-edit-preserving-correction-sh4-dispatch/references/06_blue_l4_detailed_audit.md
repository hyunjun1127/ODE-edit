# BLUE·L4-only 상세 감사: 편집 충분성, 실제 층별 부담, 보존 비용

2026-09-18 완료(9월 17일 착수). 기존 survey의 후속 감사다. 기존 01–05 문서·원자료를 유지하고, 저장된 평가의 CPU 재집계와 역사적 실행 로그 복구를 추가했다. 새 편집 run, model forward, GPU 호출은 수행하지 않았다. 모든 수치는 별도 표시가 없으면 AlphaEdit 계열이며, layer 번호는 zero-based다.

## 1. 감사의 핵심 판정

**이 fixed10k 실험에서 L4-only는 새로운 편집을 성립시키고 장기간 유지하는 강한 baseline이다. 그러나 원래 지식의 보존은 충분하지 않다. BLUE의 L8 추가는 L4의 parameter 이동량을 거의 줄이지 않았고, 최종 보존 이득으로 이어지지 않았다.**

이 결론을 세 단계로 구분한다.

1. **편집 충분성 — 강하게 관측됨.** L4-only의 at-write RS는 9,993/10,000, final RS는 9,939/10,000이다. 두 paraphrase까지 모두 성공하는 request는 final 9,285개로 BLUE의 9,286개와 거의 같다. 다만 성공하는 문항과 즉시 PS는 다르다.
2. **보존 실패 — 편집 실패와 분리해 관측됨.** L4-only에서 rewrite와 두 paraphrase를 at-write와 final 모두 유지한 9,174개 request 중 7,003개(76.34%)는 W0에서 맞혔던 neighbor를 하나 이상 잃었다. locality 손상을 편집 실패 요청 일부의 문제로 축소할 수 없다.
3. **적응적 배분 — 가능성은 관측됐으나 필요성·최적 정책은 미입증.** 별도의 cold 7-arm에서는 partial L4 뒤의 local L8이 부족한 편집을 보완하면서 full L4보다 낮은 보존 비용을 달성한 batch가 있다. 그러나 PS 손실, guard의 완화, 고정 (.75,.5) chain 부재가 남는다. 이 결과는 fixed10k BLUE의 층별 순간 비용을 측정한 결과와도 다르다.

이번 감사는 'BLUE의 두 층이 필요 없다'거나 'L4의 capacity가 무한하다'는 주장을 지지하지 않는다. **기존 BLUE 정책이 추가 자유도를 효율적인 부담 분담으로 전환했는가**를 묻고, 관측 가능한 부하와 실제 보존 비용을 구별한다.

![편집 충분성과 locality 손실의 분리](/mnt/raid5/janghj/layer_allocation/empirical/blue_audit_v1/figures/fixed10k_quality_locality.png)

그림 A/B는 각각 1,000개 요청의 도착 cohort 평균이다. B의 at-write−W0 차이는 도착 전의 누적 손상과 해당 batch의 손상을 합친 값이다. C는 at-write 성공 문항 중 이후 잃은 비율이며 회복 문항을 차감하지 않는다. D는 서로 다른 두 trained trajectory의 final cohort 비교이고 실행 가능한 routing 결과가 아니다.

## 2. 원자료·비교 조건·측정의 경계

|항목|이번 확인|
|---|---|
|기존 survey 보존|manifest의 219개 파일 + compact collection 14개, hash·크기 233건 모두 일치|
|평가 재집계|14 arms × final/at-write, 각각 1,820,000 prompt rows; W0 130,000 rows|
|기존 결과 대조|최종 numerator 42/42 일치, 기존 remote evaluation receipt 1,414개와 hash 대조|
|BLUE vs L4 설정|기록된 hparams 차이는 layers=[4,8] vs [4] 하나; 두 arm 모두 local-z, L2=1|
|native 비교의 confound|native [4–8]는 terminal-z 분배이고 L2=10; 층 수만 바꾼 비교가 아님|
|공통 stream|fixed10k, batch100×100, seed20260907, 공통 W0에서 시작|
|연산 환경|FP32/eager, TF32 matmul=False, TF32 cuDNN=True로 기록됨|
|관측 지표|RS/PS는 new NLL < true NLL, NS는 true NLL < new NLL; tie는 실패|
|PS 사용|기존 official 평가를 사후 분석; 새 paraphrase target/guard를 만들지 않음|

현재 감사는 compact raw를 다시 계산했다. 원격의 평가 JSON 1,414개를 모두 다시 다운로드했다고 주장하지 않는다. compact의 final identity, case order, shape와 수집 hash를 검증했고, at-write row identity는 기존 extraction receipt의 결속에 의존한다. 원격에서 새로 읽은 실행 로그·commit·native observation은 mechanism 감사의 별도 collection receipt로 구분한다.

검증: [입력 provenance](/mnt/raid5/janghj/layer_allocation/empirical/blue_audit_v1/input_provenance.json), [233건 hash 검사](/mnt/raid5/janghj/layer_allocation/empirical/blue_audit_v1/input_hash_checks.csv), [평가 검증](/mnt/raid5/janghj/layer_allocation/empirical/blue_audit_v1/quant/verification.json).

## 3. L4-only의 편집 충분성을 네 수준으로 확인

### 3.1 새 편집 성립과 이후 유지를 분리

|지표|L4 at-write|L4 final|L4 이후 lost / gained|BLUE at-write|BLUE final|BLUE 이후 lost / gained|
|---|---:|---:|---:|---:|---:|---:|
|RS /10,000|9,993|9,939|55 / 1|9,999|9,888|111 / 0|
|PS /20,000|19,403|19,136|423 / 156|19,529|19,155|516 / 142|
|RS+두 PS 모두 성공 /10,000|9,514|9,285|340 / 111|9,601|9,286|409 / 94|

L4의 at-write 성공 중 RS gross loss는 0.550%, PS gross loss는 2.180%다. BLUE는 각각 1.110%, 2.642%다. 따라서 'forgetting이 없다'는 표현보다 **이 stream에서 편집 유지가 매우 강하다**가 정확하다. RS만 보는 것보다 joint 성공 92.85%를 함께 제시해야 paraphrase 실패를 가리지 않는다.

BLUE는 at-write PS가 126개 더 많지만 이후 순손실도 107개 더 커서 final 차이는 19개로 줄어든다. L4는 즉시 fitting과 장기 유지의 균형에서 강하며, BLUE를 모든 문항에서 지배하지 않는다.

### 3.2 한두 batch의 평균 착시인가

|비교 시점·지표|L4 우세 batch|동률|BLUE 우세 batch|
|---|---:|---:|---:|
|at-write RS|1|92|7|
|at-write PS|22|15|63|
|at-write NS|69|3|28|
|final RS|40|46|14|
|final PS|42|12|46|
|final NS|87|2|11|

at-write RS는 92/100 batch에서 같다. 반면 BLUE의 at-write PS 우위는 넓게 나타나고, final NS는 L4 우세가 87/100 cohort에 걸쳐 나타난다. 이는 '평균 NS 개선이 몇 개 outlier batch 때문'이라는 해석을 약화한다. 그러나 서로 다른 trajectory의 비교이므로 이 표의 우세 arm을 batch마다 조합하면 같은 성능이 나온다는 뜻은 아니다.

원표: [batch별 지표·gross transition](/mnt/raid5/janghj/layer_allocation/empirical/blue_audit_v1/quant/batch_metrics.csv), [배치 이질성](/mnt/raid5/janghj/layer_allocation/empirical/blue_audit_v1/quant/batch_heterogeneity.csv), [joint 유지](/mnt/raid5/janghj/layer_allocation/empirical/blue_audit_v1/quant/joint_retention.csv).

### 3.3 오래된 label 유지가 이득을 부풀리는가

동일 subject–relation group의 마지막 occurrence는 9,783개다. 이 부분의 final RS는 L4 9,743개, BLUE 9,720개로 L4 우위 23개가 남는다. 전체 RS 우위 51개 중 28개는 뒤에 다른 target이 등장하는 212개 request에서 발생한다. 따라서 전체 +.510pp를 전부 현재 지식 유지의 개선으로 설명하면 안 된다.

Joint 성공도 이 분류에 민감하다. 마지막 occurrence에서 L4는 9,159개, BLUE는 9,180개로 L4가 21개 적다. 뒤에 다른 target이 등장하는 212개에서는 L4 121개, BLUE 101개로 20개 많다. 전체 joint 9,285 대 9,286의 거의 동률에는 이 상쇄가 들어 있다.

이 group은 string/metadata 기반이며 동의어·시간조건·복수정답의 의미 검수가 없다. '마지막 occurrence'는 운영상 분류이지 모든 이전 target이 반드시 폐기되어야 한다는 정답 주장이 아니다. [분류별 결과](/mnt/raid5/janghj/layer_allocation/empirical/blue_audit_v1/quant/strata_joint_and_metrics.csv).

### 3.4 다른 singleton과 다른 writer에서도 같은가

|AlphaEdit 정책|final RS|final PS|final NS|final RS+두 PS 모두 성공|
|---|---:|---:|---:|---:|
|BLUE [4,8]|98.880|95.775|63.726|92.86|
|L4|99.390|95.680|65.348|92.85|
|L5|99.340|94.195|62.822|90.61|
|L6|96.830|85.740|58.396|75.85|
|L7|92.400|80.970|56.277|67.29|
|L8|93.960|77.780|54.703|63.14|

L4와 L5의 RS 차이는 5개뿐이다. L4의 강점은 RS 최고점 하나보다 PS·NS·retention의 조합이다. MEMIT에서는 L7 final RS가 84.44%로 L4의 77.22%보다 높다. 따라서 'L4가 본질적으로 최고의 편집 위치'라는 설명은 부족하며 AlphaEdit의 writer/protection과의 상호작용을 고려해야 한다.

또한 이 데이터의 target은 98.37%가 one-token이고 나머지도 two-token이다. 관측된 충분성을 장문 target, 모든 relation, 다른 모델, 무제한 horizon의 capacity 보장으로 확장하지 않는다. 입력 key가 고정되는 성질은 다른 singleton에도 성립하므로 L4의 고유한 우위 원인을 그것 하나로 설명할 수도 없다.

## 4. 편집이 유지되는 요청에서도 locality가 손상되는가

**그렇다. 이 분리는 이번 감사에서 직접 정량화했다.**

L4에서 RS+두 PS가 at-write와 final 모두 성공한 요청은 9,174개다. 그 요청들의 W0-correct neighbor는 81,697개이고, 이 중 final에서 25,005개를 잃었다. 7,003/9,174개 request가 하나 이상의 원래 정답 neighbor를 잃었다. 편집 generalization이 성공한 요청에서도 원래 지식의 손상이 넓게 발생한다.

두 arm 모두 final RS+두 PS가 성공한 동일한 8,938개 request에 제한하면 다음과 같다.

|부분집합|BLUE final NS|L4 final NS|L4−BLUE|
|---|---:|---:|---:|
|양쪽 final joint 성공 /89,380 prompts|62.4513%|64.0635%|+1.6122pp|
|양쪽 at-write와 final joint 유지 /88,070 prompts|62.3470%|63.9787%|+1.6317pp|
|마지막 occurrence 중 양쪽 final joint 성공 /88,520 prompts|62.5700%|64.1358%|+1.5657pp|

따라서 전체 NS 차이가 오직 '한쪽에서 edit 자체가 실패한 요청' 때문에 생긴 것은 아니다. 다만 **이는 성공 여부에 조건을 건 사후 기술통계다.** 두 모델의 confidence·미관측 paraphrase 품질을 동일하게 맞춘 인과 실험이나 online routing 신호가 아니다.

원표: [편집 유지 요청의 locality](/mnt/raid5/janghj/layer_allocation/empirical/blue_audit_v1/quant/quality_conditioned_locality.csv), [공통 성공 부분집합 비교](/mnt/raid5/janghj/layer_allocation/empirical/blue_audit_v1/quant/quality_matched_NS.csv).

## 5. 신규 neighborhood의 손상과 과거 neighborhood forgetting

|AlphaEdit 정책|공통 W0 NS|각 요청의 at-write NS|final NS|at-write→final gross lost / gained|
|---|---:|---:|---:|---:|
|L4|89,212|72,505|65,348|11,547 / 4,390|
|BLUE|89,212|71,762|63,726|12,593 / 4,557|

L4에서는 W0 정답 89,212개 중 final까지 26,583개를 잃고, W0 오답 중 2,719개를 얻었다. 원래 정답 조건부 손실률은 29.798%다. BLUE는 28,306개를 잃어 31.729%다. NS net 점수만 보고 원지식 보존율이라고 부르지 않는다.

새로 도착한 요청에 부속된 neighborhood도 horizon 후반에 더 손상된 상태다.

|정책·cohort|해당 cohort W0 NS|at-write NS|W0 대비 차이|
|---|---:|---:|---:|
|L4 첫 1,000 requests|88.20%|83.66%|−4.54pp|
|L4 마지막 1,000 requests|89.38%|65.36%|−24.02pp|
|BLUE 첫 1,000 requests|88.20%|83.40%|−4.80pp|
|BLUE 마지막 1,000 requests|89.38%|64.37%|−25.01pp|

후반 cohort가 W0에서 더 어려워서 발생한 차이로만 설명되지는 않는다. 그러나 batch entry에서 이 N들을 평가한 값이 없으므로 **과거 batch가 미리 손상시킨 양과 새 write의 순간 손상을 이 표로 분리할 수 없다.** '신규 batch의 edit 비용이 커졌다'보다 '신규 요청의 neighborhood도 도착·쓰기 시점에 더 많이 손상되어 있다'가 정확하다.

한편 at-write 이후 N forgetting은 직접 관측된다. L4는 당시 성공한 N의 15.926%, BLUE는 17.548%를 이후 잃는다. 이 수치는 이후 회복을 차감하지 않은 gross loss다. [cohort 원표](/mnt/raid5/janghj/layer_allocation/empirical/blue_audit_v1/quant/age_extremes.csv).

## 6. B1에서 L8을 추가한 실제 효과

B1의 L4 target 100개 hash, key hash, 최종 L4 weight hash가 모두 같음을 mechanism 감사에서 확인했다. 이 batch는 공통 W0에서 시작하고 이후 두 trajectory의 분기가 누적되기 전이므로, L4-only와 BLUE의 비교가 가장 직접적이다. 비선택 parameter는 실행의 pointer/version guard로 확인했고 전체 byte를 이번에 다시 검사한 것은 아니다.

|B1 at-write|L4-only|BLUE|
|---|---:|---:|
|RS /100|100|100|
|PS /200|190|190|
|RS+두 PS joint /100|92|92|
|NS /1,000|867|867|
|W0-correct N gross loss|21|22|
|W0-incorrect N gain|2|3|
|canonical new NLL 평균|.00228767|.00220643|
|paraphrase new NLL 평균|1.61107831|1.59813712|

L8을 추가해 aggregate RS/PS/NS 성공수는 늘지 않았지만 likelihood는 변했다. N 성공 문항은 하나를 얻고 하나를 잃어 평균이 같아졌다. 따라서 '같은 점수이니 L8은 아무 작용도 하지 않았다'고 해석하면 안 된다. 반대로 이 한 batch에서 보존 개선 효과가 입증된 것도 아니다. PS strict top-1은 129→130으로 달라져 지표별 효과도 구분해야 한다.

B2 이후에는 L4 target과 최종 L4 weight 자체가 달라지므로, 두 arm의 차이를 해당 batch의 L8 순간 효과로 취급할 수 없다. [B1 paired 원표](/mnt/raid5/janghj/layer_allocation/empirical/blue_audit_v1/quant/B1_paired.csv), [100개 case 원표](/mnt/raid5/janghj/layer_allocation/empirical/blue_audit_v1/quant/B1_cases.csv).

## 7. BLUE의 실제 layer별 부담과 target 종료

### 7.1 평균 norm뿐 아니라 100개 batch의 실제 이동을 확인

|측정|BLUE|L4-only|해석|
|---|---:|---:|---|
|L4 평균 batch update Frobenius norm|10.372295|10.381361|BLUE / L4-only = 99.9127%|
|L8 평균 batch update norm|3.951418|해당 없음|추가 이동은 실제로 존재|
|평균 batch 내 L4/L8 norm share|73.8031% / 26.1969%|100% / 0%|크기의 합으로 계산한 비율|
|전체 increment energy의 L4/L8 share|85.4843% / 14.5157%|100% / 0%|각 step norm 제곱 합으로 계산|
|전체 increment energy, L4-only=1|1.167729|1|BLUE가 16.77% 더 큼|

**BLUE의 L4 이동량은 거의 유지되고 L8 이동이 추가된다.** 다만 norm share·energy share는 서로 다르며 어느 쪽도 지식량·보존 비용·편집 기여율이 아니다. 마지막 행은 Σ(batch,layer)‖ΔW‖²이고 ‖W100−W0‖²가 아니다.

이번에 상대 norm의 분모도 구분했다. 역사적 `relative_norm`은 해당 batch **entry weight norm**으로 나눈 값이다. B1에서 W0 norm을 역산해 `step_relative_w0`를 추가했지만, 이것도 한 step의 ΔW/W0다. B1에서는 step과 누적 변화가 같지만, B2 이후 누적 displacement인 ‖Wt−W0‖/‖W0‖는 현재 작은 receipt만으로 재구성할 수 없다. Σ‖ΔW‖나 Σ‖ΔW‖²로 대신하면 누적 방향의 상호작용을 잃는다.

![실제 층별 작업량](/mnt/raid5/janghj/layer_allocation/empirical/blue_audit_v1/figures/fixed10k_layer_work.png)

원표: [300개 layer–batch 측정](/mnt/raid5/janghj/layer_allocation/empirical/blue_audit_v1/mechanism/layer_batch.csv).

### 7.2 L4와 L8에서 실제로 몇 번 최적화했는가

|대상|target 수|0 Adam|1–23 Adam|24 Adam|Adam 총수|
|---|---:|---:|---:|---:|---:|
|L4-only의 L4|10,000|12|0|9,988|239,712|
|BLUE의 L4|10,000|12|1|9,987|239,710|
|BLUE의 L8|10,000|9,570|73|357|9,394|

최대25는 loss 평가 횟수이며 backward 전에 종료를 검사하므로 Adam은 최대24회다. BLUE L8의 95.7%는 초기 loss부터 중단 기준을 통과했다. 이는 full L4 뒤 대부분 요청에 대해 native target 최적화가 추가 변화 없이 종료됐다는 직접 관측이다. PS·N의 충분성이나 L8의 intrinsic capacity를 측정한 것은 아니다.

**이번 fixed10k에는 L8 중간 종료가 73개 있다.** cold 7-arm에서 관측한 'zero 또는24회' 분포를 이 run에 그대로 적용하면 틀린다. 전체 호출 순서·batch·layer·case ID와 종료 norm을 native observation에 결속해 두 실험을 구분했다.

L4-only의 active 9,988개 target은 모두24 Adam을 사용했지만, 마지막 native patch NLL은 평균 약 .001069(로그 반올림 값)이고 decay 항은 평균 .103496이다. 그 9,988개 모두 현재 δ에서 계산한 decay만 .05 이상이다. BLUE L8 active430개의 최종 patch NLL도 평균 약 .000570이다. 따라서 '끝까지 실행했으니 target 학습이 덜 되었다'는 해석은 근거가 부족하다. 반대로 gradient/KKT가 없어 수렴을 보장하지도 못한다. 이 patch NLL은 actual write 이후 canonical/PS NLL과 별개다.

원표: [30,000개 target의 로그 결속·loss](/mnt/raid5/janghj/layer_allocation/empirical/blue_audit_v1/mechanism/target_steps.csv), [종료 집계](/mnt/raid5/janghj/layer_allocation/empirical/blue_audit_v1/mechanism/target_summary.csv).

### 7.3 Zero-step이 많은데 L8 weight는 왜 움직이는가

fixed10k B1에서는 L8 target 100개 중99개가 zero-step이고, case15315 하나가24 Adam을 수행했다. 그 요청의 δ norm은4.828865, native 평균 target probability는 .946566→.999519로 바뀌었다. 이 batch의 실제 L8 weight update norm은1.202092다.

즉 이 경우 큰 update를 zero-step target 자체가 만든 것이 아니다. 남은 nonzero residual이 공동 solve를 통해 weight update를 만든다. 바뀐 weight는 다른 요청·다른 token에도 작용한다. Zero residual 요청의 key도 공동 solve의 Gram에 남으므로 '완료된 요청을 solve에서 삭제해도 같다'고 볼 수 없다. B1의 aggregate RS/PS가 같아도 NLL과 일부 N의 성공 여부가 달라진 이유를 이 구조와 함께 해석해야 한다.

### 7.4 L4의 입력은 같지만 target은 달라지는가

전체100개 batch에서 BLUE와 L4-only의 **L4 prewrite key와 history append key hash가 동일**하다. 같은 batch 내 write 전·append 시점의 L4 key도 동일하다. 이는 down projection 한 곳의 입력이 upstream 고정 계산으로 결정된다는 구조와 일치한다.

반면 B2부터 L4 target hash100개가 전부 달라지고 L4 endpoint도 달라진다. 특히 B2에서는 L4 entry weight와 initial activation norm이 같아도 target이 달라진다. Local-z는 최종 출력까지 loss를 계산하므로 downstream L8 상태가 target 최적화에 들어갈 수 있다는 코드 구조와 정합적이다. 새 intervention replay로 모든 차이의 원인을 분리한 것은 아니다.

따라서 'L4 key가 안정적이다'는 근거는 강화되지만 '두 정책에서 L4 write가 같은 방향이다'는 주장은 성립하지 않는다. Key hash 일치만으로 history tensor의 byte 동일성까지 새로 확인했다고 확대하지 않는다. [trajectory parity 100행](/mnt/raid5/janghj/layer_allocation/empirical/blue_audit_v1/mechanism/trajectory_parity.csv).

### 7.5 W0 scale이나 target 크기가 adaptive signal로 바로 충분한가

현재 저장 신호를 batch별 N 손상과 연결해 기술적 상관을 추가 계산했다. 결과의 N 손상은 at-write NS의 W0 대비 gap이며 **이번 write만의 marginal cost가 아니다.**

|설명 신호 → at-write NS의 W0 대비 손실|Spearman|batch 순위의 선형 추세를 제거한 상관|
|---|---:|---:|
|L4-only의 L4 update norm|.924|.227|
|BLUE의 L4 update norm|.940|.194|
|BLUE의 L8 update norm|.718|−.012|
|BLUE의 L8 active-target 비율|.669|−.034|

시간을 무시하면 norm이나 L8 active 비율이 손상을 잘 설명하는 것처럼 보인다. 그러나 같은 horizon 증가를 상당 부분 공유한다. 이 간단한 rank 추세 조정만으로도 관계가 크게 약해진다. 배치 내용·serial dependence를 통제한 인과 추정이나 held-out 예측은 아니므로 norm의 무용성을 증명하지는 않는다. 다만 강한 raw correlation을 바로 adaptive controller의 근거로 채택하기에는 부족하다.

**같은 layer에서는 W0 norm이 상수이므로, step norm을 W0 norm으로 나누어도 batch 순위가 그대로다.** 이는 300개 관측에서도 확인했다. 누적 displacement를 포함하거나 실제 functional sensitivity를 추가하는 것과 단순 단위 정규화를 구별해야 한다.

원표: [batch 신호 결합표](/mnt/raid5/janghj/layer_allocation/empirical/blue_audit_v1/batch_signal_observations.csv), [36개 상관 결과·한계](/mnt/raid5/janghj/layer_allocation/empirical/blue_audit_v1/cost_signal_correlations.csv), [계산 receipt](/mnt/raid5/janghj/layer_allocation/empirical/blue_audit_v1/cost_signal_receipt.json).

### 7.6 계산 비용과 원자료 접근 한계

역사적 기록의 edit 시간은 L4-only 28,456.63초, BLUE 31,365.36초다. L4-only의 target 시간은 27,532.75초로 edit 시간의96.75%다. Target call 수는 절반이지만 L8의 실제 Adam 총수가 작아 edit 시간이 절반이 되지는 않는다. BLUE 대비 L4-only의 edit 시간 감소율은 약9.27%다. 동시 GPU 사용·실행 상태가 통제된 새 speed benchmark는 아니다.

원 observer가 target 시간을 batch aggregate로 기록했으므로 per-layer target 시간을 직접 복원할 수 없다. BLUE 시간에서 별도 L4 run 시간을 빼서 L8 고유 비용으로 제시하지 않는다. [batch 비용](/mnt/raid5/janghj/layer_allocation/empirical/blue_audit_v1/mechanism/batch_cost.csv).

이번에 점검한 checkpoint 원경로24개는 모두 현재 부재한다. Native target 파일200개는 남아 있지만 해당 tensor를 다운로드·역직렬화하거나 model/activation replay를 수행하지 않았다. 따라서 현재 audit에는 actual key drift 크기, projected spectrum, target→write residual matrix, cumulative delta cosine, 최종 output Jacobian의 새 측정이 없다. Hash·norm·로그·성공률이 각각 지지하는 범위를 구분한다.

## 8. Layer별 보존 비용과 adaptive 배분을 어디까지 말할 수 있나

### 8.1 비용은 동일 entry의 순차 조건부 변화로 정의해야 함

원 fixed10k의 update norm은 layer별 parameter 이동량이다. **layer별 원지식 비용과는 다르다.** 출력 KL·N score의 변화는 upstream/downstream 상호작용을 포함하며, 다른 층의 parameter norm을 더한다고 분해되지 않는다.

같은 batch entry에서 L4를 쓴 상태를 W4, 뒤에 L8을 추가한 상태를 W48이라 하면, 실제 순서에 따른 비용은 다음처럼 정의할 수 있다.

- L4의 조건부 비용: B(W4) − B(Wentry).
- L4 이후 L8의 조건부 비용: B(W48) − B(W4).
- 두 값을 더하면 정확히 B(W48) − B(Wentry)가 된다.

여기서 B는 공통 W0 출력에 대한 고정 보존 패널의 KL처럼 일관된 함수여야 한다. S64는 후보 선택에 사용된 패널이며 선택과 독립적인 test panel은 아니다. 각 항은 음수일 수도 있으며, 순서·prefix·batch·gate에 의존한다. 이를 layer의 고유한 'capacity 소모량'으로 고정하면 안 된다. fixed10k에는 모든 batch의 stage별 B가 없으므로 이 분해를 완성할 수 없다. B1의 N 비교와 별도 cold 실험의 S64 KL을 서로 다른 근거로 구분한다.

### 8.2 기존 cold 7-arm에는 실제 조건부 비용을 계산할 자료가 있음

별도 cold 7-arm의 LD는 같은 batch entry에서 6개 후보를 평가했다. 10개 batch의 총60후보와80개 대비를 재집계했다. 이때 ownN4는 **LD 자신의 entry에서 full L4를 쓴 후보**이며, 별도 N4 arm의 누적 경로와 B2 이후부터 구별된다. E는 native rewrite contexts의 actual-write 평균 target NLL, H는 Past64 평균 NLL, B는 S64의 W0-output KL이다. B는 official Neighborhood 점수가 아니다.

|같은 LD entry의 대비|편집 측면|보존 측면|
|---|---|---|
|full L4 → .75 L4|.75 L4는10/10에서 기존 품질 guard 실패|B는10/10 감소|
|.75 L4 → .75 L4+.5 L8|E는10/10 개선, current strict는ownN4 수준으로 회복|B는10/10 증가|
|.75 L4+.5 L8 vs full L4|guard에 따라5/10 또는9/10 통과|B는10/10 감소|
|.75 L4+.5 L8 → .75 L4+1 L8|E는10/10 더 낮아짐|B는10/10 더 증가|

즉 L4를 줄여 확보한 보존 이득의 일부를 L8의 편집 보완에 쓰는 조합이 관측된다. 이 자료에서 L8의 역할은 partial L4의 손상을 직접 repair하는 것이 아니다. 또한 .75는 원래 정한 gate이므로 L4 norm25% 감소 자체를 adaptive 정책이 발견한 결과라고 부르지 않는다.

가장 단순한 cold B1의 실제 값은 다음과 같다. 이 표의 B1은 앞 절의 fixed10k B1과 **다른 실행 seed·context 설정**이다. 요청 순서는 같은 fixed10k의 앞부분을 사용했다.

|cold LD B1 후보|E|S64 KL B|current strict /100|
|---|---:|---:|---:|
|.75 L4|.18280036|.000951336|99|
|.75 L4+.5 L8|.01241446|.001252665|100|
|.75 L4+1 L8|.01028091|.002032699|100|
|own full L4|.00340461|.001720908|100|

Half L8로 full L4보다 B가 낮아져도 E는 약.00901 높다. 성공 count가 같다는 사실과 실제 likelihood 품질이 같다는 사실을 구별해야 한다. [60후보 전체](/mnt/raid5/janghj/layer_allocation/empirical/blue_audit_v1/allocation/ld_candidate_matrix.csv), [조건부 변화80개](/mnt/raid5/janghj/layer_allocation/empirical/blue_audit_v1/allocation/ld_pairwise_deltas.csv).

### 8.3 품질을 더 엄격히 유지하면 이득이 남는가

|.75 L4+.5 L8의 품질 조건|통과 batch|통과 수|
|---|---|---:|
|기존 E≤max(E_N4,.05)+1e−4, current/past strict ID 보존, H 제약|1,2,3,5,6,7,8,9,10|9/10|
|.05 plateau 제거: E≤E_N4+1e−4, 나머지 동일|2,5,7,9,10|5/10|
|E/H 평균이 ownN4보다 악화하지 않고 strict ID 보존|2,5,7,9,10|5/10|
|canonical 요청별 NLL도 +1e−4 이내 무악화|없음|0/10|

E/H 평균을 완화하지 않아도5개 batch에서는 B가 낮은 조합이 남는다. 따라서 모든 locality 이득이 느슨한 .05 guard에서만 나온다고 단정할 수 없다. 다만 같은 half 후보의 canonical NLL은81–94/100 요청에서 증가한다(ΔNLL>0 기준, 효과 크기 임계값 없음). 평균 native-context 품질 개선이 요청별 canonical 품질이나 PS 무손실을 의미하지 않는다.

B4는 H=.0065990714, ownN4 H=.0064646493이고 기존 허용치는.0065646493이어서 PastMean 조건을 약3.4422e−5 초과한다. 이것은 history guard가 실제 선택을 바꾼 사례다. [품질 조건별 집계](/mnt/raid5/janghj/layer_allocation/empirical/blue_audit_v1/allocation/quality_summary.csv), [선택 재계산](/mnt/raid5/janghj/layer_allocation/empirical/blue_audit_v1/allocation/ld_selector_views.csv).

### 8.4 Adaptive 배분의 필요성이 입증됐는가

아직 아니다. LD는9/10 batch에서 같은(.75,.5)를 선택했다. **fixed(.75,.5) 독립 cold chain이 없다.** L75는(.75,1)이므로 LD와 비교하면 강도 차이와 적응성 차이가 섞인다. .75 L4-only가 실패해도 .8/.85/.9 등의 아직 평가하지 않은 L4-only 강도까지 부족하다는 뜻은 아니다. REFIT4가 추가 layer 없이 NS를 개선한 결과도 다층 배분의 고유 이득을 제한한다.

실제 cold final은 N4 RS99.90/PS96.70/NS80.26, LD99.90/95.45/83.10이다. NS+2.84pp와 PS−1.25pp가 공존한다. 후보별 official PS/N은 없어 선택 과정에서의 matched-quality 우월성을 확인할 수 없다. 위의 강화 guard 결과도 원래 LD 경로 위의 사후 shadow이지, 그 선택으로 새로 진행한 lifelong 결과가 아니다.

두 실행의 whole-order SHA는 같고 cold7은 fixed10k 순서의 first1000이다. 다만 fixed10k는 seed20260907·100batch이고 cold7는 seed20260916·10batch이며 context provenance도 다르다. fixed10k BLUE의 full-L4→full-L8 독립 chain을 cold7에 그대로 재현한 arm은 없다. [protocol 경계](/mnt/raid5/janghj/layer_allocation/empirical/blue_audit_v1/allocation/protocol_boundaries.csv), [allocation 상세 감사](/mnt/raid5/janghj/layer_allocation/empirical/blue_audit_v1/allocation/findings-ko.md).

## 9. 현재 증거가 지지하는 방향과 남은 측정

|주장|감사 판정|
|---|---|
|이 benchmark에서 AlphaEdit local-L4는 강한 편집 baseline이다|지지. at-write, final, joint, cohort, latest-occurrence로 확인|
|L4-only면 편집 forgetting이 전혀 없다|기각. 작지만 RS/PS/joint gross loss 존재|
|L4-only의 locality 문제는 edit 실패 일부에서만 발생한다|기각. 편집 joint 유지 요청에서도 넓은 N 손실|
|BLUE는 L4의 edit 부담을 두 층으로 충분히 분산했다|parameter 이동량 관점에서 지지되지 않음|
|L8이 쓰이면 언제나 손해다|지지되지 않음. BLUE의 at-write PS 우위·배치 이질성·cold 보완 효과 존재|
|고정된 L8보다 batch별 최소비용 배분을 조사할 이유가 있다|지지. 그러나 adaptive 자체의 우위는 아직 미입증|
|W0 대비 delta norm만으로 층별 남은 budget을 정할 수 있다|미입증. functional sensitivity·누적 방향·품질 제약 필요|
|L5–L7도 conditional complement로 효율적이다|미측정. singleton sweep는 같은 prefix의 보완 효과가 아님|
|PS를 target/guard에 넣지 않아도 현재 제약이 PS를 보장한다|지지되지 않음. 본 감사의 official PS는 observer이며 현재 guard가 PS를 보장하지 않음|

남은 핵심 공백은 세 가지다. 첫째, 같은 entry·같은 품질에서 다른 층으로 옮긴 write가 실제 보존 비용을 줄이는지다. 둘째, 이를 알려주는 신호가 z의 크기나 norm보다 실제 출력 손상과 잘 연결되는지다. 셋째, full L4가 후속 local target을 거의 비활성화하는 상태에서도 partial L4 뒤의 유용성을 적은 계산으로 예측할 수 있는지다. 이 감사에서는 미측정 값을 채워 넣거나 새 method 성능으로 제시하지 않는다.

**현재의 연구 근거는 '층 수를 늘려 capacity를 확보한다'보다 '강한 L4 편집 성능을 유지하면서 더 낮은 조건부 보존 비용의 write 조합을 찾는다'에 가깝다.** 최종 RS/PS 성공과 실제 원지식 비용을 함께 확인해야 하며, norm은 기전 분석의 보조 신호로 사용한다.

## 10. 재현·산출물 안내

본 감사의 코드·원자료 receipt·CSV·검증 결과는 [blue_audit_v1 폴더](/mnt/raid5/janghj/layer_allocation/empirical/blue_audit_v1)에 있다. 기존 survey의 manifest는 변경하지 않고 별도의 audit manifest를 둔다. 모든 figure는 저장된 관측에서 만든 정적 과학 도표이며 PNG/PDF를 함께 제공한다.

평가 지표별 자세한 계산과 batch 조건부 분석은 [quant 감사](/mnt/raid5/janghj/layer_allocation/empirical/blue_audit_v1/quant/findings-ko.md), 역사적 실행 로그와 parameter 이동은 [mechanism 감사](/mnt/raid5/janghj/layer_allocation/empirical/blue_audit_v1/mechanism/REPORT.md), cold local-z의 같은-entry 후보 비교는 [allocation 감사](/mnt/raid5/janghj/layer_allocation/empirical/blue_audit_v1/allocation/findings-ko.md)에 각각 보존한다. 이 구분은 fixed10k와 cold 7-arm의 seed·horizon·selection protocol을 섞지 않기 위한 것이다. [재현 안내](/mnt/raid5/janghj/layer_allocation/empirical/blue_audit_v1/README.md), [최종 통합 검증](/mnt/raid5/janghj/layer_allocation/empirical/blue_audit_v1/validation_receipt.json), [새 산출물 manifest](/mnt/raid5/janghj/layer_allocation/empirical/blue_audit_v1/audit_manifest.json).
