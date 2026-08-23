# P1R52 target-timescale B100 — same-horizon Z0/Z1 상세 결과

> **범위:** `SAME_HORIZON_ONLY_Z0_COARSE_VS_Z1_REFINE`. 두 cell은 동일 `T_z=1`, 동일 B1 100 requests, 동일 W0·writer·cache·evaluator를 사용한다. Z0는 `m=1, dt=1/8`, Z1은 `m=2, dt=1/16`이다. Z15/Z20/Z30 결과는 별도 longer-time 보고 대상으로 이 분석과 manifest에 포함하지 않았다.

> **정정판 v2:** v1의 CSV에는 존재했지만 본문에서 생략된 Native post-W 측정, z→W gap, success/accuracy/strict/locality와 Native compute를 명시적으로 반영했다. 실험·raw identity·target-timescale 결론은 변경하지 않았다.

## 결론

- K8 accepted-z Rewrite mean NLL은 `0.075409 → 0.011389`로 `-0.064020` 감소했다. paired prompt 기준 Z1 승리는 `95/100`이다.
- K8 accepted-z Rephrase mean은 `1.424842 → 1.384642`로 `-0.040200` 감소했고 Z1 승리는 `142/200`이다. 다만 p90은 `+0.388523`로 악화했다.
- Rewrite 개선은 post-W에도 유지됐지만, Rephrase post-W mean은 `2.138524 → 2.315501`로 `+0.176977` 악화했다. accepted-z 개선이 writer의 Rephrase endpoint 개선으로 전달되지 않았다는 사실만 말할 수 있으며 원인 인과는 주장하지 않는다.
- 별도 Native에서도 z와 W는 같지 않았다. AlphaEdit post-W mean NLL은 Rewrite `0.001915`, Rephrase `1.940613`; MEMIT는 Rewrite `0.342457`, Rephrase `2.974135`였다. 따라서 Native 비교도 direct-z만으로 결론내리지 않고 실제 W endpoint를 함께 본다.
- 따라서 같은 horizon에서 `dt`를 절반으로 줄인 것은 coarse-resolution 병목의 일부를 해소했다. 그러나 Rephrase tail과 post-W transfer는 남아 있어 resolution 하나만으로 전체 병목이 해소됐다고 볼 수 없다.

## 1. 실행·불변식

|cell|T_z|m|dt|field eval|writer|clamp|PRIMARY/RESCUE/CURRENT|Slurm|post-model|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|Z0-COARSE|1.0|1|0.1250|8|8|27/800 (3.38%)|788/7/5|1151s|1103.5s|
|Z1-REFINE|1.0|2|0.0625|16|8|13/1600 (0.81%)|1593/7/0|1297s|1250.0s|

두 cell 모두 terminal valid, writer K1–K8 정확히 8회, cache entry reuse 8회·K8 뒤 append 1회, heldout K1/K4/K8만 9 evaluator groups, W0 pointer/bytes restore exact, FULL-FP32다. 각 target-timescale cell 내부 Native 실행은 0이며, 본문의 Native 수치는 별도 external reference jobs에서 왔다.

## 2. K1/K4/K8 accepted-z NLL

|K|cell|Rewrite mean/median/p90/max|Rephrase mean/median/p90/max|Rewrite success|Rephrase success/strict|
|---:|---|---|---|---:|---:|
|1|Z0-COARSE|8.171930/7.834708/14.006475/15.684274|8.136112/7.970822/13.319817/17.740170|21/100|45/200 · 18/100|
|1|Z1-REFINE|7.447817/7.362202/12.324613/14.718564|7.685157/7.643133/12.631503/17.183191|22/100|46/200 · 19/100|
|4|Z0-COARSE|0.670624/0.276079/1.221521/7.558275|2.171148/1.079140/5.791178/12.471620|99/100|191/200 · 93/100|
|4|Z1-REFINE|0.074863/0.047092/0.162490/0.660646|1.570408/0.422501/4.797185/10.822572|100/100|193/200 · 95/100|
|8|Z0-COARSE|0.075409/0.018222/0.049245/4.836492|1.424842/0.237647/4.197616/12.096726|100/100|195/200 · 96/100|
|8|Z1-REFINE|0.011389/0.008625/0.024858/0.073858|1.384642/0.166186/4.586139/11.199921|100/100|194/200 · 96/100|

K1은 공통 W0에서의 가장 깨끗한 resolution 비교다. Z1은 K1부터 Rewrite 99/100, Rephrase 185/200 prompts에서 Z0보다 낮은 NLL이었다. K8에서는 각각 95/100, 142/200으로 advantage가 유지되지만 Rephrase upper tail은 단조 개선이 아니다.

## 3. K8 z→W 전달과 locality

|cell|endpoint|Rewrite mean|Rephrase mean|Rephrase success/strict|locality|
|---|---|---:|---:|---:|---:|
|Z0-COARSE|accepted-z|0.075409|1.424842|195/200 · 96/100|889/1000|
|Z0-COARSE|pre-W|0.130960|2.205675|185/200 · 87/100|889/1000|
|Z0-COARSE|post-W|0.079716|2.138524|185/200 · 87/100|889/1000|
|Z1-REFINE|accepted-z|0.011389|1.384642|194/200 · 96/100|892/1000|
|Z1-REFINE|pre-W|0.013357|2.330200|178/200 · 80/100|892/1000|
|Z1-REFINE|post-W|0.011346|2.315501|178/200 · 80/100|892/1000|

K8 W−z Rephrase gap은 Z0 `+0.713682`, Z1 `+0.930859`이다. Rewrite gap은 각각 `+0.004307`, `-0.000043`이다.

## 4. Clamp·selection·energy

Z0는 clamp `27/800`(3.375%), Z1은 `13/1600`(0.8125%)이다. Z0의 선택은 PRIMARY/RESCUE/CURRENT `788/7/5`, Z1은 `1593/7/0`이다. Z1은 field evaluation을 2배 사용했지만 clamp-removed energy 합이 `17682.397 → 1725.868`로 감소했다. 이는 관측 association이며 개별 기제의 인과 증명은 아니다.

## 5. Native accepted-z 및 post-W reference

사용자 지시에 따라 동일 sealed B1/W0/evaluator에서 Official AlphaEdit와 Official MEMIT를 별도 실행했다. 두 실행 모두 accepted-z와 실제 weight materialization 직후 post-W를 같은 evaluator로 측정했으며, target-timescale 설정 선택에 영향 0이다.

|Native|endpoint|prompt|mean|median|p90|max|success/strict|accuracy/strict|locality|
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
|OFFICIAL-ALPHAEDIT|accepted-z|rewrite|0.001362|0.000659|0.002425|0.028182|100/100 · 100/100|100/100 · 100/100|872/1000|
|OFFICIAL-ALPHAEDIT|accepted-z|rephrase|1.103564|0.064396|3.287930|13.755802|197/200 · 98/100|151/200 · 63/100|872/1000|
|OFFICIAL-ALPHAEDIT|post-W|rewrite|0.001915|0.000728|0.002553|0.075494|100/100 · 100/100|100/100 · 100/100|872/1000|
|OFFICIAL-ALPHAEDIT|post-W|rephrase|1.940613|0.309863|5.383312|16.056799|184/200 · 87/100|122/200 · 44/100|872/1000|
|OFFICIAL-MEMIT|accepted-z|rewrite|0.000885|0.000596|0.001915|0.005452|100/100 · 100/100|100/100 · 100/100|886/1000|
|OFFICIAL-MEMIT|accepted-z|rephrase|1.092572|0.065640|3.515177|13.546903|197/200 · 98/100|150/200 · 62/100|886/1000|
|OFFICIAL-MEMIT|post-W|rewrite|0.342457|0.004158|0.076267|12.670905|98/100 · 98/100|96/100 · 96/100|886/1000|
|OFFICIAL-MEMIT|post-W|rephrase|2.974135|1.196294|8.534275|17.056204|164/200 · 76/100|100/200 · 36/100|886/1000|

|Native|prompt|post-W − accepted-z mean/median/p90/max|
|---|---|---:|
|OFFICIAL-ALPHAEDIT|rewrite|+0.000553/+0.000069/+0.000128/+0.047312|
|OFFICIAL-ALPHAEDIT|rephrase|+0.837049/+0.245466/+2.095382/+2.300997|
|OFFICIAL-MEMIT|rewrite|+0.341572/+0.003563/+0.074352/+12.665454|
|OFFICIAL-MEMIT|rephrase|+1.881562/+1.130654/+5.019098/+3.509301|

AlphaEdit의 z→W mean gap은 Rewrite `+0.000553`, Rephrase `+0.837049`였고, MEMIT는 Rewrite `+0.341572`, Rephrase `+1.881562`였다. 즉 Native direct-z가 낮은 NLL을 보였더라도 materialized W의 성능은 별도이며, 특히 MEMIT Rewrite와 두 방법 Rephrase에서 전달 손실이 관측됐다.

|Native|total|z 생성|writer core|z eval|W eval|restore|W0 restore|
|---|---:|---:|---:|---:|---:|---:|---:|
|OFFICIAL-ALPHAEDIT|383.4s|259.6s|72.7s|21.4s|18.5s|5.3s|True|
|OFFICIAL-MEMIT|355.6s|260.0s|44.7s|21.4s|18.5s|5.3s|True|

Native 대비 accepted-z mean/median/p90/max gap과 closure는 `native-gap-closure.csv`, prompt별 paired 차이와 승패는 `native-paired-nll.csv`, accepted-z와 post-W 전체 분포는 `native-reference-endpoints.csv`에 기록했다. 원문 contract mean과 새 실행 mean의 차이도 함께 남겨 재현 identity를 점검했다.

## 6. 계산량과 overhead

Z1은 field evaluation `16`으로 Z0의 2배다. post-model wall은 `1103.5s → 1250.0s`(1.133×), Slurm elapsed는 `1151s → 1297s`(1.127×, +146s)였다. GPU peak allocated는 두 cell 모두 `35.42 GiB`다.

## 7. 해석 경계

- 이 보고서는 same-horizon resolution 효과만 다룬다. Z15/Z20/Z30 결과 영향은 0이다.
- B100×1 탐색이므로 final T_z, promotion, production setting을 선정하지 않는다.
- Native 두 실행은 `USER_DIRECTED_NATIVE_REFERENCE_RUN_OVERRIDE`에 따른 external reference 통계 보강이다. target-timescale parameter·selection influence는 0이고 schedule-matched Native 또는 Native-K8 주장은 하지 않는다.
- resource cap은 제출 당시 2였고 사용자 지시로 실행 중 4로 상향됐다. 이는 동시성만 바꾸었고 cell science/result selection에는 영향 0이다.
- 제외된 pre-model/interface 기술 시도는 최종 통합 보고서에서 한 번에 provenance로 정리하며, 이 scientific denominator에는 포함하지 않는다.

세부 수치는 `microstep-trajectory.csv`, `k1-k4-k8-endpoints.csv`, `paired-accepted-z-nll.csv`, `writer-transfer-locality.csv`, `native-reference-endpoints.csv`, `native-paired-nll.csv`, `native-gap-closure.csv`, `compute.csv`, `native-compute.csv`에 있다.
