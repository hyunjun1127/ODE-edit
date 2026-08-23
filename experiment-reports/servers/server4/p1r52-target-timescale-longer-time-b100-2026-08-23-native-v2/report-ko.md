# P1R52 target-timescale B100 — longer target time 상세 결과

> **범위:** `FIXED_DT_LONGER_TARGET_TIME_Z1_Z15_Z20_Z30`. 네 cell은 동일 `dt=1/16`, sealed B1 100 requests, W0, C3 K8 writer, cache, evaluator를 공유하고 `T_z=1/1.5/2/3`만 다르다. Z0 coarse-resolution은 이 비교와 manifest에서 제외했다.

> **정정판 v2:** v1의 CSV에는 존재했지만 본문에서 생략된 Native post-W 측정, z→W gap, success/accuracy/strict/locality와 Native compute를 명시적으로 반영했다. 실험·raw identity·longer-time 결론은 변경하지 않았다.

## 핵심 관측

- K8 accepted-z rewrite mean NLL은 Z1-REFINE `0.011389` → Z15 `0.006024` → Z20 `0.005008` → Z30 `0.002264`였다. 이 B1에서 최저 관측값은 `Z30`지만 final T_z 선택 근거로 사용하지 않는다.
- K8 accepted-z rephrase mean NLL은 Z1-REFINE `1.384642` → Z15 `1.299307` → Z20 `1.319775` → Z30 `1.266131`였다. 이 B1에서 최저 관측값은 `Z30`지만 final T_z 선택 근거로 사용하지 않는다.
- 증가 시간의 marginal gain은 단조라고 가정하지 않았다. `consecutive-marginal-gain.csv`에 accepted-z와 post-W의 mean/median/p90/max를 K1/K4/K8별로 기록했다.
- 아래 Native 비교는 사용자 승인 별도 Official 실행의 external reference이며 target-timescale 설정 선택에 영향 0이다. AlphaEdit post-W mean NLL은 Rewrite `0.001915`, Rephrase `1.940613`; MEMIT는 Rewrite `0.342457`, Rephrase `2.974135`로 direct-z와 materialized W 사이의 차이가 확인됐다.

## 1. 실행·clamp·selection·overhead

|cell|T_z|m|field eval|clamp|PRIMARY/RESCUE/CURRENT|Slurm|post-model|
|---|---:|---:|---:|---:|---:|---:|---:|
|Z1-REFINE|1.0|2|16|13/1600 (0.81%)|1593/7/0|1297s|1250.0s|
|Z15|1.5|3|24|31/2400 (1.29%)|2393/7/0|1445s|1388.8s|
|Z20|2.0|4|32|127/3200 (3.97%)|3197/3/0|1638s|1584.9s|
|Z30|3.0|6|48|58/4800 (1.21%)|4790/8/2|1894s|1842.8s|

모든 cell은 terminal valid, writer 8회, cache reuse 8·append 1, W0 pointer/bytes exact restore, FULL-FP32였다. configured field eval은 16/24/32/48로 schedule과 일치했다.

## 2. K1/K4/K8 accepted-z NLL

|K|cell|Rewrite mean/median/p90/max|Rephrase mean/median/p90/max|Rewrite success|Rephrase success/strict|
|---:|---|---|---|---:|---:|
|1|Z1-REFINE|7.447817/7.362202/12.324613/14.718564|7.685157/7.643133/12.631503/17.183191|22/100|46/200 · 19/100|
|1|Z15|4.626818/4.140674/8.648135/11.721231|5.552878/5.075745/10.513036/15.135984|55/100|98/200 · 37/100|
|1|Z20|2.059115/1.656152/4.118508/9.193537|3.381675/2.655420/7.341912/13.965591|95/100|174/200 · 81/100|
|1|Z30|0.430041/0.181564/0.704281/7.873769|2.041999/0.926993/5.811227/11.551070|100/100|191/200 · 94/100|
|4|Z1-REFINE|0.074863/0.047092/0.162490/0.660646|1.570408/0.422501/4.797185/10.822572|100/100|193/200 · 95/100|
|4|Z15|0.013977/0.009658/0.030086/0.078088|1.347564/0.172630/4.369039/11.074506|100/100|196/200 · 96/100|
|4|Z20|0.014097/0.010598/0.028515/0.083965|1.383515/0.174717/4.631283/11.157700|100/100|195/200 · 96/100|
|4|Z30|0.004891/0.003825/0.009417/0.026488|1.307367/0.133072/4.488708/11.561758|100/100|195/200 · 96/100|
|8|Z1-REFINE|0.011389/0.008625/0.024858/0.073858|1.384642/0.166186/4.586139/11.199921|100/100|194/200 · 96/100|
|8|Z15|0.006024/0.004299/0.011291/0.034824|1.299307/0.124859/4.337726/11.223797|100/100|195/200 · 96/100|
|8|Z20|0.005008/0.003722/0.009430/0.032919|1.319775/0.113560/4.320387/11.325514|100/100|195/200 · 96/100|
|8|Z30|0.002264/0.001895/0.005074/0.007708|1.266131/0.107524/4.439338/11.723059|100/100|194/200 · 95/100|

## 3. K8 longer-time paired 및 marginal 변화

|cell vs Z1|prompt|delta mean/median/p90/max|longer win|Z1 win|
|---|---|---|---:|---:|
|Z15|rewrite|-0.005365/-0.003617/-0.000665/+0.001882|99/100|1/100|
|Z15|rephrase|-0.085335/-0.010487/+0.081787/+1.416773|156/200|44/200|
|Z20|rewrite|-0.006381/-0.004203/-0.000987/-0.000004|100/100|0/100|
|Z20|rephrase|-0.064867/-0.015087/+0.090278/+1.254547|160/200|40/200|
|Z30|rewrite|-0.009125/-0.006248/-0.001815/-0.000006|100/100|0/100|
|Z30|rephrase|-0.118511/-0.023491/+0.081680/+2.382028|162/200|38/200|

|increment|endpoint|prompt|K8 mean delta|K8 median delta|K8 p90 delta|
|---|---|---|---:|---:|---:|
|Z1-REFINE→Z15|accepted_z|rewrite|-0.005365|-0.004326|-0.013567|
|Z1-REFINE→Z15|accepted_z|rephrase|-0.085335|-0.041327|-0.248413|
|Z1-REFINE→Z15|post_writer_W|rewrite|-0.005337|-0.004327|-0.013497|
|Z1-REFINE→Z15|post_writer_W|rephrase|-0.041680|-0.016684|-0.188813|
|Z15→Z20|accepted_z|rewrite|-0.001016|-0.000577|-0.001861|
|Z15→Z20|accepted_z|rephrase|+0.020468|-0.011299|-0.017339|
|Z15→Z20|post_writer_W|rewrite|-0.001018|-0.000593|-0.002025|
|Z15→Z20|post_writer_W|rephrase|-0.015384|+0.009415|-0.205683|
|Z20→Z30|accepted_z|rewrite|-0.002744|-0.001827|-0.004356|
|Z20→Z30|accepted_z|rephrase|-0.053645|-0.006036|+0.118951|
|Z20→Z30|post_writer_W|rewrite|-0.002733|-0.001827|-0.004217|
|Z20→Z30|post_writer_W|rephrase|+0.011667|+0.008203|+0.217729|

음수는 longer cell의 NLL 감소다. marginal은 Rewrite에서 대체로 감소하지만 Rephrase mean/tail과 post-W는 구간별 비단조다. 이는 raw association이며 최적 T_z 선택 또는 인과 기제 판정이 아니다.

## 4. K8 accepted-z → post-W 전달

|cell|Rewrite z/W/gap|Rephrase z/W/gap|z locality|W locality|
|---|---:|---:|---:|---:|
|Z1-REFINE|0.011389/0.011346/-0.000043|1.384642/2.315501/+0.930859|892/1000|892/1000|
|Z15|0.006024/0.006009/-0.000015|1.299307/2.273821/+0.974514|891/1000|891/1000|
|Z20|0.005008/0.004991/-0.000017|1.319775/2.258437/+0.938662|892/1000|892/1000|
|Z30|0.002264/0.002258/-0.000006|1.266131/2.270103/+1.003973|893/1000|894/1000|

## 5. Native accepted-z 및 post-W reference

|method|endpoint|prompt|mean|median|p90|max|success/strict|accuracy/strict|locality|
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
|OFFICIAL-ALPHAEDIT|accepted-z|rewrite|0.001362|0.000659|0.002425|0.028182|100/100 · 100/100|100/100 · 100/100|872/1000|
|OFFICIAL-ALPHAEDIT|accepted-z|rephrase|1.103564|0.064396|3.287930|13.755802|197/200 · 98/100|151/200 · 63/100|872/1000|
|OFFICIAL-ALPHAEDIT|post-W|rewrite|0.001915|0.000728|0.002553|0.075494|100/100 · 100/100|100/100 · 100/100|872/1000|
|OFFICIAL-ALPHAEDIT|post-W|rephrase|1.940613|0.309863|5.383312|16.056799|184/200 · 87/100|122/200 · 44/100|872/1000|
|OFFICIAL-MEMIT|accepted-z|rewrite|0.000885|0.000596|0.001915|0.005452|100/100 · 100/100|100/100 · 100/100|886/1000|
|OFFICIAL-MEMIT|accepted-z|rephrase|1.092572|0.065640|3.515177|13.546903|197/200 · 98/100|150/200 · 62/100|886/1000|
|OFFICIAL-MEMIT|post-W|rewrite|0.342457|0.004158|0.076267|12.670905|98/100 · 98/100|96/100 · 96/100|886/1000|
|OFFICIAL-MEMIT|post-W|rephrase|2.974135|1.196294|8.534275|17.056204|164/200 · 76/100|100/200 · 36/100|886/1000|

|method|prompt|post-W − accepted-z mean/median/p90/max|
|---|---|---:|
|OFFICIAL-ALPHAEDIT|rewrite|+0.000553/+0.000069/+0.000128/+0.047312|
|OFFICIAL-ALPHAEDIT|rephrase|+0.837049/+0.245466/+2.095382/+2.300997|
|OFFICIAL-MEMIT|rewrite|+0.341572/+0.003563/+0.074352/+12.665454|
|OFFICIAL-MEMIT|rephrase|+1.881562/+1.130654/+5.019098/+3.509301|

AlphaEdit의 z→W mean gap은 Rewrite `+0.000553`, Rephrase `+0.837049`였고, MEMIT는 Rewrite `+0.341572`, Rephrase `+1.881562`였다. Native direct-z와 materialized W를 구분해 해석해야 하며 이 전달 손실은 target-time 설정 선택에 사용하지 않았다.

|method|total|z 생성|writer core|z eval|W eval|restore|W0 restore|
|---|---:|---:|---:|---:|---:|---:|---:|
|OFFICIAL-ALPHAEDIT|383.4s|259.6s|72.7s|21.4s|18.5s|5.3s|True|
|OFFICIAL-MEMIT|355.6s|260.0s|44.7s|21.4s|18.5s|5.3s|True|

## 6. 해석 경계

- B100×1 exploratory ablation이며 이 결과만으로 final T_z를 선택하거나 promotion하지 않는다.
- fixed-dt longer-time 효과만 다룬다. Z0 coarse resolution은 과학 분모와 raw manifest에 포함하지 않았다.
- Native는 canonical one-shot external reference다. K8/schedule-matched Native, causal target-time arm, continuous-ODE convergence를 주장하지 않는다.
- heldout은 K1/K4/K8 terminal observation-only이며 controller influence 0이다.
- 세부 request/prompt/microstep/marginal/compute 수치는 동봉 CSV에 있으며 누락값은 추정하지 않았다.
