# P1R52 Soft Sequential AlphaCache-ON / Structural-H OFF — standalone factual v5

> v4 sealed raw-free tables를 변경하지 않고 가독성 구조로 재배열한 append-only v5 판본입니다. 모델·evaluator·GPU·Slurm 재실행은 0입니다.

## 지표 정의

- **EFF** ≡ `rewrite_success`: rewrite prompt에서 target-new mean NLL < target-true mean NLL.
- **GEN** ≡ `paraphrase_success` (`rephrase_success` alias): paraphrase prompt별 NLL preference 성공.
- **GEN-strict**: request의 모든 paraphrase가 성공한 strict request count/rate.
- **Rewrite Acc / Rephrase Acc**: strict suffix-token argmax accuracy; EFF/GEN success와 별도입니다.
- **LOC**: locality-preservation accuracy. `entry-pre`는 actual W_(b-1), `post`는 W_e, `final`은 W10입니다.
- z8/W8/gap은 해당 batch의 target oracle full-six / accepted BF16 physical full-six / W8−z8입니다. Native target oracle 부재는 `NOT_RECORDED`입니다.

## 절대값: W0 + 각 B1–B10 entry-pre / immediate-post / final-W10 + B100

|Panel|Arm|EFF|Rewrite Acc|GEN|GEN-strict|Rephrase Acc|Acc-strict|LOC|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|W0 / B1 entry|P1R52 Soft AlphaCache-ON / Structural-H OFF|2/10 (20.0%)|0/10 (0.0%)|3/20 (15.0%)|1/10 (10.0%)|0/20 (0.0%)|0/10 (0.0%)|93/100 (93.0%)|
|W0 / B1 entry|Native AlphaEdit Sequential|2/10 (20.0%)|0/10 (0.0%)|3/20 (15.0%)|1/10 (10.0%)|0/20 (0.0%)|0/10 (0.0%)|93/100 (93.0%)|
|B1 entry-pre|P1R52 Soft AlphaCache-ON / Structural-H OFF|2/10 (20.0%)|0/10 (0.0%)|3/20 (15.0%)|1/10 (10.0%)|0/20 (0.0%)|0/10 (0.0%)|93/100 (93.0%)|
|B1 entry-pre|Native AlphaEdit Sequential|2/10 (20.0%)|0/10 (0.0%)|3/20 (15.0%)|1/10 (10.0%)|0/20 (0.0%)|0/10 (0.0%)|93/100 (93.0%)|
|B1 immediate-post|P1R52 Soft AlphaCache-ON / Structural-H OFF|10/10 (100.0%)|10/10 (100.0%)|19/20 (95.0%)|9/10 (90.0%)|10/20 (50.0%)|3/10 (30.0%)|92/100 (92.0%)|
|B1 immediate-post|Native AlphaEdit Sequential|10/10 (100.0%)|10/10 (100.0%)|19/20 (95.0%)|9/10 (90.0%)|13/20 (65.0%)|4/10 (40.0%)|92/100 (92.0%)|
|B1 final-W10|P1R52 Soft AlphaCache-ON / Structural-H OFF|10/10 (100.0%)|10/10 (100.0%)|19/20 (95.0%)|9/10 (90.0%)|9/20 (45.0%)|2/10 (20.0%)|95/100 (95.0%)|
|B1 final-W10|Native AlphaEdit Sequential|9/10 (90.0%)|4/10 (40.0%)|16/20 (80.0%)|8/10 (80.0%)|7/20 (35.0%)|1/10 (10.0%)|79/100 (79.0%)|
|B2 entry-pre|P1R52 Soft AlphaCache-ON / Structural-H OFF|0/10 (0.0%)|0/10 (0.0%)|1/20 (5.0%)|0/10 (0.0%)|0/20 (0.0%)|0/10 (0.0%)|86/100 (86.0%)|
|B2 entry-pre|Native AlphaEdit Sequential|0/10 (0.0%)|0/10 (0.0%)|1/20 (5.0%)|0/10 (0.0%)|0/20 (0.0%)|0/10 (0.0%)|87/100 (87.0%)|
|B2 immediate-post|P1R52 Soft AlphaCache-ON / Structural-H OFF|10/10 (100.0%)|10/10 (100.0%)|17/20 (85.0%)|7/10 (70.0%)|9/20 (45.0%)|2/10 (20.0%)|86/100 (86.0%)|
|B2 immediate-post|Native AlphaEdit Sequential|10/10 (100.0%)|10/10 (100.0%)|20/20 (100.0%)|10/10 (100.0%)|18/20 (90.0%)|8/10 (80.0%)|82/100 (82.0%)|
|B2 final-W10|P1R52 Soft AlphaCache-ON / Structural-H OFF|10/10 (100.0%)|10/10 (100.0%)|17/20 (85.0%)|7/10 (70.0%)|10/20 (50.0%)|3/10 (30.0%)|83/100 (83.0%)|
|B2 final-W10|Native AlphaEdit Sequential|8/10 (80.0%)|5/10 (50.0%)|17/20 (85.0%)|8/10 (80.0%)|13/20 (65.0%)|5/10 (50.0%)|74/100 (74.0%)|
|B3 entry-pre|P1R52 Soft AlphaCache-ON / Structural-H OFF|2/10 (20.0%)|0/10 (0.0%)|2/20 (10.0%)|1/10 (10.0%)|0/20 (0.0%)|0/10 (0.0%)|88/100 (88.0%)|
|B3 entry-pre|Native AlphaEdit Sequential|2/10 (20.0%)|0/10 (0.0%)|2/20 (10.0%)|1/10 (10.0%)|0/20 (0.0%)|0/10 (0.0%)|87/100 (87.0%)|
|B3 immediate-post|P1R52 Soft AlphaCache-ON / Structural-H OFF|10/10 (100.0%)|10/10 (100.0%)|17/20 (85.0%)|7/10 (70.0%)|8/20 (40.0%)|2/10 (20.0%)|87/100 (87.0%)|
|B3 immediate-post|Native AlphaEdit Sequential|10/10 (100.0%)|10/10 (100.0%)|19/20 (95.0%)|9/10 (90.0%)|10/20 (50.0%)|3/10 (30.0%)|86/100 (86.0%)|
|B3 final-W10|P1R52 Soft AlphaCache-ON / Structural-H OFF|10/10 (100.0%)|10/10 (100.0%)|17/20 (85.0%)|7/10 (70.0%)|8/20 (40.0%)|2/10 (20.0%)|86/100 (86.0%)|
|B3 final-W10|Native AlphaEdit Sequential|9/10 (90.0%)|7/10 (70.0%)|15/20 (75.0%)|6/10 (60.0%)|8/20 (40.0%)|2/10 (20.0%)|75/100 (75.0%)|
|B4 entry-pre|P1R52 Soft AlphaCache-ON / Structural-H OFF|0/10 (0.0%)|0/10 (0.0%)|0/20 (0.0%)|0/10 (0.0%)|0/20 (0.0%)|0/10 (0.0%)|98/100 (98.0%)|
|B4 entry-pre|Native AlphaEdit Sequential|0/10 (0.0%)|0/10 (0.0%)|0/20 (0.0%)|0/10 (0.0%)|0/20 (0.0%)|0/10 (0.0%)|97/100 (97.0%)|
|B4 immediate-post|P1R52 Soft AlphaCache-ON / Structural-H OFF|10/10 (100.0%)|10/10 (100.0%)|15/20 (75.0%)|7/10 (70.0%)|10/20 (50.0%)|4/10 (40.0%)|92/100 (92.0%)|
|B4 immediate-post|Native AlphaEdit Sequential|10/10 (100.0%)|10/10 (100.0%)|16/20 (80.0%)|8/10 (80.0%)|13/20 (65.0%)|6/10 (60.0%)|84/100 (84.0%)|
|B4 final-W10|P1R52 Soft AlphaCache-ON / Structural-H OFF|10/10 (100.0%)|10/10 (100.0%)|14/20 (70.0%)|7/10 (70.0%)|10/20 (50.0%)|4/10 (40.0%)|89/100 (89.0%)|
|B4 final-W10|Native AlphaEdit Sequential|9/10 (90.0%)|7/10 (70.0%)|16/20 (80.0%)|8/10 (80.0%)|9/20 (45.0%)|4/10 (40.0%)|84/100 (84.0%)|
|B5 entry-pre|P1R52 Soft AlphaCache-ON / Structural-H OFF|1/10 (10.0%)|0/10 (0.0%)|2/20 (10.0%)|1/10 (10.0%)|0/20 (0.0%)|0/10 (0.0%)|93/100 (93.0%)|
|B5 entry-pre|Native AlphaEdit Sequential|1/10 (10.0%)|0/10 (0.0%)|2/20 (10.0%)|1/10 (10.0%)|1/20 (5.0%)|0/10 (0.0%)|92/100 (92.0%)|
|B5 immediate-post|P1R52 Soft AlphaCache-ON / Structural-H OFF|10/10 (100.0%)|10/10 (100.0%)|16/20 (80.0%)|6/10 (60.0%)|12/20 (60.0%)|4/10 (40.0%)|92/100 (92.0%)|
|B5 immediate-post|Native AlphaEdit Sequential|10/10 (100.0%)|10/10 (100.0%)|18/20 (90.0%)|8/10 (80.0%)|14/20 (70.0%)|5/10 (50.0%)|82/100 (82.0%)|
|B5 final-W10|P1R52 Soft AlphaCache-ON / Structural-H OFF|10/10 (100.0%)|10/10 (100.0%)|17/20 (85.0%)|7/10 (70.0%)|14/20 (70.0%)|5/10 (50.0%)|91/100 (91.0%)|
|B5 final-W10|Native AlphaEdit Sequential|9/10 (90.0%)|5/10 (50.0%)|18/20 (90.0%)|8/10 (80.0%)|13/20 (65.0%)|6/10 (60.0%)|75/100 (75.0%)|
|B6 entry-pre|P1R52 Soft AlphaCache-ON / Structural-H OFF|1/10 (10.0%)|0/10 (0.0%)|6/20 (30.0%)|2/10 (20.0%)|0/20 (0.0%)|0/10 (0.0%)|87/100 (87.0%)|
|B6 entry-pre|Native AlphaEdit Sequential|0/10 (0.0%)|0/10 (0.0%)|4/20 (20.0%)|1/10 (10.0%)|0/20 (0.0%)|0/10 (0.0%)|88/100 (88.0%)|
|B6 immediate-post|P1R52 Soft AlphaCache-ON / Structural-H OFF|10/10 (100.0%)|10/10 (100.0%)|18/20 (90.0%)|8/10 (80.0%)|10/20 (50.0%)|2/10 (20.0%)|87/100 (87.0%)|
|B6 immediate-post|Native AlphaEdit Sequential|10/10 (100.0%)|10/10 (100.0%)|20/20 (100.0%)|10/10 (100.0%)|14/20 (70.0%)|5/10 (50.0%)|76/100 (76.0%)|
|B6 final-W10|P1R52 Soft AlphaCache-ON / Structural-H OFF|10/10 (100.0%)|10/10 (100.0%)|18/20 (90.0%)|8/10 (80.0%)|10/20 (50.0%)|2/10 (20.0%)|83/100 (83.0%)|
|B6 final-W10|Native AlphaEdit Sequential|9/10 (90.0%)|7/10 (70.0%)|18/20 (90.0%)|9/10 (90.0%)|11/20 (55.0%)|3/10 (30.0%)|60/100 (60.0%)|
|B7 entry-pre|P1R52 Soft AlphaCache-ON / Structural-H OFF|0/10 (0.0%)|0/10 (0.0%)|2/20 (10.0%)|0/10 (0.0%)|0/20 (0.0%)|0/10 (0.0%)|94/100 (94.0%)|
|B7 entry-pre|Native AlphaEdit Sequential|0/10 (0.0%)|0/10 (0.0%)|2/20 (10.0%)|0/10 (0.0%)|0/20 (0.0%)|0/10 (0.0%)|94/100 (94.0%)|
|B7 immediate-post|P1R52 Soft AlphaCache-ON / Structural-H OFF|10/10 (100.0%)|10/10 (100.0%)|19/20 (95.0%)|9/10 (90.0%)|14/20 (70.0%)|6/10 (60.0%)|90/100 (90.0%)|
|B7 immediate-post|Native AlphaEdit Sequential|10/10 (100.0%)|10/10 (100.0%)|20/20 (100.0%)|10/10 (100.0%)|15/20 (75.0%)|7/10 (70.0%)|72/100 (72.0%)|
|B7 final-W10|P1R52 Soft AlphaCache-ON / Structural-H OFF|10/10 (100.0%)|10/10 (100.0%)|19/20 (95.0%)|9/10 (90.0%)|14/20 (70.0%)|6/10 (60.0%)|88/100 (88.0%)|
|B7 final-W10|Native AlphaEdit Sequential|10/10 (100.0%)|8/10 (80.0%)|20/20 (100.0%)|10/10 (100.0%)|15/20 (75.0%)|6/10 (60.0%)|80/100 (80.0%)|
|B8 entry-pre|P1R52 Soft AlphaCache-ON / Structural-H OFF|2/10 (20.0%)|0/10 (0.0%)|5/20 (25.0%)|1/10 (10.0%)|0/20 (0.0%)|0/10 (0.0%)|89/100 (89.0%)|
|B8 entry-pre|Native AlphaEdit Sequential|2/10 (20.0%)|0/10 (0.0%)|4/20 (20.0%)|1/10 (10.0%)|0/20 (0.0%)|0/10 (0.0%)|93/100 (93.0%)|
|B8 immediate-post|P1R52 Soft AlphaCache-ON / Structural-H OFF|10/10 (100.0%)|10/10 (100.0%)|17/20 (85.0%)|8/10 (80.0%)|9/20 (45.0%)|3/10 (30.0%)|88/100 (88.0%)|
|B8 immediate-post|Native AlphaEdit Sequential|10/10 (100.0%)|10/10 (100.0%)|20/20 (100.0%)|10/10 (100.0%)|13/20 (65.0%)|4/10 (40.0%)|66/100 (66.0%)|
|B8 final-W10|P1R52 Soft AlphaCache-ON / Structural-H OFF|10/10 (100.0%)|10/10 (100.0%)|17/20 (85.0%)|8/10 (80.0%)|8/20 (40.0%)|3/10 (30.0%)|88/100 (88.0%)|
|B8 final-W10|Native AlphaEdit Sequential|10/10 (100.0%)|10/10 (100.0%)|20/20 (100.0%)|10/10 (100.0%)|16/20 (80.0%)|6/10 (60.0%)|67/100 (67.0%)|
|B9 entry-pre|P1R52 Soft AlphaCache-ON / Structural-H OFF|1/10 (10.0%)|0/10 (0.0%)|1/20 (5.0%)|0/10 (0.0%)|0/20 (0.0%)|0/10 (0.0%)|97/100 (97.0%)|
|B9 entry-pre|Native AlphaEdit Sequential|1/10 (10.0%)|0/10 (0.0%)|1/20 (5.0%)|0/10 (0.0%)|0/20 (0.0%)|0/10 (0.0%)|92/100 (92.0%)|
|B9 immediate-post|P1R52 Soft AlphaCache-ON / Structural-H OFF|10/10 (100.0%)|10/10 (100.0%)|17/20 (85.0%)|7/10 (70.0%)|12/20 (60.0%)|4/10 (40.0%)|95/100 (95.0%)|
|B9 immediate-post|Native AlphaEdit Sequential|10/10 (100.0%)|10/10 (100.0%)|20/20 (100.0%)|10/10 (100.0%)|19/20 (95.0%)|9/10 (90.0%)|54/100 (54.0%)|
|B9 final-W10|P1R52 Soft AlphaCache-ON / Structural-H OFF|10/10 (100.0%)|10/10 (100.0%)|17/20 (85.0%)|7/10 (70.0%)|12/20 (60.0%)|4/10 (40.0%)|95/100 (95.0%)|
|B9 final-W10|Native AlphaEdit Sequential|10/10 (100.0%)|10/10 (100.0%)|20/20 (100.0%)|10/10 (100.0%)|18/20 (90.0%)|8/10 (80.0%)|67/100 (67.0%)|
|B10 entry-pre|P1R52 Soft AlphaCache-ON / Structural-H OFF|2/10 (20.0%)|0/10 (0.0%)|6/20 (30.0%)|3/10 (30.0%)|0/20 (0.0%)|0/10 (0.0%)|82/100 (82.0%)|
|B10 entry-pre|Native AlphaEdit Sequential|3/10 (30.0%)|0/10 (0.0%)|9/20 (45.0%)|4/10 (40.0%)|0/20 (0.0%)|0/10 (0.0%)|56/100 (56.0%)|
|B10 immediate-post|P1R52 Soft AlphaCache-ON / Structural-H OFF|10/10 (100.0%)|10/10 (100.0%)|18/20 (90.0%)|8/10 (80.0%)|12/20 (60.0%)|5/10 (50.0%)|79/100 (79.0%)|
|B10 immediate-post|Native AlphaEdit Sequential|10/10 (100.0%)|10/10 (100.0%)|20/20 (100.0%)|10/10 (100.0%)|15/20 (75.0%)|5/10 (50.0%)|31/100 (31.0%)|
|B10 final-W10|P1R52 Soft AlphaCache-ON / Structural-H OFF|10/10 (100.0%)|10/10 (100.0%)|18/20 (90.0%)|8/10 (80.0%)|12/20 (60.0%)|5/10 (50.0%)|79/100 (79.0%)|
|B10 final-W10|Native AlphaEdit Sequential|10/10 (100.0%)|10/10 (100.0%)|20/20 (100.0%)|10/10 (100.0%)|15/20 (75.0%)|5/10 (50.0%)|31/100 (31.0%)|
|B100 final W10|P1R52 Soft AlphaCache-ON / Structural-H OFF|100/100 (100.0%)|100/100 (100.0%)|173/200 (86.5%)|77/100 (77.0%)|107/200 (53.5%)|36/100 (36.0%)|877/1000 (87.7%)|
|B100 final W10|Native AlphaEdit Sequential|93/100 (93.0%)|73/100 (73.0%)|180/200 (90.0%)|87/100 (87.0%)|125/200 (62.5%)|46/100 (46.0%)|692/1000 (69.2%)|

## NLL/margin 및 z8/W8/gap

|Panel|Arm|Rewrite new/true/margin|Rephrase new/true/margin|z8/W8/gap full-six|
|---|---|---:|---:|---:|
|W0 / B1 entry|P1R52 Soft AlphaCache-ON / Structural-H OFF|12.385938/6.364063/-6.021875|10.639062/5.362793/-5.276270|NOT_RECORDED|
|W0 / B1 entry|Native AlphaEdit Sequential|12.385938/6.364063/-6.021875|10.639062/5.362793/-5.276270|NOT_RECORDED|
|B1 entry-pre|P1R52 Soft AlphaCache-ON / Structural-H OFF|12.385938/6.364063/-6.021875|10.639062/5.362793/-5.276270|NOT_RECORDED_OUTER_K8_ONLY|
|B1 entry-pre|Native AlphaEdit Sequential|12.385938/6.364063/-6.021875|10.639062/5.362793/-5.276270|NOT_RECORDED|
|B1 immediate-post|P1R52 Soft AlphaCache-ON / Structural-H OFF|0.023461/12.462500/+12.439039|2.385339/9.069531/+6.684192|0.055925/0.060898/+0.004974|
|B1 immediate-post|Native AlphaEdit Sequential|0.001658/15.475000/+15.473342|1.767157/10.563281/+8.796124|NOT_RECORDED|
|B1 final-W10|P1R52 Soft AlphaCache-ON / Structural-H OFF|0.026359/12.206250/+12.179891|2.458618/9.019531/+6.560913|NOT_RECORDED_OUTER_K8_ONLY|
|B1 final-W10|Native AlphaEdit Sequential|2.568994/8.343115/+5.774121|3.241797/8.066406/+4.824609|NOT_RECORDED|
|B2 entry-pre|P1R52 Soft AlphaCache-ON / Structural-H OFF|11.965625/3.786719/-8.178906|10.163281/4.978125/-5.185156|NOT_RECORDED_OUTER_K8_ONLY|
|B2 entry-pre|Native AlphaEdit Sequential|11.981250/3.940625/-8.040625|10.141406/4.999609/-5.141797|NOT_RECORDED|
|B2 immediate-post|P1R52 Soft AlphaCache-ON / Structural-H OFF|0.031706/11.284375/+11.252669|2.435315/7.604687/+5.169373|0.027509/0.038685/+0.011176|
|B2 immediate-post|Native AlphaEdit Sequential|0.000549/17.362500/+17.361951|0.793271/10.231250/+9.437979|NOT_RECORDED|
|B2 final-W10|P1R52 Soft AlphaCache-ON / Structural-H OFF|0.036714/10.856250/+10.819536|2.312695/7.333594/+5.020898|NOT_RECORDED_OUTER_K8_ONLY|
|B2 final-W10|Native AlphaEdit Sequential|3.171631/7.987500/+4.815869|3.380811/7.528906/+4.148096|NOT_RECORDED|
|B3 entry-pre|P1R52 Soft AlphaCache-ON / Structural-H OFF|9.278540/5.055664/-4.222876|10.344739/5.911353/-4.433386|NOT_RECORDED_OUTER_K8_ONLY|
|B3 entry-pre|Native AlphaEdit Sequential|9.215222/4.916406/-4.298816|10.225964/5.879956/-4.346008|NOT_RECORDED|
|B3 immediate-post|P1R52 Soft AlphaCache-ON / Structural-H OFF|0.045762/11.112500/+11.066738|3.316881/7.411719/+4.094838|0.041262/0.059778/+0.018516|
|B3 immediate-post|Native AlphaEdit Sequential|0.000453/16.237500/+16.237047|2.648302/10.175781/+7.527479|NOT_RECORDED|
|B3 final-W10|P1R52 Soft AlphaCache-ON / Structural-H OFF|0.039105/11.046875/+11.007770|3.240351/7.260742/+4.020391|NOT_RECORDED_OUTER_K8_ONLY|
|B3 final-W10|Native AlphaEdit Sequential|1.175525/7.539062/+6.363537|2.571928/6.562695/+3.990768|NOT_RECORDED|
|B4 entry-pre|P1R52 Soft AlphaCache-ON / Structural-H OFF|11.256250/3.461530/-7.794720|9.243750/2.578638/-6.665112|NOT_RECORDED_OUTER_K8_ONLY|
|B4 entry-pre|Native AlphaEdit Sequential|10.837500/2.988959/-7.848541|9.548438/2.657788/-6.890649|NOT_RECORDED|
|B4 immediate-post|P1R52 Soft AlphaCache-ON / Structural-H OFF|0.086682/8.259375/+8.172693|3.721786/4.929956/+1.208170|0.271689/0.304906/+0.033217|
|B4 immediate-post|Native AlphaEdit Sequential|0.001225/13.125000/+13.123775|3.311117/7.867871/+4.556754|NOT_RECORDED|
|B4 final-W10|P1R52 Soft AlphaCache-ON / Structural-H OFF|0.085779/8.193750/+8.107971|3.649948/4.885596/+1.235648|NOT_RECORDED_OUTER_K8_ONLY|
|B4 final-W10|Native AlphaEdit Sequential|2.019373/5.621875/+3.602502|3.272900/4.981641/+1.708740|NOT_RECORDED|
|B5 entry-pre|P1R52 Soft AlphaCache-ON / Structural-H OFF|10.403125/5.669531/-4.733594|9.489062/5.331299/-4.157764|NOT_RECORDED_OUTER_K8_ONLY|
|B5 entry-pre|Native AlphaEdit Sequential|10.195312/5.530469/-4.664844|9.482422/5.366211/-4.116211|NOT_RECORDED|
|B5 immediate-post|P1R52 Soft AlphaCache-ON / Structural-H OFF|0.126923/9.562500/+9.435577|2.011401/6.773047/+4.761646|0.038226/0.118751/+0.080524|
|B5 immediate-post|Native AlphaEdit Sequential|0.000542/14.337500/+14.336958|1.541079/9.807813/+8.266733|NOT_RECORDED|
|B5 final-W10|P1R52 Soft AlphaCache-ON / Structural-H OFF|0.121558/9.684375/+9.562817|1.882227/6.855859/+4.973633|NOT_RECORDED_OUTER_K8_ONLY|
|B5 final-W10|Native AlphaEdit Sequential|2.540405/7.776562/+5.236157|1.815258/7.375000/+5.559742|NOT_RECORDED|
|B6 entry-pre|P1R52 Soft AlphaCache-ON / Structural-H OFF|9.875000/5.274707/-4.600293|8.710938/5.141528/-3.569409|NOT_RECORDED_OUTER_K8_ONLY|
|B6 entry-pre|Native AlphaEdit Sequential|10.115625/4.345508/-5.770117|8.928125/4.407983/-4.520142|NOT_RECORDED|
|B6 immediate-post|P1R52 Soft AlphaCache-ON / Structural-H OFF|0.022836/12.281250/+12.258414|2.050635/8.670312/+6.619678|0.012298/0.033437/+0.021139|
|B6 immediate-post|Native AlphaEdit Sequential|0.000837/16.387500/+16.386663|1.260676/12.550000/+11.289324|NOT_RECORDED|
|B6 final-W10|P1R52 Soft AlphaCache-ON / Structural-H OFF|0.022513/12.287500/+12.264987|2.027548/8.648438/+6.620889|NOT_RECORDED_OUTER_K8_ONLY|
|B6 final-W10|Native AlphaEdit Sequential|1.449539/9.171875/+7.722336|2.077778/8.081250/+6.003472|NOT_RECORDED|
|B7 entry-pre|P1R52 Soft AlphaCache-ON / Structural-H OFF|12.475000/4.710170/-7.764830|10.589844/4.517140/-6.072704|NOT_RECORDED_OUTER_K8_ONLY|
|B7 entry-pre|Native AlphaEdit Sequential|12.003125/4.310938/-7.692187|11.741406/4.716386/-7.025021|NOT_RECORDED|
|B7 immediate-post|P1R52 Soft AlphaCache-ON / Structural-H OFF|0.020165/10.450000/+10.429835|1.249432/9.202034/+7.952602|0.013854/0.020157/+0.006303|
|B7 immediate-post|Native AlphaEdit Sequential|0.001188/14.434375/+14.433187|0.593808/13.246333/+12.652525|NOT_RECORDED|
|B7 final-W10|P1R52 Soft AlphaCache-ON / Structural-H OFF|0.024921/10.137500/+10.112580|1.247598/9.109074/+7.861475|NOT_RECORDED_OUTER_K8_ONLY|
|B7 final-W10|Native AlphaEdit Sequential|0.843677/9.037516/+8.193839|1.184340/9.367514/+8.183174|NOT_RECORDED|
|B8 entry-pre|P1R52 Soft AlphaCache-ON / Structural-H OFF|10.331250/3.663208/-6.668042|10.139844/5.247168/-4.892676|NOT_RECORDED_OUTER_K8_ONLY|
|B8 entry-pre|Native AlphaEdit Sequential|9.642188/3.386096/-6.256091|9.872656/4.896582/-4.976074|NOT_RECORDED|
|B8 immediate-post|P1R52 Soft AlphaCache-ON / Structural-H OFF|0.014580/10.990625/+10.976045|3.232268/8.841016/+5.608748|0.019635/0.021258/+0.001624|
|B8 immediate-post|Native AlphaEdit Sequential|0.002659/12.350000/+12.347341|1.087395/10.787500/+9.700105|NOT_RECORDED|
|B8 final-W10|P1R52 Soft AlphaCache-ON / Structural-H OFF|0.012924/11.084375/+11.071451|3.204694/8.776074/+5.571381|NOT_RECORDED_OUTER_K8_ONLY|
|B8 final-W10|Native AlphaEdit Sequential|0.061545/9.099219/+9.037674|0.823785/9.195312/+8.371527|NOT_RECORDED|
|B9 entry-pre|P1R52 Soft AlphaCache-ON / Structural-H OFF|10.865625/2.615796/-8.249829|9.176562/3.404346/-5.772217|NOT_RECORDED_OUTER_K8_ONLY|
|B9 entry-pre|Native AlphaEdit Sequential|9.546875/2.841150/-6.705725|8.863281/3.214355/-5.648926|NOT_RECORDED|
|B9 immediate-post|P1R52 Soft AlphaCache-ON / Structural-H OFF|0.014182/10.900000/+10.885818|2.056558/6.353125/+4.296567|0.018861/0.020105/+0.001245|
|B9 immediate-post|Native AlphaEdit Sequential|0.007578/14.993750/+14.986172|0.144975/12.468750/+12.323775|NOT_RECORDED|
|B9 final-W10|P1R52 Soft AlphaCache-ON / Structural-H OFF|0.014785/10.931250/+10.916465|2.086026/6.275781/+4.189755|NOT_RECORDED_OUTER_K8_ONLY|
|B9 final-W10|Native AlphaEdit Sequential|0.060243/13.215625/+13.155382|0.291600/11.459375/+11.167775|NOT_RECORDED|
|B10 entry-pre|P1R52 Soft AlphaCache-ON / Structural-H OFF|10.585938/6.166797/-4.419141|9.378125/5.663574/-3.714551|NOT_RECORDED_OUTER_K8_ONLY|
|B10 entry-pre|Native AlphaEdit Sequential|9.920312/8.788867/-1.131445|7.959375/7.934180/-0.025195|NOT_RECORDED|
|B10 immediate-post|P1R52 Soft AlphaCache-ON / Structural-H OFF|0.021832/12.615625/+12.593793|1.755432/8.611914/+6.856482|0.016603/0.023887/+0.007285|
|B10 immediate-post|Native AlphaEdit Sequential|0.012867/14.787500/+14.774633|0.734282/12.831250/+12.096968|NOT_RECORDED|
|B10 final-W10|P1R52 Soft AlphaCache-ON / Structural-H OFF|0.021832/12.615625/+12.593793|1.755432/8.611914/+6.856482|NOT_RECORDED_OUTER_K8_ONLY|
|B10 final-W10|Native AlphaEdit Sequential|0.012867/14.787500/+14.774633|0.734282/12.831250/+12.096968|NOT_RECORDED|
|B100 final W10|P1R52 Soft AlphaCache-ON / Structural-H OFF|0.040649/10.904375/+10.863726|2.386514/7.677660/+5.291147|NOT_RECORDED|
|B100 final W10|Native AlphaEdit Sequential|1.390380/9.257985/+7.867605|1.939448/8.544935/+6.605487|NOT_RECORDED|

## Controller / routing / strength / energy / P / cache

|B|Arm|Primary/Rescue/Current|denom|active/no-direction|clamp|fallback/status|αreq/αapply|strength residual|P/capacity/energy|entropy/top1|H/cache receipt|
|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|
|B1|P1R52 Soft AlphaCache-ON / Structural-H OFF|80/0/0|80|80/0|0|0/JOINT_WRITE:8|1.7208/1.7208|4.44e-16|3.564e-03/1.025e-01/1.818e-01|1.471/0.339|cache consume=0; H-influence=0; {'ALPHA_CACHE_ON_STRUCTURAL_H_OFF': 8}|
|B2|P1R52 Soft AlphaCache-ON / Structural-H OFF|80/0/0|80|80/0|0|0/JOINT_WRITE:8|1.855/1.855|4.44e-16|3.340e-03/6.834e-02/1.257e-01|1.374/0.372|cache consume=80; H-influence=0; {'ALPHA_CACHE_ON_STRUCTURAL_H_OFF': 8}|
|B3|P1R52 Soft AlphaCache-ON / Structural-H OFF|80/0/0|80|80/0|0|0/JOINT_WRITE:8|1.4162/1.4162|2.22e-16|2.394e-03/7.032e-02/1.297e-01|1.509/0.320|cache consume=160; H-influence=0; {'ALPHA_CACHE_ON_STRUCTURAL_H_OFF': 8}|
|B4|P1R52 Soft AlphaCache-ON / Structural-H OFF|74/2/4|80|80/0|10|0/JOINT_WRITE:8|1.5237/1.5237|2.66e-15|2.624e-03/6.143e-02/1.129e-01|1.239/0.493|cache consume=240; H-influence=0; {'ALPHA_CACHE_ON_STRUCTURAL_H_OFF': 8}|
|B5|P1R52 Soft AlphaCache-ON / Structural-H OFF|80/0/0|80|80/0|5|0/JOINT_WRITE:8|1.6308/1.6308|2.22e-16|2.266e-03/7.826e-02/1.432e-01|1.494/0.320|cache consume=320; H-influence=0; {'ALPHA_CACHE_ON_STRUCTURAL_H_OFF': 8}|
|B6|P1R52 Soft AlphaCache-ON / Structural-H OFF|78/2/0|80|80/0|0|0/JOINT_WRITE:8|1.6523/1.6523|4.44e-16|2.598e-03/6.797e-02/1.248e-01|1.401/0.394|cache consume=400; H-influence=0; {'ALPHA_CACHE_ON_STRUCTURAL_H_OFF': 8}|
|B7|P1R52 Soft AlphaCache-ON / Structural-H OFF|80/0/0|80|80/0|0|0/JOINT_WRITE:8|1.7978/1.7978|4.44e-16|2.032e-03/5.365e-02/1.006e-01|1.460/0.373|cache consume=480; H-influence=0; {'ALPHA_CACHE_ON_STRUCTURAL_H_OFF': 8}|
|B8|P1R52 Soft AlphaCache-ON / Structural-H OFF|79/1/0|80|80/0|0|0/JOINT_WRITE:8|1.6045/1.6045|8.88e-16|1.966e-03/3.963e-02/7.528e-02|1.470/0.388|cache consume=560; H-influence=0; {'ALPHA_CACHE_ON_STRUCTURAL_H_OFF': 8}|
|B9|P1R52 Soft AlphaCache-ON / Structural-H OFF|80/0/0|80|80/0|0|0/JOINT_WRITE:8|1.5621/1.5621|3.47e-18|1.430e-03/3.428e-02/6.555e-02|1.436/0.380|cache consume=640; H-influence=0; {'ALPHA_CACHE_ON_STRUCTURAL_H_OFF': 8}|
|B10|P1R52 Soft AlphaCache-ON / Structural-H OFF|78/1/1|80|80/0|0|0/JOINT_WRITE:8|1.6763/1.6763|8.88e-16|2.446e-03/6.719e-02/1.231e-01|1.414/0.407|cache consume=720; H-influence=0; {'ALPHA_CACHE_ON_STRUCTURAL_H_OFF': 8}|

## Sequential state / transactions

- H-OFF Alpha solve-cache entry widths: `[0, 10, 20, 30, 40, 50, 60, 70, 80, 90]`; transaction append `100`; terminal active cache/anchors `100/100`.
- H-OFF terminal W0 pointer/byte restore: `True/True`. H-ON cache consumption is `NOT_RECORDED_IN_H_ON_SEALED_RECEIPTS` in its sealed receipt.
- All R52 arms retain physical W across B1→B10. Control routing receipt shows Structural-H decision influence=0 and observation-ledger influence=0; retry/backtracking/rollback/interbatch W0 restore remain zero in sealed terminal records.

## Compute ledger

|Arm|entry-pre evaluator F/B/generation|accuracy 추가 F/B/generation|evaluator forward|model forward|processed tokens|materialization|
|---|---:|---:|---:|---:|---:|---:|
|P1R52 Soft AlphaCache-ON / Structural-H OFF|100/0/0|0/0/0|750|242|66067|80|
|Native AlphaEdit Sequential|100/0/0|0/0/0|750|32|1547|NOT_RECORDED_NATIVE_PATH|

## Hard / low cohort, batch age, forgetting

- failed-or-transition rows `90/100`; entry rewrite-hard H-ON `89/100`; H-ON/H-OFF/Native immediate-post GEN failure `25/24/6` /100.
- post→final GEN loss H-ON/H-OFF `1/1` /100; post GEN H-ON>H-OFF `1/100`, H-OFF>H-ON `2/100`.
- Per-request machine rows retain entry hardness, batch age, pre→post gain, post→final loss, success/accuracy/NLL/margin. Routing is batch×step, so requestwise Structural-H attribution is `NOT_RECORDED`.
- Scope is `RAW_FREE_ASSOCIATION_ONLY_NO_ISOLATED_CAUSAL_ATTRIBUTION`; no isolated causal statement is made.

## 상태

- `TERMINAL_FACTUAL_REPORT_COMPLETE`; P1R52 method root is unchanged; H-aware control result is a separate immutable reference; scientific_promotion=false.
