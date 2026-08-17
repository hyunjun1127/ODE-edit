# P1R52 Soft Sequential Structural-H ON — corrected official baseline integrated v4

> 사용자 명시적 overwrite 승인에 따라 기존 v4 파일 경로를 유지하면서 corrected Official AlphaEdit cache-on 및 Official MEMIT 비교를 직접 통합했습니다.

## 핵심 요약 — success 기준 EFF / GEN / LOC

|구분|EFF|GEN|LOC|
|---|---:|---:|---:|
|**Pre-edit (original W0)**|13/100 (13.0%)|28/200 (14.0%)|892/1000 (89.2%)|
|**MEMIT (final W10)**|100/100 (100.0%)|174/200 (87.0%)|879/1000 (87.9%)|
|**AlphaEdit (final W10, cache_c ON)**|100/100 (100.0%)|190/200 (95.0%)|837/1000 (83.7%)|
|**Ours (final W10, Structural-H ON)**|100/100 (100.0%)|174/200 (87.0%)|876/1000 (87.6%)|

EFF/GEN은 모두 NLL-preference **success**이고 accuracy가 아닙니다. **Pre-edit**은 네 arm이 공유하는 원본 Llama `W0`를 동일 frozen 10×B10 stream 전체에서 평가한 공통값입니다. 아래의 `actual entry-pre B100`은 별개의 순차 진단값으로, 각 batch 진입 상태 `W_(b-1)`에서 측정됩니다. Structural-H OFF control과 superseded cache-reset AlphaEdit는 상세 비교표에서만 제시합니다.

## Aggregate pre / post / final 전체 지표

|Phase|Arm|EFF|Rewrite Acc|GEN|GEN-strict|Rephrase Acc|Acc-strict|LOC|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|actual entry-pre B100|P1R52 Soft Structural-H ON|11/100 (11.0%)|0/100 (0.0%)|26/200 (13.0%)|7/100 (7.0%)|0/200 (0.0%)|0/100 (0.0%)|903/1000 (90.3%)|
|actual entry-pre B100|P1R52 Soft AlphaCache-ON / Structural-H OFF|11/100 (11.0%)|0/100 (0.0%)|28/200 (14.0%)|9/100 (9.0%)|0/200 (0.0%)|0/100 (0.0%)|907/1000 (90.7%)|
|actual entry-pre B100|Corrected Official AlphaEdit cache_c ON|13/100 (13.0%)|0/100 (0.0%)|27/200 (13.5%)|8/100 (8.0%)|1/200 (0.5%)|0/100 (0.0%)|890/1000 (89.0%)|
|actual entry-pre B100|Official MEMIT Sequential|13/100 (13.0%)|0/100 (0.0%)|26/200 (13.0%)|9/100 (9.0%)|2/200 (1.0%)|0/100 (0.0%)|899/1000 (89.9%)|
|immediate-post B100|P1R52 Soft Structural-H ON|100/100 (100.0%)|100/100 (100.0%)|172/200 (86.0%)|75/100 (75.0%)|105/200 (52.5%)|35/100 (35.0%)|885/1000 (88.5%)|
|immediate-post B100|P1R52 Soft AlphaCache-ON / Structural-H OFF|100/100 (100.0%)|100/100 (100.0%)|173/200 (86.5%)|76/100 (76.0%)|106/200 (53.0%)|35/100 (35.0%)|888/1000 (88.8%)|
|immediate-post B100|Corrected Official AlphaEdit cache_c ON|100/100 (100.0%)|100/100 (100.0%)|190/200 (95.0%)|92/100 (92.0%)|142/200 (71.0%)|55/100 (55.0%)|857/1000 (85.7%)|
|immediate-post B100|Official MEMIT Sequential|100/100 (100.0%)|99/100 (99.0%)|171/200 (85.5%)|79/100 (79.0%)|103/200 (51.5%)|34/100 (34.0%)|885/1000 (88.5%)|
|final W10 B100|P1R52 Soft Structural-H ON|100/100 (100.0%)|100/100 (100.0%)|174/200 (87.0%)|77/100 (77.0%)|109/200 (54.5%)|37/100 (37.0%)|876/1000 (87.6%)|
|final W10 B100|P1R52 Soft AlphaCache-ON / Structural-H OFF|100/100 (100.0%)|100/100 (100.0%)|173/200 (86.5%)|77/100 (77.0%)|107/200 (53.5%)|36/100 (36.0%)|877/1000 (87.7%)|
|final W10 B100|Corrected Official AlphaEdit cache_c ON|100/100 (100.0%)|100/100 (100.0%)|190/200 (95.0%)|92/100 (92.0%)|144/200 (72.0%)|56/100 (56.0%)|837/1000 (83.7%)|
|final W10 B100|Official MEMIT Sequential|100/100 (100.0%)|98/100 (98.0%)|174/200 (87.0%)|79/100 (79.0%)|108/200 (54.0%)|38/100 (38.0%)|879/1000 (87.9%)|

## B1–B10 success 절대값: actual entry-pre → immediate-post → final-W10

|B|Phase|Arm|EFF|GEN|GEN-strict|LOC|
|---:|---|---|---:|---:|---:|---:|
|B1|entry-pre|P1R52 Soft Structural-H ON|2/10 (20.0%)|3/20 (15.0%)|1/10 (10.0%)|93/100 (93.0%)|
|B1|entry-pre|P1R52 Soft AlphaCache-ON / Structural-H OFF|2/10 (20.0%)|3/20 (15.0%)|1/10 (10.0%)|93/100 (93.0%)|
|B1|entry-pre|Corrected Official AlphaEdit cache_c ON|2/10 (20.0%)|3/20 (15.0%)|1/10 (10.0%)|93/100 (93.0%)|
|B1|entry-pre|Official MEMIT Sequential|2/10 (20.0%)|3/20 (15.0%)|1/10 (10.0%)|93/100 (93.0%)|
|B1|post|P1R52 Soft Structural-H ON|10/10 (100.0%)|19/20 (95.0%)|9/10 (90.0%)|92/100 (92.0%)|
|B1|post|P1R52 Soft AlphaCache-ON / Structural-H OFF|10/10 (100.0%)|19/20 (95.0%)|9/10 (90.0%)|92/100 (92.0%)|
|B1|post|Corrected Official AlphaEdit cache_c ON|10/10 (100.0%)|19/20 (95.0%)|9/10 (90.0%)|92/100 (92.0%)|
|B1|post|Official MEMIT Sequential|10/10 (100.0%)|18/20 (90.0%)|8/10 (80.0%)|92/100 (92.0%)|
|B1|final-W10|P1R52 Soft Structural-H ON|10/10 (100.0%)|19/20 (95.0%)|9/10 (90.0%)|95/100 (95.0%)|
|B1|final-W10|P1R52 Soft AlphaCache-ON / Structural-H OFF|10/10 (100.0%)|19/20 (95.0%)|9/10 (90.0%)|95/100 (95.0%)|
|B1|final-W10|Corrected Official AlphaEdit cache_c ON|10/10 (100.0%)|18/20 (90.0%)|8/10 (80.0%)|92/100 (92.0%)|
|B1|final-W10|Official MEMIT Sequential|10/10 (100.0%)|17/20 (85.0%)|8/10 (80.0%)|92/100 (92.0%)|
|B2|entry-pre|P1R52 Soft Structural-H ON|0/10 (0.0%)|1/20 (5.0%)|0/10 (0.0%)|86/100 (86.0%)|
|B2|entry-pre|P1R52 Soft AlphaCache-ON / Structural-H OFF|0/10 (0.0%)|1/20 (5.0%)|0/10 (0.0%)|86/100 (86.0%)|
|B2|entry-pre|Corrected Official AlphaEdit cache_c ON|0/10 (0.0%)|1/20 (5.0%)|0/10 (0.0%)|87/100 (87.0%)|
|B2|entry-pre|Official MEMIT Sequential|0/10 (0.0%)|1/20 (5.0%)|0/10 (0.0%)|87/100 (87.0%)|
|B2|post|P1R52 Soft Structural-H ON|10/10 (100.0%)|17/20 (85.0%)|7/10 (70.0%)|86/100 (86.0%)|
|B2|post|P1R52 Soft AlphaCache-ON / Structural-H OFF|10/10 (100.0%)|17/20 (85.0%)|7/10 (70.0%)|86/100 (86.0%)|
|B2|post|Corrected Official AlphaEdit cache_c ON|10/10 (100.0%)|20/20 (100.0%)|10/10 (100.0%)|82/100 (82.0%)|
|B2|post|Official MEMIT Sequential|10/10 (100.0%)|17/20 (85.0%)|7/10 (70.0%)|87/100 (87.0%)|
|B2|final-W10|P1R52 Soft Structural-H ON|10/10 (100.0%)|17/20 (85.0%)|7/10 (70.0%)|83/100 (83.0%)|
|B2|final-W10|P1R52 Soft AlphaCache-ON / Structural-H OFF|10/10 (100.0%)|17/20 (85.0%)|7/10 (70.0%)|83/100 (83.0%)|
|B2|final-W10|Corrected Official AlphaEdit cache_c ON|10/10 (100.0%)|20/20 (100.0%)|10/10 (100.0%)|84/100 (84.0%)|
|B2|final-W10|Official MEMIT Sequential|10/10 (100.0%)|18/20 (90.0%)|8/10 (80.0%)|85/100 (85.0%)|
|B3|entry-pre|P1R52 Soft Structural-H ON|2/10 (20.0%)|2/20 (10.0%)|1/10 (10.0%)|88/100 (88.0%)|
|B3|entry-pre|P1R52 Soft AlphaCache-ON / Structural-H OFF|2/10 (20.0%)|2/20 (10.0%)|1/10 (10.0%)|88/100 (88.0%)|
|B3|entry-pre|Corrected Official AlphaEdit cache_c ON|2/10 (20.0%)|2/20 (10.0%)|1/10 (10.0%)|87/100 (87.0%)|
|B3|entry-pre|Official MEMIT Sequential|2/10 (20.0%)|2/20 (10.0%)|1/10 (10.0%)|87/100 (87.0%)|
|B3|post|P1R52 Soft Structural-H ON|10/10 (100.0%)|17/20 (85.0%)|7/10 (70.0%)|87/100 (87.0%)|
|B3|post|P1R52 Soft AlphaCache-ON / Structural-H OFF|10/10 (100.0%)|17/20 (85.0%)|7/10 (70.0%)|87/100 (87.0%)|
|B3|post|Corrected Official AlphaEdit cache_c ON|10/10 (100.0%)|19/20 (95.0%)|9/10 (90.0%)|87/100 (87.0%)|
|B3|post|Official MEMIT Sequential|10/10 (100.0%)|15/20 (75.0%)|7/10 (70.0%)|87/100 (87.0%)|
|B3|final-W10|P1R52 Soft Structural-H ON|10/10 (100.0%)|17/20 (85.0%)|7/10 (70.0%)|87/100 (87.0%)|
|B3|final-W10|P1R52 Soft AlphaCache-ON / Structural-H OFF|10/10 (100.0%)|17/20 (85.0%)|7/10 (70.0%)|86/100 (86.0%)|
|B3|final-W10|Corrected Official AlphaEdit cache_c ON|10/10 (100.0%)|19/20 (95.0%)|9/10 (90.0%)|84/100 (84.0%)|
|B3|final-W10|Official MEMIT Sequential|10/10 (100.0%)|15/20 (75.0%)|7/10 (70.0%)|87/100 (87.0%)|
|B4|entry-pre|P1R52 Soft Structural-H ON|0/10 (0.0%)|0/20 (0.0%)|0/10 (0.0%)|98/100 (98.0%)|
|B4|entry-pre|P1R52 Soft AlphaCache-ON / Structural-H OFF|0/10 (0.0%)|0/20 (0.0%)|0/10 (0.0%)|98/100 (98.0%)|
|B4|entry-pre|Corrected Official AlphaEdit cache_c ON|0/10 (0.0%)|0/20 (0.0%)|0/10 (0.0%)|96/100 (96.0%)|
|B4|entry-pre|Official MEMIT Sequential|0/10 (0.0%)|0/20 (0.0%)|0/10 (0.0%)|97/100 (97.0%)|
|B4|post|P1R52 Soft Structural-H ON|10/10 (100.0%)|15/20 (75.0%)|7/10 (70.0%)|92/100 (92.0%)|
|B4|post|P1R52 Soft AlphaCache-ON / Structural-H OFF|10/10 (100.0%)|15/20 (75.0%)|7/10 (70.0%)|92/100 (92.0%)|
|B4|post|Corrected Official AlphaEdit cache_c ON|10/10 (100.0%)|16/20 (80.0%)|8/10 (80.0%)|88/100 (88.0%)|
|B4|post|Official MEMIT Sequential|10/10 (100.0%)|14/20 (70.0%)|7/10 (70.0%)|94/100 (94.0%)|
|B4|final-W10|P1R52 Soft Structural-H ON|10/10 (100.0%)|15/20 (75.0%)|7/10 (70.0%)|89/100 (89.0%)|
|B4|final-W10|P1R52 Soft AlphaCache-ON / Structural-H OFF|10/10 (100.0%)|14/20 (70.0%)|7/10 (70.0%)|89/100 (89.0%)|
|B4|final-W10|Corrected Official AlphaEdit cache_c ON|10/10 (100.0%)|16/20 (80.0%)|8/10 (80.0%)|88/100 (88.0%)|
|B4|final-W10|Official MEMIT Sequential|10/10 (100.0%)|15/20 (75.0%)|7/10 (70.0%)|93/100 (93.0%)|
|B5|entry-pre|P1R52 Soft Structural-H ON|1/10 (10.0%)|2/20 (10.0%)|1/10 (10.0%)|93/100 (93.0%)|
|B5|entry-pre|P1R52 Soft AlphaCache-ON / Structural-H OFF|1/10 (10.0%)|2/20 (10.0%)|1/10 (10.0%)|93/100 (93.0%)|
|B5|entry-pre|Corrected Official AlphaEdit cache_c ON|1/10 (10.0%)|2/20 (10.0%)|1/10 (10.0%)|92/100 (92.0%)|
|B5|entry-pre|Official MEMIT Sequential|1/10 (10.0%)|2/20 (10.0%)|1/10 (10.0%)|88/100 (88.0%)|
|B5|post|P1R52 Soft Structural-H ON|10/10 (100.0%)|14/20 (70.0%)|5/10 (50.0%)|92/100 (92.0%)|
|B5|post|P1R52 Soft AlphaCache-ON / Structural-H OFF|10/10 (100.0%)|16/20 (80.0%)|6/10 (60.0%)|92/100 (92.0%)|
|B5|post|Corrected Official AlphaEdit cache_c ON|10/10 (100.0%)|18/20 (90.0%)|8/10 (80.0%)|88/100 (88.0%)|
|B5|post|Official MEMIT Sequential|10/10 (100.0%)|15/20 (75.0%)|7/10 (70.0%)|88/100 (88.0%)|
|B5|final-W10|P1R52 Soft Structural-H ON|10/10 (100.0%)|17/20 (85.0%)|7/10 (70.0%)|91/100 (91.0%)|
|B5|final-W10|P1R52 Soft AlphaCache-ON / Structural-H OFF|10/10 (100.0%)|17/20 (85.0%)|7/10 (70.0%)|91/100 (91.0%)|
|B5|final-W10|Corrected Official AlphaEdit cache_c ON|10/10 (100.0%)|19/20 (95.0%)|9/10 (90.0%)|78/100 (78.0%)|
|B5|final-W10|Official MEMIT Sequential|10/10 (100.0%)|17/20 (85.0%)|7/10 (70.0%)|88/100 (88.0%)|
|B6|entry-pre|P1R52 Soft Structural-H ON|1/10 (10.0%)|5/20 (25.0%)|1/10 (10.0%)|85/100 (85.0%)|
|B6|entry-pre|P1R52 Soft AlphaCache-ON / Structural-H OFF|1/10 (10.0%)|6/20 (30.0%)|2/10 (20.0%)|87/100 (87.0%)|
|B6|entry-pre|Corrected Official AlphaEdit cache_c ON|0/10 (0.0%)|4/20 (20.0%)|1/10 (10.0%)|88/100 (88.0%)|
|B6|entry-pre|Official MEMIT Sequential|2/10 (20.0%)|5/20 (25.0%)|2/10 (20.0%)|87/100 (87.0%)|
|B6|post|P1R52 Soft Structural-H ON|10/10 (100.0%)|18/20 (90.0%)|8/10 (80.0%)|86/100 (86.0%)|
|B6|post|P1R52 Soft AlphaCache-ON / Structural-H OFF|10/10 (100.0%)|18/20 (90.0%)|8/10 (80.0%)|87/100 (87.0%)|
|B6|post|Corrected Official AlphaEdit cache_c ON|10/10 (100.0%)|19/20 (95.0%)|9/10 (90.0%)|85/100 (85.0%)|
|B6|post|Official MEMIT Sequential|10/10 (100.0%)|18/20 (90.0%)|8/10 (80.0%)|85/100 (85.0%)|
|B6|final-W10|P1R52 Soft Structural-H ON|10/10 (100.0%)|18/20 (90.0%)|8/10 (80.0%)|82/100 (82.0%)|
|B6|final-W10|P1R52 Soft AlphaCache-ON / Structural-H OFF|10/10 (100.0%)|18/20 (90.0%)|8/10 (80.0%)|83/100 (83.0%)|
|B6|final-W10|Corrected Official AlphaEdit cache_c ON|10/10 (100.0%)|19/20 (95.0%)|9/10 (90.0%)|80/100 (80.0%)|
|B6|final-W10|Official MEMIT Sequential|10/10 (100.0%)|18/20 (90.0%)|8/10 (80.0%)|83/100 (83.0%)|
|B7|entry-pre|P1R52 Soft Structural-H ON|0/10 (0.0%)|2/20 (10.0%)|0/10 (0.0%)|92/100 (92.0%)|
|B7|entry-pre|P1R52 Soft AlphaCache-ON / Structural-H OFF|0/10 (0.0%)|2/20 (10.0%)|0/10 (0.0%)|94/100 (94.0%)|
|B7|entry-pre|Corrected Official AlphaEdit cache_c ON|0/10 (0.0%)|2/20 (10.0%)|0/10 (0.0%)|92/100 (92.0%)|
|B7|entry-pre|Official MEMIT Sequential|0/10 (0.0%)|1/20 (5.0%)|0/10 (0.0%)|95/100 (95.0%)|
|B7|post|P1R52 Soft Structural-H ON|10/10 (100.0%)|19/20 (95.0%)|9/10 (90.0%)|90/100 (90.0%)|
|B7|post|P1R52 Soft AlphaCache-ON / Structural-H OFF|10/10 (100.0%)|19/20 (95.0%)|9/10 (90.0%)|90/100 (90.0%)|
|B7|post|Corrected Official AlphaEdit cache_c ON|10/10 (100.0%)|19/20 (95.0%)|9/10 (90.0%)|90/100 (90.0%)|
|B7|post|Official MEMIT Sequential|10/10 (100.0%)|19/20 (95.0%)|9/10 (90.0%)|94/100 (94.0%)|
|B7|final-W10|P1R52 Soft Structural-H ON|10/10 (100.0%)|19/20 (95.0%)|9/10 (90.0%)|88/100 (88.0%)|
|B7|final-W10|P1R52 Soft AlphaCache-ON / Structural-H OFF|10/10 (100.0%)|19/20 (95.0%)|9/10 (90.0%)|88/100 (88.0%)|
|B7|final-W10|Corrected Official AlphaEdit cache_c ON|10/10 (100.0%)|19/20 (95.0%)|9/10 (90.0%)|91/100 (91.0%)|
|B7|final-W10|Official MEMIT Sequential|10/10 (100.0%)|19/20 (95.0%)|9/10 (90.0%)|93/100 (93.0%)|
|B8|entry-pre|P1R52 Soft Structural-H ON|2/10 (20.0%)|5/20 (25.0%)|1/10 (10.0%)|89/100 (89.0%)|
|B8|entry-pre|P1R52 Soft AlphaCache-ON / Structural-H OFF|2/10 (20.0%)|5/20 (25.0%)|1/10 (10.0%)|89/100 (89.0%)|
|B8|entry-pre|Corrected Official AlphaEdit cache_c ON|2/10 (20.0%)|4/20 (20.0%)|1/10 (10.0%)|87/100 (87.0%)|
|B8|entry-pre|Official MEMIT Sequential|2/10 (20.0%)|5/20 (25.0%)|1/10 (10.0%)|87/100 (87.0%)|
|B8|post|P1R52 Soft Structural-H ON|10/10 (100.0%)|18/20 (90.0%)|8/10 (80.0%)|88/100 (88.0%)|
|B8|post|P1R52 Soft AlphaCache-ON / Structural-H OFF|10/10 (100.0%)|17/20 (85.0%)|8/10 (80.0%)|88/100 (88.0%)|
|B8|post|Corrected Official AlphaEdit cache_c ON|10/10 (100.0%)|20/20 (100.0%)|10/10 (100.0%)|85/100 (85.0%)|
|B8|post|Official MEMIT Sequential|10/10 (100.0%)|17/20 (85.0%)|8/10 (80.0%)|85/100 (85.0%)|
|B8|final-W10|P1R52 Soft Structural-H ON|10/10 (100.0%)|17/20 (85.0%)|8/10 (80.0%)|89/100 (89.0%)|
|B8|final-W10|P1R52 Soft AlphaCache-ON / Structural-H OFF|10/10 (100.0%)|17/20 (85.0%)|8/10 (80.0%)|88/100 (88.0%)|
|B8|final-W10|Corrected Official AlphaEdit cache_c ON|10/10 (100.0%)|20/20 (100.0%)|10/10 (100.0%)|81/100 (81.0%)|
|B8|final-W10|Official MEMIT Sequential|10/10 (100.0%)|17/20 (85.0%)|7/10 (70.0%)|85/100 (85.0%)|
|B9|entry-pre|P1R52 Soft Structural-H ON|1/10 (10.0%)|1/20 (5.0%)|0/10 (0.0%)|97/100 (97.0%)|
|B9|entry-pre|P1R52 Soft AlphaCache-ON / Structural-H OFF|1/10 (10.0%)|1/20 (5.0%)|0/10 (0.0%)|97/100 (97.0%)|
|B9|entry-pre|Corrected Official AlphaEdit cache_c ON|2/10 (20.0%)|1/20 (5.0%)|0/10 (0.0%)|95/100 (95.0%)|
|B9|entry-pre|Official MEMIT Sequential|1/10 (10.0%)|1/20 (5.0%)|0/10 (0.0%)|96/100 (96.0%)|
|B9|post|P1R52 Soft Structural-H ON|10/10 (100.0%)|17/20 (85.0%)|7/10 (70.0%)|95/100 (95.0%)|
|B9|post|P1R52 Soft AlphaCache-ON / Structural-H OFF|10/10 (100.0%)|17/20 (85.0%)|7/10 (70.0%)|95/100 (95.0%)|
|B9|post|Corrected Official AlphaEdit cache_c ON|10/10 (100.0%)|20/20 (100.0%)|10/10 (100.0%)|89/100 (89.0%)|
|B9|post|Official MEMIT Sequential|10/10 (100.0%)|19/20 (95.0%)|9/10 (90.0%)|93/100 (93.0%)|
|B9|final-W10|P1R52 Soft Structural-H ON|10/10 (100.0%)|17/20 (85.0%)|7/10 (70.0%)|95/100 (95.0%)|
|B9|final-W10|P1R52 Soft AlphaCache-ON / Structural-H OFF|10/10 (100.0%)|17/20 (85.0%)|7/10 (70.0%)|95/100 (95.0%)|
|B9|final-W10|Corrected Official AlphaEdit cache_c ON|10/10 (100.0%)|20/20 (100.0%)|10/10 (100.0%)|88/100 (88.0%)|
|B9|final-W10|Official MEMIT Sequential|10/10 (100.0%)|19/20 (95.0%)|9/10 (90.0%)|93/100 (93.0%)|
|B10|entry-pre|P1R52 Soft Structural-H ON|2/10 (20.0%)|5/20 (25.0%)|2/10 (20.0%)|82/100 (82.0%)|
|B10|entry-pre|P1R52 Soft AlphaCache-ON / Structural-H OFF|2/10 (20.0%)|6/20 (30.0%)|3/10 (30.0%)|82/100 (82.0%)|
|B10|entry-pre|Corrected Official AlphaEdit cache_c ON|4/10 (40.0%)|8/20 (40.0%)|3/10 (30.0%)|73/100 (73.0%)|
|B10|entry-pre|Official MEMIT Sequential|3/10 (30.0%)|6/20 (30.0%)|3/10 (30.0%)|82/100 (82.0%)|
|B10|post|P1R52 Soft Structural-H ON|10/10 (100.0%)|18/20 (90.0%)|8/10 (80.0%)|77/100 (77.0%)|
|B10|post|P1R52 Soft AlphaCache-ON / Structural-H OFF|10/10 (100.0%)|18/20 (90.0%)|8/10 (80.0%)|79/100 (79.0%)|
|B10|post|Corrected Official AlphaEdit cache_c ON|10/10 (100.0%)|20/20 (100.0%)|10/10 (100.0%)|71/100 (71.0%)|
|B10|post|Official MEMIT Sequential|10/10 (100.0%)|19/20 (95.0%)|9/10 (90.0%)|80/100 (80.0%)|
|B10|final-W10|P1R52 Soft Structural-H ON|10/10 (100.0%)|18/20 (90.0%)|8/10 (80.0%)|77/100 (77.0%)|
|B10|final-W10|P1R52 Soft AlphaCache-ON / Structural-H OFF|10/10 (100.0%)|18/20 (90.0%)|8/10 (80.0%)|79/100 (79.0%)|
|B10|final-W10|Corrected Official AlphaEdit cache_c ON|10/10 (100.0%)|20/20 (100.0%)|10/10 (100.0%)|71/100 (71.0%)|
|B10|final-W10|Official MEMIT Sequential|10/10 (100.0%)|19/20 (95.0%)|9/10 (90.0%)|80/100 (80.0%)|

## Accuracy/strict 별도표

|B|Phase|Arm|Rewrite Acc|Rephrase Acc|Rephrase Acc strict|
|---:|---|---|---:|---:|---:|
|B1|entry-pre|P1R52 Soft Structural-H ON|0/10 (0.0%)|0/20 (0.0%)|0/10 (0.0%)|
|B1|entry-pre|P1R52 Soft AlphaCache-ON / Structural-H OFF|0/10 (0.0%)|0/20 (0.0%)|0/10 (0.0%)|
|B1|entry-pre|Corrected Official AlphaEdit cache_c ON|0/10 (0.0%)|0/20 (0.0%)|0/10 (0.0%)|
|B1|entry-pre|Official MEMIT Sequential|0/10 (0.0%)|0/20 (0.0%)|0/10 (0.0%)|
|B1|post|P1R52 Soft Structural-H ON|10/10 (100.0%)|10/20 (50.0%)|3/10 (30.0%)|
|B1|post|P1R52 Soft AlphaCache-ON / Structural-H OFF|10/10 (100.0%)|10/20 (50.0%)|3/10 (30.0%)|
|B1|post|Corrected Official AlphaEdit cache_c ON|10/10 (100.0%)|13/20 (65.0%)|4/10 (40.0%)|
|B1|post|Official MEMIT Sequential|10/10 (100.0%)|11/20 (55.0%)|3/10 (30.0%)|
|B1|final-W10|P1R52 Soft Structural-H ON|10/10 (100.0%)|10/20 (50.0%)|2/10 (20.0%)|
|B1|final-W10|P1R52 Soft AlphaCache-ON / Structural-H OFF|10/10 (100.0%)|9/20 (45.0%)|2/10 (20.0%)|
|B1|final-W10|Corrected Official AlphaEdit cache_c ON|10/10 (100.0%)|15/20 (75.0%)|5/10 (50.0%)|
|B1|final-W10|Official MEMIT Sequential|10/10 (100.0%)|12/20 (60.0%)|4/10 (40.0%)|
|B2|entry-pre|P1R52 Soft Structural-H ON|0/10 (0.0%)|0/20 (0.0%)|0/10 (0.0%)|
|B2|entry-pre|P1R52 Soft AlphaCache-ON / Structural-H OFF|0/10 (0.0%)|0/20 (0.0%)|0/10 (0.0%)|
|B2|entry-pre|Corrected Official AlphaEdit cache_c ON|0/10 (0.0%)|0/20 (0.0%)|0/10 (0.0%)|
|B2|entry-pre|Official MEMIT Sequential|0/10 (0.0%)|0/20 (0.0%)|0/10 (0.0%)|
|B2|post|P1R52 Soft Structural-H ON|10/10 (100.0%)|9/20 (45.0%)|2/10 (20.0%)|
|B2|post|P1R52 Soft AlphaCache-ON / Structural-H OFF|10/10 (100.0%)|9/20 (45.0%)|2/10 (20.0%)|
|B2|post|Corrected Official AlphaEdit cache_c ON|10/10 (100.0%)|18/20 (90.0%)|8/10 (80.0%)|
|B2|post|Official MEMIT Sequential|10/10 (100.0%)|12/20 (60.0%)|5/10 (50.0%)|
|B2|final-W10|P1R52 Soft Structural-H ON|10/10 (100.0%)|10/20 (50.0%)|3/10 (30.0%)|
|B2|final-W10|P1R52 Soft AlphaCache-ON / Structural-H OFF|10/10 (100.0%)|10/20 (50.0%)|3/10 (30.0%)|
|B2|final-W10|Corrected Official AlphaEdit cache_c ON|10/10 (100.0%)|18/20 (90.0%)|8/10 (80.0%)|
|B2|final-W10|Official MEMIT Sequential|9/10 (90.0%)|13/20 (65.0%)|5/10 (50.0%)|
|B3|entry-pre|P1R52 Soft Structural-H ON|0/10 (0.0%)|0/20 (0.0%)|0/10 (0.0%)|
|B3|entry-pre|P1R52 Soft AlphaCache-ON / Structural-H OFF|0/10 (0.0%)|0/20 (0.0%)|0/10 (0.0%)|
|B3|entry-pre|Corrected Official AlphaEdit cache_c ON|0/10 (0.0%)|0/20 (0.0%)|0/10 (0.0%)|
|B3|entry-pre|Official MEMIT Sequential|0/10 (0.0%)|0/20 (0.0%)|0/10 (0.0%)|
|B3|post|P1R52 Soft Structural-H ON|10/10 (100.0%)|8/20 (40.0%)|2/10 (20.0%)|
|B3|post|P1R52 Soft AlphaCache-ON / Structural-H OFF|10/10 (100.0%)|8/20 (40.0%)|2/10 (20.0%)|
|B3|post|Corrected Official AlphaEdit cache_c ON|10/10 (100.0%)|10/20 (50.0%)|3/10 (30.0%)|
|B3|post|Official MEMIT Sequential|10/10 (100.0%)|5/20 (25.0%)|1/10 (10.0%)|
|B3|final-W10|P1R52 Soft Structural-H ON|10/10 (100.0%)|8/20 (40.0%)|2/10 (20.0%)|
|B3|final-W10|P1R52 Soft AlphaCache-ON / Structural-H OFF|10/10 (100.0%)|8/20 (40.0%)|2/10 (20.0%)|
|B3|final-W10|Corrected Official AlphaEdit cache_c ON|10/10 (100.0%)|10/20 (50.0%)|3/10 (30.0%)|
|B3|final-W10|Official MEMIT Sequential|10/10 (100.0%)|6/20 (30.0%)|2/10 (20.0%)|
|B4|entry-pre|P1R52 Soft Structural-H ON|0/10 (0.0%)|0/20 (0.0%)|0/10 (0.0%)|
|B4|entry-pre|P1R52 Soft AlphaCache-ON / Structural-H OFF|0/10 (0.0%)|0/20 (0.0%)|0/10 (0.0%)|
|B4|entry-pre|Corrected Official AlphaEdit cache_c ON|0/10 (0.0%)|0/20 (0.0%)|0/10 (0.0%)|
|B4|entry-pre|Official MEMIT Sequential|0/10 (0.0%)|0/20 (0.0%)|0/10 (0.0%)|
|B4|post|P1R52 Soft Structural-H ON|10/10 (100.0%)|10/20 (50.0%)|4/10 (40.0%)|
|B4|post|P1R52 Soft AlphaCache-ON / Structural-H OFF|10/10 (100.0%)|10/20 (50.0%)|4/10 (40.0%)|
|B4|post|Corrected Official AlphaEdit cache_c ON|10/10 (100.0%)|12/20 (60.0%)|5/10 (50.0%)|
|B4|post|Official MEMIT Sequential|9/10 (90.0%)|11/20 (55.0%)|5/10 (50.0%)|
|B4|final-W10|P1R52 Soft Structural-H ON|10/10 (100.0%)|10/20 (50.0%)|4/10 (40.0%)|
|B4|final-W10|P1R52 Soft AlphaCache-ON / Structural-H OFF|10/10 (100.0%)|10/20 (50.0%)|4/10 (40.0%)|
|B4|final-W10|Corrected Official AlphaEdit cache_c ON|10/10 (100.0%)|12/20 (60.0%)|5/10 (50.0%)|
|B4|final-W10|Official MEMIT Sequential|9/10 (90.0%)|11/20 (55.0%)|5/10 (50.0%)|
|B5|entry-pre|P1R52 Soft Structural-H ON|0/10 (0.0%)|0/20 (0.0%)|0/10 (0.0%)|
|B5|entry-pre|P1R52 Soft AlphaCache-ON / Structural-H OFF|0/10 (0.0%)|0/20 (0.0%)|0/10 (0.0%)|
|B5|entry-pre|Corrected Official AlphaEdit cache_c ON|0/10 (0.0%)|1/20 (5.0%)|0/10 (0.0%)|
|B5|entry-pre|Official MEMIT Sequential|0/10 (0.0%)|1/20 (5.0%)|0/10 (0.0%)|
|B5|post|P1R52 Soft Structural-H ON|10/10 (100.0%)|11/20 (55.0%)|3/10 (30.0%)|
|B5|post|P1R52 Soft AlphaCache-ON / Structural-H OFF|10/10 (100.0%)|12/20 (60.0%)|4/10 (40.0%)|
|B5|post|Corrected Official AlphaEdit cache_c ON|10/10 (100.0%)|14/20 (70.0%)|5/10 (50.0%)|
|B5|post|Official MEMIT Sequential|10/10 (100.0%)|12/20 (60.0%)|5/10 (50.0%)|
|B5|final-W10|P1R52 Soft Structural-H ON|10/10 (100.0%)|14/20 (70.0%)|5/10 (50.0%)|
|B5|final-W10|P1R52 Soft AlphaCache-ON / Structural-H OFF|10/10 (100.0%)|14/20 (70.0%)|5/10 (50.0%)|
|B5|final-W10|Corrected Official AlphaEdit cache_c ON|10/10 (100.0%)|14/20 (70.0%)|5/10 (50.0%)|
|B5|final-W10|Official MEMIT Sequential|10/10 (100.0%)|12/20 (60.0%)|5/10 (50.0%)|
|B6|entry-pre|P1R52 Soft Structural-H ON|0/10 (0.0%)|0/20 (0.0%)|0/10 (0.0%)|
|B6|entry-pre|P1R52 Soft AlphaCache-ON / Structural-H OFF|0/10 (0.0%)|0/20 (0.0%)|0/10 (0.0%)|
|B6|entry-pre|Corrected Official AlphaEdit cache_c ON|0/10 (0.0%)|0/20 (0.0%)|0/10 (0.0%)|
|B6|entry-pre|Official MEMIT Sequential|0/10 (0.0%)|0/20 (0.0%)|0/10 (0.0%)|
|B6|post|P1R52 Soft Structural-H ON|10/10 (100.0%)|11/20 (55.0%)|3/10 (30.0%)|
|B6|post|P1R52 Soft AlphaCache-ON / Structural-H OFF|10/10 (100.0%)|10/20 (50.0%)|2/10 (20.0%)|
|B6|post|Corrected Official AlphaEdit cache_c ON|10/10 (100.0%)|13/20 (65.0%)|5/10 (50.0%)|
|B6|post|Official MEMIT Sequential|10/10 (100.0%)|10/20 (50.0%)|2/10 (20.0%)|
|B6|final-W10|P1R52 Soft Structural-H ON|10/10 (100.0%)|11/20 (55.0%)|3/10 (30.0%)|
|B6|final-W10|P1R52 Soft AlphaCache-ON / Structural-H OFF|10/10 (100.0%)|10/20 (50.0%)|2/10 (20.0%)|
|B6|final-W10|Corrected Official AlphaEdit cache_c ON|10/10 (100.0%)|13/20 (65.0%)|5/10 (50.0%)|
|B6|final-W10|Official MEMIT Sequential|10/10 (100.0%)|11/20 (55.0%)|2/10 (20.0%)|
|B7|entry-pre|P1R52 Soft Structural-H ON|0/10 (0.0%)|0/20 (0.0%)|0/10 (0.0%)|
|B7|entry-pre|P1R52 Soft AlphaCache-ON / Structural-H OFF|0/10 (0.0%)|0/20 (0.0%)|0/10 (0.0%)|
|B7|entry-pre|Corrected Official AlphaEdit cache_c ON|0/10 (0.0%)|0/20 (0.0%)|0/10 (0.0%)|
|B7|entry-pre|Official MEMIT Sequential|0/10 (0.0%)|0/20 (0.0%)|0/10 (0.0%)|
|B7|post|P1R52 Soft Structural-H ON|10/10 (100.0%)|14/20 (70.0%)|6/10 (60.0%)|
|B7|post|P1R52 Soft AlphaCache-ON / Structural-H OFF|10/10 (100.0%)|14/20 (70.0%)|6/10 (60.0%)|
|B7|post|Corrected Official AlphaEdit cache_c ON|10/10 (100.0%)|15/20 (75.0%)|7/10 (70.0%)|
|B7|post|Official MEMIT Sequential|10/10 (100.0%)|11/20 (55.0%)|4/10 (40.0%)|
|B7|final-W10|P1R52 Soft Structural-H ON|10/10 (100.0%)|14/20 (70.0%)|6/10 (60.0%)|
|B7|final-W10|P1R52 Soft AlphaCache-ON / Structural-H OFF|10/10 (100.0%)|14/20 (70.0%)|6/10 (60.0%)|
|B7|final-W10|Corrected Official AlphaEdit cache_c ON|10/10 (100.0%)|15/20 (75.0%)|7/10 (70.0%)|
|B7|final-W10|Official MEMIT Sequential|10/10 (100.0%)|13/20 (65.0%)|6/10 (60.0%)|
|B8|entry-pre|P1R52 Soft Structural-H ON|0/10 (0.0%)|0/20 (0.0%)|0/10 (0.0%)|
|B8|entry-pre|P1R52 Soft AlphaCache-ON / Structural-H OFF|0/10 (0.0%)|0/20 (0.0%)|0/10 (0.0%)|
|B8|entry-pre|Corrected Official AlphaEdit cache_c ON|0/10 (0.0%)|0/20 (0.0%)|0/10 (0.0%)|
|B8|entry-pre|Official MEMIT Sequential|0/10 (0.0%)|0/20 (0.0%)|0/10 (0.0%)|
|B8|post|P1R52 Soft Structural-H ON|10/10 (100.0%)|8/20 (40.0%)|3/10 (30.0%)|
|B8|post|P1R52 Soft AlphaCache-ON / Structural-H OFF|10/10 (100.0%)|9/20 (45.0%)|3/10 (30.0%)|
|B8|post|Corrected Official AlphaEdit cache_c ON|10/10 (100.0%)|13/20 (65.0%)|4/10 (40.0%)|
|B8|post|Official MEMIT Sequential|10/10 (100.0%)|8/20 (40.0%)|2/10 (20.0%)|
|B8|final-W10|P1R52 Soft Structural-H ON|10/10 (100.0%)|8/20 (40.0%)|3/10 (30.0%)|
|B8|final-W10|P1R52 Soft AlphaCache-ON / Structural-H OFF|10/10 (100.0%)|8/20 (40.0%)|3/10 (30.0%)|
|B8|final-W10|Corrected Official AlphaEdit cache_c ON|10/10 (100.0%)|13/20 (65.0%)|4/10 (40.0%)|
|B8|final-W10|Official MEMIT Sequential|10/10 (100.0%)|7/20 (35.0%)|2/10 (20.0%)|
|B9|entry-pre|P1R52 Soft Structural-H ON|0/10 (0.0%)|0/20 (0.0%)|0/10 (0.0%)|
|B9|entry-pre|P1R52 Soft AlphaCache-ON / Structural-H OFF|0/10 (0.0%)|0/20 (0.0%)|0/10 (0.0%)|
|B9|entry-pre|Corrected Official AlphaEdit cache_c ON|0/10 (0.0%)|0/20 (0.0%)|0/10 (0.0%)|
|B9|entry-pre|Official MEMIT Sequential|0/10 (0.0%)|0/20 (0.0%)|0/10 (0.0%)|
|B9|post|P1R52 Soft Structural-H ON|10/10 (100.0%)|12/20 (60.0%)|4/10 (40.0%)|
|B9|post|P1R52 Soft AlphaCache-ON / Structural-H OFF|10/10 (100.0%)|12/20 (60.0%)|4/10 (40.0%)|
|B9|post|Corrected Official AlphaEdit cache_c ON|10/10 (100.0%)|17/20 (85.0%)|7/10 (70.0%)|
|B9|post|Official MEMIT Sequential|10/10 (100.0%)|12/20 (60.0%)|4/10 (40.0%)|
|B9|final-W10|P1R52 Soft Structural-H ON|10/10 (100.0%)|12/20 (60.0%)|4/10 (40.0%)|
|B9|final-W10|P1R52 Soft AlphaCache-ON / Structural-H OFF|10/10 (100.0%)|12/20 (60.0%)|4/10 (40.0%)|
|B9|final-W10|Corrected Official AlphaEdit cache_c ON|10/10 (100.0%)|17/20 (85.0%)|7/10 (70.0%)|
|B9|final-W10|Official MEMIT Sequential|10/10 (100.0%)|12/20 (60.0%)|4/10 (40.0%)|
|B10|entry-pre|P1R52 Soft Structural-H ON|0/10 (0.0%)|0/20 (0.0%)|0/10 (0.0%)|
|B10|entry-pre|P1R52 Soft AlphaCache-ON / Structural-H OFF|0/10 (0.0%)|0/20 (0.0%)|0/10 (0.0%)|
|B10|entry-pre|Corrected Official AlphaEdit cache_c ON|0/10 (0.0%)|0/20 (0.0%)|0/10 (0.0%)|
|B10|entry-pre|Official MEMIT Sequential|0/10 (0.0%)|1/20 (5.0%)|0/10 (0.0%)|
|B10|post|P1R52 Soft Structural-H ON|10/10 (100.0%)|12/20 (60.0%)|5/10 (50.0%)|
|B10|post|P1R52 Soft AlphaCache-ON / Structural-H OFF|10/10 (100.0%)|12/20 (60.0%)|5/10 (50.0%)|
|B10|post|Corrected Official AlphaEdit cache_c ON|10/10 (100.0%)|17/20 (85.0%)|7/10 (70.0%)|
|B10|post|Official MEMIT Sequential|10/10 (100.0%)|11/20 (55.0%)|3/10 (30.0%)|
|B10|final-W10|P1R52 Soft Structural-H ON|10/10 (100.0%)|12/20 (60.0%)|5/10 (50.0%)|
|B10|final-W10|P1R52 Soft AlphaCache-ON / Structural-H OFF|10/10 (100.0%)|12/20 (60.0%)|5/10 (50.0%)|
|B10|final-W10|Corrected Official AlphaEdit cache_c ON|10/10 (100.0%)|17/20 (85.0%)|7/10 (70.0%)|
|B10|final-W10|Official MEMIT Sequential|10/10 (100.0%)|11/20 (55.0%)|3/10 (30.0%)|

## NLL / margin / z8-W8

|B|Phase|Arm|Rewrite new/true/margin|Rephrase new/true/margin|z8/W8/gap full-six|
|---:|---|---|---:|---:|---:|
|B1|entry-pre|P1R52 Soft Structural-H ON|12.385938/6.364063/-6.021875|10.639062/5.362793/-5.276270|NOT_RECORDED_OUTER_K8_ONLY|
|B1|entry-pre|P1R52 Soft AlphaCache-ON / Structural-H OFF|12.385938/6.364063/-6.021875|10.639062/5.362793/-5.276270|NOT_RECORDED_OUTER_K8_ONLY|
|B1|entry-pre|Corrected Official AlphaEdit cache_c ON|12.385938/6.364063/-6.021875|10.639062/5.362793/-5.276270|NOT_RECORDED_OFFICIAL_BASELINE|
|B1|entry-pre|Official MEMIT Sequential|12.385938/6.364063/-6.021875|10.639062/5.362793/-5.276270|NOT_RECORDED_OFFICIAL_BASELINE|
|B1|post|P1R52 Soft Structural-H ON|0.023461/12.462500/+12.439039|2.385339/9.069531/+6.684192|0.055925/0.060898/+0.004974|
|B1|post|P1R52 Soft AlphaCache-ON / Structural-H OFF|0.023461/12.462500/+12.439039|2.385339/9.069531/+6.684192|0.055925/0.060898/+0.004974|
|B1|post|Corrected Official AlphaEdit cache_c ON|0.001658/15.475000/+15.473342|1.767157/10.563281/+8.796124|NOT_RECORDED_OFFICIAL_BASELINE|
|B1|post|Official MEMIT Sequential|0.005394/13.318750/+13.313356|2.626630/9.331250/+6.704620|NOT_RECORDED_OFFICIAL_BASELINE|
|B1|final-W10|P1R52 Soft Structural-H ON|0.025919/12.237500/+12.211581|2.433472/9.032422/+6.598950|NOT_RECORDED_OUTER_K8_ONLY|
|B1|final-W10|P1R52 Soft AlphaCache-ON / Structural-H OFF|0.026359/12.206250/+12.179891|2.458618/9.019531/+6.560913|NOT_RECORDED_OUTER_K8_ONLY|
|B1|final-W10|Corrected Official AlphaEdit cache_c ON|0.002040/15.375000/+15.372960|1.761728/10.509375/+8.747647|NOT_RECORDED_OFFICIAL_BASELINE|
|B1|final-W10|Official MEMIT Sequential|0.005729/14.187500/+14.181771|2.162412/10.090234/+7.927822|NOT_RECORDED_OFFICIAL_BASELINE|
|B2|entry-pre|P1R52 Soft Structural-H ON|11.965625/3.786719/-8.178906|10.163281/4.978125/-5.185156|NOT_RECORDED_OUTER_K8_ONLY|
|B2|entry-pre|P1R52 Soft AlphaCache-ON / Structural-H OFF|11.965625/3.786719/-8.178906|10.163281/4.978125/-5.185156|NOT_RECORDED_OUTER_K8_ONLY|
|B2|entry-pre|Corrected Official AlphaEdit cache_c ON|11.981250/3.940625/-8.040625|10.141406/4.999609/-5.141797|NOT_RECORDED_OFFICIAL_BASELINE|
|B2|entry-pre|Official MEMIT Sequential|12.062500/3.982813/-8.079688|10.235938/5.100488/-5.135449|NOT_RECORDED_OFFICIAL_BASELINE|
|B2|post|P1R52 Soft Structural-H ON|0.030971/11.403125/+11.372154|2.383398/7.635156/+5.251758|0.028600/0.040879/+0.012280|
|B2|post|P1R52 Soft AlphaCache-ON / Structural-H OFF|0.031706/11.284375/+11.252669|2.435315/7.604687/+5.169373|0.027509/0.038685/+0.011176|
|B2|post|Corrected Official AlphaEdit cache_c ON|0.000570/17.331250/+17.330680|0.796198/10.218750/+9.422552|NOT_RECORDED_OFFICIAL_BASELINE|
|B2|post|Official MEMIT Sequential|0.052026/11.531250/+11.479224|2.589008/7.139844/+4.550836|NOT_RECORDED_OFFICIAL_BASELINE|
|B2|final-W10|P1R52 Soft Structural-H ON|0.033751/11.000000/+10.966249|2.287329/7.506250/+5.218921|NOT_RECORDED_OUTER_K8_ONLY|
|B2|final-W10|P1R52 Soft AlphaCache-ON / Structural-H OFF|0.036714/10.856250/+10.819536|2.312695/7.333594/+5.020898|NOT_RECORDED_OUTER_K8_ONLY|
|B2|final-W10|Corrected Official AlphaEdit cache_c ON|0.000790/17.506250/+17.505460|0.779749/10.531250/+9.751501|NOT_RECORDED_OFFICIAL_BASELINE|
|B2|final-W10|Official MEMIT Sequential|0.208564/13.956250/+13.747686|1.535216/8.347266/+6.812050|NOT_RECORDED_OFFICIAL_BASELINE|
|B3|entry-pre|P1R52 Soft Structural-H ON|9.266016/5.051465/-4.214551|10.358801/5.919080/-4.439722|NOT_RECORDED_OUTER_K8_ONLY|
|B3|entry-pre|P1R52 Soft AlphaCache-ON / Structural-H OFF|9.278540/5.055664/-4.222876|10.344739/5.911353/-4.433386|NOT_RECORDED_OUTER_K8_ONLY|
|B3|entry-pre|Corrected Official AlphaEdit cache_c ON|9.227429/4.918262/-4.309167|10.230457/5.886060/-4.344397|NOT_RECORDED_OFFICIAL_BASELINE|
|B3|entry-pre|Official MEMIT Sequential|9.343848/5.018408/-4.325439|10.345880/5.940356/-4.405524|NOT_RECORDED_OFFICIAL_BASELINE|
|B3|post|P1R52 Soft Structural-H ON|0.052677/10.828125/+10.775448|3.280365/7.391016/+4.110650|0.042461/0.071162/+0.028701|
|B3|post|P1R52 Soft AlphaCache-ON / Structural-H OFF|0.045762/11.112500/+11.066738|3.316881/7.411719/+4.094838|0.041262/0.059778/+0.018516|
|B3|post|Corrected Official AlphaEdit cache_c ON|0.000427/16.518750/+16.518323|2.642706/10.210156/+7.567450|NOT_RECORDED_OFFICIAL_BASELINE|
|B3|post|Official MEMIT Sequential|0.018636/12.282813/+12.264177|4.345114/8.144824/+3.799710|NOT_RECORDED_OFFICIAL_BASELINE|
|B3|final-W10|P1R52 Soft Structural-H ON|0.049213/10.743750/+10.694537|3.226418/7.243945/+4.017527|NOT_RECORDED_OUTER_K8_ONLY|
|B3|final-W10|P1R52 Soft AlphaCache-ON / Structural-H OFF|0.039105/11.046875/+11.007770|3.240351/7.260742/+4.020391|NOT_RECORDED_OUTER_K8_ONLY|
|B3|final-W10|Corrected Official AlphaEdit cache_c ON|0.000492/16.218750/+16.218258|2.614275/10.043359/+7.429085|NOT_RECORDED_OFFICIAL_BASELINE|
|B3|final-W10|Official MEMIT Sequential|0.028711/13.028906/+13.000195|4.152859/8.746460/+4.593601|NOT_RECORDED_OFFICIAL_BASELINE|
|B4|entry-pre|P1R52 Soft Structural-H ON|11.262500/3.461267/-7.801233|9.226562/2.574289/-6.652274|NOT_RECORDED_OUTER_K8_ONLY|
|B4|entry-pre|P1R52 Soft AlphaCache-ON / Structural-H OFF|11.256250/3.461530/-7.794720|9.243750/2.578638/-6.665112|NOT_RECORDED_OUTER_K8_ONLY|
|B4|entry-pre|Corrected Official AlphaEdit cache_c ON|10.946875/3.181317/-7.765558|9.384375/2.557324/-6.827051|NOT_RECORDED_OFFICIAL_BASELINE|
|B4|entry-pre|Official MEMIT Sequential|11.265625/3.590692/-7.674933|9.292188/2.570306/-6.721881|NOT_RECORDED_OFFICIAL_BASELINE|
|B4|post|P1R52 Soft Structural-H ON|0.071910/9.353125/+9.281215|3.620538/5.054492/+1.433954|0.289644/0.308589/+0.018944|
|B4|post|P1R52 Soft AlphaCache-ON / Structural-H OFF|0.086682/8.259375/+8.172693|3.721786/4.929956/+1.208170|0.271689/0.304906/+0.033217|
|B4|post|Corrected Official AlphaEdit cache_c ON|0.000711/13.125000/+13.124289|3.517872/7.895508/+4.377636|NOT_RECORDED_OFFICIAL_BASELINE|
|B4|post|Official MEMIT Sequential|0.660022/10.121875/+9.461853|3.671301/6.627637/+2.956336|NOT_RECORDED_OFFICIAL_BASELINE|
|B4|final-W10|P1R52 Soft Structural-H ON|0.082506/9.259375/+9.176869|3.546207/5.015039/+1.468832|NOT_RECORDED_OUTER_K8_ONLY|
|B4|final-W10|P1R52 Soft AlphaCache-ON / Structural-H OFF|0.085779/8.193750/+8.107971|3.649948/4.885596/+1.235648|NOT_RECORDED_OUTER_K8_ONLY|
|B4|final-W10|Corrected Official AlphaEdit cache_c ON|0.000739/12.925000/+12.924261|3.447289/7.972998/+4.525710|NOT_RECORDED_OFFICIAL_BASELINE|
|B4|final-W10|Official MEMIT Sequential|0.541682/10.325000/+9.783318|3.518386/6.835352/+3.316966|NOT_RECORDED_OFFICIAL_BASELINE|
|B5|entry-pre|P1R52 Soft Structural-H ON|10.398438/5.652344/-4.746094|9.483984/5.354492/-4.129492|NOT_RECORDED_OUTER_K8_ONLY|
|B5|entry-pre|P1R52 Soft AlphaCache-ON / Structural-H OFF|10.403125/5.669531/-4.733594|9.489062/5.331299/-4.157764|NOT_RECORDED_OUTER_K8_ONLY|
|B5|entry-pre|Corrected Official AlphaEdit cache_c ON|10.273438/5.787891/-4.485547|9.389453/5.344824/-4.044629|NOT_RECORDED_OFFICIAL_BASELINE|
|B5|entry-pre|Official MEMIT Sequential|10.226562/5.922266/-4.304297|9.326172/5.569775/-3.756396|NOT_RECORDED_OFFICIAL_BASELINE|
|B5|post|P1R52 Soft Structural-H ON|0.124084/9.443750/+9.319666|2.162231/6.666797/+4.504565|0.035147/0.127528/+0.092381|
|B5|post|P1R52 Soft AlphaCache-ON / Structural-H OFF|0.126923/9.562500/+9.435577|2.011401/6.773047/+4.761646|0.038226/0.118751/+0.080524|
|B5|post|Corrected Official AlphaEdit cache_c ON|0.000577/14.893750/+14.893173|1.596143/9.587891/+7.991748|NOT_RECORDED_OFFICIAL_BASELINE|
|B5|post|Official MEMIT Sequential|0.130315/11.018750/+10.888435|2.733736/7.984766/+5.251029|NOT_RECORDED_OFFICIAL_BASELINE|
|B5|final-W10|P1R52 Soft Structural-H ON|0.127118/9.531250/+9.404132|1.925513/6.757031/+4.831519|NOT_RECORDED_OUTER_K8_ONLY|
|B5|final-W10|P1R52 Soft AlphaCache-ON / Structural-H OFF|0.121558/9.684375/+9.562817|1.882227/6.855859/+4.973633|NOT_RECORDED_OUTER_K8_ONLY|
|B5|final-W10|Corrected Official AlphaEdit cache_c ON|0.000649/14.793750/+14.793101|1.585985/9.648438/+8.062452|NOT_RECORDED_OFFICIAL_BASELINE|
|B5|final-W10|Official MEMIT Sequential|0.030116/12.093750/+12.063634|2.308370/8.535937/+6.227568|NOT_RECORDED_OFFICIAL_BASELINE|
|B6|entry-pre|P1R52 Soft Structural-H ON|9.840625/5.283984/-4.556641|8.678906/5.163843/-3.515063|NOT_RECORDED_OUTER_K8_ONLY|
|B6|entry-pre|P1R52 Soft AlphaCache-ON / Structural-H OFF|9.875000/5.274707/-4.600293|8.710938/5.141528/-3.569409|NOT_RECORDED_OUTER_K8_ONLY|
|B6|entry-pre|Corrected Official AlphaEdit cache_c ON|9.962500/4.989697/-4.972803|8.728125/4.851318/-3.876807|NOT_RECORDED_OFFICIAL_BASELINE|
|B6|entry-pre|Official MEMIT Sequential|10.037500/5.539111/-4.498389|8.746875/5.152197/-3.594678|NOT_RECORDED_OFFICIAL_BASELINE|
|B6|post|P1R52 Soft Structural-H ON|0.018686/11.825000/+11.806314|1.903076/8.421094/+6.518018|0.012150/0.030843/+0.018694|
|B6|post|P1R52 Soft AlphaCache-ON / Structural-H OFF|0.022836/12.281250/+12.258414|2.050635/8.670312/+6.619678|0.012298/0.033437/+0.021139|
|B6|post|Corrected Official AlphaEdit cache_c ON|0.000711/16.218750/+16.218039|1.247578/10.742969/+9.495391|NOT_RECORDED_OFFICIAL_BASELINE|
|B6|post|Official MEMIT Sequential|0.043232/13.068750/+13.025518|2.109558/8.007031/+5.897473|NOT_RECORDED_OFFICIAL_BASELINE|
|B6|final-W10|P1R52 Soft Structural-H ON|0.018903/11.931250/+11.912347|1.839575/8.411328/+6.571753|NOT_RECORDED_OUTER_K8_ONLY|
|B6|final-W10|P1R52 Soft AlphaCache-ON / Structural-H OFF|0.022513/12.287500/+12.264987|2.027548/8.648438/+6.620889|NOT_RECORDED_OUTER_K8_ONLY|
|B6|final-W10|Corrected Official AlphaEdit cache_c ON|0.003489/15.775000/+15.771511|1.223346/10.553125/+9.329779|NOT_RECORDED_OFFICIAL_BASELINE|
|B6|final-W10|Official MEMIT Sequential|0.011819/14.387500/+14.375681|1.742734/9.012891/+7.270156|NOT_RECORDED_OFFICIAL_BASELINE|
|B7|entry-pre|P1R52 Soft Structural-H ON|12.418750/4.660953/-7.757797|10.570312/4.510889/-6.059423|NOT_RECORDED_OUTER_K8_ONLY|
|B7|entry-pre|P1R52 Soft AlphaCache-ON / Structural-H OFF|12.475000/4.710170/-7.764830|10.589844/4.517140/-6.072704|NOT_RECORDED_OUTER_K8_ONLY|
|B7|entry-pre|Corrected Official AlphaEdit cache_c ON|12.550000/4.860943/-7.689057|11.231250/4.801831/-6.429419|NOT_RECORDED_OFFICIAL_BASELINE|
|B7|entry-pre|Official MEMIT Sequential|12.621875/5.027367/-7.594508|11.068750/4.833759/-6.234991|NOT_RECORDED_OFFICIAL_BASELINE|
|B7|post|P1R52 Soft Structural-H ON|0.020410/10.428125/+10.407715|1.417386/9.105134/+7.687748|0.014237/0.022629/+0.008392|
|B7|post|P1R52 Soft AlphaCache-ON / Structural-H OFF|0.020165/10.450000/+10.429835|1.249432/9.202034/+7.952602|0.013854/0.020157/+0.006303|
|B7|post|Corrected Official AlphaEdit cache_c ON|0.001394/14.406251/+14.404858|0.905770/12.627690/+11.721920|NOT_RECORDED_OFFICIAL_BASELINE|
|B7|post|Official MEMIT Sequential|0.059418/12.140626/+12.081208|1.728072/9.620767/+7.892695|NOT_RECORDED_OFFICIAL_BASELINE|
|B7|final-W10|P1R52 Soft Structural-H ON|0.020508/10.328125/+10.307618|1.378459/9.116859/+7.738400|NOT_RECORDED_OUTER_K8_ONLY|
|B7|final-W10|P1R52 Soft AlphaCache-ON / Structural-H OFF|0.024921/10.137500/+10.112580|1.247598/9.109074/+7.861475|NOT_RECORDED_OUTER_K8_ONLY|
|B7|final-W10|Corrected Official AlphaEdit cache_c ON|0.001427/14.196877/+14.195450|0.895570/12.530800/+11.635230|NOT_RECORDED_OFFICIAL_BASELINE|
|B7|final-W10|Official MEMIT Sequential|0.013517/11.771876/+11.758359|1.715175/10.220783/+8.505608|NOT_RECORDED_OFFICIAL_BASELINE|
|B8|entry-pre|P1R52 Soft Structural-H ON|10.353125/3.655640/-6.697485|10.096094/5.229199/-4.866895|NOT_RECORDED_OUTER_K8_ONLY|
|B8|entry-pre|P1R52 Soft AlphaCache-ON / Structural-H OFF|10.331250/3.663208/-6.668042|10.139844/5.247168/-4.892676|NOT_RECORDED_OUTER_K8_ONLY|
|B8|entry-pre|Corrected Official AlphaEdit cache_c ON|10.310937/4.299048/-6.011890|10.378125/5.573193/-4.804932|NOT_RECORDED_OFFICIAL_BASELINE|
|B8|entry-pre|Official MEMIT Sequential|10.879687/4.251755/-6.627933|10.537500/5.523340/-5.014160|NOT_RECORDED_OFFICIAL_BASELINE|
|B8|post|P1R52 Soft Structural-H ON|0.025361/10.925000/+10.899639|3.047363/8.524414/+5.477051|0.019813/0.025302/+0.005489|
|B8|post|P1R52 Soft AlphaCache-ON / Structural-H OFF|0.014580/10.990625/+10.976045|3.232268/8.841016/+5.608748|0.019635/0.021258/+0.001624|
|B8|post|Corrected Official AlphaEdit cache_c ON|0.000833/13.312500/+13.311667|1.503124/9.505469/+8.002345|NOT_RECORDED_OFFICIAL_BASELINE|
|B8|post|Official MEMIT Sequential|0.006640/10.359375/+10.352735|2.832425/8.439453/+5.607028|NOT_RECORDED_OFFICIAL_BASELINE|
|B8|final-W10|P1R52 Soft Structural-H ON|0.023513/10.915625/+10.892112|3.014746/8.487695/+5.472949|NOT_RECORDED_OUTER_K8_ONLY|
|B8|final-W10|P1R52 Soft AlphaCache-ON / Structural-H OFF|0.012924/11.084375/+11.071451|3.204694/8.776074/+5.571381|NOT_RECORDED_OUTER_K8_ONLY|
|B8|final-W10|Corrected Official AlphaEdit cache_c ON|0.000806/13.268750/+13.267944|1.476257/9.526172/+8.049915|NOT_RECORDED_OFFICIAL_BASELINE|
|B8|final-W10|Official MEMIT Sequential|0.003946/10.643750/+10.639804|2.536590/8.603711/+6.067121|NOT_RECORDED_OFFICIAL_BASELINE|
|B9|entry-pre|P1R52 Soft Structural-H ON|10.943750/2.648779/-8.294971|9.335938/3.423242/-5.912695|NOT_RECORDED_OUTER_K8_ONLY|
|B9|entry-pre|P1R52 Soft AlphaCache-ON / Structural-H OFF|10.865625/2.615796/-8.249829|9.176562/3.404346/-5.772217|NOT_RECORDED_OUTER_K8_ONLY|
|B9|entry-pre|Corrected Official AlphaEdit cache_c ON|10.275000/2.493066/-7.781934|9.071875/3.216162/-5.855713|NOT_RECORDED_OFFICIAL_BASELINE|
|B9|entry-pre|Official MEMIT Sequential|10.900000/2.632959/-8.267041|9.601562/3.457617/-6.143945|NOT_RECORDED_OFFICIAL_BASELINE|
|B9|post|P1R52 Soft Structural-H ON|0.013228/11.012500/+10.999272|2.035648/6.346875/+4.311227|0.018804/0.019809/+0.001006|
|B9|post|P1R52 Soft AlphaCache-ON / Structural-H OFF|0.014182/10.900000/+10.885818|2.056558/6.353125/+4.296567|0.018861/0.020105/+0.001245|
|B9|post|Corrected Official AlphaEdit cache_c ON|0.000766/15.000000/+14.999234|0.727843/11.281250/+10.553407|NOT_RECORDED_OFFICIAL_BASELINE|
|B9|post|Official MEMIT Sequential|0.005456/12.712500/+12.707044|1.341887/8.586719/+7.244832|NOT_RECORDED_OFFICIAL_BASELINE|
|B9|final-W10|P1R52 Soft Structural-H ON|0.013343/11.106250/+11.092907|2.107652/6.319531/+4.211879|NOT_RECORDED_OUTER_K8_ONLY|
|B9|final-W10|P1R52 Soft AlphaCache-ON / Structural-H OFF|0.014785/10.931250/+10.916465|2.086026/6.275781/+4.189755|NOT_RECORDED_OUTER_K8_ONLY|
|B9|final-W10|Corrected Official AlphaEdit cache_c ON|0.000753/14.987500/+14.986747|0.694644/11.242188/+10.547543|NOT_RECORDED_OFFICIAL_BASELINE|
|B9|final-W10|Official MEMIT Sequential|0.004342/13.018750/+13.014408|1.329067/8.675391/+7.346324|NOT_RECORDED_OFFICIAL_BASELINE|
|B10|entry-pre|P1R52 Soft Structural-H ON|10.450000/6.180078/-4.269922|9.439844/5.640332/-3.799512|NOT_RECORDED_OUTER_K8_ONLY|
|B10|entry-pre|P1R52 Soft AlphaCache-ON / Structural-H OFF|10.585938/6.166797/-4.419141|9.378125/5.663574/-3.714551|NOT_RECORDED_OUTER_K8_ONLY|
|B10|entry-pre|Corrected Official AlphaEdit cache_c ON|10.550000/7.973438/-2.576563|9.145312/6.797754/-2.347559|NOT_RECORDED_OFFICIAL_BASELINE|
|B10|entry-pre|Official MEMIT Sequential|10.634375/6.301953/-4.332422|8.929297/5.222070/-3.707227|NOT_RECORDED_OFFICIAL_BASELINE|
|B10|post|P1R52 Soft Structural-H ON|0.021262/12.478125/+12.456863|1.892896/8.486328/+6.593433|0.016665/0.029074/+0.012409|
|B10|post|P1R52 Soft AlphaCache-ON / Structural-H OFF|0.021832/12.615625/+12.593793|1.755432/8.611914/+6.856482|0.016603/0.023887/+0.007285|
|B10|post|Corrected Official AlphaEdit cache_c ON|0.001149/16.768750/+16.767601|0.646593/11.707031/+11.060438|NOT_RECORDED_OFFICIAL_BASELINE|
|B10|post|Official MEMIT Sequential|0.009984/13.412500/+13.402516|1.999236/9.637891/+7.638654|NOT_RECORDED_OFFICIAL_BASELINE|
|B10|final-W10|P1R52 Soft Structural-H ON|0.021262/12.478125/+12.456863|1.892896/8.486328/+6.593433|NOT_RECORDED_OUTER_K8_ONLY|
|B10|final-W10|P1R52 Soft AlphaCache-ON / Structural-H OFF|0.021832/12.615625/+12.593793|1.755432/8.611914/+6.856482|NOT_RECORDED_OUTER_K8_ONLY|
|B10|final-W10|Corrected Official AlphaEdit cache_c ON|0.001149/16.768750/+16.767601|0.646593/11.707031/+11.060438|NOT_RECORDED_OFFICIAL_BASELINE|
|B10|final-W10|Official MEMIT Sequential|0.009984/13.412500/+13.402516|1.999236/9.637891/+7.638654|NOT_RECORDED_OFFICIAL_BASELINE|

## P1R52 controller / routing; official baselines는 N/A

|B|Arm|Primary/Rescue/Current|request-step|clamp|route status|H/cache|
|---:|---|---:|---:|---:|---|---|
|B1|P1R52 Soft Structural-H ON|80/0/0|80|0|JOINT_WRITE:8|H statuses={'H_EMPTY_EXACT_ATOMIC_EQUIVALENCE': 8}|
|B1|P1R52 Soft AlphaCache-ON / Structural-H OFF|80/0/0|80|0|JOINT_WRITE:8|Alpha solve consume=0; H influence=0|
|B2|P1R52 Soft Structural-H ON|80/0/0|80|0|JOINT_WRITE:8|H statuses={'H_ACTIVE_CERTIFIED': 8}|
|B2|P1R52 Soft AlphaCache-ON / Structural-H OFF|80/0/0|80|0|JOINT_WRITE:8|Alpha solve consume=80; H influence=0|
|B3|P1R52 Soft Structural-H ON|80/0/0|80|0|JOINT_WRITE:8|H statuses={'H_ACTIVE_CERTIFIED': 8}|
|B3|P1R52 Soft AlphaCache-ON / Structural-H OFF|80/0/0|80|0|JOINT_WRITE:8|Alpha solve consume=160; H influence=0|
|B4|P1R52 Soft Structural-H ON|72/4/4|80|8|JOINT_WRITE:8|H statuses={'H_ACTIVE_CERTIFIED': 8}|
|B4|P1R52 Soft AlphaCache-ON / Structural-H OFF|74/2/4|80|10|JOINT_WRITE:8|Alpha solve consume=240; H influence=0|
|B5|P1R52 Soft Structural-H ON|80/0/0|80|5|JOINT_WRITE:8|H statuses={'H_ACTIVE_CERTIFIED': 8}|
|B5|P1R52 Soft AlphaCache-ON / Structural-H OFF|80/0/0|80|5|JOINT_WRITE:8|Alpha solve consume=320; H influence=0|
|B6|P1R52 Soft Structural-H ON|77/3/0|80|0|JOINT_WRITE:8|H statuses={'H_ACTIVE_CERTIFIED': 8}|
|B6|P1R52 Soft AlphaCache-ON / Structural-H OFF|78/2/0|80|0|JOINT_WRITE:8|Alpha solve consume=400; H influence=0|
|B7|P1R52 Soft Structural-H ON|79/0/1|80|0|JOINT_WRITE:8|H statuses={'H_ACTIVE_CERTIFIED': 8}|
|B7|P1R52 Soft AlphaCache-ON / Structural-H OFF|80/0/0|80|0|JOINT_WRITE:8|Alpha solve consume=480; H influence=0|
|B8|P1R52 Soft Structural-H ON|79/1/0|80|5|JOINT_WRITE:8|H statuses={'H_ACTIVE_CERTIFIED': 8}|
|B8|P1R52 Soft AlphaCache-ON / Structural-H OFF|79/1/0|80|0|JOINT_WRITE:8|Alpha solve consume=560; H influence=0|
|B9|P1R52 Soft Structural-H ON|80/0/0|80|0|JOINT_WRITE:8|H statuses={'H_ACTIVE_CERTIFIED': 8}|
|B9|P1R52 Soft AlphaCache-ON / Structural-H OFF|80/0/0|80|0|JOINT_WRITE:8|Alpha solve consume=640; H influence=0|
|B10|P1R52 Soft Structural-H ON|78/0/2|80|0|JOINT_WRITE:8|H statuses={'H_ACTIVE_CERTIFIED': 8}|
|B10|P1R52 Soft AlphaCache-ON / Structural-H OFF|78/1/1|80|0|JOINT_WRITE:8|Alpha solve consume=720; H influence=0|
|B1–B10|Corrected Official AlphaEdit cache_c ON|N/A|N/A|N/A|official method; P1R52 router 없음|dynamic cache_c|
|B1–B10|Official MEMIT Sequential|N/A|N/A|N/A|official method; P1R52 router 없음|static COV_CACHE|

## Official method cache semantics

|B|Alpha reset|Alpha width|Alpha consumed|Alpha cache norm entry→exit|MEMIT COV reused|MEMIT reset|historical decision influence|
|---:|---:|---:|---:|---:|---:|---:|---:|
|B1|True|0→10|False|0.0000→92.9110|0|0|0|
|B2|False|10→20|True|92.9110→141.8159|5|0|0|
|B3|False|20→30|True|141.8159→181.2386|5|0|0|
|B4|False|30→40|True|181.2386→213.4830|5|0|0|
|B5|False|40→50|True|213.4830→251.6050|5|0|0|
|B6|False|50→60|True|251.6050→290.6044|5|0|0|
|B7|False|60→70|True|290.6044→328.0390|5|0|0|
|B8|False|70→80|True|328.0390→363.3246|5|0|0|
|B9|False|80→90|True|363.3246→397.5502|5|0|0|
|B10|False|90→100|True|397.5502→432.4002|5|0|0|

- AlphaEdit B1만 reset하고 B2–B10은 official `cache_c`를 연속 소비합니다. static null-space P와 dynamic cache_c는 별도 receipt입니다.
- MEMIT `COV_CACHE`는 frozen second-moment computation cache이며 request-history decision state가 아닙니다. optional target-z cache는 비활성입니다.

## Compute ledger

|Arm|model forward|backward|target backward|evaluator forward|processed tokens|entry-pre evaluator F/B/gen|accuracy-added F/B/gen|
|---|---:|---:|---:|---:|---:|---:|---:|
|P1R52 Soft Structural-H ON|242|0|0|750|66067|100/0/0|0/0/0|
|P1R52 Soft AlphaCache-ON / Structural-H OFF|242|0|0|750|66067|100/0/0|0/0/0|
|Corrected Official AlphaEdit cache_c ON|2640|2358|2358|750|368772|100/0/0|0/0/0|
|Official MEMIT Sequential|2640|2400|2400|750|323417|100/0/0|0/0/0|

## 실패/hard cohort와 forgetting (raw-free association)

|Arm|entry rewrite fail|entry rephrase-any fail|post rewrite fail|post rephrase-any fail|final rewrite fail|final rephrase-any fail|post→final rewrite loss|post→final rephrase loss|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|P1R52 Soft Structural-H ON|89/100|93/100|0/100|25/100|0/100|23/100|0/100|1/100|
|P1R52 Soft AlphaCache-ON / Structural-H OFF|89/100|91/100|0/100|24/100|0/100|23/100|0/100|1/100|
|Corrected Official AlphaEdit cache_c ON|87/100|92/100|0/100|8/100|0/100|8/100|0/100|1/100|
|Official MEMIT Sequential|87/100|91/100|0/100|21/100|0/100|21/100|0/100|2/100|

- 이 표는 동일 request의 관찰 연관만 요약합니다. hard sample의 원인이나 개별 preservation component의 인과효과를 주장하지 않습니다.
- 정확한 400개 request-level pre/post/final bit/NLL/margin/delta는 machine table에 보존했습니다.

## 실행·무결성

- corrected AlphaEdit/MEMIT job `20440`, source `9ccf0b42dfdb54e593f57daba46e6004470fa2e9`; 두 task COMPLETED exit0; B1–B10 10/10; failures0.
- 두 official baseline 모두 physical W가 B1→B10 유지되고 interbatch W0 restore=0; terminal 후 W0 pointer/bytes exact restore PASS.
- request/order pairing 400/400 exact; unmatched/imputation=0/0; old R52 result/model/evaluator rerun=0.
- in-place update 이전 H-aware v4 SHA `6d55a15469d24ba7cd3f76f7791bd990280bfb79041bfd0ecf74c09397b5ad44`; three-arm v5 SHA `4c97731e80379a3a45ede38e23581cb8dfbf0e53006093627fea28372b8120a2`. 사용자의 명시적 overwrite 승인을 receipt에 기록했습니다.
- scientific_promotion=false; factual comparison only.

## Machine-readable tables

- `/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-s05-p1r52-llama-soft-seq-nohistorical-v1/local/odebf/reports/p1r52-sequential-alphacache-on-structuralh-off-3arm-v1/p1r52-sequential-four-arm-readable-metrics-v6.json`
- `/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-s05-p1r52-llama-soft-seq-nohistorical-v1/local/odebf/reports/p1r52-sequential-alphacache-on-structuralh-off-3arm-v1/p1r52-sequential-four-arm-cache-compute-v6.json`
- `/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-s05-p1r52-llama-soft-seq-nohistorical-v1/local/odebf/reports/p1r52-sequential-alphacache-on-structuralh-off-3arm-v1/p1r52-sequential-four-arm-hard-cohort-v6.json`
- `/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-s05-p1r52-llama-soft-seq-nohistorical-v1/local/odebf/reports/p1r52-sequential-alphacache-on-structuralh-off-3arm-v1/p1r52-sequential-four-arm-per-request-v6.json`
