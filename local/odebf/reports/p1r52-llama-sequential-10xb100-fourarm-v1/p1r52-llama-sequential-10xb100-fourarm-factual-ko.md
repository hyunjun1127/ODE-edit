# P1R52 Llama Sequential 10×B100 — 4팔 최종 사실 보고서

> 범위: 10개 순차 B100, 총 1,000 edits/arm. 모델·evaluator 재실행 없이 봉인된 raw-free 결과만 사용했습니다.

## 판정 및 경계

- 네 팔 모두 10/10 배치, 1,000/1,000 requests, terminal W0 pointer+byte restore를 완료했습니다.
- 사용자 지시에 따라 method-specific W_(b-1) batch-entry aggregate는 `NONCANONICAL_UNUSED`입니다. 아래 표에는 단일 공통 W0(B1 entry)와 immediate-post/final-W10만 포함합니다.
- EFF ≡ rewrite_success, GEN ≡ paraphrase_success(rephrase_success alias). Rewrite/Rephrase Acc는 별도 strict suffix-token accuracy입니다.
- 결과는 이 1,000-edit Llama 패키지에 한정하며 isolated Structural-H causality를 주장하지 않습니다.

## 공통 original-W0 B1 baseline

|EFF|Rewrite Acc|GEN|GEN strict|Rephrase Acc|Acc strict|LOC|
|---:|---:|---:|---:|---:|---:|---:|
|13/100 (13.00%)|0/100 (0.00%)|28/200 (14.00%)|8/100 (8.00%)|1/200 (0.50%)|0/100 (0.00%)|892/1000 (89.20%)|

## 1,000-request 절대값: immediate-post aggregate와 final W10

|Arm|Panel|EFF|Rewrite Acc|GEN|GEN strict|Rephrase Acc|Acc strict|LOC|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|Official EasyEdit MEMIT Sequential|10× immediate-post|996/1000 (99.60%)|991/1000 (99.10%)|1825/2000 (91.25%)|862/1000 (86.20%)|1332/2000 (66.60%)|512/1000 (51.20%)|8056/10000 (80.56%)|
|Official EasyEdit MEMIT Sequential|final W10/B1000|942/1000 (94.20%)|819/1000 (81.90%)|1777/2000 (88.85%)|838/1000 (83.80%)|1287/2000 (64.35%)|512/1000 (51.20%)|7214/10000 (72.14%)|
|Official EasyEdit AlphaEdit Sequential (cache_c ON)|10× immediate-post|1000/1000 (100.00%)|1000/1000 (100.00%)|1922/2000 (96.10%)|937/1000 (93.70%)|1495/2000 (74.75%)|606/1000 (60.60%)|8177/10000 (81.77%)|
|Official EasyEdit AlphaEdit Sequential (cache_c ON)|final W10/B1000|998/1000 (99.80%)|996/1000 (99.60%)|1909/2000 (95.45%)|929/1000 (92.90%)|1479/2000 (73.95%)|598/1000 (59.80%)|7763/10000 (77.63%)|
|P1R52 Repair-R1 Soft Sequential / Structural-H ON|10× immediate-post|998/1000 (99.80%)|995/1000 (99.50%)|1687/2000 (84.35%)|748/1000 (74.80%)|1019/2000 (50.95%)|321/1000 (32.10%)|8586/10000 (85.86%)|
|P1R52 Repair-R1 Soft Sequential / Structural-H ON|final W10/B1000|999/1000 (99.90%)|992/1000 (99.20%)|1696/2000 (84.80%)|751/1000 (75.10%)|1023/2000 (51.15%)|324/1000 (32.40%)|8394/10000 (83.94%)|
|P1R52 Repair-R1 Soft Sequential / AlphaCache ON, Structural-H OFF|10× immediate-post|998/1000 (99.80%)|995/1000 (99.50%)|1680/2000 (84.00%)|737/1000 (73.70%)|1018/2000 (50.90%)|315/1000 (31.50%)|8574/10000 (85.74%)|
|P1R52 Repair-R1 Soft Sequential / AlphaCache ON, Structural-H OFF|final W10/B1000|998/1000 (99.80%)|992/1000 (99.20%)|1696/2000 (84.80%)|748/1000 (74.80%)|1032/2000 (51.60%)|324/1000 (32.40%)|8367/10000 (83.67%)|

## final W10 NLL / margin

|Arm|Rewrite new/true/margin|Rephrase new/true/margin|
|---|---:|---:|
|Official EasyEdit MEMIT Sequential|0.861355/9.603316/+8.741960|1.753286/8.282058/+6.528772|
|Official EasyEdit AlphaEdit Sequential (cache_c ON)|0.027169/14.741400/+14.714231|1.262916/10.363093/+9.100177|
|P1R52 Repair-R1 Soft Sequential / Structural-H ON|0.056457/10.752489/+10.696031|2.430740/7.162800/+4.732060|
|P1R52 Repair-R1 Soft Sequential / AlphaCache ON, Structural-H OFF|0.058027/10.822771/+10.764744|2.423899/7.196311/+4.772412|

## B1–B10 immediate-post 절대 EFF / GEN / LOC

|B|Arm|EFF|GEN|GEN strict|LOC|history width|
|---:|---|---:|---:|---:|---:|---:|
|B1|Official EasyEdit MEMIT Sequential|98/100 (98.00%)|163/200 (81.50%)|75/100 (75.00%)|886/1000 (88.60%)|0|
|B2|Official EasyEdit MEMIT Sequential|100/100 (100.00%)|170/200 (85.00%)|78/100 (78.00%)|905/1000 (90.50%)|100|
|B3|Official EasyEdit MEMIT Sequential|99/100 (99.00%)|174/200 (87.00%)|80/100 (80.00%)|879/1000 (87.90%)|200|
|B4|Official EasyEdit MEMIT Sequential|100/100 (100.00%)|187/200 (93.50%)|89/100 (89.00%)|828/1000 (82.80%)|300|
|B5|Official EasyEdit MEMIT Sequential|100/100 (100.00%)|184/200 (92.00%)|85/100 (85.00%)|834/1000 (83.40%)|400|
|B6|Official EasyEdit MEMIT Sequential|100/100 (100.00%)|184/200 (92.00%)|87/100 (87.00%)|734/1000 (73.40%)|500|
|B7|Official EasyEdit MEMIT Sequential|100/100 (100.00%)|190/200 (95.00%)|92/100 (92.00%)|789/1000 (78.90%)|600|
|B8|Official EasyEdit MEMIT Sequential|100/100 (100.00%)|191/200 (95.50%)|93/100 (93.00%)|720/1000 (72.00%)|700|
|B9|Official EasyEdit MEMIT Sequential|100/100 (100.00%)|194/200 (97.00%)|94/100 (94.00%)|719/1000 (71.90%)|800|
|B10|Official EasyEdit MEMIT Sequential|99/100 (99.00%)|188/200 (94.00%)|89/100 (89.00%)|762/1000 (76.20%)|900|
|B1|Official EasyEdit AlphaEdit Sequential (cache_c ON)|100/100 (100.00%)|184/200 (92.00%)|88/100 (88.00%)|871/1000 (87.10%)|0|
|B2|Official EasyEdit AlphaEdit Sequential (cache_c ON)|100/100 (100.00%)|187/200 (93.50%)|90/100 (90.00%)|867/1000 (86.70%)|100|
|B3|Official EasyEdit AlphaEdit Sequential (cache_c ON)|100/100 (100.00%)|187/200 (93.50%)|90/100 (90.00%)|849/1000 (84.90%)|200|
|B4|Official EasyEdit AlphaEdit Sequential (cache_c ON)|100/100 (100.00%)|194/200 (97.00%)|95/100 (95.00%)|825/1000 (82.50%)|300|
|B5|Official EasyEdit AlphaEdit Sequential (cache_c ON)|100/100 (100.00%)|195/200 (97.50%)|95/100 (95.00%)|832/1000 (83.20%)|400|
|B6|Official EasyEdit AlphaEdit Sequential (cache_c ON)|100/100 (100.00%)|194/200 (97.00%)|95/100 (95.00%)|751/1000 (75.10%)|500|
|B7|Official EasyEdit AlphaEdit Sequential (cache_c ON)|100/100 (100.00%)|193/200 (96.50%)|96/100 (96.00%)|796/1000 (79.60%)|600|
|B8|Official EasyEdit AlphaEdit Sequential (cache_c ON)|100/100 (100.00%)|197/200 (98.50%)|97/100 (97.00%)|748/1000 (74.80%)|700|
|B9|Official EasyEdit AlphaEdit Sequential (cache_c ON)|100/100 (100.00%)|196/200 (98.00%)|96/100 (96.00%)|810/1000 (81.00%)|800|
|B10|Official EasyEdit AlphaEdit Sequential (cache_c ON)|100/100 (100.00%)|195/200 (97.50%)|95/100 (95.00%)|828/1000 (82.80%)|900|
|B1|P1R52 Repair-R1 Soft Sequential / Structural-H ON|99/100 (99.00%)|177/200 (88.50%)|81/100 (81.00%)|885/1000 (88.50%)|0|
|B2|P1R52 Repair-R1 Soft Sequential / Structural-H ON|100/100 (100.00%)|179/200 (89.50%)|82/100 (82.00%)|908/1000 (90.80%)|100|
|B3|P1R52 Repair-R1 Soft Sequential / Structural-H ON|100/100 (100.00%)|172/200 (86.00%)|75/100 (75.00%)|891/1000 (89.10%)|200|
|B4|P1R52 Repair-R1 Soft Sequential / Structural-H ON|100/100 (100.00%)|180/200 (90.00%)|84/100 (84.00%)|843/1000 (84.30%)|300|
|B5|P1R52 Repair-R1 Soft Sequential / Structural-H ON|100/100 (100.00%)|159/200 (79.50%)|68/100 (68.00%)|867/1000 (86.70%)|400|
|B6|P1R52 Repair-R1 Soft Sequential / Structural-H ON|100/100 (100.00%)|166/200 (83.00%)|74/100 (74.00%)|805/1000 (80.50%)|500|
|B7|P1R52 Repair-R1 Soft Sequential / Structural-H ON|100/100 (100.00%)|169/200 (84.50%)|74/100 (74.00%)|844/1000 (84.40%)|600|
|B8|P1R52 Repair-R1 Soft Sequential / Structural-H ON|99/100 (99.00%)|168/200 (84.00%)|73/100 (73.00%)|813/1000 (81.30%)|700|
|B9|P1R52 Repair-R1 Soft Sequential / Structural-H ON|100/100 (100.00%)|167/200 (83.50%)|73/100 (73.00%)|854/1000 (85.40%)|800|
|B10|P1R52 Repair-R1 Soft Sequential / Structural-H ON|100/100 (100.00%)|150/200 (75.00%)|64/100 (64.00%)|876/1000 (87.60%)|900|
|B1|P1R52 Repair-R1 Soft Sequential / AlphaCache ON, Structural-H OFF|99/100 (99.00%)|177/200 (88.50%)|81/100 (81.00%)|885/1000 (88.50%)|0|
|B2|P1R52 Repair-R1 Soft Sequential / AlphaCache ON, Structural-H OFF|100/100 (100.00%)|176/200 (88.00%)|80/100 (80.00%)|909/1000 (90.90%)|100|
|B3|P1R52 Repair-R1 Soft Sequential / AlphaCache ON, Structural-H OFF|100/100 (100.00%)|171/200 (85.50%)|73/100 (73.00%)|889/1000 (88.90%)|200|
|B4|P1R52 Repair-R1 Soft Sequential / AlphaCache ON, Structural-H OFF|100/100 (100.00%)|180/200 (90.00%)|84/100 (84.00%)|842/1000 (84.20%)|300|
|B5|P1R52 Repair-R1 Soft Sequential / AlphaCache ON, Structural-H OFF|100/100 (100.00%)|159/200 (79.50%)|68/100 (68.00%)|871/1000 (87.10%)|400|
|B6|P1R52 Repair-R1 Soft Sequential / AlphaCache ON, Structural-H OFF|100/100 (100.00%)|166/200 (83.00%)|74/100 (74.00%)|803/1000 (80.30%)|500|
|B7|P1R52 Repair-R1 Soft Sequential / AlphaCache ON, Structural-H OFF|100/100 (100.00%)|172/200 (86.00%)|75/100 (75.00%)|840/1000 (84.00%)|600|
|B8|P1R52 Repair-R1 Soft Sequential / AlphaCache ON, Structural-H OFF|99/100 (99.00%)|165/200 (82.50%)|70/100 (70.00%)|810/1000 (81.00%)|700|
|B9|P1R52 Repair-R1 Soft Sequential / AlphaCache ON, Structural-H OFF|100/100 (100.00%)|164/200 (82.00%)|69/100 (69.00%)|852/1000 (85.20%)|800|
|B10|P1R52 Repair-R1 Soft Sequential / AlphaCache ON, Structural-H OFF|100/100 (100.00%)|150/200 (75.00%)|63/100 (63.00%)|873/1000 (87.30%)|900|

## hard / forgetting cohort

|Arm|requests|post EFF fail|final EFF fail|post GEN-strict fail|final GEN-strict fail|EFF success→fail|GEN success→fail|
|---|---:|---:|---:|---:|---:|---:|---:|
|Official EasyEdit MEMIT Sequential|1000|4|58|138|162|56|68|
|Official EasyEdit AlphaEdit Sequential (cache_c ON)|1000|0|2|63|71|2|16|
|P1R52 Repair-R1 Soft Sequential / Structural-H ON|1000|2|1|252|249|0|21|
|P1R52 Repair-R1 Soft Sequential / AlphaCache ON, Structural-H OFF|1000|2|2|263|252|0|16|

## R52 routing / history 사실

- P1R52 Repair-R1 Soft Sequential / Structural-H ON: 80/80 writes; negative actual `0`; fallback `0`; H status `{'H_EMPTY_EXACT_ATOMIC_EQUIVALENCE': 8, 'H_ACTIVE_CERTIFIED': 72}`; max |strength residual| `8.882e-16`; cache entry widths `[0,100,...,900]`.
- P1R52 Repair-R1 Soft Sequential / AlphaCache ON, Structural-H OFF: 80/80 writes; negative actual `0`; fallback `0`; H status `{'ALPHA_CACHE_ON_STRUCTURAL_H_OFF': 80}`; max |strength residual| `8.882e-16`; cache entry widths `[0,100,...,900]`.

## Compute / transaction

|Arm|wall s|completed K|model F/B|evaluator F|tokens|materializations|entry evaluator canonical|W0 restore|
|---|---:|---:|---:|---:|---:|---:|---|---|
|Official EasyEdit MEMIT Sequential|4272.1|0|21865/20525|7500|2864223|NOT_RECORDED|NONCANONICAL_UNUSED_RAW_RECEIPT_ONLY|True/True|
|Official EasyEdit AlphaEdit Sequential (cache_c ON)|4442.8|0|23188/21606|7500|3481547|NOT_RECORDED|NONCANONICAL_UNUSED_RAW_RECEIPT_ONLY|True/True|
|P1R52 Repair-R1 Soft Sequential / Structural-H ON|5567.2|0|442/0|6500|517957|NOT_RECORDED|0|True/True|
|P1R52 Repair-R1 Soft Sequential / AlphaCache ON, Structural-H OFF|6670.3|0|442/0|6500|517957|NOT_RECORDED|0|True/True|

## Machine artifacts

- aggregate: `/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-s05-p1r52-llama-seq-10xb100-fourarm-r1-v1/local/odebf/reports/p1r52-llama-sequential-10xb100-fourarm-v1/p1r52-b100-four-arm-aggregate.json`
- batch rows: `/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-s05-p1r52-llama-seq-10xb100-fourarm-r1-v1/local/odebf/reports/p1r52-llama-sequential-10xb100-fourarm-v1/p1r52-b100-four-arm-batch-checkpoints.json` (40)
- request rows: `/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-s05-p1r52-llama-seq-10xb100-fourarm-r1-v1/local/odebf/reports/p1r52-llama-sequential-10xb100-fourarm-v1/p1r52-b100-four-arm-request-retention.json` (4000)
- R52 controller rows: `/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-s05-p1r52-llama-seq-10xb100-fourarm-r1-v1/local/odebf/reports/p1r52-llama-sequential-10xb100-fourarm-v1/p1r52-b100-r52-controller-routing.json` (160)
- integrity/compute: `/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-s05-p1r52-llama-seq-10xb100-fourarm-r1-v1/local/odebf/reports/p1r52-llama-sequential-10xb100-fourarm-v1/p1r52-b100-four-arm-integrity-compute.json`

scientific_promotion=false
