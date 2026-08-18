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

## GH 추가 점검 — hard cohort, writer realization, layer routing, history/cache

> 이 절은 봉인된 raw-free request/step/terminal receipt와 구현 소스를 추가로 교차 점검한 결과다. 모델·evaluator 재실행은 없으며, 아래의 `hard`는 특별한 언급이 없으면 final W10에서 두 paraphrase가 모두 성공하지 못한 `GEN strict failure`를 뜻한다. prompt 원문은 raw-free package에 없으므로 언어학적 유형이 아니라 수치적 cohort만 분석한다.

### 추가 진단 요약

- P1R52의 주된 문제는 sequential edit forgetting이 아니다. H-ON의 immediate-post→final 변화는 EFF `+1/1000`, GEN `+9/2000`, LOC `-192/10000`이다. 낮은 GEN은 편집 직후부터 존재한다.
- Rewrite objective 기준 writer 전달은 유지된다. 8,000 request-step의 z-target improvement와 physical-W progress 상관은 `0.93863`이고, 누적 progress는 `9702.553→9723.234`이다.
- 반면 intended latent delta와 realized delta의 기하학적 일치는 약하다. 유효 intended norm 7,941행에서 cosine 중앙값은 `0.4454`, p10은 `0.1257`, norm-gain 중앙값/p90은 `1.925/10.677`이다.
- R52 H-ON final strict GEN failure 249건 중 AlphaEdit 성공은 200건, MEMIT 성공은 172건이다. 네 방법 모두 실패한 요청은 36건뿐이다. 따라서 대부분은 dataset-intrinsic hard가 아니라 R52-specific hard다.
- R52-only hard cohort는 rewrite NLL 기반 allocation을 적게 받는다. strict 성공 cohort의 평균 request-step energy share는 `1.063%`지만, R52 실패·Alpha 성공 200건은 `0.503%`다.
- RESCUE/CURRENT는 전체 GEN 저하의 주원인이 아니다. H-ON 8,000 request-step 중 `PRIMARY/RESCUE/CURRENT=7870/75/55`이고, R52-only hard 200건도 `1582/6/12`로 98.88%가 PRIMARY다.
- H barrier는 layer route와 historical damage를 실제로 바꾸지만 최종 성능 차이는 작다. H-ON−H-OFF는 EFF `+1/1000`, GEN strict `+3/1000`, Rephrase Acc `-9/2000`, LOC `+27/10000`이다.

### Immediate-post→final W10: sequential retention

|Arm|EFF 변화|Rewrite Acc 변화|GEN 변화|GEN strict 변화|Rephrase Acc 변화|LOC 변화|
|---|---:|---:|---:|---:|---:|---:|
|Official MEMIT|-54/1000|-172/1000|-48/2000|-24/1000|-45/2000|-842/10000|
|Official AlphaEdit|-2/1000|-4/1000|-13/2000|-8/1000|-16/2000|-414/10000|
|R52 Structural-H ON|+1/1000|-3/1000|+9/2000|+3/1000|+4/2000|-192/10000|
|R52 Structural-H OFF|0/1000|-3/1000|+16/2000|+11/1000|+14/2000|-207/10000|

R52는 이후 batch 때문에 rewrite/GEN이 무너진 형태가 아니다. sequential 누적에서 확인되는 손실은 주로 LOC 약 `1.92–2.07%p`다. 반대로 MEMIT은 EFF `5.4%p`, Rewrite Acc `17.2%p`, LOC `8.42%p`를 잃으며 가장 큰 forgetting을 보인다.

### Final GEN hard cohort 구성

R52 H-ON의 final GEN은 prompt 기준 `1696/2000`, strict request 기준 `751/1000`이다. 이를 요청 단위로 분해하면 다음과 같다.

|상태|요청 수|
|---|---:|
|두 paraphrase 모두 성공|751|
|하나만 성공|194|
|둘 다 실패|55|
|합계 strict failure|249|

네 arm의 strict outcome 주요 패턴은 다음과 같다.

|패턴|요청 수|
|---|---:|
|H-ON/H-OFF/AlphaEdit/MEMIT 모두 성공|642|
|두 R52 모두 실패, AlphaEdit/MEMIT 모두 성공|150|
|네 방법 모두 실패|36|
|H-ON만 성공하고 H-OFF 실패|20|
|H-OFF만 성공하고 H-ON 실패|17|

H-ON strict failure 249건에서:

- AlphaEdit 성공: `200/249 = 80.32%`
- MEMIT 성공: `172/249 = 69.08%`
- AlphaEdit와 MEMIT 모두 실패: `38/249 = 15.26%`
- H-OFF가 대신 성공: `17/249 = 6.83%`

따라서 `hard`라는 표기는 R52에 대한 hard cohort를 주로 뜻하며, 모든 method에서 본질적으로 어려운 sample과 동일하지 않다.

### Hard cohort의 수치적 공통점

|지표|R52 strict 성공 751|R52 strict 실패 249|R52 실패·Alpha 성공 200|
|---|---:|---:|---:|
|Final rephrase target-new NLL 평균|1.7542|4.4713|4.2072|
|Final rephrase old−new margin 평균|+6.2528|+0.1454|+0.3162|
|Immediate-post rephrase target-new NLL 평균|1.8282|4.5487|4.2862|
|Immediate-post rephrase margin 평균|+6.2032|+0.1660|+0.3183|
|평균 allocation energy share/request-step|1.0629%|0.8102%|0.5031%|
|현재 rewrite NLL 평균|2.4074|2.1559|1.8938|
|Target NLL improvement 평균|1.2258|1.1737|1.1146|
|Semantic→KDC direction cosine 평균|0.9244|0.9201|0.9231|
|Intended→writer realized cosine 평균|0.4329|0.4241|0.4228|
|Final locality rate 평균|82.58%|88.03%|87.00%|

관측상 가장 강한 공통점은 `paraphrase-hard지만 rewrite urgency가 낮아 allocation을 적게 받는 것`이다. R52의 amplitude는 현재 request별 rewrite target-new NLL에 비례한다. 따라서 rewrite는 비교적 쉬우나 paraphrase가 어려운 요청은 underweight된다. KDC cosine과 writer cosine의 cohort 차이는 작으므로, hard cohort 전체를 direction collapse만으로 설명할 수는 없다.

Strict failure 평균 margin이 양수일 수 있는 이유는 두 paraphrase의 평균을 기록하기 때문이다. 249건 중 194건은 한 paraphrase만 실패하여 평균 margin은 양수여도 strict criterion은 실패한다.

### 수치상 최악 sample 예시

|B/index|case_id|Final rephrase margin|Alpha/MEMIT strict|P/R/C (8 steps)|평균 energy share|해석|
|---|---:|---:|---:|---:|---:|---|
|B1/36|9846|-12.5376|0/0|3/1/4|10.7683%|모든 method 실패, CURRENT 반복 intrinsic-hard 후보|
|B1/37|15078|-9.5781|0/0|8/0/0|1.8832%|모든 method 실패지만 target 선택은 전부 PRIMARY|
|B10/11|15043|-8.0759|1/1|8/0/0|0.7870%|두 baseline 성공, R52-specific hard|
|B10/86|8573|-8.0278|1/1|8/0/0|0.7195%|두 baseline 성공, R52-specific hard|
|B7/44|2133|-7.5820|1/1|8/0/0|0.9777%|두 baseline 성공, all-PRIMARY|
|B6/94|13806|-5.4961|1/1|8/0/0|0.1532%|심한 under-allocation, 두 baseline 성공|

H-ON final rewrite_success 실패는 단 한 건이다: `B1/index78, case_id=17660`. Immediate-post부터 rewrite margin이 `0.0`이었고 final margin은 `-0.6328`이다. 따라서 249건의 GEN hard cohort와 rewrite failure는 거의 별개의 현상이다.

### PRIMARY / RESCUE / CURRENT 상세

|Arm|PRIMARY|RESCUE|CURRENT|Clamp|Fallback|
|---|---:|---:|---:|---:|---:|
|H-ON|7870 (98.375%)|75 (0.938%)|55 (0.688%)|99|0|
|H-OFF|7888 (98.600%)|64 (0.800%)|48 (0.600%)|114|0|

H-ON strict failure cohort 1,992 request-step는 `PRIMARY/RESCUE/CURRENT=1955/18/19`이고, R52 실패·Alpha 성공 cohort 1,600 step는 `1582/6/12`다. 즉 각각 98.14%, 98.88%가 PRIMARY다.

- RESCUE를 한 번이라도 경험한 요청: 49건, strict failure 11건(`22.45%`)
- RESCUE가 없던 요청의 strict failure율: `25.03%`
- CURRENT를 한 번이라도 경험한 요청: 33건, strict failure 10건(`30.30%`)
- CURRENT가 없던 요청의 strict failure율: `24.72%`

CURRENT는 extreme-hard 요청에 다소 농축되지만 hard 249건 중 CURRENT 경험 요청은 10건뿐이다. RESCUE는 실패율을 높이지 않는다. 또한 B5의 P/R/C는 `790/8/2`인데 strict GEN이 68%, B10은 `799/0/1`인데 64%다. 낮은 GEN을 fallback 빈도로 설명할 수 없다.

### Writer: objective 전달과 latent geometry를 분리한 판정

#### Rewrite NLL objective 전달

|Arm|z target improvement 평균/합|Physical-W progress 평균/합|z/W progress 상관|negative W request-step|CURRENT-held|
|---|---:|---:|---:|---:|---:|
|H-ON|1.212819 / 9702.553|1.215404 / 9723.234|0.93863|35/8000|55|
|H-OFF|1.212530 / 9700.237|1.214743 / 9717.940|0.93978|35/8000|48|

Aggregate rewrite objective는 writer에서 소실되지 않는다. 보고서 본문의 `negative actual=0`은 aggregate scalar 기준이고, request-level로는 35/8000(`0.44%`)의 negative progress가 존재한다.

Batch-terminal full-six target-new NLL 평균은 H-ON `z=0.03887`, `W=0.05562`, gap `0.01675`; H-OFF `z=0.04233`, `W=0.05789`, gap `0.01557`이다. Atomic Llama Soft의 gap `0.03780`보다 작으므로 rewrite NLL 기준의 writer collapse는 아니다.

#### Intended latent delta의 기하학적 realization

유효 intended norm(`>=1e-4`)을 가진 H-ON 7,941 request-step에서:

- cosine mean/median/p10: `0.4340 / 0.4454 / 0.1257`
- negative cosine: `78`
- norm-gain median/p90: `1.925 / 10.677`
- residual-ratio median/p90: `1.726 / 10.528`

K8 request-level median cosine은 B2 `0.243`, B3 `0.069`, B7 `0.151`, B8 `0.089`, B10 `0.198`이다. 따라서 writer는 rewrite NLL을 개선하지만 intended request별 latent vector를 같은 방향·크기로 충실히 재현하지는 못한다. 100개 request-specific target을 다섯 layer scalar로 공유하는 geometry가 paraphrase transfer를 제한했을 가능성이 크다.

### Allocation concentration

H-ON 80 physical steps의 request-wise energy allocation은 다음과 같다.

- top-1 request share mean/median/p90/max: `32.45% / 19.79% / 84.58% / 98.73%`
- top-3 share 평균: `50.27%`
- effective request support mean/median/min: `31.74 / 17.18 / 1.10` requests out of 100
- semantic→KDC cosine mean/median/p10/min: `0.9233 / 0.9924 / 0.7514 / 0.0018`
- preservation/semantic norm ratio mean/median/p90: `0.5619 / 0.1273 / 0.9080`
- positive preservation-conflict projection: `6453/8000`

평균적으로 한 요청이 전체 batch target energy의 32%를 차지하며, 일부 step은 사실상 한 요청에 99%가 집중된다. B8 top-1 평균은 54.6%, effective support는 21.9였다. 다만 B10은 support가 약 40인데도 GEN이 가장 낮으므로 concentration 하나만으로 모든 batch 차이를 설명하지는 못한다.

### Layer별 routing 및 BF16 energy 분배

`routing.pi`는 최종 parameter norm이 아니라 해당 step의 progress share다.

|Arm/지표|L4|L5|L6|L7|L8|
|---|---:|---:|---:|---:|---:|
|H-ON mean π|10.90%|16.03%|19.82%|21.02%|32.23%|
|H-ON realized BF16 step-energy share|16.29%|18.83%|19.39%|19.38%|26.11%|
|H-OFF mean π|11.67%|14.91%|18.65%|21.86%|32.91%|
|H-OFF realized BF16 step-energy share|17.92%|16.74%|17.80%|20.91%|26.63%|

80회 realized step-energy 합은 H-ON `76.7665`, H-OFF `80.1790`으로 H-ON이 약 4.26% 작다. 평균적으로 L8이 가장 큰 비중을 갖는다.

마지막 B10/K8 routing은 평균과 달리 H의 작동을 강하게 보여준다.

|Arm|L4 π|L5 π|L6 π|L7 π|L8 π|
|---|---:|---:|---:|---:|---:|
|H-ON|11.64%|23.38%|19.17%|16.19%|29.61%|
|H-OFF|25.18%|35.91%|22.31%|10.32%|6.29%|

B10 K8 batch-local cumulative BF16 capacity는 H-ON `3.135/3.144/3.180/3.788/4.749`, H-OFF `3.852/3.285/2.987/3.457/4.037`이다. H barrier가 local layer route를 크게 바꾸지만 최종 GEN/LOC 차이는 작다.

전체 sequential 최종 `||W10−W0||`의 layer별 norm은 산출물에 없다. terminal에서 W0를 정확히 복원하고 final physical values는 hash만 봉인했기 때문에, 위 표는 accepted step energy와 batch-local capacity accounting이며 최종 parameter norm으로 해석하면 안 된다.

### Alpha/Woodbury history key cache와 Structural-H

R52 H-ON/H-OFF 모두 Alpha solve key cache는 동일하게 누적·소비한다.

|Batch|Entry width|Consumed|Appended|Post active width|Anchor total|
|---|---:|---:|---:|---:|---:|
|B1|0|0|100|100|100|
|B2|100|100|100|200|200|
|B5|400|400|100|500|500|
|B10|900|900|100|1000|1000|

- history transaction version: `0→1→…→10`
- obsolete count: 모든 batch `0`
- 다섯 layer 각각 post-commit raw solve key와 projected risk key를 capture하고 SHA를 봉인
- 최종 logical history: layer당 1,000 request columns
- 다섯 layer accounting: solve-key 5,000 columns + risk-key 5,000 columns이지만 고유 요청은 1,000개

H-ON은 B1에서 history가 비어 `H_EMPTY_EXACT_ATOMIC_EQUIVALENCE 8/8`, B2–B10에서 `H_ACTIVE_CERTIFIED 72/72`다. Structural-H decision width는 `100→900`, batch당 influence는 8이다. 전체 `h_selected−h_disabled` 합/평균은 `-0.09994/-0.001249`이고, B10 합이 `-0.03453`으로 가장 크다. H-aware와 H-disabled velocity의 step별 L1 차이는 평균 `0.10498`이다.

H-OFF도 Alpha solve cache를 B2–B10에서 전부 소비하지만 Structural-H decision width/influence는 항상 `0/0`이다. risk/anchor ledger는 observation-only로 유지된다. 따라서 H-OFF는 history 전체를 끈 실험이 아니라 `AlphaCache ON / Structural-H OFF` control이다.

### Historical damage와 hard cohort

B1–B9의 900 prior request에 대해 B10 terminal anchor drift를 결합한 결과다.

|Arm/cohort|Lifetime target-new NLL increase mean|B10-local increase mean|
|---|---:|---:|
|H-ON strict 성공 687|0.02171|0.00486|
|H-ON strict 실패 213|0.06393|0.03942|
|H-OFF strict 성공 685|0.02473|0.00570|
|H-OFF strict 실패 215|0.06884|0.03858|

Hard cohort는 lifetime drift가 약 3배, 마지막 B10에 의한 positive NLL damage가 약 8배 크다. 그러나 hard cohort의 immediate-post NLL도 이미 높으므로 history가 failure를 처음 만든 것은 아니다. 원래 약한 paraphrase representation이 후속 edit에 더 민감하게 흔들리는 형태다. H-ON은 H-OFF보다 lifetime damage를 조금 낮추지만 final strict GEN 개선은 `3/1000`에 그친다.

### Official baseline cache 확인

Official AlphaEdit는 EasyEdit `apply_AlphaEdit_to_model`을 직접 호출한다. B1에서만 dynamic `cache_c`를 reset하고 B2–B10은 이전 cache를 solver가 소비한다. static null-space projector P와 dynamic cache는 분리돼 있다.

- cache shape: `[5,14336,14336]`, FP32
- Frobenius norm B1 exit/B2 exit/B5 exit/B10 entry→exit: `416.85 / 773.17 / 1809.94 / 3206.33→3544.50`
- B2–B10 `solver_consumed_entry_cache=true`

Official MEMIT도 EasyEdit `apply_memit_to_model(copy=False)` 직접 호출이다. request history decision cache는 없고, 다섯 layer의 static covariance computation cache(`[14336,14336]` each)를 B1에서 생성한 뒤 B2–B10 동일 identity로 재사용한다.

### 추가 판정

관측 근거가 강한 순서로 보면:

1. Rewrite-NLL 기반 request allocation이 paraphrase-hard/rewrite-easy 요청을 underweight한다.
2. 100개의 target을 5개 shared layer coefficient로 구현하면서 latent vector realization이 크게 왜곡된다.
3. 이 paraphrase-hard cohort는 후속 edit의 historical drift에도 더 민감하다.
4. CURRENT와 Structural-H route는 일부 extreme case와 layer 배치를 바꾸지만 전체 GEN 저하의 주원인은 아니다.
5. Overall edit forgetting, RESCUE 빈도, aggregate rewrite-objective writer failure는 주원인이 아니다.

다음 실험에서 직접 대응되는 변수는 H threshold보다 allocation 쪽이다. Rewrite urgency 외의 minimum per-request allocation, effective-support floor, top-1 share cap을 우선 검토할 근거가 있다. 동시에 5-scalar shared writer의 intended→realized cosine을 개선하지 않으면 EFF는 유지되어도 GEN ceiling이 낮게 남을 가능성이 크다. 이 절은 관측 연관 분석이며 개별 요소의 isolated causal claim은 아니다.

### 추가 분석 한계

- raw-free package에는 prompt 원문이 없어 entity/문장 유형별 hard-pattern은 확인할 수 없다.
- final physical W는 terminal에서 W0로 복원됐고 layer별 최종 parameter norm은 저장되지 않았다.
- R52 상위 compute ledger의 `completed_k_total=0`은 child K 집계 누락이므로 Official 대비 F/B 비용 직접 비교에는 사용할 수 없다.
- 사용자 지시로 method-specific batch-entry aggregate는 비정규/미사용이다. 모든 1,000개에 대한 공통 pre-edit aggregate로 해석하면 안 된다.

scientific_promotion=false


## Append-only v2: target/write/controller 상세 및 compute 정정

- v1 상위 job ledger의 `completed_k_total=0`은 child sequential step을 집계하지 않은 raw field입니다. 실제 R52 물리 전이는 accepted-k/materialization 영수증 기준 각 80회입니다.
- Official 방법의 ODE K는 적용 대상이 아니므로 `NOT_APPLICABLE`; Official materialization count는 현재 공통 영수증에서 `NOT_RECORDED`입니다.
- 모든 팔에서 K-step heldout evaluator 접근은 0입니다. R52 canonical batch-entry evaluator도 0; Official entry-evaluator 원자료는 사용자 지시에 따라 비정규/미사용입니다.

### final W10 직접 산술 델타

|비교|EFF|Rewrite Acc|GEN|GEN strict|Rephrase Acc|Acc strict|LOC|
|---|---:|---:|---:|---:|---:|---:|---:|
|R52 ON−OFF|+1/1000|+0/1000|+0/2000|+3/1000|-9/2000|+0/1000|+27/10000|
|R52 ON−AlphaEdit|+1/1000|-4/1000|-213/2000|-178/1000|-456/2000|-274/1000|+631/10000|
|R52 OFF−AlphaEdit|+0/1000|-4/1000|-213/2000|-181/1000|-447/2000|-274/1000|+604/10000|
|R52 ON−MEMIT|+57/1000|+173/1000|-81/2000|-87/1000|-264/2000|-188/1000|+1180/10000|

### R52 controller totals 및 분포

|Arm|Primary/Rescue/Current|active/request-steps|clamp|fallback|P mean/med/p90/max|capacity mean/med/p90/max|energy mean/med/p90/max|BF16 energy mean/med/p90/max|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|R52 Structural-H ON|7870/75/55|8000/8000|99|0|0.01549/0.01226/0.03625/0.05794|0.6751/0.4985/1.403/2.725|0.8226/0.6312/1.649/2.745|0.9596/0.769/1.823/2.938|
|R52 Structural-H OFF|7888/64/48|8000/8000|114|0|0.01593/0.01241/0.03585/0.05794|0.7198/0.5433/1.399/2.725|0.8625/0.6936/1.654/2.745|1.002/0.8403/1.828/2.938|

### Batch-terminal z8/W8 full-six target-new NLL

|B|Arm|z8|W8|W−z|
|---:|---|---:|---:|---:|
|B1|R52 Structural-H ON|0.048306|0.094739|+0.046433|
|B2|R52 Structural-H ON|0.039480|0.051547|+0.012067|
|B3|R52 Structural-H ON|0.020351|0.047520|+0.027169|
|B4|R52 Structural-H ON|0.029267|0.038427|+0.009160|
|B5|R52 Structural-H ON|0.014250|0.031922|+0.017672|
|B6|R52 Structural-H ON|0.036013|0.042554|+0.006541|
|B7|R52 Structural-H ON|0.056493|0.074765|+0.018272|
|B8|R52 Structural-H ON|0.104820|0.110990|+0.006170|
|B9|R52 Structural-H ON|0.014877|0.023297|+0.008419|
|B10|R52 Structural-H ON|0.024843|0.040436|+0.015594|
|B1|R52 Structural-H OFF|0.048306|0.094739|+0.046433|
|B2|R52 Structural-H OFF|0.045228|0.060449|+0.015220|
|B3|R52 Structural-H OFF|0.023988|0.034833|+0.010844|
|B4|R52 Structural-H OFF|0.028420|0.041499|+0.013079|
|B5|R52 Structural-H OFF|0.013824|0.027281|+0.013457|
|B6|R52 Structural-H OFF|0.030736|0.036952|+0.006216|
|B7|R52 Structural-H OFF|0.079218|0.085612|+0.006394|
|B8|R52 Structural-H OFF|0.104850|0.115200|+0.010350|
|B9|R52 Structural-H OFF|0.018000|0.023719|+0.005719|
|B10|R52 Structural-H OFF|0.030680|0.058651|+0.027971|

### v2 machine artifacts

- step rows: `/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-s05-p1r52-llama-seq-10xb100-fourarm-r1-v1/local/odebf/reports/p1r52-llama-sequential-10xb100-fourarm-v1/p1r52-b100-r52-target-writer-step-v2.json` (160)
- terminal z/W rows: `/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-s05-p1r52-llama-seq-10xb100-fourarm-r1-v1/local/odebf/reports/p1r52-llama-sequential-10xb100-fourarm-v1/p1r52-b100-r52-terminal-z-w-v2.json` (20)
- controller/compute summary: `/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-s05-p1r52-llama-seq-10xb100-fourarm-r1-v1/local/odebf/reports/p1r52-llama-sequential-10xb100-fourarm-v1/p1r52-b100-r52-controller-compute-summary-v2.json`

scientific_promotion=false
