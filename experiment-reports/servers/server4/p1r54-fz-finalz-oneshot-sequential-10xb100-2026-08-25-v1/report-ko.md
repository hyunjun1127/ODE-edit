# P1R54 FZ Final-Z One-shot Sequential 10×B100 factual review

**결론:** `p1r54_fz_finalz_r1_s4`(job24175)는 10/10 batches, 1000/1000 requests, FULL-FP32, W-chain 9/9, cache 0→1000, terminal W0 restore를 모두 통과했다. 각 B100에서 K1–K8 target flow를 먼저 끝낸 뒤 K8 final commanded z를 한 번만 썼다. writer 80→10, layer apply 400→50으로 줄면서 post-preflight runtime은 11,918.27s→5,670.86s(−52.42%, 2.10×)가 됐다. 반면 K-step FZ 대비 final W Gen은 84.95%→81.20%(−3.75pp), strict Gen은 75.60%→70.00%(−5.60pp)로 낮아졌다. final locality는 84.53%→85.62%(+1.09pp)다. `scientific_promotion=false`.

## 최종 핵심 표

모든 NLL 셀은 target-new request mean의 `mean/median/p90/max`이며, sequential 방법은 terminal W10의 동일 B1→B10 1,000-request denominator다.

| Method | final W Eff | final W Gen | final W strict | final W Loc | z Rewrite NLL | W Rewrite NLL | z Rephrase NLL | W Rephrase NLL |
|---|---:|---:|---:|---:|---|---|---|---|
| Official AlphaEdit-cache | 99.40% | 96.70% | 94.30% | 78.10% | 0.001392/0.000743/0.003357/0.025194 | 0.043526/0.001017/0.005216/8.861880 | 0.797691/0.069259/2.721582/12.862535 | 1.200882/0.318129/3.407061/12.973390 |
| Official MEMIT | 95.40% | 90.35% | 86.10% | 72.39% | 0.002097/0.000903/0.004589/0.043856 | 0.757153/0.038796/2.473372/19.905504 | 0.771799/0.081747/2.646872/12.733783 | 1.667522/0.755956/4.704381/14.999562 |
| P1R54 FZ K-step writer8 | 100.00% | 84.95% | 75.60% | 84.53% | 0.008817/0.000617/0.004828/2.775532 | 0.018627/0.000580/0.005625/8.714952 | 1.265033/0.249023/3.850843/18.477956 | 2.306031/1.447106/6.100811/13.292747 |
| **P1R54 FZ final-z writer1** | **100.00%** | **81.20%** | **70.00%** | **85.62%** | **0.019942/0.000582/0.004561/4.438613** | **0.027707/0.000538/0.005348/7.182264** | **1.343384/0.239663/4.063507/14.361774** | **2.715791/1.947146/6.675395/17.598382** |

Native 두 행은 server1의 exact same-stream sealed final-W10 backfill이며 재실행하지 않았다. FZ 두 행만 writer cadence의 직접 비교다.

## 정확히 어떻게 write했나

- 각 B100 entry에서 W, target origin, teacher, rho를 한 번 봉인했다.
- K1–K7은 target controller만 진행했다. writer/apply/materialization, physical W mutation, cache/history append는 모두 0이다.
- K8 final commanded/accepted z만 Official EasyEdit AlphaEdit C3 writer에 1회 전달했다.
- 따라서 batch당 field8/writer1/layer5, 전체 field80/writer10/layer50이다.
- 성공한 K8 write 뒤에만 현재 batch key100을 cache에 append했고, 다음 B100은 방금 commit된 실제 W activation에서 새로 시작했다.

## Rewrite/Rephrase 세부

| Endpoint | Rewrite success | Rewrite accuracy | Rephrase success | Rephrase accuracy | Rephrase strict success | Rephrase strict accuracy | Loc |
|---|---:|---:|---:|---:|---:|---:|---:|
| final-z accepted z | 1000/1000 (100.00%) | 995/1000 (99.50%) | 1912/2000 (95.60%) | 1446/2000 (72.30%) | 926/1000 (92.60%) | 586/1000 (58.60%) | 8806/10000 (88.06%) |
| final-z immediate post-W | 1000/1000 (100.00%) | 995/1000 (99.50%) | 1624/2000 (81.20%) | 1050/2000 (52.50%) | 697/1000 (69.70%) | 339/1000 (33.90%) | 8698/10000 (86.98%) |
| final-z terminal W10 | 1000/1000 (100.00%) | 995/1000 (99.50%) | 1624/2000 (81.20%) | 1054/2000 (52.70%) | 700/1000 (70.00%) | 340/1000 (34.00%) | 8562/10000 (85.62%) |
| K-step terminal W10 | 1000/1000 (100.00%) | 997/1000 (99.70%) | 1699/2000 (84.95%) | 1161/2000 (58.05%) | 756/1000 (75.60%) | 410/1000 (41.00%) | 8453/10000 (84.53%) |

| Method/endpoint | Prompt | target-new mean/median/p90/max | target-true mean/median/p90/max |
|---|---|---|---|
| final-z accepted z | Rewrite | 0.019942/0.000582/0.004561/4.438613 | 14.369881/14.114128/19.534552/28.259150 |
| final-z accepted z | Rephrase | 1.343384/0.239663/4.063507/14.361774 | 10.411660/10.157923/15.640906/28.545965 |
| final-z terminal W10 | Rewrite | 0.027707/0.000538/0.005348/7.182264 | 14.207958/14.017750/19.465893/27.247036 |
| final-z terminal W10 | Rephrase | 2.715791/1.947146/6.675395/17.598382 | 7.953322/7.756338/12.973017/22.449608 |
| K-step accepted z | Rewrite | 0.008817/0.000617/0.004828/2.775532 | 14.519902/14.051021/19.683716/28.370306 |
| K-step accepted z | Rephrase | 1.265033/0.249023/3.850843/18.477956 | 10.604270/10.305470/15.767402/26.311680 |
| K-step terminal W10 | Rewrite | 0.018627/0.000580/0.005625/8.714952 | 14.517836/14.114930/20.189289/30.168802 |
| K-step terminal W10 | Rephrase | 2.306031/1.447106/6.100811/13.292747 | 8.516335/8.261639/13.544448/22.684467 |

## Writer cadence paired delta

| Metric | final-z writer1 | K-step writer8 | writer1−writer8 |
|---|---:|---:|---:|
| final W Eff | 100.00% | 100.00% | 0.00pp |
| final W Gen | 81.20% | 84.95% | −3.75pp |
| final W strict Gen | 70.00% | 75.60% | −5.60pp |
| final W Loc | 85.62% | 84.53% | +1.09pp |
| final W Rewrite target-new NLL mean | 0.027707 | 0.018627 | +0.009080 |
| final W Rephrase target-new NLL mean | 2.715791 | 2.306031 | +0.409760 |
| final W Rephrase target-new NLL median | 1.947146 | 1.447106 | +0.500041 |
| final W Rephrase target-new NLL p90 | 6.675395 | 6.100811 | +0.574585 |

Request-paired `final-z−K-step` final-W rephrase NLL delta는 mean +0.409760, median +0.142841, p90 +1.831917, max +11.692910이다. 분포 통계의 차이와 paired-delta 통계를 혼동하지 않는다.

Accepted-z→immediate-W Rephrase mean gap은 final-z +1.373218, K-step +1.078709로 final-z가 +0.294509 더 크다. terminal W10 기준 z→W gap도 +1.372406 대 +1.040998이다. Rewrite의 terminal z→W mean gap은 +0.007766 대 +0.009810이다. 즉 열화는 주로 Rephrase/Gen 쪽이다.

## B1–B10 final W10

| B | final-z Gen | final-z strict | final-z Loc | K-step Gen | K-step strict | K-step Loc | paired Rephrase NLL Δ | final-z immediate→W10 NLL Δ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 83.0% | 72.0% | 85.2% | 91.0% | 85.0% | 83.4% | +0.567299 | −0.049857 |
| 2 | 84.5% | 76.0% | 87.5% | 88.0% | 80.0% | 86.7% | +0.563792 | −0.029786 |
| 3 | 81.0% | 70.0% | 87.5% | 81.5% | 71.0% | 86.4% | +0.192999 | +0.005411 |
| 4 | 83.0% | 72.0% | 84.6% | 87.0% | 78.0% | 84.2% | +0.245490 | +0.135005 |
| 5 | 78.0% | 65.0% | 86.4% | 84.0% | 76.0% | 84.2% | +0.435923 | −0.009430 |
| 6 | 79.0% | 68.0% | 82.1% | 83.5% | 73.0% | 80.7% | +0.424317 | +0.001689 |
| 7 | 79.5% | 69.0% | 85.8% | 83.0% | 74.0% | 85.1% | +0.685208 | −0.018480 |
| 8 | 84.5% | 74.0% | 81.8% | 87.5% | 79.0% | 80.9% | +0.241249 | −0.024564 |
| 9 | 81.0% | 69.0% | 86.4% | 84.0% | 73.0% | 85.2% | +0.453234 | −0.018102 |
| 10 | 78.5% | 65.0% | 88.9% | 80.0% | 67.0% | 88.5% | +0.288084 | 0.000000 |

Rephrase NLL paired delta는 10/10 batch 모두 양수다. 반면 immediate→final W10 변화는 부호가 섞이고 작아서, 전체 Gen 차이를 sequential forgetting 하나로 설명할 수 없다.

## K1→K8 target curve와 selector

| K | final-z target-new train objective mean | K-step target-new train objective mean |
|---:|---:|---:|
| 1 | 5.947676 | 5.687967 |
| 2 | 2.732503 | 2.504001 |
| 3 | 1.101926 | 1.031163 |
| 4 | 0.423573 | 0.423077 |
| 5 | 0.157192 | 0.166614 |
| 6 | 0.060279 | 0.062145 |
| 7 | 0.029357 | 0.022545 |
| 8 | 0.016738 | 0.008779 |

B1 W0와 K1 selected target bytes는 두 cadence에서 exact match했다. 이후 final-z는 W를 고정한 채 target만 진행하고 K-step은 매 K write 후 새 W에서 다시 target을 계산하므로 K2부터 궤적이 달라지는 것이 계약상 예상된다. 전체 8,000 request-step에서 final-z selector P/R/C는 5419/2423/158, K-step은 5216/2681/103이다. 두 방법 모두 clamp 0/8000이다.

## 계산량과 overhead

| Counter | final-z writer1 | K-step writer8 | 변화 |
|---|---:|---:|---:|
| target field evaluations | 80 | 80 | 0 |
| Official writer calls | 10 | 80 | −87.50% |
| layer applies | 50 | 400 | −87.50% |
| finite-demand/materialization | 10/10 | 80/80 | −87.50% |
| model forwards | 4,742 | 8,592 | −44.81% |
| processed tokens | 2,258,801 | 9,517,241 | −76.27% |
| writer edit-core wall | 465.19s | 3,650.18s | −87.26% |
| batch wall sum | 5,293.39s | 11,512.18s | −54.02% |
| post-preflight runtime | 5,670.86s | 11,918.27s | −52.42%; 2.10× |
| peak allocated GPU bytes | 38,027,805,184 | 38,027,805,184 | 동일 |
| peak reserved GPU bytes | 44,010,831,872 | 43,260,051,456 | +750,780,416 |

속도 이득은 명확하지만 GPU peak memory는 줄지 않았다. wall-time은 단일 paired 실행의 관측값이며 보편적 성능 보장은 아니다.

## 무결성 및 판정

- job24175: `COMPLETED`, exit 0, elapsed 01:35:15, 1GPU/8CPU/56000M.
- terminal/manifest/final-W10 및 10개 batch terminal canonical identity와 SHA 결속 전부 PASS.
- 10 batches, 1000 requests, field80, writer10, layer50, cache entry widths 0→900, exits 100→1000, commit→next entry 9/9.
- FULL-FP32: 291/291 parameter tensors FP32, quantization/autocast/cast0.
- W0 pointer/bytes restore PASS, nonfinite/scientific failure/imputation/retry0.
- job24078은 distinct 기술 실패 lineage로 denominator0이며, 성공한 TECH-R1 결과가 이를 참조한다.

**사실 기반 판정:** final-z one-shot은 계산 효율과 locality를 개선했지만, 동일 stream의 K-step writer 대비 Rephrase/Gen과 strict Gen이 낮다. 관측상 병목은 Rewrite가 아니라 Rephrase의 target-to-W realization이다. 한 번의 큰 final write가 여덟 번의 receding physical realization을 완전히 대체했다는 근거는 없다. 다만 이는 단일 Llama sequential stream의 writer-cadence 비교이며, 보편적 인과 주장이나 promotion 근거로 사용하지 않는다.

