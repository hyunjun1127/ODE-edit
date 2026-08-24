# P1R54 Post-write Realization Reset — Sequential B1→B10

**결론:** FZ/PDZ 모두 10/10 batches, 1000/1000 requests, W-chain9/9, cache0→1000으로 유효하다. FZ reset은 continuation보다 Gen −0.30pp, strict −0.20pp, Loc +0.16pp다. PDZ continuation은 `ABSENT_CONTROL`. `scientific_promotion=false`.

## 최종 핵심 표

| Method | W Eff | W Gen | W strict | W Loc | z Rewrite mean/median/p90/max | W Rewrite mean/median/p90/max | z Rephrase mean/median/p90/max | W Rephrase mean/median/p90/max |
|---|---:|---:|---:|---:|---|---|---|---|
| FZ-RESET | 100.00% | 84.65% | 75.40% | 84.69% | 0.017847/0.000657/0.006295/4.761465 | 0.029981/0.000597/0.006453/11.387056 | 1.330886/0.067031/4.909337/17.165323 | 2.404130/0.652997/7.434225/18.031172 |
| PDZ-RESET | 99.90% | 83.75% | 73.70% | 84.79% | 0.049261/0.018961/0.062375/10.023932 | 0.055955/0.017321/0.063027/9.849946 | 1.447819/0.309397/4.792752/16.431705 | 2.631030/1.361871/7.262716/17.820286 |
| OFFICIAL-ALPHAEDIT-CACHE | 99.40% | 96.70% | 94.30% | 78.10% | 0.001392/0.000743/0.003357/0.025194 | 0.043526/0.001017/0.005216/8.861880 | 0.797691/0.069259/2.721582/12.862535 | 1.200882/0.318129/3.407061/12.973390 |
| OFFICIAL-MEMIT | 95.40% | 90.35% | 86.10% | 72.39% | 0.002097/0.000903/0.004589/0.043856 | 0.757153/0.038796/2.473372/19.905504 | 0.771799/0.081747/2.646872/12.733783 | 1.667522/0.755956/4.704381/14.999562 |
| P1R54-FZ-SEQUENTIAL | 100.00% | 84.95% | 75.60% | 84.53% | 0.008817/0.000617/0.004828/2.775532 | 0.018627/0.000580/0.005625/8.714952 | 1.265033/0.065512/4.660754/21.401110 | 2.306031/1.447106/6.100811/13.292747 |

Reset W와 세 baseline W NLL은 모두 terminal W10의 동일 10-cohort/1,000-request 평가다. Native 두 arm은 server1 기존 sealed final-W10 raw, FZ continuation은 server4 기존 sealed final-W10 raw를 사용했으며 신규 native 재실행은 0이다.


## Baseline final-W10 NLL backfill

동일 canonical B1→B10 stream의 terminal W10 1,000-request raw에서 request-level prompt mean을 집계했다. Official AlphaEdit-cache/MEMIT는 server1의 기존 sealed raw를 create-once 전송해 재사용했고 신규 native 재실행은 0이다. FZ continuation은 기존 server4 sealed raw를 재분석했다. imputation=0.

| Method | Prompt | target-new mean/median/p90/max | target-true mean/median/p90/max | requests |
|---|---|---|---|---:|
| OFFICIAL-ALPHAEDIT-CACHE | Rewrite | 0.043526/0.001017/0.005216/8.861880 | 14.608628/14.390605/20.425499/31.089413 | 1000 |
| OFFICIAL-ALPHAEDIT-CACHE | Rephrase | 1.200882/0.318129/3.407061/12.973390 | 10.384222/10.370014/15.267340/22.683067 | 1000 |
| OFFICIAL-MEMIT | Rewrite | 0.757153/0.038796/2.473372/19.905504 | 9.669079/9.445754/15.330941/24.699993 | 1000 |
| OFFICIAL-MEMIT | Rephrase | 1.667522/0.755956/4.704381/14.999562 | 8.297380/8.211417/13.323760/21.764063 | 1000 |
| P1R54-FZ-SEQUENTIAL | Rewrite | 0.018627/0.000580/0.005625/8.714952 | 14.517836/14.114930/20.189289/30.168802 | 1000 |
| P1R54-FZ-SEQUENTIAL | Rephrase | 2.306031/1.447106/6.100811/13.292747 | 8.516335/8.261639/13.544448/22.684467 | 1000 |

호환 필드 `target_old_nll_mean`은 이 전송 schema에서 `target_true_nll_by_request`의 request mean에 대응한다.

기존 report의 `NR`은 source package에 raw final-W10 NLL이 없었다는 뜻이며 성능 0을 뜻하지 않는다. 본 v2는 새 native raw와 기존 FZ raw를 SHA로 결속해 해당 칸을 채운 분석-only revision이다.

## Rewrite/Rephrase 세부

### FZ-RESET

| Prompt | success | accuracy | strict success | strict accuracy |
|---|---:|---:|---:|---:|
| Rewrite | 100.00% | 99.40% | — | — |
| Rephrase | 84.65% | 57.05% | 75.40% | 39.00% |

z→W mean NLL gap은 Rewrite +0.012133, Rephrase +1.073244. reset c=0 8000/8000, anchor match 80/80, last-command 보존 80/80; PRC P/R/C=[5155, 2715, 130]; clamp=0/8000; 추가 F/B/M/E=[0, 0, 0, 0].

### PDZ-RESET

| Prompt | success | accuracy | strict success | strict accuracy |
|---|---:|---:|---:|---:|
| Rewrite | 99.90% | 99.50% | — | — |
| Rephrase | 83.75% | 50.05% | 73.70% | 30.50% |

z→W mean NLL gap은 Rewrite +0.006694, Rephrase +1.183211. reset c=0 8000/8000, anchor match 80/80, last-command 보존 80/80; PRC P/R/C=[7951, 39, 10]; clamp=0/8000; 추가 F/B/M/E=[0, 0, 0, 0].

## Sequential 고유 표

| Arm | batches/requests | W-chain | cache | append/consume | field/writer/layer | wall(s) |
|---|---:|---:|---:|---:|---:|---:|
| FZ | 10/1000 | 9/9 | 0→1000 | 10/80 | 80/80/400 | 11711.38 |
| PDZ | 10/1000 | 9/9 | 0→1000 | 10/80 | 80/80/400 | 11352.64 |

Accepted-z→immediate-W Rephrase mean gap은 FZ +1.117365, PDZ +1.214867. z-success→W-failure는 238/2000와 262/2000, Rewrite failure0. immediate→final W10 NLL mean은 FZ −0.0441, PDZ −0.0317이라 하락을 terminal forgetting 하나에 귀속할 수 없다. actual layer update energy/GPU peak memory는 `NOT_RECORDED`.
