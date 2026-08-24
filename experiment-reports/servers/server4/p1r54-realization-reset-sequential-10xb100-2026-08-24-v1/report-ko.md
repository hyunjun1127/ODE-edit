# P1R54 Post-write Realization Reset — Sequential B1→B10

**결론:** FZ/PDZ 모두 10/10 batches, 1000/1000 requests, W-chain9/9, cache0→1000으로 유효하다. FZ reset은 continuation보다 Gen −0.30pp, strict −0.20pp, Loc +0.16pp다. PDZ continuation은 `ABSENT_CONTROL`. `scientific_promotion=false`.

## 최종 핵심 표

| Method | W Eff | W Gen | W strict | W Loc | z Rewrite mean/median/p90/max | W Rewrite mean/median/p90/max | z Rephrase mean/median/p90/max | W Rephrase mean/median/p90/max |
|---|---:|---:|---:|---:|---|---|---|---|
| FZ-RESET | 100.00% | 84.65% | 75.40% | 84.69% | 0.017847/0.000657/0.006295/4.761465 | 0.029981/0.000597/0.006453/11.387056 | 1.330886/0.067031/4.909337/17.165323 | 2.404130/0.652997/7.434225/18.031172 |
| PDZ-RESET | 99.90% | 83.75% | 73.70% | 84.79% | 0.049261/0.018961/0.062375/10.023932 | 0.055955/0.017321/0.063027/9.849946 | 1.447819/0.309397/4.792752/16.431705 | 2.631030/1.361871/7.262716/17.820286 |
| OFFICIAL-ALPHAEDIT-CACHE | 99.40% | 96.70% | 94.30% | 78.10% | 0.001392/0.000743/0.003357/0.025194 | NR/NR/NR/NR | 0.797691/0.069259/2.721582/12.862535 | NR/NR/NR/NR |
| OFFICIAL-MEMIT | 95.40% | 90.35% | 86.10% | 72.39% | 0.002097/0.000903/0.004589/0.043856 | NR/NR/NR/NR | 0.771799/0.081747/2.646872/12.733783 | NR/NR/NR/NR |
| P1R54-FZ-SEQUENTIAL | 100.00% | 84.95% | 75.60% | 84.53% | 0.008817/0.000617/0.004828/2.775532 | NR/NR/NR/NR | 1.265033/0.065512/4.660754/21.401110 | NR/NR/NR/NR |

Reset W는 terminal W10의 10 cohort 평가다. Native v5의 terminal W10 NLL 미기록 필드는 NR이다.

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
