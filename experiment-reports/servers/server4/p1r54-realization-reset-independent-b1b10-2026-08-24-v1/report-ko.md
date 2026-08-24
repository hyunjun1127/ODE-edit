# P1R54 Post-write Realization Reset — Independent B1–B10

**결론:** 10/10 W0/cold-cache slices, 1000/1000 requests가 유효하다. FZ reset은 continuation FZ와 거의 동일하고, PDZ reset은 FZ보다 Gen/strict가 낮다. `scientific_promotion=false`.

## 최종 핵심 표

| Method | W Eff | W Gen | W strict | W Loc | z Rewrite mean/median/p90/max | W Rewrite mean/median/p90/max | z Rephrase mean/median/p90/max | W Rephrase mean/median/p90/max |
|---|---:|---:|---:|---:|---|---|---|---|
| FZ-RESET | 99.90% | 92.15% | 87.00% | 87.47% | 0.036535/0.001239/0.023193/3.425783 | 0.037463/0.001234/0.022775/3.842672 | 1.287966/0.099642/4.678692/17.360403 | 1.969347/0.470816/6.129216/18.019650 |
| PDZ-RESET | 99.90% | 90.20% | 84.10% | 87.41% | 0.076053/0.034465/0.112898/3.305080 | 0.076466/0.034626/0.109659/3.655180 | 1.560413/0.398864/5.040854/17.339769 | 2.359244/1.086654/6.510426/18.191282 |
| OFFICIAL-ALPHAEDIT | 99.90% | 93.00% | 88.00% | 86.26% | 0.002307/0.000690/0.002685/0.998165 | 0.003315/0.000747/0.002938/1.877316 | 1.100522/0.051957/4.308492/15.602556 | 1.848039/0.293467/5.901396/17.445372 |
| OFFICIAL-MEMIT | 98.30% | 82.55% | 74.50% | 87.89% | 0.001098/0.000645/0.002308/0.025492 | 0.196725/0.004256/0.109747/12.664020 | 1.104760/0.053958/4.294096/15.612510 | 2.875626/1.155112/8.353469/20.605066 |
| P1R54-FZ-INDEPENDENT | 99.80% | 92.10% | 86.70% | 87.47% | 0.036367/0.001228/0.020374/3.373269 | 0.037832/0.001213/0.020455/3.746590 | 1.290592/0.097049/4.662909/16.324144 | 1.965954/0.475650/6.100472/17.943241 |

NLL은 target-new이며 mean/median/p90/max 순서다. Native는 exact same-stream independent sealed package다.

## Rewrite/Rephrase 세부

### FZ-RESET

| Prompt | success | accuracy | strict success | strict accuracy |
|---|---:|---:|---:|---:|
| Rewrite | 99.90% | 99.00% | — | — |
| Rephrase | 92.15% | 61.45% | 87.00% | 44.70% |

z→W mean NLL gap은 Rewrite +0.000928, Rephrase +0.681381. reset c=0 8000/8000, anchor match 80/80, last-command 보존 80/80; PRC P/R/C=[6884, 1093, 23]; clamp=0/8000; 추가 F/B/M/E=[0, 0, 0, 0].

### PDZ-RESET

| Prompt | success | accuracy | strict success | strict accuracy |
|---|---:|---:|---:|---:|
| Rewrite | 99.90% | 98.90% | — | — |
| Rephrase | 90.20% | 54.50% | 84.10% | 37.80% |

z→W mean NLL gap은 Rewrite +0.000413, Rephrase +0.798831. reset c=0 8000/8000, anchor match 80/80, last-command 보존 80/80; PRC P/R/C=[7984, 16, 0]; clamp=0/8000; 추가 F/B/M/E=[0, 0, 0, 0].

## Independent 고유 표

| Arm | endpoints/requests | W0/cold | cross-state | wall sum(s) | writer/layer |
|---|---:|---:|---:|---:|---:|
| FZ | 10/1000 | 10/10 | 0 | 10189.62 | 80/400 |
| PDZ | 10/1000 | 10/10 | 0 | 10728.50 | 80/400 |

FZ reset−continuation final-W는 Eff +0.1pp, Gen +0.05pp, strict +0.3pp, Loc 약 +0.01pp로 near-identity다. PDZ control은 B1만 있어 full causal 비교가 아니다. per-B, target-true, K별 reset/PRC/trajectory 분포는 CSV/JSON에 있다. actual layer update energy/GPU peak memory는 `NOT_RECORDED`. residual ratio 극단값은 near-zero 분모 경계다.
