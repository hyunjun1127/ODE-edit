# P1R52 Joint P+C TECH-R6 동표본 baseline 비교

- 범위: Llama3-8B-Instruct, B1의 동일 100개 요청, original W0 시작
- 신규 model/evaluator/GPU/Slurm 실행: `0`
- 비교 패널: Pre-edit, Official MEMIT, Official AlphaEdit, P1R52 J0, accepted-z, C0/C1/C2
- scientific_promotion: `false`

## 1. 한눈에 보는 절대값

| 패널 | Rewrite success | Rewrite acc | Rewrite NLL | Rephrase success | Strict Gen | Rephrase acc | Strict acc | Rephrase NLL | LOC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Pre-edit original W0 | 13/100 | 0/100 | 11.096726 | 28/200 | 8/100 | 1/200 | 0/100 | 9.816823 | 892/1000 |
| Official MEMIT B1 | 98/100 | 97/100 | 0.245842 | 163/200 | 75/100 | 98/200 | 35/100 | 2.935170 | 886/1000 |
| Official AlphaEdit B1 | 100/100 | 100/100 | 0.001309 | 184/200 | 88/100 | 127/200 | 48/100 | 1.816861 | 871/1000 |
| P1R52 J0 B1 | 99/100 | 99/100 | 0.073150 | 177/200 | 81/100 | 103/200 | 31/100 | 2.332472 | 885/1000 |
| Accepted-z common | 100/100 | 100/100 | 0.022487 | 196/200 | 97/100 | 141/200 | 56/100 | 1.346694 | N/A |
| C0 PIU/PIR-U | 100/100 | 99/100 | 0.028274 | 179/200 | 82/100 | 106/200 | 34/100 | 2.276502 | 891/1000 |
| C1 Joint P+C + suffix | 100/100 | 99/100 | 0.026397 | 179/200 | 82/100 | 105/200 | 34/100 | 2.297181 | 889/1000 |
| C2 Joint P+C fixed quota | 100/100 | 93/100 | 0.211536 | 174/200 | 79/100 | 92/200 | 28/100 | 2.633401 | 892/1000 |

`success`와 suffix-token `accuracy`는 별도 정의이며 혼합하지 않았다. Rephrase success 분모는 prompt 200, strict Gen 분모는 request 100이다.

## 2. 동표본 및 W0 identity

| 항목 | 값 |
|---|---|
| stream root | `467e5946ec0eb975284ca25e16f63f3b8ae0093503ca8b84948409689e0ad25a` |
| stream order | `018be113361157d6f4050c37a4fec14fff78e60388e3898253d66f070d78cfc3` |
| B1 request-order SHA256 | `bb2e661ad44bf8dc20e0224f21dc90164ed4b3dcf3316d751afb7d8a00bfaed6` |
| Joint pilot request-order SHA256 | 위 값과 exact |
| cross-run W0 layer hashes | 5/5 exact |
| C0/C1/C2 accepted-z SHA256 | `467a195af3602208e97606ed5c2a78f1d148b4afd35bb2d2332e4f1b3410e461` |
| C0/C1/C2 accepted-z identity | exact |
| baseline history width at B1 entry | MEMIT/AlphaEdit/J0 모두 0 |
| J0 B1 Structural-H 상태 | 8/8 `H_EMPTY_EXACT_ATOMIC_EQUIVALENCE` |

W0 layer SHA256는 두 패키지에서 모두 `10c24f6f…`, `7c006100…`, `73db4a10…`, `e814d631…`, `2acbde8f…`로 일치한다. 따라서 baseline B1과 Joint pilot은 동일 sample/order 및 동일 W0 출발점 비교다.

## 3. J0 대비 C0/C1/C2

| delta (Joint−J0) | C0 | C1 | C2 |
|---|---:|---:|---:|
| Rewrite success | +1/100 | +1/100 | +1/100 |
| Rewrite accuracy | 0/100 | 0/100 | −6/100 |
| Rewrite NLL | −0.044875 | −0.046753 | +0.138386 |
| Rephrase success | +2/200 | +2/200 | −3/200 |
| Strict Gen | +1/100 | +1/100 | −2/100 |
| Rephrase accuracy | +3/200 | +2/200 | −11/200 |
| Strict rephrase accuracy | +3/100 | +3/100 | −3/100 |
| Rephrase NLL | −0.055970 | −0.035291 | +0.300929 |
| LOC | +6/1000 | +4/1000 | +7/1000 |

C0와 C1은 J0 대비 rewrite/rephrase success와 NLL에서 위 표의 양의 count 또는 음의 NLL delta를 기록했다. C2는 rewrite success와 LOC는 유지했지만 rewrite accuracy, Gen, accuracy, NLL에서 J0보다 낮은 count 또는 높은 NLL을 기록했다.

## 4. Official baseline과의 위치

- C0/C1은 MEMIT보다 rewrite success `+2/100`, strict Gen `+7/100`, rephrase success `+16/200`, LOC `+5/+3`을 기록했다.
- C0/C1은 AlphaEdit와 rewrite success가 `100/100`으로 같고, strict Gen은 `−6/100`, rephrase success는 `−5/200`, LOC는 `+20/+18`이다.
- AlphaEdit의 rewrite/rephrase NLL `0.001309/1.816861`은 C0 `0.028274/2.276502`, C1 `0.026397/2.297181`보다 낮다.
- C2는 MEMIT보다 rewrite success는 `+2/100`, rewrite NLL은 `−0.034307`이지만 strict Gen은 `+4/100`; AlphaEdit보다 strict Gen `−9/100`, rephrase NLL `+0.816540`이다.

## 5. 판정

- 동표본 baseline 패널 completeness: `PASS`
- baseline 재실행 필요: `NO`
- rerun/model/evaluator/GPU/Slurm action count: `0/0/0/0/0`
- 기존 TECH-R6 판정: `INDEPENDENT_PRODUCTION_GATE=PASS_VIA_C1`, 변경 없음
- C2 gate: `FAIL`, 변경 없음
- 이 패널은 단일 B100 비교이며 10×B100 production 결과가 아니다.

## 6. Evidence

- Joint terminal: `/mnt/raid5/janghj/.codex/worktrees/odeeditsh2-s05-p1r52-joint-pc-c1-c2-independent-b100-v1/local/odebf/results/s05-p1r52-joint-pc-c1-c2-pilot-b100-tech-r6-v1/raw/case-01/terminal.json`
- Joint report: `/mnt/raid5/janghj/.codex/worktrees/odeeditsh2-s05-p1r52-joint-pc-c1-c2-independent-b100-v1/local/odebf/reports/p1r52-joint-pc-c1-c2-pilot-tech-r6-v1/p1r52-joint-pc-c1-c2-pilot-tech-r6-factual-ko.md`
- Baseline aggregate: `/mnt/raid5/janghj/.codex/worktrees/odeeditsh2-s05-p1r52-joint-pc-c1-c2-independent-b100-v1/local/odebf/reports/p1r52-llama-sequential-10xb100-fourarm-v1/p1r52-b100-four-arm-aggregate.json`
- Baseline B1 rows: `/mnt/raid5/janghj/.codex/worktrees/odeeditsh2-s05-p1r52-joint-pc-c1-c2-independent-b100-v1/local/odebf/reports/p1r52-llama-sequential-10xb100-fourarm-v1/p1r52-b100-four-arm-batch-checkpoints.json`
