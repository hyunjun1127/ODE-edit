# P1R52 Joint P+C C0/C1/C2 TECH-R6 1×B100 pilot factual report

- 작성 시각: 2026-08-21 KST
- instruction: `ODEEDIT-S05-P1R52-JOINT-PC-C1-C2-INDEPENDENT-B100-V1`
- 범위: Llama3-8B-Instruct, 독립 1×B100 pilot, C0/C1/C2, terminal-only evaluator
- scheduler: job `22295`, `COMPLETED`, ExitCode `0:0`, elapsed `00:07:28`
- source HEAD/tree: `20911aa13a6689577b62f53b7096f061bb5151a0` / `e22899378f571394948816989b62e148df84b10c`
- authoritative contract: `/mnt/raid5/janghj/ODE-edit/local/state/p1r52-joint-pc-c1-c2-independent-b100-v1/authoritative-contract.txt`, SHA256 `9f24ca3062e91a903f4fb92e80c73c372b8fc03f76e8e12d545ae97c3df4f0b8`, 13,564 bytes, 495 lines
- scientific_promotion: `false`

## 1. 결론

TECH-R6 pilot은 1/1 case, 100/100 requests, C0/C1/C2 세 arm 모두 terminal이며 technical failure 0, typed failure 0, imputation 0이다. 세 arm은 accepted-z, writer-entry W0, stream/order가 동일하고 arm마다 물리 materialization 1회 후 W0가 복원됐다.

계약의 lenient paired production gate에서 C1은 C0와 rephrase success 179/200, strict success 82/100을 동일하게 유지했다. C1의 final residual norm은 1.278058로 C0 1.294668보다 0.016610 낮고, BF16 update energy는 40.939220으로 C0 41.404755보다 0.465536 낮다. 따라서 `INDEPENDENT_PRODUCTION_GATE=PASS_VIA_C1`; C2는 같은 gate에서 `FAIL`이다. 이 report 이후 production/job 제출은 0이다.

## 2. 동일 입력·격리 증명

| 항목 | 값 |
|---|---:|
| request attempts / valid cases | 100 / 1 |
| accepted-z SHA256 | `467a195af3602208e97606ed5c2a78f1d148b4afd35bb2d2332e4f1b3410e461` |
| accepted-z exact across C0/C1/C2 | PASS |
| writer-entry W hash exact across C0/C1/C2 | PASS (5/5 layer tensors) |
| target compute / per-arm recompute | 1 / 0 |
| stream root | `467e5946ec0eb975284ca25e16f63f3b8ae0093503ca8b84948409689e0ad25a` |
| stream order | `018be113361157d6f4050c37a4fec14fff78e60388e3898253d66f070d78cfc3` |
| request order SHA256 | `bb2e661ad44bf8dc20e0224f21dc90164ed4b3dcf3316d751afb7d8a00bfaed6` |
| C1/C2 joint pi exact | PASS |
| cross-arm weight contamination | 0 |
| materializations C0/C1/C2 | 1 / 1 / 1 |
| retry / backtracking / imputation | 0 / 0 / 0 |
| heldout writer decision access | 0 |
| W0 restored / pointer identity | PASS / PASS (all arms) |

Joint P+C pi는 `[0.185813740, 0.214082792, 0.218292132, 0.201513946, 0.180297390]`이다. Router solve는 success/status `true/0`, simplex residual `0`, stationarity residual `7.02e-16`, minimax `t=0.994325649`; normalized P/C는 각각 `0.994325649/0.994325649`이다.

## 3. 공통 accepted-z endpoint

| metric | numerator / denominator | rate 또는 mean |
|---|---:|---:|
| rewrite success | 100 / 100 | 1.000 |
| rewrite accuracy | 100 / 100 | 1.000 |
| rewrite target-new NLL | — | 0.022487 |
| rephrase success | 196 / 200 | 0.980 |
| strict rephrase success | 97 / 100 | 0.970 |
| rephrase accuracy | 141 / 200 | 0.705 |
| strict rephrase accuracy | 56 / 100 | 0.560 |
| rephrase target-new NLL | — | 1.346694 |

## 4. C0/C1/C2 terminal writer metrics

| metric | C0 PIU/PIR-U | C1 Joint P+C + suffix | C2 Joint P+C fixed quota |
|---|---:|---:|---:|
| rewrite success | 100/100 | 100/100 | 100/100 |
| rewrite accuracy | 99/100 | 99/100 | 93/100 |
| rewrite target-new NLL | 0.028274 | 0.026397 | 0.211536 |
| rewrite W−z NLL gap | 0.005787 | 0.003910 | 0.189048 |
| rephrase success | 179/200 | 179/200 | 174/200 |
| strict rephrase success | 82/100 | 82/100 | 79/100 |
| rephrase accuracy | 106/200 | 105/200 | 92/200 |
| strict rephrase accuracy | 34/100 | 34/100 | 28/100 |
| rephrase target-new NLL | 2.276502 | 2.297181 | 2.633401 |
| rephrase W−z NLL gap | 0.929808 | 0.950487 | 1.286708 |
| locality | 891/1000 | 889/1000 | 892/1000 |
| final residual norm | 1.294668 | 1.278058 | 25.581469 |
| entry→final residual reduction | 26.011137 | 26.027747 | 1.724336 |
| residual reduction / entry | 0.952586 | 0.953195 | 0.063149 |
| total BF16 update energy | 41.404755 | 40.939220 | 12.038146 |
| layer-8 energy share | 0.472752 | 0.468026 | 0.147598 |
| final virtual W-only NLL | 0.054444 | 0.053937 | 0.290496 |

C1−C0: final residual `−0.016610` (−1.283%), BF16 energy `−0.465536` (−1.124%), layer-8 share `−0.004727`, rewrite NLL `−0.001877`, rephrase NLL `+0.020679`; success/strict-success counts are 동일하다. Locality는 `−2/1000`이다.

C2−C0: BF16 energy `−29.366609` (−70.926%)이지만 final residual은 `+24.286801`, rewrite accuracy는 `−6/100`, rephrase success는 `−5/200`, strict rephrase success는 `−3/100`이다. C2는 suffix normalization/catch-up/second pass가 모두 0이며 residual reduction fraction은 0.063149다.

## 5. 기술 무결성 및 이력

- C0 source decision delta count 0; C0 exact control regression PASS.
- C1/C2 joint pi identity PASS. C1 final beta=1. C2 suffix normalization/debt/catch-up/second-pass/current-slope-inverse count는 모두 0.
- C2 intended quota identity max-abs는 모든 layer에서 0; layer 4 entry field reuse 1, layers 5–8 current key/q recapture 4; total writer solve count 9.
- router hard budget/weighted-sum/lexicographic/materialization/model F/B 영향 count 0.
- 각 arm post-commit BF16 hash와 virtual BF16 hash exact; 물리 commit 1; arm 종료 후 W0 restore PASS.
- TECH-R1 job22287, R2 job22289, R3 job22290, R4 job22291, R5 job22292는 create-once 기술 실패 이력으로 보존됐다. R6 job22295만 scientific endpoint로 사용했다. R1–R6 source commits는 `9475bafc`, `e4fd198e`, `cc9f5f30`, `d757c690`, `847acb40`, `20911aa1`이며 R6는 `COMPLETED/0:0`이다.

## 6. 판정 경계

- `TECHNICAL_VALIDITY=PASS`
- `INDEPENDENT_PRODUCTION_GATE_C1=PASS` (C0 수준 Gen/strict Gen 유지 및 residual realization 유지)
- `INDEPENDENT_PRODUCTION_GATE_C2=FAIL` (C0 대비 Gen/strict Gen 및 residual realization 미유지)
- `SEQUENTIAL_NOT_EVALUATED`
- `PRODUCTION_NOT_SUBMITTED=true`
- `SCIENTIFIC_PROMOTION=false`

단일 B100 pilot 수치이며 10×B100 production 결과로 해석하지 않는다. Official AlphaEdit는 이 pilot에서 재실행·paired 분석하지 않았다.

## 7. Immutable evidence paths

- result: `/mnt/raid5/janghj/.codex/worktrees/odeeditsh2-s05-p1r52-joint-pc-c1-c2-independent-b100-v1/local/odebf/results/s05-p1r52-joint-pc-c1-c2-pilot-b100-tech-r6-v1`
- state: `/mnt/raid5/janghj/.codex/worktrees/odeeditsh2-s05-p1r52-joint-pc-c1-c2-independent-b100-v1/local/odebf/state/p1r52-joint-pc-c1-c2-pilot-tech-r6-v1`
- log: `/mnt/raid5/janghj/.codex/worktrees/odeeditsh2-s05-p1r52-joint-pc-c1-c2-independent-b100-v1/local/odebf/logs/p1r52-joint-pc-c1-c2-pilot-tech-r6-v1`

