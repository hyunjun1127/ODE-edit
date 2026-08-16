# P1R35 Full-Current-Residual Atomic — 사실 결과

- 생성 시각: 2026-08-13 22:54:23 KST
- instruction: `ODEEDIT-S05-P1R35-P1R34-FULL-CURRENT-RESIDUAL-V1`
- scientific parent: P1R34 execution `ac321d86a634a20888cb57003a65467d40aa5ddd`
- P1R35 scientific checkpoint: `a625e3d1cded3ced0e9128ef7a44205953041447`
- P1R35 TECH-R1 execution/source: `117ece2ee1132e7277a0b43ec5ac77feebfb2646`
- tree: `1b88c6f36482234a78c6e34fe8f3b5f6faddba3c`
- source manifest SHA-256: `5406c6e58318c1a144e77b3da1cac1be057f8776049bd673d65a87344eccbc79`
- numerical lock SHA-256: `136c4286bc496373b0f14f8f84ddc4982abcba8c7dc8e30be9d11260b9766c89`
- seal: `3d38b76de81c21e65068b1c1e22466ec55a0c5973ae289081d773391ed92f628`
- order: `984fe6ecc419fd0f701dae6032f33556f65a94dbb2850b57196be56c353b015b`
- method delta: `u=d+lag=target_next-current_terminal`; remaining-horizon division 0
- scope: Llama/Qwen × RS × Neutral/Soft, one sealed B10 Atomic endpoint per arm
- scientific promotion: `false`

이 문서는 raw-free 사실·수치 표만 제공한다. 과학적 해석, 인과 판정, 추천, 후속 설계는 포함하지 않는다.

## 실행 및 무결성

| 항목 | 값 |
|---|---|
| B1 attempt | job `19457`; Llama completed0, Qwen TECH-R1 이전 technical fail |
| B1 TECH-R1 replacement | job `19459`; Qwen completed0 |
| production | job `19460`, array `0-1%2`; Llama/Qwen completed0 |
| production scheduler | Llama `00:05:00`, MaxRSS `7,104,320 KiB`; Qwen `00:05:28`, MaxRSS `10,635,196 KiB` |
| attempted/completed arms | 4/4 |
| accepted transitions | 32/32 |
| typed scientific failure | 0 |
| K8 / tau1 | 4/4 |
| action freeze before heldout | 2 paired tasks PASS |
| official terminal evaluator | arm당 1회 |
| W0 pointer/byte restore | 4/4 PASS |
| maximum `u=d+lag=target_next-y` residual | 0 |
| maximum `h*write_velocity=u` residual | 0 |
| maximum routing equality residual | `4.440892098500626e-16` |
| remaining/fresh/lag division counts | 0 / 0 / 0 |
| semantic debt / second-h counts | 0 / 0 |
| physical h application count | 32 (accepted step당 1) |
| finite endpoint objective | logical 32; microbatch model forward 160; backward 0 |
| retry/backtracking/history/inner-heldout | 0 / 0 / 0 / 0 |

TECH-R1 사실: 최초 Qwen B1에서 독립 FP32 cast된 P1R24 target displacement와 `target_next-current_target` 사이의 1 rounding-unit 차이가 P1R35 exact-coordinate identity를 넘겼다. TECH-R1은 accepted FP32 state로부터 `d`를 직접 재구성했다. scientific formula와 tolerance는 변경하지 않았고 실패 root는 보존했다.

## Terminal endpoint

| model | arm | status | Eff | Gen | Loc | Eff new/old/margin NLL | Gen new/old/margin NLL | Loc new/old/margin NLL | z8 full6 new-NLL | W8 full6 new-NLL | z-W gap | cumulative P | capacity sum | functional-P mean |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama | Neutral | K8 COMPLETE | 10/10 | 20/20 | 82/100 | 0.046172 / 10.762500 / 10.716328 | 0.924577 / 8.771875 / 7.847298 | 8.312031 / 4.303210 / 4.008821 | 0.047648 | 0.085726 | 0.038078 | 0.00310104 | 2.259457 | 0.00587365 |
| Llama | Soft | K8 COMPLETE | 10/10 | 20/20 | 82/100 | 0.042726 / 10.831250 / 10.788524 | 0.898883 / 8.757812 / 7.858929 | 8.327813 / 4.305581 / 4.022231 | 0.043347 | 0.081244 | 0.037897 | 0.00316229 | 2.324124 | 0.00653954 |
| Qwen | Neutral | K8 COMPLETE | 10/10 | 20/20 | 80/100 | 0.241483 / 11.465625 / 11.224142 | 2.318524 / 9.789844 / 7.471320 | 8.850781 / 5.172231 / 3.678550 | 0.129565 | 0.137112 | 0.007547 | 0.07180831 | 8.076977 | 0.04436076 |
| Qwen | Soft | K8 COMPLETE | 10/10 | 20/20 | 80/100 | 0.163577 / 12.431250 / 12.267673 | 2.236621 / 9.795313 / 7.558691 | 8.847969 / 5.174609 / 3.673359 | 0.035103 | 0.078203 | 0.043100 | 0.06013332 | 5.802717 | 0.04045231 |

## Step aggregate

`actual`은 다음 accepted state의 refreshed W-only full-six target-new NLL 또는 k8 terminal objective로 계산되었다. 모든 arm에서 negative actual step 수는 0이다.

| model | arm | mean realization | minimum actual | k8 predicted | k8 actual | k8 realization | negative steps | edit-core s | F/B/materializations | tokens |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama | Neutral | 0.725789 | 0.056329 | 0.095681 | 0.056329 | 0.588724 | 0 | 122.647 | 220/125/8 | 35,263 |
| Llama | Soft | 0.730170 | 0.058674 | 0.097813 | 0.058674 | 0.599860 | 0 | 93.226 | 220/125/8 | 35,263 |
| Qwen | Neutral | 0.761301 | 0.037010 | 0.045324 | 0.037010 | 0.816559 | 0 | 144.414 | 220/125/8 | 32,023 |
| Qwen | Soft | 0.761951 | 0.094187 | 0.132497 | 0.094187 | 0.710858 | 0 | 86.950 | 220/125/8 | 32,023 |

전체 32-step 상세값은 `p1r35-per-step.csv`에 있다. 열은 `L_base`, `L_endpoint`, old/finite rho, predicted/actual/realization, next W NLL, target loss, d/lag/u norm, q, P/capacity, cumulative P, equality 및 coordinate counters/hashes를 포함한다.

## 동일 seal/order의 immutable endpoint 숫자

아래는 source reports의 기록값을 전사한 사실 표다. P1R34 Qwen은 paired official endpoint가 없으므로 `NOT_RECORDED`다.

| model | arm | P1R33 E/G/L | P1R33 Eff NLL | P1R33 z8/W8 full6 NLL | P1R34 E/G/L | P1R34 Eff NLL | P1R34 z8/W8 full6 NLL | P1R35 E/G/L | P1R35 Eff NLL | P1R35 z8/W8 full6 NLL |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama | Neutral | 4/7/83 | 4.388672 | 0.053980 / 4.210408 | 10/20/81 | 0.070964 | 0.044640 / 0.119840 | 10/20/82 | 0.046172 | 0.047648 / 0.085726 |
| Llama | Soft | 4/7/83 | 4.291406 | 0.051199 / 4.148334 | 10/20/82 | 0.069351 | 0.043170 / 0.120770 | 10/20/82 | 0.042726 | 0.043347 / 0.081244 |
| Qwen | Neutral | 9/16/80 | 1.109961 | 0.042414 / 0.884423 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED / 0.16415 internal only | 10/20/80 | 0.241483 | 0.129565 / 0.137112 |
| Qwen | Soft | 8/12/80 | 2.570996 | 0.015626 / 2.092753 | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | 10/20/80 | 0.163577 | 0.035103 / 0.078203 |

Reference identities:

- P1R34 report SHA-256: `df54b3fb30956187da2a37f78638b3e7e3a9ceb9328b233208727afcbe6693d2`
- P1R33 report SHA-256: `fea3bf33280798662e2d1ec629e890b00631e16775f1e128fb9117401ece74d1`
- P1R33 endpoint JSON SHA-256: `806a8590d1d7f7e8ae5c6089211bc546ae9b06248f8053de14560d26c5c0f8a6`

## Authoritative artifacts

- Llama terminal: `local/odebf/results/s05-p1r35-llama3-8b-inst-b10-rs-full-current-residual-neutral-soft-pair-v1/terminal.json`, SHA-256 `220f56e3bf51be32663643efdc5531d0648fa407c962375c71fb2de9538ae6f6`
- Llama manifest: SHA-256 `2da9d4780c26e5611510e72a402b309e3da279e2aedfdf99659c7018a831c52e`
- Qwen terminal: `local/odebf/results/s05-p1r35-qwen2.5-7b-inst-b10-rs-full-current-residual-neutral-soft-pair-v1/terminal.json`, SHA-256 `07fc95bfaf55e6cb6aa3c9de04cd4edc10889ccdab588e5ff49b470cdd5279c9`
- Qwen manifest: SHA-256 `0066e3cfac0584350c6894bd76d47da069946e965529e513324db99b90b5285d`

## NOT_RECORDED

- k0..k8 heldout Eff/Gen/Loc: 계약상 terminal-only이므로 `NOT_RECORDED`.
- formal scientific verdict, causal interpretation, recommendation, promotion claim: 이 결과 전달 범위에서 작성하지 않음.
- scientific promotion: `false`.
