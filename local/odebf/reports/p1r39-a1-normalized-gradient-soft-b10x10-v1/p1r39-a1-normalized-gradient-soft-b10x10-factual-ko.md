# P1R39 정규화 복합-gradient Soft 확장 B10×10 결과 (사실 기록)

- 생성 시각(Asia/Seoul): 2026-08-14T14:04:26.218399+09:00
- instruction: `ODEEDIT-S05-P1R39-A1-NORMALIZED-GRADIENT-SOFT-B10X10-V1`
- source: `670132cf333da88ffef51a4b3c0a0da6b4870804`; P1R39 Neutral base: `763457560f2efb177a56310dfd87526772cf8158`
- scheduler: `19702`, array `0-1%2`; tasks COMPLETED0 2/2
- attempts/endpoints/failures: 20/20/0; accepted steps: 160/160
- stream root/order: `74d6896535fe46211e3f11d3d9b420c1ab36f1d503ee81f9eaace23c2fcb89e6` / `abe62c071168789b4a1e5ff57d2645ea16a328946362ea7bff166d6d5c76cd5c`
- scientific_promotion: `false`

## 1. Soft terminal 집계

|model|A/E/F|Eff|Gen|Loc|Eff NLL|z8 NLL|W8 NLL|gap|Structural-P|functional-P mean|capacity|energy|realization|negative steps|edit-core s|eval s|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|llama3-8b-inst|10/10/0|100/100|181/200|875/1000|0.125622|0.082181|0.128860|0.046678|0.006274|0.020192|0.035811|0.058079|0.735312|0|1039.32|190.65|
|qwen2.5-7b-inst|10/10/0|96/100|156/200|845/1000|0.770430|0.616813|0.668303|0.051490|0.121404|0.014228|0.121591|0.109191|0.732045|0|1359.39|213.33|

## 2. Case-paired 산술 비교

`Δ`는 Soft − comparison이다.

|model|comparison|matched/attempts|ΔEff|ΔGen|ΔLoc|ΔEff NLL|Δz NLL|ΔW NLL|Δgap|P ratio|capacity ratio|energy ratio|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|llama3-8b-inst|P1R39-Neutral|10/10|0|2|0|-0.022403|-0.010101|-0.015352|-0.005251|0.951700|0.919505|0.936914|
|llama3-8b-inst|P1R38-Soft|10/10|1|-3|48|-0.394012|-0.334236|-0.424090|-0.089855|0.378317|0.403039|0.456999|
|llama3-8b-inst|P1R36-RS-Soft|8/10|0|3|1|0.001239|0.027582|0.009077|-0.018505|1.037906|1.454352|1.441305|
|llama3-8b-inst|Official-AlphaEdit|10/10|0|-4|16|0.124453|NOT_RECORDED|NOT_RECORDED|NOT_RECORDED|NOT_RECORDED|NOT_RECORDED|NOT_RECORDED|
|qwen2.5-7b-inst|P1R39-Neutral|10/10|0|1|1|-0.058775|-0.063698|-0.065855|-0.002157|0.599142|0.356866|0.424230|
|qwen2.5-7b-inst|P1R38-Soft|10/10|-2|-19|3|0.243372|0.319154|0.279851|-0.039303|0.286000|0.157249|0.213492|
|qwen2.5-7b-inst|P1R36-RS-Soft|9/10|-3|-2|1|0.387157|0.397423|0.387357|-0.010065|1.038911|1.257640|1.019579|
|qwen2.5-7b-inst|Official-AlphaEdit|10/10|-4|-36|17|0.737784|NOT_RECORDED|NOT_RECORDED|NOT_RECORDED|NOT_RECORDED|NOT_RECORDED|NOT_RECORDED|

- Official AlphaEdit latent-z/W/P/capacity/energy: `NOT_RECORDED`.
- Stepwise heldout Eff/Gen/Loc: `NOT_RECORDED`.

## 3. 불변식 및 파일

- K8/action-freeze/W0: 20/20/20.
- Adam/LR/bias/raw-cap/velocity-decay/remaining/debt/retry/backtracking/history: 0.
- matched-strength equality residual max: `<=1e-8` (160/160).
- Neutral raw-free package: `/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-s05-p1r39-pr-p1r38-normalized-gradient-neutral-b10x10-v1/local/odebf/reports/p1r39-normalized-gradient-neutral-b10x10-v1`; report SHA `97d7a6d36818dd5445d29850d03031b19b715c94434c236124c9289e01bd7548`.
- Soft raw result terminals: `e8ed3e44ec52ca93132718ad49886d7ad104f14c2c8263694baeb1cc480b23e0`, `0ad42d64ddaf9ce82c8337e1fbfd3d98a020de28c7811ea87d4b455fbb9649e6`.
- 사실·수치·산술 delta·identity·NOT_RECORDED만 포함.
