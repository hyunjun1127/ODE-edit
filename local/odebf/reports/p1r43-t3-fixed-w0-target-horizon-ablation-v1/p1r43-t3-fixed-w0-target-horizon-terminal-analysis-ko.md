# P1R43-T3 Fixed-W0 Target-Horizon Causal Ablation 종결 분석

- 생성 시각(Asia/Seoul): `2026-08-15T13:39:56.323271+09:00`
- checkpoint: `d646ddf74e7cc9eb85fd5c9c668ef729d00f3464`
- exact P1R43 parent: `11508b6da11d606521b703037034e1814b70d8a8`
- stream/order: `74d6896535fe46211e3f11d3d9b420c1ab36f1d503ee81f9eaace23c2fcb89e6` / `abe62c071168789b4a1e5ff57d2645ea16a328946362ea7bff166d6d5c76cd5c`
- 시도/endpoint/typed failure: `20 / 20 / 0`
- scientific_promotion=false

## 한 페이지 5문항 요약

1. **T8→T24가 target weakness를 해결했는가?** 두 모델 모두 10/10 case에서 full-six case-mean NLL이 감소하여 계약의 horizon-positive 방향 조건을 충족했다. Llama는 full-six mean `0.260137→0.174681`, Eff `99/100→99/100`, Gen `195/200→196/200`; Qwen은 `1.008113→0.888438`, Eff `93/100→94/100`, Gen `166/200→168/200`이다. 다만 Qwen T24의 full-six mean `0.888438`과 Gen `168/200`은 target weakness가 완전히 해소된 endpoint는 아니다.
2. **두 모델 공통인가?** full-six 10/10 case 개선과 Eff/Gen 비감소는 공통이다. 개선 절대량과 최종 강도는 다르며, Llama T24가 Qwen T24보다 낮은 full-six NLL과 높은 Eff/Gen을 기록했다.
3. **P2R1과 충분히 가까운가?** Llama T24−P2R1 full-six mean delta는 `+0.081329`, Eff `99−100`, Gen `196−197`; Qwen은 `+0.887794`, Eff `94−100`, Gen `168−193`이다. Qwen은 P2R1에 가깝지 않고, Llama도 exact equality는 아니다. 따라서 두 모델 공통으로 단순 P1R43-T3로 복귀할 수 있다는 조건은 충족되지 않았다.
4. **남은 병목은 무엇인가?** 후반 CURRENT 비율은 Llama `140/1600`, Qwen `340/1600`; path/net mean ratio는 `0.9764`, `0.9225`라 큰 진동 지표는 없다. full-six와 Gen이 함께 개선되어 TRAIN_HELDOUT_GAP/LONG_HORIZON_OVERFIT에는 해당하지 않는다. Qwen은 추가 이동 후에도 높은 full-six NLL이 남아 `NORMALIZED_FLOW_UNDERSTRENGTH`; Llama는 소수 hard-tail이 남는 혼합 상태다. `FINITE_SELECTION_STALL`은 Qwen 후반 CURRENT 증가의 동반 telemetry이나 net motion이 작지 않아 단독 최종 분류로 쓰지 않았다.
5. **다음 writer gate target operator는?** 계약 행렬 기준으로 P1R43-T3는 두 모델 공통 target gate를 닫지 못했고 P2R1과의 차이가 Qwen에서 크다. 따라서 writer gate에 넘길 target operator로 단순 P1R43-T3를 선택하는 조건은 충족되지 않았다. 이 문장은 계약 5문항의 gate 답변이며 추가 실험 추천은 아니다.

## 무결성 및 scientific operator

- exact P1R43 requestwise primary/rescue/current selector, entry norm 1회, FP64 field/update, FP32 selected target를 사용했다.
- 동일 trajectory의 z8과 z24를 저장했고 h=`0.125`, target updates=`24`, reset@8/16=`0/0`이다.
- evaluator adapter는 T8/T24 모두 snapshot_index=8, accepted_snapshot_count=9, fixed_budget_slots_completed=8이다.
- P1R43 decision-operator replay이며 full executable-trace replay가 아니다. KL/decay/RMS/tangent/gamma/clamp decision influence=0이다.
- case terminal/W0 pointer/W0 bytes/calibration1/reset8·16 zero/forbidden zero: `20/20/20/20/20/20`.

## Primary causal: 동일 trajectory T8↔T24

Primary statistical unit은 10개 B10 case-mean paired delta이고, request 분포는 hard-tail 보조 통계다.

| Model | full6 case mean T8→T24 | 개선 cases | pooled request med T8→T24 | >=3 | <.05 | Eff | Gen | Eff NLL | Gen NLL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llama3-8b-inst | 0.260137→0.174681 | 10/10 | 0.041545→0.005286 | 2→2 | 57→94 | 99/100→99/100 | 195/200→196/200 | 0.230080→0.152482 | 1.703101→1.475637 |
| qwen2.5-7b-inst | 1.008113→0.888438 | 10/10 | 0.032229→0.004246 | 11→10 | 66→87 | 93/100→94/100 | 166/200→168/200 | 1.104646→0.947664 | 3.571337→3.456451 |

| Model | case Δfull6 mean/median/p90/worst | request Δfull6 mean/median/p90/worst | early P/R/C | late P/R/C | path mean | net mean | net/path |
|---|---|---|---:|---:|---:|---:|---:|
| llama3-8b-inst | -0.085456/-0.067830/-0.050997/-0.050522 | -0.085456/-0.031610/-0.002167/0.000000 | 782/5/13 | 1449/11/140 | 0.308962 | 0.301679 | 0.976427 |
| qwen2.5-7b-inst | -0.119675/-0.052536/-0.018269/-0.013958 | -0.119675/-0.016352/0.000000/0.000000 | 736/19/45 | 1248/12/340 | 1.985887 | 1.831977 | 0.922498 |

## Secondary method: P2R1 target-only z24

`SECONDARY_METHOD_MATCHED_COMPARISON`: model alias, tokenizer/model artifacts, frozen stream/order/case, full-six context plan, evaluator 및 exact P1R43 parent가 일치한다. Target operator/hparams는 비교의 의도된 방법 차이이며 동일하다고 주장하지 않는다.

| Model | T3/P2 full6 mean | ΔT3−P2 | Eff T3/P2 | Gen T3/P2 | Eff NLL T3/P2 | Gen NLL T3/P2 |
|---|---:|---:|---:|---:|---:|---:|
| llama3-8b-inst | 0.174681/0.093352 | 0.081329 | 99/100 / 100/100 | 196/200 / 197/200 | 0.152482/0.135148 | 1.475637/1.185918 |
| qwen2.5-7b-inst | 0.888438/0.000644 | 0.887794 | 94/100 / 100/100 | 168/200 / 193/200 | 0.947664/0.001119 | 3.456451/1.664718 |

## Mandatory historical reference: original P1R43 Neutral/Soft

이 절의 비교 역할은 `HISTORICAL_MATCHED_REFERENCE`이며 `DIRECT_CAUSAL_COMPARATOR`가 아니다. Original P1R43은 각 step에서 W를 변경하지만 P1R43-T3는 전체 trajectory에서 W0를 고정한다. 인과 주장은 하지 않는다.

| Model | Arm | P1R43 z8 full6 mean | T8−P1R43 | T24−P1R43 | P1R43 Eff | P1R43 Gen | P1R43 Eff NLL/margin | P1R43 Gen NLL/margin |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| llama3-8b-inst | Neutral | 0.236418 | 0.023719 | -0.061737 | 100/100 | 182/200 | 0.216326/10.475705 | 2.148835/5.871847 |
| llama3-8b-inst | Soft | 0.174896 | 0.085241 | -0.000215 | 100/100 | 184/200 | 0.159384/10.510616 | 2.163182/5.836767 |
| qwen2.5-7b-inst | Neutral | 0.789759 | 0.218354 | 0.098679 | 96/100 | 161/200 | 0.883366/10.384253 | 3.821950/3.924242 |
| qwen2.5-7b-inst | Soft | 0.753546 | 0.254567 | 0.134892 | 96/100 | 157/200 | 0.829520/10.789582 | 3.887184/3.902646 |

| Model | Arm | Eff NLL ΔT8/ΔT24 | Eff margin ΔT8/ΔT24 | Gen NLL ΔT8/ΔT24 | Gen margin ΔT8/ΔT24 |
|---|---|---:|---:|---:|---:|
| llama3-8b-inst | Neutral | 0.013753/-0.063844 | 0.201403/2.292750 | -0.445733/-0.673197 | 1.450654/2.610091 |
| llama3-8b-inst | Soft | 0.070695/-0.006902 | 0.166492/2.257839 | -0.460081/-0.687545 | 1.485734/2.645171 |
| qwen2.5-7b-inst | Neutral | 0.221280/0.064298 | -0.395277/1.638022 | -0.250613/-0.365498 | 0.866170/1.265860 |
| qwen2.5-7b-inst | Soft | 0.275126/0.118144 | -0.800606/1.232693 | -0.315847/-0.430732 | 0.887766/1.287456 |

Original P1R43 terminal full-six per-request median/p90/worst와 threshold별 hard-tail count는 원 raw-free P1R43 package에 `NOT_RECORDED`이다. 따라서 해당 필드는 상세 historical table에서도 `NOT_RECORDED`로 유지했다. Eff/Gen NLL·margin의 mean/median/p90/worst는 원 panel table에 기록되어 machine table에 보존된다.

## Compute ledger

| Model | semantic F/B | primary F | rescue F | endpoint F | target wall s | evaluator wall s | total case wall s | materialization |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| llama3-8b-inst | 1200/1200 | 1200 | 660 | 100 | 287.282123 | 16.815865 | 419.876648 | 0 |
| qwen2.5-7b-inst | 1200/1200 | 1200 | 910 | 100 | 252.455292 | 15.722467 | 389.172100 | 0 |

## Gate 및 taxonomy

- `TECHNICAL_FAIL`: 0/20.
- `HORIZON_POSITIVE`: 두 모델 모두 full-six case mean 10/10 감소, pooled median delta<0, Gen correct 비감소, Gen NLL 개선, >=3 hard-tail 비증가를 충족했다.
- Llama: horizon-positive이며 P2R1과의 잔여 차이는 full-six mean +0.081329, Eff -1, Gen -1이다. 단일 request worst가 T8/T24 모두 높아 mixed hard-tail이 남는다.
- Qwen: `NORMALIZED_FLOW_UNDERSTRENGTH`; 추가 path/net motion은 존재하지만 T24 full-six mean 0.888438, Eff 94/100, Gen 168/200이며 P2R1 대비 full-six +0.887794, Gen -25다.
- `TRAIN_HELDOUT_GAP`, `LONG_HORIZON_OVERFIT`: 해당 없음. 두 모델 모두 full-six와 Gen NLL이 함께 개선됐다.
- `OSCILLATORY_LONG_HORIZON`: 해당 없음. net/path mean ratio는 Llama 0.9764, Qwen 0.9225다.
- `FINITE_SELECTION_STALL`: Qwen late CURRENT 340/1600이 기록됐으나 post-T8 net motion이 작지 않아 단독 최종 분류로 확정하지 않았다.

## FACT / INFERENCE / NOT_RECORDED

### FACT

- attempts/endpoints/failures=20/20/0, W0 pointer+bytes 20/20, forbidden influence0 20/20.
- T8와 T24는 동일한 continuous z0→z24 trajectory의 immutable endpoints다.
- case/request exact paired raw values와 arithmetic deltas는 machine tables에 있다.

### INFERENCE

- 장기 horizon은 두 모델에서 target objective와 Gen 지표를 개선했지만, P2R1과의 잔여 차이는 특히 Qwen에서 크다.
- fixed-W0 causal contrast로 horizon 부족은 부분 원인이지만 단독으로 전체 target weakness를 설명하지 못한다.

### NOT_RECORDED / NOT_APPLICABLE

- stepwise heldout Eff/Gen/Loc: `NOT_RECORDED` (endpoint-only evaluator).
- W-only, physical locality, Structural-P, capacity, update norm/energy: `NOT_APPLICABLE` (target-only, W=W0).
- Original P1R43 terminal full-six per-request median/p90/worst 및 hard-tail count: `NOT_RECORDED`.
- task-level scheduler elapsed/MaxRSS: scientific case receipt에 `NOT_RECORDED`; job id는 submission receipt에 기록됨.

## Artifact identities

- contract SHA: `d9245782bf3339bfabe48296a9e673ba33763d1bca43be60982a182084ea67e7`
- P2R1 report SHA: `9920ddc361f38f8d8c1f7f17bb4991572f78170d8cb03456fd8227d6ff638c3d`
- original P1R43 report SHA: `ed60bd3bc57bf2ee14d35d558674734b5ab7c446125131b2a281f58b7d3bfbf7`
- detailed exact identities and all file hashes are in `analysis-manifest.json`.
