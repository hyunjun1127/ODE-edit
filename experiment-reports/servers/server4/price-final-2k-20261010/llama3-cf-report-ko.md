# Llama Ours CF: 최종 PRICE method 2K 결과 반영

사용자 결정(2026-10-10): Llama는 **β = c = 1.0 + early exit + HC**로 확정하고 2K CF를 돌린다. 이번 반영도 사용자 지시("main table에 최신화해")에 따른 것이다.
이 결과가 README 본표 Llama `PRICE (Ours)` 행의 CF 값(이전 FREE100 job 60103)을 대체한다.

## 실행

| 항목 | 값 |
|---|---|
| server / job | server4, Slurm **62604** `pf2k-llama3-P-beta100-ee` (2026-10-10 13:06–17:39 KST, COMPLETED) |
| attempt | `server4:/data/janghj/ODE-edit/local/price-final-2k-20261010/execution-llama3-r2` |
| source | commit `885a0e2624e40a2355aaba10a9b644319b2398e3`, archive sha256 `8acf7ef0…` |
| arm | official `llama3-P-beta100`, resolved sha256 `1208f9ce35b89fd530e522d8ee49b35fa212cd8057e94983a9f6d3702a8ff737` |
| method | β_base = c = β_max_scale = 1.0, anchor-unit lr ρ0.05, γ1, cap-endpoint FP32 cast, HC-PRICE, per-request early exit |
| sample | eval-2K first 2,000, B1–B20(100건씩). 순서 sha256 `0b912d11…`(baseline과 동일), slice identity `17358c33…` |
| context | 공용 `llama3.json`, semantic `cf14b857…`(baseline과 동일) |
| 평가 | 기존 strict NLL preference R/P/N(PRICE observer). W0는 README W0 행과 일치(8.20 / 10.95 / 88.55) |

- **early exit:** F < τ_F(0.05)이면 그 요청을 멈추고 R을 고정한다. F ≥ 2τ_F이면 재개한다. 업데이트 중인 요청이 없으면 fit을 끝낸다.
- early exit는 runner patch(`project/run_scripts/price_early_exit`)로 적용했고, official fit 코드는 그대로다.

## W20 (2,000 edits, all-seen)

| 지표 | 분자/분모 | 정확한 성공률(%) | 표 표시 |
|---|---:|---:|---:|
| Eff (R) | 1998/2000 | 99.90 | 99.90 |
| Gen (P) | 3740/4000 | 93.50 | 93.50 |
| Loc (N) | 16300/20000 | 81.50 | 81.50 |
| Score | 반올림 전 세 성공률의 조화평균 | 90.97763890410414 | 90.98 |

- 원 summary는 `tier-P-beta100/batch-20/post/summary.json`(sha256 `f46f5936…`)이다.
- 표시는 decimal half-up이다. CF Flu/Con은 아직 평가하지 않아 `DEFERRED`로 둔다.

## 경과와 이전 행 비교

| 시점 | 최종 method | 이전 FREE100 (60103) |
|---|---|---|
| W5 | 100 / 95.00 / 86.54, 93.51 | 100 / 93.80 / 86.74, 93.20 |
| W10 | 100 / 95.55 / 84.88, 93.03 | 99.9 / 93.90 / 85.09, 92.56 |
| W15 | 99.87 / 94.57 / 83.10, 91.96 | 99.7 / 93.43 / 83.84, 91.85 |
| W20 | 99.90 / 93.50 / 81.50, **90.98** | 99.70 / 92.78 / 82.21, **90.98** |

- W20 Score는 같다(90.978 대 90.982). P는 +0.72, N은 −0.71이다.
- 모든 baseline보다 높다. MEMIT-BLUE(90.72)보다 +0.26, AlphaEdit-BLUE(89.84)보다 +1.14다.

## checkpoint (500 edit마다, resumable)

| 시점 | editable weights sha256 | writer history H sha256 |
|---|---|---|
| W5 | `bcb724d9…` | `3a2bd517…` |
| W10 | `f9b28161…` | `d1a90923…` |
| W15 | `c57fccb6…` | `afb8d665…` |
| W20 | `a440c536…` | `bba86fff…` |

- 각 checkpoint에는 `W{n}-checkpoint.json`(HC 누적 통계, cursor, state·RNG 식별값)이 함께 있다.
- 정확한 수치와 전체 SHA: [results.json](../../../../audits/servers/server4/price-final-2k-20261010/llama3-cf-results.json).
- zsRE 2K는 같은 설정으로 server4 job **62889**에 제출했다(official zsRE stream·evaluator). Flu/Con은 W20 checkpoint로 별도 평가한다.
