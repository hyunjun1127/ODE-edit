# P1R24 Atomic Strength Recovery — paired 종합

## 한 줄 결론

**최종 계약 분류: `ATOMIC_NOT_RECOVERED`.** Llama는 RS에서 강도를 회복했지만 일반적인 Soft 보존 신호는 없었고, Qwen RS Neutral은 Native/P1R23 strength 기준을 놓쳤으며 Qwen BG는 typed boundary로 endpoint가 없었다. 누락 arm을 보간하지 않으며 승격은 false다.

## FACT

| model | alloc | Neutral E/G/L | Soft E/G/L | Soft−Neutral 핵심 변화 |
|---|---|---|---|---|
| Llama | RS | 10/20/82 | 10/20/81 | Eff/Gen NLL 악화, Loc -1, terminal P/capacity 증가, fKL 소폭 감소 |
| Llama | BG | 10/20/81 | 10/20/82 | Eff NLL·fKL·Loc 개선, Gen NLL/P/capacity 악화 |
| Qwen | RS | 10/19/80 | 10/20/80 | Eff/Gen NLL, P, fKL, capacity 개선; Neutral strength gate 실패 |
| Qwen | BG | typed k4 boundary | not started | endpoint 비교 불가 |

- 유효 smoke 2/2, 완결 production arm 6/8, accepted production fields 52개다. 완결 arm은 모두 action-freeze/manifest/W0 restore를 통과했다.
- 모든 accepted field는 full-six physical W-only slope, `a^T v=rho_write`, h 정확히 1회, second remaining division 0을 기록했다. H/history 영향은 0이다.
- Soft가 실행된 24/24 field에서 same-state selected P proxy는 Neutral shadow보다 낮고 strength equality를 유지했다. 이 local optimization fact는 terminal preservation 보장을 뜻하지 않는다.
- Qwen BG는 Neutral k1–k4 후 `NO_POSITIVE_DIRECTION/Q_NUMERICAL_DEGENERACY`; retry/rescue 없이 종료했다. Soft 및 endpoint를 `NOT_RECORDED`로 둔다.

## 이전 P1R23 및 Native reference

- 동일-seal P1R23은 모든 8 arm에서 `10/20`, Loc Llama 81, Qwen RS 81, Qwen BG N/S 80/81이었다.
- P1R24 Llama RS Neutral은 Loc `81→82`이나 Gen NLL은 `0.526→0.961`; Qwen RS Neutral은 Gen `20→19`, Eff NLL `0.038→0.538`로 악화했다. P1R24 Qwen RS Soft는 Gen 20을 회복하고 Eff NLL `0.021`을 달성했지만 Neutral strength prerequisite를 대신할 수 없다.
- Official AlphaEdit endpoint reference는 Llama `10/20/79`, Qwen `10/20/80`; wall 51.90s/49.88s다. Continuous Native NLL 및 호환 pure-edit phase 분모는 없다.

## Compute

- 완결 arm당 `180F/125B`, materialization 8회; tokens Llama `27,207`, Qwen `24,687`이다.
- edit-core 범위는 Llama `78.96–89.75s`, Qwen RS `88.42–107.23s`. Terminal evaluator wall은 약 `0.99–1.08s`다.
- P1R24의 AlphaEdit KL-gradient geometry 때문에 이전 P1R23 `130F/80B`와 직접 같은 compute method가 아니다. Official AlphaEdit와의 formal ratio는 `NOT_RECORDED`다.

## INFERENCE

1. Llama: absolute edit count는 회복됐지만 Soft 효과는 RS와 BG에서 상충하며 cumulative-P/capacity를 보편적으로 낮추지 못했다.
2. Qwen: RS Soft만 보면 강한 긍정 신호지만, RS Neutral prerequisite와 BG 완결성이 깨져 package recovery라고 결론낼 수 없다.
3. cumulative Structural-P router는 local proxy를 안정적으로 최적화했으나 BF16 actual realization과 terminal P/Loc는 state-dependent하게 어긋났다. 보존 보장은 성립하지 않는다.

## NOT_RECORDED

- Qwen BG endpoint/Soft/restore
- Native continuous NLL, compatible pure-edit profiler denominator
- fresh-sample, sequential/Historical, B100, 보편성

## 주장 경계

Outcome-selected 재사용 B10의 atomic, locally-linearized, matched-predicted-strength mechanistic/descriptive 결과다. 이 실험은 fresh promotion gate가 아니며 `scientific_promotion=false`; 후속 모델/GPU/Slurm 작업은 이 보고서가 승인하지 않는다.
