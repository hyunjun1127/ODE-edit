# Qwen C1 독립 post-run 분석

## 결론과 범위

Qwen `mv1mix_qwen_c1_v1`의 허용된 sanitized 산출물은 유효하다. Slurm step
`15576.1`은 `COMPLETED`, exit `0:0`이고, run summary는
`run_status=completed`, `all_pass=true`, planned/attempted/pass
`12/12/12`, outcomes `72`다.

이 결과에서 계산된 것은 **Qwen 한 모델의 C1 gate input**뿐이다.
`clear_continue_input=true`, `gray_input=false`,
`scientific_kill_input=false`, `controller_pivot_input=false`이지만, 이는
pair 판정이나 후속 실행 승인이 아니다. Pair/cross-model 판정, method gain,
MV-2 및 ODE 주장은 이 분석에서 계산하거나 승인하지 않았다.

## Artifact 및 exact lock 검증

- Artifact validation: `valid=true`, `error_codes=[]`,
  `panel_complete=true`.
- Exact fold: selection seed
  `ode-edit-motivation-counterfact-v1`, fold seed
  `ode-edit-mv1-score-mix-confirmatory-folds-v1`, confirmatory fold `0/5`,
  exact 12 cases
  `5349, 609, 13925, 17345, 2700, 14151, 8769, 12891, 15915, 135, 17238, 8583`.
  Selected-case hash는
  `20d5249f481ae367bb6447140f90681a34becc08951e60c0324ea571d3c142ac`,
  selection manifest ID는
  `186372487e0a79d64a4895dfa5ad142a5e9b1dd3f675c9a3468205147c38c4ce`,
  fold manifest ID는
  `8b428f5c619df6b056cb9507042745e438dc50cbe261cd547e16f5d8469515a1`이다.
- Exact panel order:
  `score_mix`, `frozen_static_mix`, `uniform`,
  `ordered_global_alpha`, `native_memit_full`, `no_op_replay`.
  12 cases × 6 arms = 72 outcomes가 모두 있다.
- Matched-`C`: `q=1/256=0.00390625`. 앞의 네 matched-`C` arm은 12개
  case 모두에서 `c_distance`가 정확히 같았고, 48개 row가 검증됐다.
  부동소수 재측정 `c_energy`의 case 내 최대 span은
  `9.576245081199114e-11`이며 analyzer의 equal-`C` contract를 통과했다.
  `native_memit_full`과 `no_op_replay`는 matched-`C` oracle에서 제외되는
  reference arm이다.
- Firewall/receipt: sanitized JSON만 사용했고
  `outcome_firewall_and_receipts_valid=true`; feature/action 및 forecast
  commitment receipt 12개와 receipt hash map이 일치한다. Forecast policy의
  `confirmatory_outcomes_used=[]`도 유지됐다.
- Rollback/replay: 72/72 outcome의 `rollback_exact=true`; 12/12 no-op의
  `logits_hash_equal=true`, no-op progress는 모두 0이다.
  C1/calibration/combined replay envelope는 모두 `1e-12`다.
- 원시 artifact SHA-256:
  - `manifest.json`:
    `b5a746828931fea757242451226dc75c14ae1179da82189913073b5c2c20cdb8`
  - `features.jsonl`:
    `be683914f3be9163bfc75f8b971a517c9150e594422a14843a9caa18a5495aad`
  - `actions.jsonl`:
    `7cf4b985eabbe10d88e900706c3414f5d86e06803a9acde023a46536ea6b7b24`
  - `outcomes.jsonl`:
    `b6ea6125b7853151841564bd4a5cdac76e05161761983b1c9401086da328602f`
  - `events.jsonl`:
    `0050e70a2dab99081f634f06947bc948d409089b7dce57ac50f76029a779d63f`

## Primary와 matched-C oracle

Primary estimand은
`progress(score_mix) - progress(frozen_static_mix)`다.

| metric | Qwen C1 primary | matched-C finite-panel oracle |
| --- | ---: | ---: |
| mean | `0.059492299954096474` | `0.0627475877602895` |
| 20% trimmed mean | `0.03591315448284149` | `0.0359298437833786` |
| median | `0.037488460540771484` | `0.037488460540771484` |
| sign above replay | `11/12` (`0.9166666666666666`) | `11/12` (`0.9166666666666666`) |
| paired case-bootstrap mean 95% CI | `[0.017130000640948613, 0.12305578043063482]` | `[0.02319556698203087, 0.1264144552250703]` |
| observed range | `[-0.03737926483154297, 0.3771786689758301]` | `[0, 0.3771786689758301]` |

Primary bootstrap은 seed `20260731`, oracle bootstrap은 `20260732`, 각각
4000 replicates다. Oracle mean과 primary mean 차이는
`0.0032552878061930315`다. CI endpoint 하나는 hard gate가 아니다.

## Predicted vs realized와 calibration forecast

Locked C1 action forecast의 predicted mean은 `0.0919819535484502`,
realized primary mean은 `0.059492299954096474`다. Realized-minus-predicted
mean error는 `-0.03248965359435372`, MAE는 `0.03510902746891767`,
median absolute error는 `0.013232614757795899`다. Zero-intercept
realized-on-predicted slope는 `0.8224793064689951`, Pearson은
`0.8996197195922011`, positive-direction concordance는 `11/12`다.

D0+D1 calibration forecast는 beta `1.1595737381191145`, expected realized gap
`0.11201531043591464`, bootstrap interval
`[0.06476441445994054, 0.17629666495305207]`, residual envelope
`0.03236109868826046`였다. 차이는 다음과 같다.

- C1 predicted mean − calibration expected gap:
  `-0.02003335688746445`
- C1 realized mean − calibration expected gap:
  `-0.05252301048181817`
- C1 realized mean − C1 predicted mean:
  `-0.03248965359435372`
- C1 realized mean − calibration interval lower endpoint:
  `-0.0052721145058440624`
- absolute C1 mean prediction error − calibration residual envelope:
  `0.00012855490609325892`

따라서 realized mean은 두 forecast보다 작았지만 양수이며, 예측오차의 절댓값은
calibration residual envelope와 거의 같은 크기다. 이 비교는 forecast
calibration 진단이지 별도의 방법 효과 주장이 아니다.

## Policy, hash, code 경계

- Static policy hash:
  `381e22334e0e3d07c37f07db16c22e8cdd1b689deba66ec7b032dd32ba0b2955`
  (file SHA-256
  `1c785d1b0a4bf0be70a8eeb9a1ea9a6ddea85899becc950fefed3b6688230f0c`)
- Forecast policy hash:
  `a1feb93481f64785539c102a1ac58340fee71ba4e91a9ccc4b8f699cc6c04c99`
  (file SHA-256
  `edab0e6b077457e2a57bd34ba1e0db04fd935dd2012e80b158e23314d5ffaf92`)
- Calibration hash:
  `440f1eadd495d7e9a36cde258d711acfc0207867dab4d468b4c920765df0f122`
- Forecast analysis internal hash:
  `cabff79e285f252b7219de3f016e2ef280ad506a4b79264d5806dd8909ba6e22`
  (file SHA-256
  `97fc3855e1cb61c1b8c7ed274e212b8e78c220cf93e10ebe16dbc23290a1d69a`)

Raw run manifest의 ODE-edit commit은
`03a5df6e65c4933b66178361ac5683ee4e02b1a6` (`03a5df6`)이고 당시
tracked worktree는 clean이었다. Post-run analyzer 경계는 요청에서 고정한
`26d5ed2`다. 즉 raw run을 재해석하되 변경하지 않았으며, legacy C1 analysis
shape는 유지하고 fold01용 enriched wrapper를 별도로 내는
`test_default_c1_shape_is_legacy_and_enriched_input_is_separate`가 보존·통과했다.
Git은 호출하지 않았으므로 `26d5ed2`에 관한 추가 repository-history 추정은
하지 않는다.

## 네 가지 증거 범주

### 1. Proposal

Proposal 원문은 이 독립 분석에서 열거나 재평가하지 않았다. 구현 spec §10.9가
Proposal에 귀속한 “작은 diagnostic 뒤 살아남을 때만 다음 단계” 원칙만 gate
문맥으로 유지했다. Proposal의 더 넓은 주장으로 결과를 확장하지 않는다.

### 2. Repo 사실

위 completion, sanitized artifact identity, exact fold/panel, firewall,
equal-`C`, rollback, locked policy, primary/oracle/forecast 수치, hash,
결정적 재실행과 test 결과는 허용된 repository 산출물 및 분석 코드에서 직접
확인했다.

### 3. GitHub 추정

없다. Git/GitHub를 사용하지 않았고 branch, PR, remote, review 상태를
추정하지 않는다.

### 4. 사용자 확인 필요

이 완료된 read-only Qwen 분석을 보존하는 데 추가 확인은 필요하지 않다.
그러나 pair-level review나 어떤 후속 실행도 이 산출물의 권한 밖이며, 별도의
판정·권한 없이 추정하거나 실행하지 않는다.

## 재현성 검증

Analyzer CLI는 `--fold01-aggregate-input`과 default bootstrap
seed `20260731` / replicates `4000`으로 실행했다.

- Aggregate input:
  `experiment-reports/global/2026-07-31-mv1mix-qwen-c1-v1.fold01-input.json`
- Aggregate internal `analysis_hash`:
  `89af4e5ea6882464f6508789e5498557c9ace9855390103ec28e001cb4eddae9`
- Aggregate file SHA-256:
  `4920d798ab61fbb0ef5704e10f379562c932399a0c779d819f682f5398ab1dde`
- 임시 재실행:
  `/tmp/mv1mix-qwen-c1-rerun.ZmaJx4/2026-07-31-mv1mix-qwen-c1-v1.fold01-input.json`
  — byte-identical, 같은 SHA-256; 검증 뒤 임시 copy는 삭제했다.
- Focused tests:
  `test_mv1_score_mix_confirmatory_analysis` +
  `test_mv1_score_mix_followup_analysis` — `Ran 20 tests`, `OK`.
- Aggregate와 small summary는 non-finite JSON 값을 거부하도록 별도 검사하고,
  각 internal hash를 재계산해 일치시킨다.

이 보고서는 Qwen C1의 허용된 sanitized 범위만 사용했다. `direct_z/*.pt`는
열거나 역직렬화하지 않았고, 타 모델 raw/report/policy도 열지 않았다.
