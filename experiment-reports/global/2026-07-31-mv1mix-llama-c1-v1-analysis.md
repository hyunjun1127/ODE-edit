# Llama C1 독립 post-run 분석

## 결론 경계

`mv1mix_llama_c1_v1`의 sanitized artifact는 유효하다. Slurm step
`15576.0`은 `COMPLETED`, exit code `0:0`이며 run은 `completed`,
planned/attempted/pass가 `12/12/12`, outcome이 `72`, failure가 `0`이다.

이 문서는 `llama3-8b-inst` 한 모델의 C1 fold 0 gate input만 기록한다.
`clear_continue_input=true`, `gray_input=false`,
`scientific_kill_input=false`, `controller_pivot_input=false`는 단일 모델
입력값일 뿐이다. Pair-level decision은 계산하지 않았다.

## Artifact와 실행 lock

- 검증: `artifact_validation.valid=true`, `error_codes=[]`.
- fold: canonical confirmatory 60의 `confirmatory[0::5]`, fold `0/5`,
  exact 12 case. Selected-case hash는
  `20d5249f481ae367bb6447140f90681a34becc08951e60c0324ea571d3c142ac`,
  selection manifest는
  `186372487e0a79d64a4895dfa5ad142a5e9b1dd3f675c9a3468205147c38c4ce`,
  fold manifest는
  `8b428f5c619df6b056cb9507042745e438dc50cbe261cd547e16f5d8469515a1`이다.
- panel: case마다 정확히 6 arm이며 순서는 `score_mix`,
  `frozen_static_mix`, `uniform`, `ordered_global_alpha`,
  `native_memit_full`, `no_op_replay`다. Primary estimand는
  `progress(score_mix)-progress(frozen_static_mix)`다.
- firewall/receipt: 12개 commitment와 12개 receipt가 모두 고유하며,
  `outcomes_observed_before_commitment=false`다. Action은
  write → flush → fsync → exclusive receipt 뒤 outcome을 연다.
  Static/adaptive weight는 outcome을 사용하지 않았고 forecast policy의
  confirmatory outcome 사용 목록도 비어 있다.
- equal-C/rollback: `q=1/256=0.00390625`. 네 matched-C arm의 case별
  C-distance 최대 편차는 `0`, C-energy 최대 편차는
  `3.930214223594455e-13`이다. 72개 outcome과 12개 event의 rollback은
  전부 exact다.
- replay: C1 no-op absolute progress 최대값은 `0`; 수치 floor를 적용한
  C1/calibration/effective replay envelope는 모두 `1e-12`다.
- 분석 과정에서 `direct_z/*.pt`를 열거나 역직렬화하지 않았다.

Exact case 순서는 `5349`, `609`, `13925`, `17345`, `2700`, `14151`,
`8769`, `12891`, `15915`, `135`, `17238`, `8583`이다.

## Primary와 matched-C oracle

| metric | adaptive − frozen static | matched-C finite-panel oracle opportunity |
| --- | ---: | ---: |
| case count | 12 | 12 |
| mean | 0.008755276600519816 | 0.008755276600519816 |
| 20% trimmed mean | 0.006667003035545349 | 0.006667003035545349 |
| median | 0.00555419921875 | 0.00555419921875 |
| positive sign above replay | 12/12 (1.0) | 12/12 (1.0) |
| paired bootstrap mean 95% CI | [0.00441025123000145, 0.013934093713760377] | [0.004272228479385376, 0.013962519168853762] |
| observed range | [0.00049591064453125, 0.02647113800048828] | [0.00049591064453125, 0.02647113800048828] |
| bootstrap | seed 20260731, 4000 | seed 20260732, 4000 |

Oracle은 locked matched-C panel
`{score_mix, frozen_static_mix, uniform, ordered_global_alpha}` 안에서만
계산한 opportunity다. 이 표는 단일 모델 gate input이며 별도 효능 주장을
추가하지 않는다. CI endpoint 하나만으로 hard gate를 만들지 않았다.

## Predicted vs realized와 calibration forecast

D0+D1 17-case calibration lock은 non-negative zero-intercept robust transfer
`beta=0.9497786623102332`, calibration hash
`1f1ecf1a393df1f5474e523be49ca7d1dbecc9de3b812d766a1c3e41f9ca51a0`,
residual envelope `0.0012365315172465315`다.

- C1 predicted mean: `0.009918976144819487`
- C1 realized mean: `0.008755276600519816`
- realized − predicted mean: `-0.0011636995442996718`
- MAE / median AE: `0.00436933714406695` /
  `0.0005172882016112145`
- Pearson: `0.6803640896042981`
- zero-intercept realized-on-predicted slope: `0.5864924590968326`
- positive-direction concordance: `1.0`

사전에 고정한 calibration forecast의 expected realized gap은
`0.010963672438775659`, case-bootstrap 95% interval은
`[0.005291678279052575, 0.017507945627132517]`다. C1 realized mean과의
차이(realized − locked forecast)는 `-0.0022083958382558434`,
C1 predicted mean과의 차이(predicted − locked forecast)는
`-0.0010446962939561714`다. Realized mean은 이 locked interval 안에 있다.
이 forecast는 local progress-gap calibration일 뿐이다.

## Policy와 hash

- Static policy hash:
  `e37443fa075245aa52828773ffc062217d983fb46254d1f40010d396be3d5e06`
  (file SHA-256
  `935ec07fb72907c79ccaf9563e314bd2513c462888180cb97810af9662886ce6`)
- Forecast policy hash:
  `da5b352cbae20a44ff24e22f9c598f4a5df88aeab373995df3025ce2d0894533`
  (file SHA-256
  `51d09aaf8c3a98256d482a0c9bddda309055cde66e33cf1f012f112194a3d0a6`)
- Forecast analysis internal hash:
  `d8b371b37c76877c06349251d7abd3f9c5d828421c441f31e272cacd2b2e2094`
  (file SHA-256
  `6f977b55ed6c4d85761ef65e809f90a231cf8d97357f40a3e630c005d43a8d7f`)
- Fold01 input internal analysis hash:
  `4cfae701eca06e335ba7f86376c807c48b1288fd74574168b3d724b23448681e`
  (file SHA-256
  `792b9c7900bb1452f8e3fd6bb7f7c1bffff90f9295c7f309655d66b1c22c751f`)

Raw source hashes는 small summary에 모두 기록했다. Analyzer가 manifest,
stream 4개, receipt hash map을 raw summary와 exact 대조했고 오류는 없었다.

## Run/analysis code 경계

Run manifest는 ODE-edit commit
`03a5df6e65c4933b66178361ac5683ee4e02b1a6` (`03a5df6`)과 당시
tracked-worktree clean을 기록한다. Post-run 분석 경계는 task에 고정된
analysis code commit `26d5ed2`이며, 실행한 analyzer 파일 SHA-256은
`5510710d3b2ed7c65d837bafc8cb6021cb9785cd83aef76e2827f35fd37f5556`이다.
Git/GitHub 조회로 이 label을 재추정하지 않았다.

즉 run action/outcome은 `03a5df6` 산출물이고, `26d5ed2` 분석기는 sanitized
artifact를 읽어 post-run envelope만 생성했다. 기존 C1 기본 반환 shape와
새 enriched fold01 input의 분리를 검증하는 legacy test가 통과했으므로
기존 C1 동작 보존도 확인했다.

## 네 가지 근거 범주

| 범주 | 기록 |
| --- | --- |
| Proposal | §10.8–10.9가 fold, panel, estimand, bootstrap과 gate 규칙을 사전 고정한다. 이는 설계 근거이지 관측 효과 증거가 아니다. |
| Repo 사실 | Slurm/raw completion, exact hashes, receipt/firewall, equal-C, rollback, analyzer metric, deterministic rerun, focused test 결과만 직접 확인했다. |
| GH 추정 | 사용하지 않았다. Git/GitHub를 조회하거나 상태를 추정하지 않았다. |
| 사용자 확인 필요 | 이 분석 작성 자체에는 추가 확인이 필요 없었다. Submit/cancel, pair review 또는 후속 실행은 수행하지 않았으며 이 단일 모델 문서가 승인하지 않는다. |

## 재현·검증

요청된 CLI를 bootstrap override 없이 실행했다: seed `20260731`,
replicate `4000`, `--fold01-aggregate-input`. 생성 경로는
`experiment-reports/global/2026-07-31-mv1mix-llama-c1-v1.fold01-input.json`이다.
같은 명령을 임시 output에 재실행한 결과 원본과 byte-identical이었고 두
SHA-256은 모두
`792b9c7900bb1452f8e3fd6bb7f7c1bffff90f9295c7f309655d66b1c22c751f`였다.

집중 테스트는 confirmatory analyzer 전용 7개와 fold01/legacy 경계 3개,
총 10개를 실행해 `Ran 10 tests in 0.518s`, `OK`였다. 포함된 경계 test는
legacy C1 shape 분리, canonical fixed lock 강제, invalid envelope의
untrusted selection ID redaction이다.

Small summary는 모든 숫자가 finite인지 `allow_nan=false`로 검사하고,
`summary_hash`를 제외한 canonical JSON의 SHA-256을 내부 hash로 둔다.
