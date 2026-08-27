# BGODE-R3 G2 수치 경계 사실 보고서

## 판정

`R3_G2_ALL_PREFIX_JVP_FD_NUMERICAL_BOUNDARY_UNLOCALIZED_SCHEMA_GAP`이다. job 26916의 Llama/Qwen 두 task 모두 locked all-prefix normalized-actuator serial-JVP 대 central-FD identity에서 `NumericalBoundary`로 종료했다. 이는 Cauchy 결과를 보고 threshold를 조정하는 기술 실패가 아니라 predeclared numerical gate의 실패이다. 따라서 tolerance·threshold·ridge·damping·fallback·pinv를 변경하거나 재제출하지 않았다.

## 모델별 사실

| 모델 | scheduler | elapsed | boundary | arm/N/node | prefix | max abs error | panel endpoint |
|---|---|---:|---|---|---|---:|---:|
| llama3-8b-inst | FAILED 1:0 | 00:27:18 | all-prefix JVP/FD | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | 0 |
| qwen2.5-7b-inst | FAILED 1:0 | 05:18:14 | all-prefix JVP/FD | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | 0 |

G2 구현은 node receipt와 trajectory panel을 terminal JSON에서만 원자적으로 publish한다. 예외에는 `NumericalBoundary.receipt`가 있었지만 top-level failure artifact로 serialize하지 않았으므로 실패 arm/N/node/prefix, 최대 절대오차, 예외 전 dynamic write 수는 raw에 남지 않았다. 실행 시간으로 이를 역추정하거나 G1 값으로 대체하지 않았다. 이 누락은 `NOT_RECORDED_SCHEMA_GAP`이며 imputation은 0이다.

## 분모와 보존 경계

- scheduler failure: 2/2 tasks
- terminal endpoint: 0/2 models
- trajectory panel endpoint: 0/24 (3 arms × 4 N × 2 models)
- Cauchy arm assessment: 0/6
- sample: sealed ordinal 26 / case 17454 / natural unequal-nonprefix
- run source HEAD/tree: `423782776354b6779b8f5f312248e20111f75ba3` / `d672d23f049fc2f9bfeb569f2c4a2e84c179661b`
- 기존 result/log/private prefix는 immutable이다.
- R1/R2/P1R55/unrelated mutation 0, resubmit 0, scientific promotion false이다.

## 해석 한계

G1의 같은 natural sample은 locked all-prefix FD gate를 통과했지만, 그 값으로 G2의 실패 값을 대신할 수 없다. G2에서 어느 trajectory node가 처음 실패했는지는 저장되지 않았으므로 Full/Fisher/Plain 또는 N별 과학 비교와 Cauchy 판정은 모두 수행 불가이다. 이 보고서는 실패를 숨기지 않고 numerical boundary와 failure-schema limitation만 결속한다.
