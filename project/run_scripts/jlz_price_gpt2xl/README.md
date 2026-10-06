# GPT2-XL ours PRICE — six-cell 실행 구현

역사적 method-only source는 85a09238로 보존한다. 신규 실행 지시
`USER-GH-PRICE-MODEL-RUNS-TRACKING-20261007-SERVER1`가 이 범위의
구현-only 제한을 대체했다. 정본의 exact authority bytes를 검사하고
별도 immutable source/config/input/resource lock에서 six-cell을 제출한다.
실제 job/관측 상태는 제출 receipt와 보고서에서 구분한다.

## 구현

- GPT2 serial attention→MLP, learned positions(1024), tied head, final LN 1회.
- 실제 Conv1D Parameter는 [6400,1600]. bias 포함 native addmm와
  `W_entry + (R.double() @ Q.T).T.float()`를 사용한다.
- physical L13..17, anchor17/readout47. R=[1600,B], H=[6400,6400].
- MEMIT: `(20000*C0 + H + K@K.T)Q=K`.
- AlphaEdit: `(10*I + N@(H+K@K.T))Q=N@K`; .02 projector 재사용.
  C0는 Alpha 비용 진단에만 사용하며 writer에 혼합하지 않는다.
- native YAML 실제 parser/constructor로 lr=.5, 평가20/갱신19/terminal19를 결속한다.
  ours EfficiencyAdamAbs·가격·cap/controller 의미는 유지한다.
- candidate no-grad BUILD, whole-owner subject loss와 same-layer/off-owner pullback.
  전체 builder 역미분을 수행하는 exact-gradient 방법이라는 주장은 하지 않는다.
- terminal native payload copy와 CPU FP32 H append 1회, RAM transaction,
  평가 비변이 검사 및 독립 scalar/row collector 구현.
- W&B current/pre·current/post·all_seen/post 분리 및 job ID/immutable receipt.
  fit/global_candidate는 20-slot batch 축이며 early-stop 시 빈 슬롯을 보간하지 않는다.
- NoCP. 기존 model/C0/P/다른 source 불변. 새 context는 아직 생성하지 않았다.

## CPU 확인

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /mnt/raid5/janghj/EasyEdit/.venv/bin/python -m unittest \
  project.run_scripts.jlz_price_gpt2xl.test_contract \
  project.run_scripts.experiment_tracking.test_method \
  project.run_scripts.experiment_tracking.test_job_identity \
  project.run_scripts.experiment_tracking.test_tracking
```

47 tests: 45 통과, 2 skip, 0 실패. CPU operator fixture/fake SDK 검사이며
target-model forward/backward, B1/B2, 실제 online scientific logging은 NOT_RUN이다.
이 결과는 역사적 owner 검사다. 실행 recall의 독립 source 검토와 재검사는
별도 receipt에 기록하며 target-model PASS로 승격하지 않는다.

## 남은 실행 검증

실제 자산 재결속/CPU token inventory, native context READY, A6000 actual B1의
native↔staged parity·normal equation·LOO·KKT·terminalcopy/Honce·B2 연결,
W&B online readback, 6-cell 자원 admission·봉인·제출은 수행하지 않았다.
준비/제출 보조 코드도 실제 환경에서 실행 검증되었다고 주장하지 않는다.
