# B1 r3: T0 완료 증거와 FD 미확립

51057은 FAILED(1:0), B1은 실행되지 않았다. 실행 source는 e3f92b43788d2491cad5b77328eb0229ccb69ca7이다. JSON serialization 수리는 정상 적용되었고 T0의 유일한 차단 항목은 pair_factor_AD_FD다. 누락 B1 수치를 0으로 채우지 않았다.

## 실제 차이

4개 reference의 direct/cached gradient 상대차는 모두 0이다. 아래는 고정 12 FD scale 중 최소 상대오차이며, 최소치 하나를 선택해 통과 처리한 것이 아니다. 원 기준은 인접 2 scale에서 각각 1% 이하다.

| Reference | AD | 최소 FD 상대오차 | 원 FD 판정 |
| --- | ---: | ---: | --- |
| 57 | 0.00639223662735 | 0.626061% | 인접 두 scale 미충족 |
| 125 | 0.00380245550018 | 3.107715% | 미충족 |
| 332 | -0.00608286868131 | 1.295147% | 미충족 |
| 337 | 42.3283160609 | 0.000447% | PASS |

57의 처음 세 scale 오차는 0.6261%, 11.2988%, 0.6261%로, 두 유효 scale이 인접하지 않았다. 반복 동일점 noise와 perturbation roundoff는 같은 의미가 아니며 실패 원인을 단정하지 않는다. 전체 scale은 [FD-grid.csv](FD-grid.csv)에 보존한다.

## FD 이외 저장된 실제 검사

Native 4-request FP32 write replay exact, Current Q geometry, reference repeat, panel gradient/basis, Current affine/physical invariant, STEP/CUM cross-term, W/M/RNG restore가 통과했다. Current physical/cached NLL 및 logits 차이는 0, strict/pair ID는 동일하다. 실제 보호 probe logit 최대차 3.5047531128e-5, RMS 3.2131345794e-6, NLL 최대차 3.0517578125e-5. 이는 full512 과학 실행 또는 전체 numerical validation PASS가 아니다.

사용자 후속 승인 “통과할테니 task 이어서 진행해”는 FD 미확립을 기록한 B1 진행 허용으로 별도 r4 lock에 결속한다. 원 FD threshold/grid/result는 변경하지 않고 full_numerical_validation=NOT_ESTABLISHED를 유지한다. 기존 완료 T0를 재사용하며 별도 GPU T0 반복은 하지 않는다. 방법의 geometry/Current/reference/Armijo 등 수용 조건은 유지한다.

## 비용·범위

51057 parent allocation 538 GPU-sec, program 533.910165초, nested T0 468.085079초다. 중첩 시간을 더하지 않는다. 앞선 50974(136), 51055(104), 51056(335)와 합한 실패/기술 parent allocation은 1113 GPU-sec다. 이 값은 B1 비용이 아니다. B1 결과/commit/history/성능은 모두 NOT_RUN. 새 W/M disk checkpoint 없음, exact resume NOT_AVAILABLE. S3/S10 실행 권한 없음.

## 검산과 근거

CPU 독립 FD 재분류는 저장된 48 scale과 원 판정을 일치 확인했다. R/P/N 독립 reducer는 B1 부재를 확인하고 [first-table.csv](first-table.csv)에 NA를 기록했다. Raw/source/실패 기록은 수정하지 않았다. [T0-summary.json](T0-summary.json), [input-manifest.json](input-manifest.json), [rooted-receipt.json](rooted-receipt.json), [independent-reducer.json](independent-reducer.json)을 참조한다. 별도 독립 agent 감사는 수행하지 않았다. HTML renderer 패키지 미설치로 실제 HTML 렌더는 NOT_RUN이다.

재현 모듈: project.run_scripts.single_layer_mechanism_first.review_t0 및 review_b1. 원 terminal output을 입력으로 새 분석 디렉터리에 출력한다. 모델/forward/GPU 재실행 없이 수행한다.
