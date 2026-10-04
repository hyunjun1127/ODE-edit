# V14 B1 사실 보고

상태: COMPLETED
실행 source: `2ab04d0b4d339573ee54fd85c7240d090a01e2b1`
cold W0/H0, 고정 first100, V14_RD fit 1회·commit 1회. B2/추가 fit/CP 없음.
CPU 검산과 실제 모델 검산은 별도 기록. 독립 reviewer 미사용(owner audit).

| endpoint | R | P | N |
|---|---:|---:|---:|
| W0 | 5/100 | 20/200 | 886/1000 |
| V14_RD | 100/100 | 181/200 | 882/1000 |

정확도/실현 telemetry는 comparison-B1.csv, realization.json, writers.json에 분리했다.
margin_true_minus_new = true NLL − new NLL; 반대 부호는 부호를 바꾼 별도 필드만 사용한다.
Raw와 tensor는 Git 미게시. NO_BROADCAST_NOT_REQUIRED: 동일 서버 B1 관측, compact 보고/manifest만 공유.
원자료: `/data/janghj/ODE-edit/local/jlz-v14-native-writer-aware-b1/20261005-v1/attempt-r1/B1`
V13 비교: PENDING_COMPARISON (기존 paused monitor 재개 및 결과 조회 없음).
과학적 해석·후속 승격 없음.
