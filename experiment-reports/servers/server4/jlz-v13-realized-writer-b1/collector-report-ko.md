# V13 B1 사실 보고

상태: COMPLETED
실행 source: `082300955e21a2c29218d66d98c5d2c37bc53a20`
cold W0/H0, 고정 first100, fit 1회 허용, RT/RD/MT/MD/CD 동일 계획의 별도 endpoint. B2/추가 fit/CP 없음.
CPU 검산과 실제 모델 검산은 별도 기록. 독립 reviewer 미사용(owner audit).

| endpoint | R | P | N |
|---|---:|---:|---:|
| W0 | 5/100 | 20/200 | 886/1000 |
| RT | 99/100 | 177/200 | 878/1000 |
| RD | 89/100 | 147/200 | 886/1000 |
| MT | 100/100 | 188/200 | 879/1000 |
| MD | 100/100 | 188/200 | 879/1000 |
| CD | 100/100 | 190/200 | 875/1000 |

정확도/실현 telemetry는 comparison-B1.csv, realization.json, writers.json에 분리했다.
margin_true_minus_new = true NLL − new NLL; 반대 부호는 부호를 바꾼 별도 필드만 사용한다.
Raw와 tensor는 Git 미게시. NO_BROADCAST_NOT_REQUIRED: 동일 서버 B1 관측, compact 보고/manifest만 공유.
원자료: `/data/janghj/ODE-edit/local/jlz-v13-realized-writer-b1/20261005-v1/attempt-r1/B1`
과학적 해석·후속 승격 없음.
