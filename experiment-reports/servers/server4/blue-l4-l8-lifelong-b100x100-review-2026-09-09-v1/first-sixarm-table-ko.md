# 6-chain 첫 표: 실제 최종 W100에서 전체 10,000 요청 재평가

|arm|job|status|batches|requests|gpu_hours|RS|PS|NS|rewrite_TF_exact|rephrase_TF_exact|
|---|---|---|---|---|---|---|---|---|---|---|
|MEMIT_ORIGINAL|39307|TERMINAL_VALID|100|10000|11.6208|7183/10000 (71.83%)|13530/20000 (67.65%)|53287/100000 (53.29%)|3603/10000|4817/20000|
|AlphaEdit_ORIGINAL|392831|TERMINAL_VALID|100|10000|12.045|9888/10000 (98.88%)|19155/20000 (95.78%)|63726/100000 (63.73%)|9465/10000|13229/20000|
|MEMIT_L4_ONLY|392832|TERMINAL_VALID|100|10000|10.6289|7722/10000 (77.22%)|14966/20000 (74.83%)|57848/100000 (57.85%)|4265/10000|7129/20000|
|AlphaEdit_L4_ONLY|392833|TERMINAL_VALID|100|10000|10.9336|9939/10000 (99.39%)|19136/20000 (95.68%)|65348/100000 (65.35%)|9529/10000|13362/20000|
|MEMIT_L8_ONLY|392834|TERMINAL_VALID|100|10000|7.02333|8177/10000 (81.77%)|14460/20000 (72.30%)|48425/100000 (48.43%)|4448/10000|4859/20000|
|AlphaEdit_L8_ONLY|392835|TERMINAL_VALID|100|10000|8.59306|9396/10000 (93.96%)|15556/20000 (77.78%)|54703/100000 (54.70%)|7694/10000|6822/20000|

RS/PS: new 평균-token NLL < true NLL. NS: true NLL < new NLL. Tie 실패. TF exact는 teacher-forced 모든 target token top-1 일치로 primary preference와 다르다. Current-B100/online 합계가 아니다. 6개 final 평가 member SHA, sample/order, prompt pair 및 endpoint identity 독립 대조 완료; 전체 chain/checkpoint CPU 검산은 후속 단계다. 원본1k/JVP와의 설정·target-policy 차이를 통제한 인과비교가 아니다. scientific_promotion=false.
