# SH4: FD-only 사용자 승인 / B1 지속

최신 사용자 “통과할테니 task 이어서 진행해”를 앞선 FD-only 질문에 대한 승인으로 결속한다. 51057의 완료 actual T0에서 direct/cached gradient 및 FD 이외 항목을 재사용한다. FD는 원 판정 미확립, full_numerical_validation=NOT_ESTABLISHED이며 T0_PASS로 바꾸지 않는다.

새 immutable B1 r4 attempt는 T0 GPU 반복 없이 B1 공유 native와 네 정책을 진행한다. 수치/방법 보호조건 불변, checkpoint 저장 없음, 최대 batch=1, sequential 미승인. B1 완료 및 사실 보고까지 모니터링하며 다른 task는 재개하지 않는다.

CPU 227 tests PASS (9.240초). 실제 B1 결과는 아직 NOT_RUN이며 제출 receipt는 이후 별도로 남긴다. 과거 allocation 1113 GPU-sec는 신규 B1 비용과 구분한다.
