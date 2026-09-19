# 51056 기록 오류 수리 / B1-only 사전검산

## 실제 오류와 수정

51056은 FAILED 1:0, parent allocation 335 GPU초다. basis._append의
residual_norm > absolute_cutoff가 NumPy bool을 반환하여
basis.components[0].added에서 JSON 직렬화가 실패했다. NumPy 2.2.6의
실제 build_functional_basis 결과로 CPU 재현했다. 부분 JSON은 불변 보존한다.
비교 결과만 Python bool로 변환하며 계산식·threshold·선택 결과는 바꾸지 않는다.

앞선 50974/51055/51056 allocation은 각각 136/104/335초, 합575초다.
50983은 이미 취소·미할당이며 이번에 다른 job을 변경하지 않는다.
51055의 유효한 fixed4 hook component만 기존 exact-source 검증으로 재사용한다.
51056의 native replay/geometry/repeat 부분 검사는 전체 T0 PASS가 아니다.
후속 FD 및 B1은 미실행이다.

## 새 실행 범위

- maximum_batch=1, sequential_authorized=false, auto_continue=false.
- B1_N4/EN_KL_Q/DEC_LINE/DEC_MODES_CUM: 동일 B100 native 공유, 원 방법 불변.
- B1 gate는 보고하되 PASS/FAIL 모두 B1 후 종료한다.
- science 진입점에서도 B2 또는 S3/S10을 차단한다.
- W0 observer는 이번 B1의 같은 first100만 관측한다. 다른 요청을 실험하지 않는다.
- noCP 유지. 메모리 transaction은 유지하며 기존 artifact 변경 없음.
- project cap2, 새 job1GPU/8CPU/60416MiB, admission 시 다른 점유 합산.
- 기존 24GiB 초기 추정치를 실제 admission에서 확인하며 storage waiver 없음.

## CPU 검사

220 tests PASS, 9.266초. 신규7 tests는 실제 basis 및 covariance/solver
receipt를 원 strict JSON writer로 저장·재읽기하고, 기존 NumPy bool 오류 재현,
중복쓰기 거부, B1 gate PASS/FAIL 모두 B2 차단, T0 실패의 B1 차단을 검사했다.
기존 213 checks도 포함한다. 이는 GPU 수치검증 또는 B1 완료 증거가 아니다.
독립 agent red는 미실행이며 SH 자체 source/CPU 검산이다.
기존 hook trajectory gradient 경고 정책 외 수치 gate 완화 없음.
