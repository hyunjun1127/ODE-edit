# T1 routing 최소수리 및 재제출 준비

최신 사용자 정정: “아니 점검에서 오류 사항있으면 수리 재제출해 / job 모니터링만 하지 말라는거였어”. 이전 점검-only 해석을 대체한다. 수리·CPU검증·정확 기존 job 교체·제출 확인까지만 수행하며 이후 진행/로그/결과 모니터링은 하지 않는다.

## 원인과 수정

원 실행6ef71ed2의 Backend.state는 ACTUAL recipe에 force_removal이 주어지면 존재하지 않는 construction_endpoint/removed_cohort_indices를 먼저 조회했다. T1 대각선 호출이 이 경로를 사용하며 CPU 최소반례 KeyError를 이전 보고에 보존했다. 이번 수정은 force_removal을 우선 해석하고, 나머지를 ACTUAL/COUNTERFACTUAL로 분기하는 순서 수정뿐이다.

Whole U 5개 weight·FP64 뺄셈·한 번 FP32변환·원 endpoint copy_ 복원·기존 tolerance·MB16·evaluator·패널·T0→T4·noCP는 변경하지 않는다. 이미 별도로 검증된 collector의 report/index 후 최종 COMPLETED 기록 수정은 새 source에도 포함한다.

## CPU 검증

기존16+추가6=22 tests PASS, 5.624초. 추가 검사는 실제 Backend.state 메서드를 CPU 작은5tensor로 호출한다: ACTUAL, T1 forced diagonal 3구간, 단일제거, 순서고정 이중제거, 예외시 원 endpoint 정확 복원, malformed recipe 거부. immutable 입력 hash도 검사한다. 모델 생성/CUDA 초기화/평가0, actual GPU 수치검증은 NOT_OBSERVED다.

기존53176/53177/53179는 교체를 위한 한정 확인에서 모두 PENDING, RunTime0, AllocTRES없음이었다. 반복 진행 모니터링이 아니라 중복실행 방지/정확 교체 대상 확인이다. 제출 직전 다시 exact owner/source/argv/dependency와 미시작 상태를 확인하여 해당3개만 교체한다. 다른 task는 변경하지 않는다. 원본 source/raw/receipt/24CP를 삭제하지 않는다.

공통 T0 및 원24CP는 기존 fullSHA+현재 stat로 재사용한다. 새 대형 재해시/전송/복제0. 새 immutable attempt-r2-routing/source/archive/lock과 원 attempt 연결을 남긴다. GPU각1/CPU8/60416MiB, task/projectcap2; CPU collector afterany는 새2GPU ID에 연결한다. 자율 등록된 프로그램만 자연 진행하며 agent monitoring_active=false/automatic_resume=false를 유지한다.

Owner 직접 검토·CPU 회귀검사, 독립 red agent 미사용. NO_BROADCAST_NOT_REQUIRED. 실제 job mapping은 제출 receipt로 별도 기록한다. 본 사전기록은 재제출/실험완료/GPU PASS 증거가 아니다.
