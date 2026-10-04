# Held 등록 검사 최소 수리

실행 source `2ff0ecc63dda2c830fbd68e6a2d76f3317a446f7`로 MD58442/CD58443/collector58444를 모두 held 등록했다.
Release 직전 새 submitter의 NO_ARM_SERIALIZATION 검사에서 `NoneType` TypeError가 발생했다.
공통 dependency helper는 parent가 없으면 `None`을 반환한다. 새 문자열 membership 검사가 이 표현을 정규화하지 않았다.
모든 job이 PENDING(JobHeldUser), release0/GPU실행0인 것을 실제 queue에서 확인했다.

원 archive/lock/config/launcher/source와 최초 failure receipt는 그대로 보존한다.
새 `verify_parallel_dependencies`는 원 `expected_dependencies`로 None/empty/외부 barrier를 정규화하고
MD↔CD 과학 dependency 금지 및 collector afterany 양 parent를 검사한다. 해당 CPU 회귀를 추가한다.
신규 `release_held.py`는 명시된 원 attempt의 기존 세 held job만 fresh owner/source/argv/resources/dep/bytes 검산하여 release한다.
이 복구 경로에는 sbatch, cancel, retry, 모델 실행 또는 자동 반복이 없다.
실행 source/수학/precision/tolerance/계수/fit 예산을 변경하지 않는다. 등록 수리 source와 실제 GPU source는 별도로 기록한다.
이 failure는 GPU/품질 실패가 아니며, CPU fixture가 모든 등록 오류를 보장했다는 주장도 하지 않는다.
