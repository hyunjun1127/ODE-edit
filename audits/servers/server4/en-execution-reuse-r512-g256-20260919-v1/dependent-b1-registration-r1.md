# B1 의존성 대기 등록 — 사용자 후속 요청

사용자: “B1 부분 pending으로 걸어놓을 수 있나?”

이 호출은 준비 job 50410 뒤에 B1 한 job을 등록하는 범위다. 이전 모니터링
중단은 유지하며, 등록·held inspection·release 확인 이후 agent polling을 하지 않는다.
Sequential/B2, 준비 job 변경, 다른 job 변경, 새 teacher/native 생성 권한은 추가하지 않는다.

- Slurm `afterok:50410`, task 동시 1 GPU / project cap 2. 두 matched schedule은 같은 B1 job 안에서 순차 실행한다.
- 준비 중인 source/lock/output은 불변이다. 새 admission lock은 정확 producer lock/submission/source와 예상 READY 경로를 봉인하며 아직 없는 READY SHA를 만들어내지 않는다.
- Job 시작 시 CPU bootstrap이 READY source/640문서/manifest/input identity를 한 번 확인한다. 새 `execution.ready.lock.json`은 원 admission lock identity와 실제 READY SHA를 결속한다. 원 lock을 덮지 않는다.
- 기존 full teacher 검증 및 B1 actual checks/geometry/guard/optimizer/observer는 유지한다. bootstrap에는 polling/sleep/submit 기능이 없다.
- 준비 실패 시 afterok가 시작을 막는다. READY 누락/실패 receipt/identity 불일치/실제 I/O·공간 부족은 모델 로드 전에 기술 실패다. 자동 재제출하지 않는다.
- 제출 전 보수적 full 준비 payload + B1 reserve를 확인한다(현재 retained payload를 빼지 않은 보수적 산정). 실제 시작 시 이미 저장된 teacher를 제외한 B1 incremental reserve를 다시 검사한다. 과거 storage waiver를 상속하지 않는다.
- 변화는 admission/dependency bootstrap과 그 CPU tests뿐이며 수치·method/runtime core는 변경하지 않았다. 신규 B1 execution은 기존 prep execution과 별도로 봉인한다.

과거 CPU fixture 편집 중 indentation error 1회는 제출 전 수정했다. 기존/신규 CPU 검사의 raw stdout은 local receipt에만 보존한다. CPU 검사를 actual Llama PASS로 표시하지 않는다.

CPU regression: **145 tests PASS / 40.744초**, shell syntax PASS.
Local receipt: `local/en-execution-reuse/20260919-v1/receipts/cpu-dependent-b1-r1.json`.
Source/model/teacher actual B1 검증은 미실행이다.

완료 후 상세 report에는 admission lock과 effective READY lock을 모두 연결하고,
분석 입력은 `execution.ready.lock.json`을 사용한다. 준비/본실험 제출과 실제
실험 완료·성능·parity PASS를 구분한다.
