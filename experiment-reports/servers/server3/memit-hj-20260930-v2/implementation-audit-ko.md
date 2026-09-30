# MEMIT HJ v2 전체 구현 및 제출 전 점검

상태: **IMPLEMENTED / CPU_CHECKED / NOT_SUBMITTED**. 이 문서는 실제 GPU 성공이나 실험 완료 보고가 아니다.

## 권한과 source

부모 nonce `ODEEDIT-GH-SH3-MEMIT-HJ-V2-20260930-R1`, 추가 nonce
`ODEEDIT-GH-SH3-MEMIT-HJ-ALL-IMPL-AUDIT-DAG-20260930-R1`을 수신했다.
정본9개 및 부모·추가 envelope를 FULL_READ하고 exact SHA/size를 검산했다.
정본은 수정하지 않았다. `full-read.json`, `additional-authority.json`, `coverage.csv`가 읽기 및 구현 대응 증거다.

실제 MEMIT-H 54007의 execution `3a904be9261d162239c6b780a62c52f3e59a9142`와
BLUE `311b076a92e4ed0f14f5c8b4909732da781bc5f7`의 source/config/context/C0/evaluator를 재사용한다.
Pinned `apply_memit_seq_to_model`의 code object를 복제 globals에 연결하고, pinned execute의
residual 배분·phase 관측만 task-local AST adapter로 확장했다. native z/solve/FP32 materialization/
post-all-layer history append 및 cache_c 반환은 원 경로다. joint/SPG/refresh 변경은 새 namespace에만 있다.
기존 공유 source/env/model/과거 raw는 변경하지 않았다. 새 자산 다운로드·C4 복구·타 task 재개는 없다.

## 28 cell과 physical DAG

|GPU group|과학 cell 및 기술 단계|Slurm 의존성|
|---|---|---|
|P|actual T0a BS100 native/adapter, RAM/임시CP roundtrip, W0 전체10k observer|기존 project 점유가 있으면 afterany|
|A|W0 writer4(E0/T1), 000/001, native 첫1k T0b, writer anchor 나머지8/history8: 총22cell|afterok P|
|B|100/101: 2cell|afterok P + afterany A|
|C|010/011: 2cell|afterok P + afterany B; 정확 calibration PASS 필수|
|D|110/111: 2cell|afterok P + afterany C; 정확 calibration PASS 필수|
|CPU|5 GPU group 모두 afterany로 수집·검산·표/그림/보고/manifest|GPU0|

각 GPU job은1GPU/8CPU/119GiB이고 최대 동시1이다. 모든 RAM-only 분기와 진단은 해당 persistent
process 안에서 이어진다. 8논리main48k, main physical32–44k/320–440write, 진단20k/2000write를
구분한다. 총52–64k/2320–2440write는 전체 Z 구현이 허용될 때의 계획 범위이며 실측값이 아니다.
NOT_FIRED는 부모 alias이고 독립 실행으로 계수하지 않는다. 교정 BLOCKED는 네 Z cell만 막는다.
각 group이 source/config/input-bound readiness를 검사하며, 순서용 afterany를 과학 PASS로 쓰지 않는다.

## 구현 및 CPU 점검

Owner head-server3가 정적 검토·production 함수 fixture·dry DAG·오류 주입을 수행했다.
**독립 reviewer는 수행하지 않았다.** CPU 회귀23개, Python AST/5 CLI/2 launcher syntax 검사가 통과했다.

- 비가환·rank-deficient PSD Gram, 직접 solve fallback, L8 직접 residual, FP32 actual displacement.
- 실제 pinned wrapper/execute를 호출한 작은 CPU fixture로 prior-H와 post5층 key append/B1→B2 전달,
  energy-matched 두 shadow의 복원 및 최종1회 append를 검산했다.
- SPG zero policy, 총 call budget/최종 재평가, Armijo 강제 수락 금지, 6수락점 plateau,
  교정 미도달/실패의 포함, cap400 초과 시 clip하지 않는 차단을 검사했다.
- 실제 LlamaRMSNorm의 FP64 forward/backward 및 예외 시 dtype/함수 복원을 검사했다.
  실제 모델 suffix의96점 FP64 교정은 미실행이다.
- native key의 context-group 평균과 정본 dispersion의 equal-context RMS를 각각 보존했다.
  모든 occurrence 포함 rebuild, reference reset, full Gram/probe 일치를 작은 fixture로 확인했다.
- CP atomic 실패, TorchVersion scalar, 독립 RAM/RNG/context clone, source/파일 오염 거부,
  refcount/rolling/정확 path 삭제와 5k recovery routing을 검사했다.
- 28 cell의 trigger 없음/최초trigger/Z차단/재개에서 physical 요청·write 수와 중복prefix 부재를 확인했다.
- NLL preference tie/NS true 방향, TF micro/macro/strict, NLL 평균과 raw prompt identity 검산,
  W5 first500/at-write/birth/active/superseded/paired 전이는 CPU reducer에 연결했다.
- production submit의 6개 held inspection, dependency 종류, 역순 release, create-once 중복 거부를
  모의 Slurm fixture로 검사했다. 실제 Slurm 등록과 구분한다.
- collector는 raw 검산·보고·manifest 생성 뒤 terminal을 게시한다. 누락/기술 실패는
  TECHNICAL_INCOMPLETE이며 성능 음성이나 Z차단과 구분한다.

점검 중 수정한 사항: stale SPG 반환점/호출 예산, 실제 FP64 norm/softmax scratch,
원 native KL teacher 고정, equal-context dispersion, history shadow 복원,
복원 시 immutable T0 증거 연결, NLL/identity 독립검산, Z차단의 모델-load 전 처리.
과학 threshold/표본/arm/hparams를 성능에 따라 바꾸지 않았다.
초기 toy fixture의 CUDA 이동 mock 누락으로 작은 행렬 연산이 장치에 도달한 적이 있다.
모델/과학 job은 없었고 해당 준비 비용은 미측정이다. 수정 후 최종 회귀는 CUDA를 숨겨 실행했다.

## 자원·checkpoint·비용

Host96GiB/GPU122GiB는 제출 전 추정이며 actual peak PASS가 아니다. 최대5개의 editable W/H CPU
clone24.61GiB, liveH/C0, FP64 factor/workspace, CP serialization/reload를 포함한다.
Local 저장180GiB 계획은 입력 복사 없이 long000/100/110/111 latest2와 pinned branch,
atomic temporary/raw/IO 여유를 포함한다. 정확 현재 free는 freeze/admission receipt로 갱신한다.
각 GPU wall720h는 partition 최대값에 따른 provisional request이며 GPUh hard budget이 아니다.
총 GPUh는 PENDING_CALIBRATION. T0b 이후 native/진단/SPG P95/관측/refresh 실측 성분으로
cost-lock forecast를 만들고, nested timer나 shared prefix 비용을 중복 더하지 않는다.

임시 CP는 이번 manifest에 기록한 long path만 허용한다. 전체 W/H/RNG/context/cache binding/
ledger/sample reference/trigger/cursor/source를 저장·SHA·실제 reload/output parity 후 기록한다.
각 path의 owner/hash/refcount를 확인하고 rolling 또는 완료·artifact검산·dependency 해제 후
정확 파일만 tombstone과 함께 삭제한다. 실패 CP 및 과거 자료는 보존한다.
단기/진단 CP 예외를 추가하지 않았다. 같은 frozen 계약의 복원 CLI는 새 output namespace에서
검증된 CP를 사용하며 scalar metadata로 edited state를 복원했다고 주장하지 않는다.

## 실제 증거와 종료 경계

현재 CPU/구현 점검만 완료했고 actual T0/교정/첫 science write는 NOT_OBSERVED다.
전체 여섯 Slurm job의 등록·held검사·release receipt를 별도 제출 보고에 연결한다.
대표 E0/T1 첫BS10 commit/observer→다음entry 또는 전량등록 뒤 실제 resource-pending을
확인하여 인계한 뒤 agent monitoring을 중단한다. 프로그램과 collector는 자연 진행하며
완료 상세 리뷰는 사용자 recall 때 수행한다. GH 추가승인을 기다리지 않는다.
