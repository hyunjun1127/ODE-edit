# Qwen FE current-entry-z 수리: 공용 source 인계

Nonce: `USER-FE-QWEN-WORKING-CURRENT-Z-20261012-R1`.
수신 turn: `01a12d3b-5ac2-7a83-b5c1-5b13c658e30a`.
최신 사용자 정정에 따라 실행 소유는 requester root/rent, 공용 구현 소유는 SH1이다.
SH2 신규 제출도 HOLD이며 공용 API 지원을 제출 지시로 해석하지 않는다.
SH1 신규 제출/취소/GPU forward/CP 삭제는 모두 0이다. README 본표는 수정하지 않았다.

## 구현

- 원고 `74e36c38`의 current batch-entry 모델에서 현재100건 L4 compute_z와
  later-layer firstforward를 모두 만든 뒤 native FE writer를 호출한다.
  W0 fallback은 거부한다. 원저자 writer/solve/C0/history 함수는 AST 불변이다.
- 저자 native `algs.L2`만 0/.1/1/10으로 변경한다. 0은 held-out 대조군이며
  본평가 승격은 .1/1/10 중 사전 gate와 Score/tie 규칙으로만 결정한다.
  명시적 수치 실패는 −∞, 자원·통신·스키마·미완료 오류는 선택 보류다.
- W0 factual, 매 current100 및 500단위 all-seen raw/query/지표를 저장한다.
  독립 CPU reducer가 token/NLL/정답 여부·분모·work·20commit·CP identity를
  검산한 W20만 선택 입력으로 사용한다. 실제 실패 run/과거 지표는 보존한다.
- native 별도 full export를 저장측 변경으로 제거했다. 평가/raw 저장 후
  host 공용 lock에서 latest 하나만 atomic overwrite한다. W0 CP는 없다.
- H는 기존 CPU tensor를 동기 serialization 동안 참조하고, hash는
  memoryview/chunk로 계산한다. full-H clone/tobytes/finite boolean 사본을
  피하며 기존 digest와 reader 형식은 유지한다. solver/dtype 수리는 없다.
- W&B task/group, baseline, arm, profile, config/source/stream/selection hash를
  held-out 선택용으로 분리한다. current/all-seen 공용 metric/축은 유지하며
  held-out 값을 본표 FE 최종값으로 승격하거나 이전 run을 덮어쓰지 않는다.

## CPU 검사와 제한

새 검사 + 기존 native + fake-SDK transport 총 **67 PASS**.
원 native tiny writer의 L2 소비, 관측 전후 weight/H/RNG 일치, BF16 digest
동일성, 작은 latest roundtrip, 실제 current100/all-seen500 scalar 경로,
raw 변조 거부, 수치 실패 veto, host reserve binding을 포함한다.
실제 pretrained/GPU/온라인 W&B/59GiB 내 실행 성공을 입증한 결과는 아니다.
역사적 MaxRSS 약46.98/48.79GiB는 수신된 자원 근거이며 새 실행 실측과 구별한다.

## 실행 owner API 및 다음 단계

`official/diagnostics/fe_qwen_current/README.md`의 bindings JSON으로
`prepare OUTPUT BINDINGS`, frozen source의 `launch CONFIG LOCK`, 단발 CPU
`collect OUTPUT`를 사용한다. Python/HF/assets/C0/stream/lock/reserve/CPU/server는
실제 rent 경로로 결속한다. 기존 실제 alt stream은 4,155,152 bytes,
SHA `7c8324f8de7dd2b128a65af82c4d562a687d80a24a26576e452d96239eb9b296`이다.
64GiB server1 reserve를 rent/server2에 강제하지 않는다. 최소32GiB는 유지한다.
후속 arm은 같은 task의 이미 저장된 latest 할당량을 lock 아래 공제하되
미저장 latest+tmp1+raw+reserve를 유지한다. 후속 시작/실제 부족 거부 CPU 검사를 통과했다.

rent 실제 준비·등록 및 네 held-out 결과는 아직 이 source 검사로 대체할 수 없다.
선택 통과 후에만 동일 tuned 설정의 원 CF/zsRE cold 본평가를 연결한다.
collector는 모델/GPU 호출이나 자동 sbatch/재시도를 하지 않는다.
GPU 성능 회복/본평가 완료는 **미관측**이다.
