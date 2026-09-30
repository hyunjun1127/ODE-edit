# 최신 job mapping — CPU 보고 보완

현재 등록·release된 plan: **GPU P56007/A56008/B56009/C56010/D56011 + CPU56019**.
GPU source `a8126eb65cbe9a1814b381007798d857003555ed`, CPU analysis source `6c9e16522cb2534da2ce4d7c600a98edac01fd34`.
CPU56019는 `afterany:56007:56008:56009:56010:56011`, GPU0/8CPU/32GiB/4h다.
기존CPU56012의 정확owner/source/fullargv/PENDING/미배정 상태를 확인한 후 그 job만 취소했다.
새CPU는 별도source를 소비하며 held owner/argv/resource/dependency/source 검사 PASS 뒤 release했다.
GPU job·source·raw 변경0, 원CPU submission/cancel receipt 보존. 새GPU나 과학arm을 추가하지 않았다.
대응 receipt: `audits/servers/server3/memit-hj-20260930-v2/CPU-collector-replacement.json`.
Actual T0a PASS; 초기 science gate/terminal NOT_OBSERVED. W0 전체10k 관측이 진행 중이다.

---
아래는 최초 제출 및 T0 시점의 역사다. 최초CPU56012는 위의56019로 대체되었다.

# MEMIT HJ v2 전체 DAG 제출 인계

Source: `a8126eb65cbe9a1814b381007798d857003555ed`.
Lock: `1aa28cb09de1cae5c00824bdafb86eb2a7f83dda24422cf5cf11906a055d43c5`.
Local lock: `/data/janghj/ODE-edit/local/memit-hj/20260930-v2/attempt-v1/execution-full-sha.lock.json`.

첫 관측: 2026-09-30T22:41:01.401932+09:00. P56007은 RUNNING, 나머지는 Dependency PENDING.
**전량 REGISTERED / held 검사 PASS / RELEASED. Actual initial gate와 terminal은 NOT_OBSERVED.**

|job|group/내용|dependency|자원|
|---|---|---|---|
|56007|P: T0a+W0 전체10k|없음|1GPU/8CPU/119GiB/720h|
|56008|A: E0/T1+000/001+T0b+20진단|afterok:56007|1GPU/8CPU/119GiB/720h|
|56009|B: 100/101|afterok:56007,afterany:56008|1GPU/8CPU/119GiB/720h|
|56010|C: 010/011 조건부Z|afterok:56007,afterany:56009|1GPU/8CPU/119GiB/720h|
|56011|D: 110/111 조건부Z|afterok:56007,afterany:56010|1GPU/8CPU/119GiB/720h|
|56012|CPU: 최종 독립 reducer/collector|afterany:56007:56008:56009:56010:56011|GPU0/8CPU/32GiB/4h|

28 logical cell→group 전량 mapping과 fullargv는 `audits/servers/server3/memit-hj-20260930-v2/submission.json`에 있다.
실제 admission에서 기존 본인 project active/admitted job은0이었다. 각 GPU group의 직렬 의존성이 동시1을 보장한다.
Manual hold는 모두 해제했다. P actual T0/readiness가 실패하면 afterok 소비자는 실행하지 않으며 CPU afterany가 실패를 수집한다.
교정 실패는 C/D의 네 Z cell만 차단한다. A/B의 native 경로를 교정 성능으로 선별하지 않는다.

제출 직전 free 671989698560 B, 필요 저장 계획 193273528320 B. 독점 예약은 아니다.
기존 model/context/C0/source와 신규 task source 총364파일 full SHA 검산 완료. GPU job에서는 큰 입력의 동일 dev/inode/mtime/ctime/size이면 이미 검산한 SHA를 재사용하며, source 및 변경 identity는 SHA 재검산한다.
처음 준비 lock의 source 축약8자리만40자리로 확장한 `execution-full-sha.lock.json`을 실제 제출에 사용했다. 기존 준비 lock은 보존했고 source/config/input bytes는 동일하다.
CPU23검사/정적CLI 점검과 실제 T0를 구분한다. 별도 independent reviewer는 없으며 owner audit이다.
등록 프로그램은 승인된 전체 계획을 자율 실행한다. 현재는 부모 지시의 대표 초기 gate까지 한정 관찰 중이며, actual PASS 이후 능동 monitoring을 중단한다.
총 종료 예상은 T0b 성분 실측 전 PENDING_CALIBRATION이다. 720h는 partition wall request이며 GPUh 예상/실측 또는 hard budget이 아니다.

## 실제 T0a 중간 증거

Job56007에서 BS100 원 native/adapter 비교200요청을 기술비용으로 수행했다. z/key/W/H/RNG/context/ledger 및 R/P/N 평가행이 모두 exact 일치했다. L4 key 불변도 확인했다.
임시CP 5,284,843,077B를 저장·SHA·실제재로드/state/output 검산한 후, refcount0/owner/path/SHA를 확인하고 지정된 단일 파일만 tombstone과 함께 삭제했다. 과거자료 삭제0.
초기 cached/native loss 차이0, gradient 상대오차1.0194003152719233e-5, gradient bitwise=false. Oracle precision은 NOT_ESTABLISHED_AT_ORIGIN이며 이후 T0b의32요청/96점 FP64 교정 이전 Z PASS로 읽지 않는다.
T0a 측정829.284초와 CP roundtrip20.797초는 본체 과학요청/중복 nested timer에서 분리한다. 최초science BS10 gate는 아직 NOT_OBSERVED이며 W0 전체10k observer가 이어진다.
GH actual제출 전달은 accepted=true로 확인했으며 응답/추가승인은 기다리지 않았다.
