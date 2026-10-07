# GPT-J OURS Alpha CAP075 — 60619 실패 수리·우선 재제출

사용자의 “60619 FAIL 되었으니 RUN 다시 올려”, “저거 OURS니깐 우선순위 높여서 올려”를 수행했다. 해당 arm **하나만** 새 cold first2000 BS100×20으로 등록·held 검사·release했다.

| 역할 | 실제 job | dependency | bounded 초기 상태 |
|---|---:|---|---|
| GPT-J OURS Alpha CAP075 | **61003** | afterany:60618 | PENDING / Dependency |
| 독립 CPU collector | **61004** | afterany:61003 | PENDING / Dependency |

이는 제출·release 사실이며 실제 새 B1/gradient/W20 완료 또는 W&B 원격 검증을 뜻하지 않는다. 새 GPU 검증과 W20은 **NOT_OBSERVED**다. agent recurring monitor/heartbeat/autoretry는 없고 봉인 runner/collector가 자연 진행한다.

## 실패 증거와 수리 범위

60619는 11 commits/1100 edits 후 B12 첫 proposal에서 `POSTCAST_PRICED_FEASIBILITY`로 FAILED(1:0)했다. L3 owner29의 cap은 35.888365745544434, 저장 FP32 norm은 35.8883670137107로 초과량 **1.2681662653e-6**이 기존 절대 허용오차 1e-6을 넘었다. shared 최대 초과량 2.65023129e-8은 기존 허용오차 이내다. FP64 projection/KKT를 통과한 뒤 FP32 nearest 반올림에서 생긴 local 경계 초과다. B12 rollback verified=true/logical_commit=false이며 이전 1100 prefix와 실패 raw/source를 보존했다.

이번 profile에만 명시적 `cap_endpoint_toward_zero_v1`을 적용한다. 이상적 FP64 길이가 정확 CAP인 양수 block의 **첫 FP32 저장**에서, 성분이 이상적 절댓값보다 커지는 nearest 값만 인접 toward-zero 값으로 선택한다. ideal Euclidean projection/τ/KKT는 그대로이고 interior는 원 nearest다. 규칙은 feasibility 결과를 보기 전에 고정하며 norm 기반 scalar rescue/shrink, 반복 projection, pruning, moment reset, cap/beta 및 tolerance 확대는 없다. positive interior/zero/reentry를 유지한다.

이는 새로운 수치 실현 규칙이다. 이후 trajectory가 달라질 수 있으며 원 시도와 bitwise 동일하거나 다른 arm들과 동일 구현이라고 주장하지 않는다. 다른 arm·원 frozen archive를 수리본으로 hotpatch/재실행하지 않았다.

기존 B12의 600개 cap scalar를 사용한 CPU cast component 회귀, default/interior/zero 비변이, AST/import/CLI와 단일-arm 구조 테스트 4개가 통과했다. 이후 release 영수증 비변이 회귀 1개도 통과했다. 별도 toy/model load/forward/backward/pilot/fit은 0이다. 원 R payload는 noCP로 없으므로 **원 tensor replay와 실제 GPT-J GPU parity는 검증하지 못했다**. 검토 수준은 owner 및 collaborating worker의 한정 source/DAG 검토이며 외부 독립 GPU 인증이 아니다.

## OURS 우선·cap2 scheduling

실행 중 **60618·60620은 변경 0**이다. 60620이 준비 도중 RUNNING으로 바뀐 것을 확인하고 그대로 보존했다. 새 61003은 60618 종료 뒤 그 lane을 사용한다. 아직 시작하지 않은 정확 PENDING 60621·60106·60917만 일시 hold 후 원 dependency에 61003을 추가하고 모두 release했다. 취소·기존 ID/source/config 변경은 없다.

- 60621: afterany:60620:61003
- 60106: afterany:60618:61003, 60107은 원 afterany:60106 유지
- baseline 60917: 모든 기존 OURS frontier와 61003 후 시작; 나머지 baseline의 기존 두 lane은 유지

실제 owner/node/source 결속과 전체 server4 GPU DAG의 최대 antichain 폭 **2**를 검산했다. 새 retry 1GPU와 현재 할당/후속까지 합산 project cap2다. OURS 후 baseline이며 과학 성능 성공을 선행 조건으로 삼지 않는다. GPU가 비기를 agent가 기다리지 않는다.

각 GPU1/CPU8/59392MiB, hard60416MiB, 48h 요청상한, exportNONE/Requeue0; collector GPU0/CPU8/24576MiB/4h다. 현재 node/QoS·메모리 정책을 검산했고 무관 파일 삭제/환경 변경은 없다. 기존 6-arm 전체 기준의 보수적 storage reserve 36,590,583,808B를 줄이지 않고 준비 시 free95,475,646,464B에 결속했다. 요청 wall은 ETA가 아니다.

## Source·입력·W&B 및 비용

실제 봉인 실행 source: **`e16a1014dee40d9fd44df9d549dc74a051792610`**. Config SHA `bab92540da015aa7e286f3887af3848a95427c15e5a0731373cbf0c73904ae46`, lock SHA `cff0b4b48c5015d89874e27e9c6ba4a99ed89e5a8727c944f53a68b69283f914`. 이후 publication/control 수정과 실행 archive를 구분한다. EasyEdit GPT-J **L3–L8/lr .5/Alpha λ10**, PRICE 목적·native KL/norm/25eval24update·평가 분모는 그대로다.

NoCP이므로 1100 prefix부터 resume하지 않는다. 새 W0/H0로 시작하며 exact identity-qualified 기존 native input 및 W0 first2000 raw **R2000/P4000/N20000**만 재사용한다. context 재생성·W0 새 forward·추가 fit은 0이다. 예상은 20 commits/19 own-state joins/120 H appends이며 actual 아직 미관측이다.

기존 W&B method scalar/schema·current100/all-seen 실측·job ID/name·immutable run identity·privacy를 유지한다. parent는 실패 run `e4a3d16d7145461d`, 새 run ID/URL/online remote readback은 실제 startup에서 생성·확인하므로 현재 NOT_OBSERVED다. SDK 접수를 remote PASS라고 하지 않는다.

실패 60619의 parent allocated GPU 비용은 **12,229 GPU-sec (3.39694 GPUh)**다. 다른 arm/B1/새 run 비용과 합치거나 새 ETA로 쓰지 않는다. 새 pending 시간은 GPU allocation 비용이 아니다.

## 등록 영수증 예외와 한계

첫 freeze는 최종 collector 코드와 오래된 CPU source 참조 SHA 차이를 발견해 **archive/job 생성 전에** 차단됐다. 이전 준비 bytes를 보존하고 새 source-bound config/static receipt로 결속했다.

두 scheduler release는 성공했지만 immutable held `submission.json`을 상태 갱신하려던 control 코드가 이후 `IMMUTABLE_RECEIPT_CONFLICT`를 냈다. 실제 새 job을 다시 제출/release하지 않았다. held receipt를 KEEP하고 별도 local `release-reconciled.json` 및 아래 compact RELEASED receipt에 scheduler release 증거와 bounded snapshot을 기록했다. 후속 source에서 별도 `release.json`을 쓰도록 수정·mock CPU 검산했으며 **이미 봉인된 실행 archive는 변경하지 않았다**.

[제출 receipt](../../../../../runs/jlz-price-gptj-2k/cast-repair-20261008/submission.json), [source/RCA 검산](../../../../../audits/servers/server4/jlz-price-gptj-2k/cast-repair-20261008/source-review.json), [실제 resource/우선순위 DAG](../../../../../audits/servers/server4/jlz-price-gptj-2k/cast-repair-20261008/resource-priority-proof.json), [수치 realization 명세](../../../../../plans/updates/server4/jlz-price-gptj-2k/cap075-cast-repair-20261008.json).

Raw/model/prompt/tensor/fullstdout와 기존 실패 원자료는 ignored local KEEP, source/compact facts만 Git이다. NO_BROADCAST_NOT_REQUIRED: 현물 입력 재사용으로 추가 대형 전송이 필요 없다. 새 자동 retry·다른 실패 arm 재제출·Qwen 재개는 없다.

Generic access helper는 승인된 task-specific `runs/jlz-price-gptj-2k/**` prefix를 지원하지 않아 NOT_PASS(exit7)다. 이전 동일 task 예외와 현재 사용자/parent own-scope publication 권한을 근거로 compact exact receipt만 게시하며 shared helper 수정이나 허위 PASS는 없다. [게시 검산·정확 예외](../../../../../audits/servers/server4/jlz-price-gptj-2k/cast-repair-20261008/publication-checks.json).
