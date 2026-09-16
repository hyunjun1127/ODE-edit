# Local-z adaptive allocation — 준비·등록·PENDING 인계

상태: **MONITORING_PAUSED_AWAITING_USER / PENDING_GATE_NOT_RUN**.
기술 준비와 7개 과학 프로그램을 등록했지만, 마지막 관측에서 모두 PENDING이다.
기술 READY·teacher 재현·실제 dynamic gate·70batch 완료를 관측한 것이 아니다.

## 범위와 정확한 source

- Instruction: `ODEEDIT-S06-LOCAL-Z-ADAPTIVE-ALLOCATION-SEQ1000-SH4-V1`. main 시작 `cee9447330e0eabe097775ca6185ea4c45cf267d`.
- 실제 실행 source `32a92ad6f3fff2f258d8778f3936d152e975ac1b`, tree `75a96b2e2122d2be6b096af12c22df8a4365a8e1`.
- Archive SHA `6378df3a237d5a6c489b87c90f6d99c0221b409857632852916f2f5bffa7f901`.
- Execution lock SHA `093bb13dd6b42b8f2f8b8478248b067add1d94533547afa8ca9af416e6f14629`.
- FULL_READ SHA `a8de455e9294f82acecd38b250c88c1c7662800d892a91a87e04057cc07986fa`; 정본과 최신 PROTOCOL, 원 native/fitter 연결 전체를 결속했다.
- cold W0/M4=M8=0, seed20260916, FP32/eager, matmul/cudnn TF32off.
- 동일 first1000 / B100×10, 신규7chain·70batch·7000 arm-request observations / unique1000.
  기존 warm N4/REFIT4/EP/CAKE 결과는 이번 paired 실행을 대체하지 않는다.

## 제출 및 cap1

| 프로그램 | Job | 범위 | 마지막 상태 |
|---|---|---|---|
| 공통 준비/기술검사 | 48679 | W0/context/teacher 재현/필수 native 연결 | PENDING |
| 과학 array | 48680_[0-6] %1 | 0 LD, 1 N4, 2 REFIT4, 3 L75, 4 T75, 5 L4D, 6 TD | PENDING |

과학 array는 준비 job afterok 및 exact lock에 결속된 READY를 모두 요구한다.
각1GPU/8CPU/60416MiB/exportNONE/Requeue0/12h, 최대 동시실행가능 용량은 **1GPU**다.
등록 직전 기존 project reservation은 없었다. held owner/source/node/resource/args/dependency를 검사한 뒤 release했다.
7개가 등록된 사실과 기술검증을 통과해 과학 실행이 시작된 사실은 구분한다.
마지막 관측: `2026-09-16T11:36:50.122707+00:00`. 그 뒤 scheduler·결과·로그를 재조회하지 않는다.

## 구현 및 검증 수준

원 NativeSingletonFitter와 compute_z는 수정하지 않았다. LOCAL은 local residual,
TERMINAL은 own-entry Z8을 고정하고 L4에서도 actual h8를 읽는 별도 adapter다.
원 K/repeat/direct-solve/add AST를 재사용하고 CPU FP32 endpoint gate 0/1/.5/.75를 적용한다.
기존 warm sequential runner나 EP quality/alpha policy를 실행하지 않는다.

Past64는 received-event 최신 fact ledger와 고정 SHA priority이며 현재 overwrite fact를 제외한다.
selector는 E/H·canonical rewrite strict ID·S64 D만 받는다. P/N/Dev는 observer다.
NaN/손상은 기술 실패이며 품질 부적격으로 숨기지 않는다. 최종 history는 eligible layer마다1회이고,
LD/TD의 L8 zero-write/commonN4에도 M8를1회 갱신한다.

12개 CPU 테스트(7arm 실제 proposal routing mock 포함), syntax/import, frozen 2136-member
identity/stat 연결, native adapter AST, memory-policy audit, staged access/diff 검사를 통과했다.
현재 실제 Llama/model-level 검증은 **미실행 관측 상태**다. 준비 프로그램의 동일상태 E/H 5e-5,
D 5e-7, strict 동일 기준을 완화하지 않았다. generation order, candidate score order, native 연결,
history/save-restore 검사를 준비에 포함한다. 기술 비교는 saved targets를 원 native writer에
명시 재생하는 별도 기술비용이며 baseline sequential chain이 아니다. FD/ULP/KKT는 추가하지 않았다.
별도 red agent는 사용하지 않았으며 parent 자체 검토 수준이다.

## 자산·저장·비용

기존 model/P/fixed10k/reference768/teacher192를 read-only identity로 연결한다.
teacher의 실제 W0 TF32off 재현을 공통 준비에서 확인하고, 5e-7 규약 불충족 때만 동일192 tokens로
한 번 새 teacher를 생성한다. 이전 teacher 완료를 PENDING으로 되돌리지 않는다.
모든 arm은 준비가 봉인한 동일 context text/token/RNG를 받으며 arm 간 mutable state는 공유하지 않는다.

source freeze 때 free `431185588224` bytes, reserve `227633266688` bytes.
B1/B5/B10 ×7 =21 selected W4/W8/M4/M8/context/RNG/received-ledger CP,
매batch native-fit/target·후보 점수·selected delta·commit/link·평가를 보존한다.
CAKE no-checkpoint 지시는 적용하지 않는다. FP32 delta만으로 exact replay를 주장하지 않는다.
GPU off/on continuation은 별도 미검증이다. 과거 파일 삭제/재전송은 없다.

과학 계획 12000 target calls / 최대288000 Adam /300000 loss /150 solve /190 candidate는 **산술 예상**이다.
기술 계획700 fresh target calls +400 explicit target replay calls/13 solves는 과학 비용과 별도다.
12h wall은 reserve이며 실측/과학 gate/GPU-hour hardcap이 아니다. 신규 실측 allocation은 PENDING 시점 미확인이다.
Native target/loss/Adam 반환 counters 및 final anchor/delta/radius를 보존한다.
native iteration별 clamp-hit counter는 NOT_RECORDED; nested fit/total timer는 합산하지 않는다.

## 대기 인계

Local resume: `/data/janghj/ODE-edit/local/local-z-adaptive-allocation/20260916-v1/resume-manifest.json`

SHA `541012971022dd245df9aa64d085379d931116e7eb4dad2eba085a31e0920ebe`.
`monitoring_active=false`, `automatic_resume=false`, `resume_trigger=explicit_user_call`.
등록된 프로그램은 자연 진행하며 추가 submit/callback/heartbeat/완료대기/최종분석을 예약하지 않는다.
기술 실패 시 READY가 만들어지지 않아 과학 경로는 fail-closed로 남는다. 사용자 recall 이후 사실을 확인한다.
다른 paused task는 변경하지 않았다. NO_BROADCAST_NOT_REQUIRED.
