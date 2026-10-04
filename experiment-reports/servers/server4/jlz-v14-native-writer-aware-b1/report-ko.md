# V14 B1 등록 사실 보고

상태: SUBMITTED_RELEASED_RESOURCE_PENDING. 실제 GPU 검증·B1 결과는 NOT_OBSERVED/NOT_MEASURED.

- GPU: 58391 (`odeedit_jlz_v14_s4_B1`), PENDING(Resources).
- CPU collector: 58392, PENDING(Dependency), afterany:58391.
- GPU 제출 의존성: afterany:58381. V13 GPU의 terminal로 충족됐으며 V13 source/job/result는 변경하지 않았다.
- 실행 source: `2ab04d0b4d339573ee54fd85c7240d090a01e2b1`.
- lock SHA256: `4891b4e21686628e71463d3c07b1c4697e2cc1270f9f44724f866a8e50d92874`.
- GPU1 / CPU8 / 59392MiB / 24h; collector GPU0 / CPU8 / 24576MiB / 4h.
  export NONE, Requeue0. requested wall은 ETA가 아니다.
- 모든 job의 owner/name/fullargv/script/source/lock/resources/dependency를 held 상태에서
  검증한 뒤 release했다. 중복 제출·기존 job 변경 없음.

승인 nonce: ODEEDIT-USER-GH-SH4-JLZ-V14-NATIVE-WRITER-B1-20261005-R1.
정본: plans/global/2026-10-05-jlz-v14-native-writer-aware/.

cold W0/H0 first100, L4–L8 V14_RD fit 1회·commit 1회, W0/B1 R100/P200/N1000 평가만
실행한다. B2, 추가 full-B fit, 별도 BS2 순차 pilot, baseline, sweep, checkpoint는 없다.

정본 11파일 SHA/정독, native token/lookup 700행, observer identity 1300행 검산 완료.
CPU reference 8개 및 tiny-model 생산 회귀 8개 PASS. 실제 pretrained Llama GPU PASS로
간주하지 않는다. owner audit이며 독립 reviewer는 사용하지 않았다.

구현은 requested R=a*u, context별 actual-v native loss, whole-B causal solve VJP,
공통 stop, EfficiencyAdam, requested norm analytic gradient 1회, 마지막 evaluated
materialized weight exact commit과 final native H 누적을 분리한다.

local 자료: /data/janghj/ODE-edit/local/jlz-v14-native-writer-aware-b1/20261005-v1/.
sealed 실행: 위 경로의 attempt-r1/source 및 config.json/execution.lock.json.
CPU 최종 보고 예정: attempt-r1/cpu-report/report-ko.md (아직 결과 없음).
NO_BROADCAST_NOT_REQUIRED: source/compact receipt만 공유하고 raw/tensor/prompt는 local 보존.
V13 결과 비교는 PENDING_COMPARISON이며 기존 paused scientific monitoring을 재개하지 않는다.

등록 후 한정 initial resource snapshot에서 agent monitoring을 종료했다.
monitoring_active=false / automatic_resume=false. 봉인 B1/collector는 자동 retry 없이 자연 진행한다.
