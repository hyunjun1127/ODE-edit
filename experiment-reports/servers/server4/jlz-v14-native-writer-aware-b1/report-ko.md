# V14 B1 실행 준비

상태: CPU_AND_SOURCE_CHECKED_NOT_SUBMITTED. 실제 GPU 검증과 B1 결과는 아직 NOT_MEASURED.

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
NO_BROADCAST_NOT_REQUIRED: source/compact receipt만 공유하고 raw/tensor/prompt는 local 보존.
V13 결과 비교는 PENDING_COMPARISON이며 기존 paused scientific monitoring을 재개하지 않는다.
