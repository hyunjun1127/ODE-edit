# E0–E4 자율 수리 r2 — 제출 전 사실 기록

권한 nonce: `ODEEDIT-GH-SH4-ALPHA-KEY-AUTONOMOUS-RESUME-20260923-R1`.
정본 main `0498b22f7f1d67395131a7d2f9a6c8c1b2d1d32c` 및 원 E0–E4 사양을 유지한다.

## 재사용한 확정 원인

이전 gate 52527은 `WRITER_UNEXPECTED_BOS: 0`으로 실패했다. 원 native와 새 runtime 모두 동일한 pinned tokenizer와 설정을 사용했다. `PreTrainedTokenizerFast.add_bos_token=False` 속성은 backend의 BOS postprocessor를 바꾸지 않았으며, 실제 12개 native 시퀀스에는 BOS가 있었다. 새 검증 코드의 BOS 부재 가정이 잘못된 것이지 native 토큰화 변경이 아니다.

이전 실행 source `a95876f8e5c4cf59df9cd9d7f824d1ac99f8bc77`, lock `4a80070051cbbcc1b5e1d01f0e94124ddc92f79bedaba3772c42dcbbc7530725` 및 실패 자료는 보존한다. 별도 진단 branch commit `a4c583d61735c695d638d3fa37ceda299245bc10`의 CPU 증거를 재사용했다. 이전 parent allocation은 44 GPU-sec, native z100/SHAM/과학 family 완료는 0이다. 52528/52529는 미시작 CANCELLED, CPU reducer 52530은 COMPLETED였다. 추가 취소할 기존 작업은 없었다.

## 최소 수정과 검사

- native tokenizer 호출·BOS 처리 자체·원 native source·정밀도·임계값은 변경하지 않았다.
- 잘못된 BOS 부재 assertion을 원 native의 tokenizer/backend/context/실제 token ID/마스크/lookup exact 결속으로 교체했다. 결속 witness SHA는 `1b6273a0ffeb8cf9ed49933adc9a682a1826efc78d31507ee1cce74356341efd`다.
- `attempt-r2`의 source/archive/lock/log/output/reducer package를 r1과 분리했다. 이전 binding receipt도 덮어쓰지 않고 exact SHA로 재사용한다.
- CPU 142 tests PASS/0 failure/0 error/0 skip. 실제 tokenizer 12시퀀스 및 토큰·BOS·context·lookup 변조 거부를 포함한다. CPU synthetic model fixtures는 수행했지만 pretrained model load/실제 model forward/GPU 할당은 0이다. 실제 G1/G2 PASS가 아니다.
- reducer가 이전 generated-r1을 보존하고 새 lock/generated-r2를 쓰는 회귀검사와 경로 이탈 거부를 포함한다. Owner 검토이며 이번 수리를 별도 독립 agent가 감사했다고 주장하지 않는다.

## 실행 계획과 보존

사전 등록 graph는 gate → afterok geometry/writers(각1GPU), 세 parent의 afterany CPU reducer다. 동시 최대2GPU이며 기존 owner admission이 있으면 자동 취소하지 않는다. E0+94 contrast family와 400 native target 계획만 유지하고 SEQ/ORDER/FUTURE는 미제출이다. 실제 GPU gate는 runner 안에서 fail-closed다.

기존 input CP12와 small input23/전체 binding1819 검증은 재사용하며 12CP를 전량 다시 읽거나 전송하지 않았다. 런타임에서 기존 content 검증의 stat 결속을 다시 확인한다. 새 full W/M/optimizer resume CP는 만들지 않고 승인된 diagnostic K/R/Δ/target/bank만 저장한다.

자원은 각1GPU/8CPU/60416MiB/exportNONE/Requeue0, operational wall7일(과학시간 budget 아님). 기존 future storage 계획232,700,000,000B를 유지하며 새 freeze/admission에서 실제 free/inode를 기록한다. CPU 계획 host peak49GiB는 추정이며 actual GPU peak/timing은 미관측이다. 직전 한정 admission의 owner queue는 비었고 server4 GPU7/8 할당이었다. 실제 제출 때 다시 확인한다.

GPU 부족 pending으로 인계할 때는 전량 정상등록/검사/release 및 실제 부족 근거를 먼저 남기고 G0–G3를 미관측으로 기록한다. 실행 중이면 actual 초기 gate까지 확인한다. 이후 agent monitoring/자동 재개는 중지하며 branch만 게시한다. GH main 통합은 별도다. NO_BROADCAST_NOT_REQUIRED.

CPU receipt: [cpu-validation.json](../../../../../audits/servers/server4/alpha-key-causal-20260923-r1/autonomous-repair-r2/cpu-validation.json).

## CPU 봉인 실패 보존 및 attempt-r3

source `52dfb6a64c82f9f47199180c5cc342279ca871ed`의 첫 r2 freeze는 source member 목록이 attempt 경로 변수를 가린 `TypeError`로 lock 생성 전에 중단됐다. Slurm 제출0/GPU 비용0이며 생성된 archive/extracted source는 보존했다. 변수명을 분리하고 소형 archive를 이용한 freeze→lock→4phase script→중복 거부 전체 CPU 회귀검사를 추가한다. 다음 제출은 기존 r2를 덮지 않는 `attempt-r3`다. 이는 제어 경로 수정이며 native/과학 수치 변경이 아니다.
