# Qwen2.5 CF CAKE 2K: server1 → server4

사용자 지정 Qwen CF 한 개만 이관했다. 2026-10-11 07:21:34 KST 기준
server1 **63300은 CANCELLED**, server4 **63324 / official-s4-qwen25-cf-cake-2k는 PENDING**이다.
새 job을 held 제출·검사한 뒤 원 job을 취소하고 새 job을 release했다. 중복 실행은 없다.
다른 CAKE 5개 및 기존 실행 job은 변경하지 않았다.

server4에서 `afterany:63284`로 BLUE 뒤에 실행한다. 현재 PRICE 두 job과 OR/AND 자원 의존성,
MEMIT·AlphaEdit·W0·BLUE·새 CAKE를 포함한 시작/종료 순서 50개 상태를 CPU 열거하여 최대 GPU 2개를 검증했다.
자원은 GPU 1개, CPU 8개, RAM 58 GiB, 48시간, `lab_gpu_s4`, 자동 requeue 없음이다.

## 동일한 실험 설정

- 모델 revision `a09a35458c702b33eeacc393d103063234e8bc28`, FP32/eager, TF32 off, seed 0.
- 원 CF first-2K 순서, batch 100, Qwen 공용 context `5c01bc1a…`, projector SHA `d3a9687d…` 동일.
- CAKE hparams·causal scores·native 편집/평가/순차 history 계산은 원 63300과 동일하다.
- Qwen AlphaEdit 기반 layers 4–8, clamp 4, lr 0.5, 25 steps, L2 1, temperature 0.1을 유지한다.
- **500/1000/1500/2000 edits의 가중치·history·RNG checkpoint 네 개를 모두 보존**한다.
- CF FLU/CON은 W20 checkpoint 후속 평가로 유지한다. 현재 GPU 실행·W&B online·성능은 미관측이다.

원 source는 `a247c7925911990a0dc7230885c54710d540f8f7`, 이관 실행 source는
`40292095c0ab1020ee22ea777b3d593b5add3c05`이다. 공통 CAKE runner가 server4 Qwen CF 설정을 받아
실제 서버를 W&B에 기록하도록 수정하고 `official/runners/server4/cake_run.py` 진입점을 추가했다.
새 config SHA는 `4c36de11b2f6aba9bee5f2890093d9612e67cbeb86b349d5adb10a1f67885a6e`이다.
원 native 수치 함수·hparams 회귀 검사, history/RNG 및 milestone 복원 검사를 포함한 CPU 테스트 **9개 PASS**.
사전 GPU qualification은 사용자 방침에 따라 실행하지 않았다.

## 서버 자산·저장 공간

동일 모델 파일·projector·패키지 버전을 검증하고 서버 로컬 자산을 재사용했다.
모델·projector 대용량 복사는 없으며, 기존 SPHERE checkpoint도 삭제하지 않았다.
CAKE는 projector와 누적 H를 소비하며 C0 NPZ는 읽지 않는다. 원 C0 provenance는 parent manifest에 보존했다.
W&B는 공통 로더에 맞는 기존 server4 `local/jlz-price-cap-base-repair-2k/tracking.env`를 사용한다.
서버 루트의 다른 형식 `servers/local/wandb.env`를 수정하지 않았다.

출력은 `/data/janghj/ODE-edit/local/cake-qwen25-cf-server4-20261011/output`이다.
한 checkpoint 예상 7.98 GiB, 영구 milestone 4개와 추가 쓰기 1개 여유, raw 2 GiB,
다른 작업 증가분 16 GiB, 최종 여유 8 GiB를 합쳐 **65.90 GiB**를 요구했고, 준비 시 **70.14 GiB**를 확인했다.
server1의 전체 6-run용 64 GiB 여유 하한은 이 단일 run에서 8 GiB로 바꾸되,
원자적 checkpoint 저장 및 별도 milestone hard link 보존 방식은 유지했다.

[실제 제출·교체·자원 영수증](../../audits/servers/server4/cake-qwen25-cf-move-20261011/submission.json) ·
[CPU 자산·저장 검증](../../audits/servers/server4/cake-qwen25-cf-move-20261011/cpu-preparation.json).
