# 단일 B1 D1/D2 제출 인계

실험 완료 보고가 아니다. GPU **57780**과 CPU 수집기 **57781**을 held 검사 후
정상 release했다. GPU는 의존성 없이 실행하며 수집기는 `afterany:57780`이다.
실제 모델 로드와 B1 entry 기록까지 확인했고 qualification·D1/D2 결과는 아직 미회수다.

실행 source는 `9cf106713e587fa08ddcfed89628b2770a55b74b`, tree는
`128b0ac1070d843c3d8e26481ae5ca4d32a8e3d6`다. 실행 lock SHA256은
`bd93594b34db719098f811294a7a8e04966356bd9981525b2b5df1145d57ceb5`다.
제출 controller만 `007bb011e3b9e9d99ed3ae652ebd360feea42856`으로 구분한다.
Slurm의 pending CPU topology 표기 `8-14`를 요청 CPU8과 구별하도록 inspector를
수리한 것으로, 기존 held 57780을 그대로 사용했고 runner/과학식은 변경하지 않았다.

정본 5파일과 동결 31 source 및 공용 prompts를 결속했다. 원 optimizer에는
관측 callback만 추가했고 CPU 28개 검사가 통과했다. 제출기 수리 후 workflow
7개도 통과했다. 이 CPU 결과는 실제 GPU 수치 qualification PASS가 아니다.
독립 reviewer agent는 사용하지 않았으며 별도 구현한 CPU reducer가 결과를 검산한다.

범위는 cold W0/H0, first100, v10 Tprime A 한 fit의 25후보/24갱신이다.
c9/13/17/21/25 weight는 RAM에만 보존하고 D1 W0+5후보를 R100/P200/N1000으로
평가한다. D2 네 mask 공통 식별 대상은 사전 고정874/1000이다. 126개 D2 제외가
D1 N1000 분모를 바꾸지 않는다. c25만 공식 terminal이며 추가 fit/B2/armB는 없다.

S1 실제 runtime은 A6000/Torch2.9.1+cu128/Transformers4.57.1이다. 원 S3 H200/
4.44.2와의 교차 플랫폼 동등성은 주장하지 않는다. 이번 task 외 job/source는
변경하지 않았고 기존 STOP도 유지한다. 신규 checkpoint 저장0, exact resume 불가다.

Local attempt는
`/mnt/raid5/janghj/ODE-edit/local/jlz-v10-a-b1-diagnostics/20261003-v1/attempt-r1`이다.
`output/`에 실행 증거, `collection/`에 CPU 검산·report/CSV/그림/manifest가 생성된다.
6h wall은 예약상한이며 actual runtime/비용이 아니다. 실제 할당비용은 collector의
exact parent accounting과 program timer를 구분해 보고한다.

새 반복 monitor·자동 retry는 만들지 않았다. sealed 단일 runner와 수집기가
등록 범위를 진행한다. 결과 미측정을 0이나 PASS로 채우지 않으며 과학 판정은 GH 소유다.
같은 host에서 GH가 raw를 읽을 수 있어 `NO_BROADCAST_NOT_REQUIRED`로 기록한다.
Git에는 소스·검사·compact 기록만 포함하며 raw/prompt/weight는 local에 남긴다.
