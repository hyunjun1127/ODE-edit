# SH4 six-arm sequential 수신·CPU 준비

Instruction: ODEEDIT-S06-LOWCOST-SIXARM-SEQUENTIAL-B100X10-SH4-V1.
신규 envelope/amendment, 원문/설계/cells/schema-v2, 기존 실행 지시와 PROTOCOL을 정독했다. 실제 시작 main `1c379d9c3194460d6c9a1c4a3a5d513fadec5ba7` / tree `43330418e5e4204a7a0b76aae20d84809437f8b4`; shared checkout 변경0. 전용 branch `codex/server4-lowcost-sixarm-seq10-v1`.

- 공통 prepared `4ea9e5a7733725ab513571845691a28e4ed321884d749156e256fb992a55f933` 및 original W50/M50 `0ecc3a4b790ca7ab4d837cab785adc329f47c9b33c5b7092827c6ddac0571704`: 현재 full SHA 및 CPU weights_only/W4/M4/context/RNG/P mapping 대조. 새 M8 재구성0.
- 로컬 root `/data/janghj/ODE-edit/local/low-cost-write-donor-seq10/20260913-v1/attempt-v1/`. `prepared-CPU-receipt.json` SHA `7a1247a84af9b87d09224c74cc2a5b19a97399a21f786b4add1ff94ccb3671d4`; `sample-sequential.lock.json` SHA `1f57f8888bb13801f0748081acb61c1de448adb5bf04b6a9d02aef952a63d3bb`.
- B51 `[5000,5100)`부터 B60 `[5900,6000)`까지 동일 순서. 6 fresh chains / 60 batches / 6000 arm-request observations / unique suffix1000. 이전 static endpoint 이월0.
- array mapping `0=N4,1=RES8,2=S875,3=S75,4=FULL8,5=REFIT4`, `%2`, 1GPU/8CPU/60416M/exportNONE. 기존 프로젝트 admission이 존재할 때만 afterany 안전 연결, 기존 job 변경0.
- 매 batch fresh first fit, 실제 partial-state second fit, final selected history append1. 프로그램은 B51→B52 actual state 전달을 검사하고 초기 receipt를 남긴다. GPU 초기 gate는 아직 NOT_RUN.
- B55 suffix500 및 B60 fullseen6000 평가에서 Current/하위 패널은 동일 raw rows 재사용. 별도 rebatching의 수치 동등성 주장0. B51/55/60 selected checkpoint 18개와 증분/연대 journal 저장, GPU replay NOT_TESTED.
- W0/entry reference는 기존 static의 동일 패널 자료만 연결한다. 다른 current batch의 W0를 측정했다고 주장하지 않는다. Audit128/MMLU68 평가0.
- 예상치: 2–8 GPUh/chain, 전체12–48 GPUh, scheduler walltime24h, GPU-hour 사용자 상한null. 고정 평가·fullseen·I/O와 fit 비용을 구분하며 과거 M8 setup을 새 allocated 비용에 더하지 않는다. 출력80GiB 계획, 관측 당시 가용 약362GB(독점 예약 아님).
- CPU tests 25 PASS: sequential runtime6/evaluation5/submission2 + 재사용 fitting5/evaluation7. 독립 red 16 tests와 별도 [감사](../../../audits/servers/server4/2026-09-13-lowcost-sixarm-seq10/red-preflight.md). 실제 GPU gate 또는 전체 실험 완료 PASS로 부르지 않는다.

GH M0 peer-direct 응답 수신 완료. 제출·held검사·release 후 초기 gate 또는 all-PENDING 상태에서 `MONITORING_PAUSED_AWAITING_USER`; 이후 polling/자동 분석/main 통합0. NO_BROADCAST_NOT_REQUIRED, raw tensor/prompt/checkpoint/log Git0, 타 task 변경0.
