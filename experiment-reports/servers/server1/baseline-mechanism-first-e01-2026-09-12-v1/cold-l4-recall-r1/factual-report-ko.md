# E0/E1 cold L4 B100 사실 보고 — 45719 부분 완료

이 보고는 cold L4의 첫 B100 한 개와 기존 관측 대조만 포함한다. E0 원 checkpoint 동등성, warm continuation, 20 cell 전체 또는 E1-A 전체 완료 보고가 아니다. Scientific synthesis는 GH 소유다.

## 실행·무결성

Scheduler COMPLETED / 0:0. 단일 GPU 할당 1175초 (0.326389 GPUh); `.batch`/`.extern`을 중복 더하지 않았다. 프로그램 elapsed 1168.112초, 초기 gate 885.320초는 별도다.
실제 native B100 1회, compute-z 100회, key 2회, dense solve 1회, 원본 FP32 history append 1회. W0/context exact, endpoint finite 및 selected/nonselected 보호 확인. 종료 후 pointer/bytes/RNG 복원 PASS; copy에 따른 version 증가 1개는 원복했다고 주장하지 않는다.

## Current100 평가

RS/PS는 NLLnew < NLLtrue, NS는 NLLtrue < NLLnew의 strict 비교이며 tie는 실패다. 100/200/1000은 request/2 paraphrase/10 neighborhood 분모다. 모든 NLL은 낮을수록 해당 continuation의 예측확률이 높다.

| endpoint | RS | PS | NS |
|---|---:|---:|---:|
| W0 | 5/100 (5.0%) | 20/200 (10.0%) | 886/1000 (88.6%) |
| NATIVE | 100/100 (100.0%) | 190/200 (95.0%) | 867/1000 (86.7%) |

## 재현 상태와 계측 범위

재계산 target은 0/100 byte-exact이고 최대 절대차는 0.00138332229다. 단순 forward noise나 hardware 차이로 원인을 확정하지 않는다. 원본 trajectory와 동등하다고 사용하지 않으며 stored CP와 actual W/M 대조가 남아 있다.
대표 signed contraction 2.63113093, ±2^-8 central FD 2.63305664, relative error 0.000731893209; 기존 lock 기준 PASS. Alpha 성능 sweep이나 native update 대체가 아니다.
Peak allocated 37846147584 bytes, reserved 43278925824 bytes. Forward 2856회, input positions 441106개. Token counts는 FLOP 또는 유효 GPU kernel 시간과 동일하지 않다.
compute-z 내부 최종 loss/iteration/stop/clamp 및 native backward 횟수는 NOT_OBSERVED. General corpus, historical panel, projected spectrum/full exposure, 나머지 19 cells와 E0 warm continuation은 남아 있다. 미기록 값을 0으로 채우지 않는다.

## 재현

`python -m project.run_scripts.baseline_mechanism_first.cold_analysis --attempt <sealed attempt> --input-lock <sealed input.lock.json> --destination <new directory>`

기존 raw/source/input은 읽기 전용. 신규 GPU/model load/evaluation replay=0. `scientific_promotion=false`.
