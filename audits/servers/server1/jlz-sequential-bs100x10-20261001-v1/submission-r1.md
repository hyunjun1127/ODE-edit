# JLZ B100×10 제출 기록

Instruction/nonce: `ODEEDIT-GH-SH1-JLZ-BS100X10-20261001-R1`.
2026-10-01 제출. 이 문서는 제출 사실이며 actual BS100 또는 10batch 완료 증명이 아니다.

| 항목 | 값 |
|---|---|
| GPU | 56684 / odeedit_jlz_sequential_s1 |
| CPU collector | 56685 / odeedit_jlz_collect_s1 |
| dependency | afterany:56684 |
| frozen scientific source | c91962dd4952eb58a16b6b00dcaa3018be71693f |
| frozen scientific tree | 620a5ed510569a09bac405fc220e11afa0724182 |
| submission controller | d742e49b (scientific lock와 별도) |
| execution lock SHA256 | 3fb6601ac22352f214c17d3b0a3ebcf495c4897fba0e9523a9afa36bf8a396cf |
| GPU launcher SHA256 | 4b810643e1ddda43fea19514a7328df01347083b611adbae03753a15d1d4b894 |
| collector launcher SHA256 | edb7773685c93d469a9f6ddbe516644b2215b187ca6561078f5ae8d62e49da65 |
| GPU resource | devbox/gpu/lab_gpu_s1, 1GPU, 8CPU, 131072MiB, 48h, exportNONE, Requeue0 |
| CPU resource | 8CPU, 8192MiB, 2h, GPU0 |
| checkpoint / exact resume | false / NOT_AVAILABLE |

Local run: `/mnt/raid5/janghj/ODE-edit/local/jlz-sequential/20261001-v1/attempt-r1/`.
`submitted-gpu.json`, `held-inspection.json`, `submission.json`을 create-once 보존했다.
두 job 모두 held에서 owner/name/argv/resource/dependency를 검사했다. CPU collector를 먼저 release하고 GPU를 release했다.
제출 전과 release 직전 project active+admitted pending GPU 계수는 각각 0이며, 신규 1로 cap2 이내다.
기존 job 변경/취소, Server3 접근, 새 native fit, checkpoint 저장, 원자료 전송 없음.

## 실행 전 검증 범위

31개 dispatch member, frozen source25개와 inputs17개 SHA/size 결속.
모델/데이터/contexts/C0는 기존 local 파일만 사용한다. C0 다섯 층은 각각 float32 14336²,
positive mom2 count 66019200이며 count로 FP32 나눈 뒤 FP64 solve에 사용한다.
실제 첫1000의 batch별 rewrite/KL packing은 700rows, key packing은 600rows,
최대 padded length32다. 이는 CPU packing 증거이며 모델 forward PASS는 아니다.
모델 runtime 기록: torch2.9.1+cu128 / transformers4.57.1 / RTX A6000, load27.895756s.
48h는 예약상한이며 실측 ETA가 아니다. 최초 BS100 비교/fit 비용은 관측 후 따로 기록한다.

## 독립 red의 제출 도구 추가 WARN

과학 kernel과 별개인 submission controller에는 향후 compressed pending array 계수(`squeue --array`),
복수 GPU-type GRES 계수, collector sbatch 직후 별도 ID receipt 보완 권고가 있다.
이번에는 두 resource 조회 모두 빈 project queue였으며 두 ID 및 held/release 결과가 모두 영속 receipt에 남았다.
따라서 이 edge는 이번 0+1 admission에 영향을 주지 않았다. 이 사유로 실행 중 source/config/job을 변경하거나 중복 제출하지 않는다.
이 기록은 향후 controller 일반성이 완전 검증됐다는 주장이 아니다.

## 게시와 다음 경계

Source 및 controller를 clean integration으로 nonforce main 게시했다:
`f184af2bea3c8c13e1224b8cd4412b28c80141e3`, tree `785fd988b550e6b8189e33aa417ce5a9c31325e3`.
실행 source와 publication source는 구분한다.
B1 실제 commit/다섯 H append/observer/B2 entry continuity까지만 관측하고 이후 agent monitoring을 pause한다.
10batch program/afterany collector는 계속하며 상세 terminal 검토는 사용자 recall에서 수행한다.
Raw Git0 / NO_BROADCAST_NOT_REQUIRED.
