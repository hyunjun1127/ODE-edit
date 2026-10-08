# Qwen OURS M1 2K 실행

사용자 nonce `USER-SH4-QWEN-OURS-M1-2K-LIVE-WANDB-20261008-R1`로 단일 QWEN_M1_CAP075 cold2K 실행을 승인받았다. 이전 취소된 Qwen run을 재활성화하지 않는다.

준비 source `9ecdf8342c52ee7e6a85a6ed741fc4f5468bc1ef`는 이 repo의 Git object에 없지만 지정된 archive/config/lock/run.sh 네 SHA와 frozen 3470파일·runtime/input 검증이 통과했다. `preparation-v1/run.sh` 원본을 그대로 실행한다. 별도 운영 wrapper의 Git commit은 scientific source와 구분한다.

Qwen L4–8/anchor8/ridge15000/lr.1/CAP075/M1만 유지한다. 20×100 편집, R/P/N current pre/post 및5/10/15/20 all-seen. 신규 W0/contexts/model forward 사전검산0. 기존W0 26000행을 재사용하고 generation은 W20 전체 edit commit 이후에만 수행한다. generation 실패 시 W20 RPN은 유지한다.

## W&B 및 종료 수집

원 cap_tracking/TrackedEvents가 `wkdguswns2256 / layer allocation`에 모델·writer·실제 job ID·source/config·별도 UUID와 scalar history를 전송한다. SDK 접수는 remote PASS와 구분한다.

job-lifetime CPU companion은 원 history를 추가로 쓰지 않는다. 시작, B1/B2/B5/B10/B15/B20 및 finish에서 동일 run/config와 실제 logged row/step을 bounded readback한다. price/budget/realization·M1 norm/context NLL·first500·NS delta 등 scalar allowlist 밖 세부값은 숫자-only JSON 파일로 동일 run에 연결한다. 원문/케이스ID/tensor/모델/credential/fullstdout는 제외한다. 원격 검증 실패는 명시 degraded이며 scientific 재실행 사유가 아니다.

같은 frozen collector로 milestone 및 종료 후 CPU 재집계한다. 최종 RS>=99%, NS>=80.14%를 수치로 판정하고 PS를 보고한다. 아직 실제 GPU/지표/완료는 미관측이다. 운영 wrapper는 science 종료 후 종료하며 별도 agent heartbeat/Slurm 자동 retry는 없다.

## 자원

GPU1 RTX PRO 6000/CPU8/59392MiB/24h, exportNONE/requeue0. 합산 project cap3, Qwen task cap1. 기존61418/61420 유지,60917–60923 baseline hold 유지. disk reserve24536416256bytes를 제출 직전 확인한다. GPU/host 추정74.62/49.74GiB는 실측이 아니다.

등록 결과와 W&B 실제 online 여부는 후속 receipt로 남긴다. 정상 자원 PENDING을 완료로 쓰지 않는다.
