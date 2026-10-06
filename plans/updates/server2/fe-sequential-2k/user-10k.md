# 사용자 FE 10k 확대 — 2026-10-06

현재 SH2 대화의 사용자 원문: “10k까지 진행하는 걸로 하자.”

기존 `USER-GH-SH2-FE-SEQUENTIAL-2K-20261006` 및 로그인 재개 지시의 동일 FE run을 제출 전에 확대한다. 아직 FE Slurm ID는 없다. 원 2k 설계·CPU7·입력 준비는 역사 그대로 보존한다. 전용 branch/local/report namespace는 기존 승인 경로를 사용하고 새 실행 profile/job은 `fe-sequential-10k`로 구분한다.

- fixed10k 전부, BS100×100, cold W0/H0, W0 target fit 10,000회 후 canonical replay 10,000회. 상한 350,000 loss evaluations / 340,000 Adam updates.
- 100 commit / 500 layer solves / 500 prewrite history append / 99 interbatch 연결. B101 없음.
- 기존 current pre/post 및 5-batch마다 all-seen 관측을 W100까지 확장. 최종 R10,000/P20,000/N100,000. first100/500 및 birth/active cohort는 저장된 동일 row에서 계산한다.
- 수식·FE35 budget·컨텍스트·FP32/FP64·W0 target 고정·noCP 그대로. 기존 다른 task 재개/변경 없음.
- target table 819,200,000 bytes를 CPU RAM에만 유지. 신규 edited W/H/delta/resume 저장 없음.
- task 1GPU/6CPU/59392MiB, project cap2 또는 더 엄격한 현재 정책. 48h는 요청 상한이며 완료 ETA가 아니다. 첫 실제 fit 시간은 runner가 기록하며 agent의 장기 polling은 없다.
- 새 원문 결속/input/source/resource lock 후 공통 W&B logger 사용. 사용자 로그인은 online 3-point smoke/readback으로 별도 확인.

원 global 2k contract를 수정하지 않는다. 충돌하는 horizon/분모만 이 사용자 지시로 대체한다. 품질·성과/실제 GPU PASS는 아직 미측정이다.
