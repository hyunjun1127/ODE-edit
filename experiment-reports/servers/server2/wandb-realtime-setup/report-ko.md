# Server2 W&B 설정 — 로그인 재개 결과

상태: **READY_ONLINE_VERIFIED**. USER_REPORTED 로그인 이후 한 번의 CPU smoke에서 3점 업로드·원격 readback3점, dropped0을 확인했다. 실제 FE 모델 검증은 아니다.

- entity `wkdguswns2256`, project `layer allocation`(공백 유지), API `https://api.wandb.ai`.
- [검증된 smoke run](https://wandb.ai/wkdguswns2256/layer%20allocation/runs/d2b83c62f25547fd), ID `d2b83c62f25547fd`.
- SDK0.30.0, Python `/mnt/raid5/janghj/ODE-edit/local/wandb-setup/sdk-env/bin/python`; 과학 Python/torch/transformers 변경0.
- nonsecret 설정 `/mnt/raid5/janghj/ODE-edit/servers/local/wandb.env`; SH1 shared helper source93912c3f를 read-only 사용.
- 자동 code/git/console/metadata/stats/artifact 업로드 비활성. smoke 원격 파일 boundary 검사 포함. 키/원문 prompt/tensor/fullstdout 전송0.
- 기존 인증실패 receipt `local/wandb-setup/initial/receipt.json`는 보존. 새 정본 `local/wandb-setup/login-resume-r1/result.json`.
- FE 새 source3645b408에 init→scalar log→bounded finish 연결; startup online은 model load 전에 검사. 네트워크 저하가 과학 재실행 사유가 되지 않는다.
- 이번 smoke GPU0/Slurm0. 별도 승인된 FE10k의 job59878/59879 등록과 구분한다. 실험 자체의 run URL은 아직 agent 미관측.

독립 reviewer0, owner 확인. credential/SDK/spool broadcast0; NO_BROADCAST_NOT_REQUIRED. 추가 반복 모니터 없음.
