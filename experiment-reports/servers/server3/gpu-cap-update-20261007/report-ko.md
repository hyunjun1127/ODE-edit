# SH3 합산 GPU cap1 최신 정정

**APPLIED_CAP1_NO_QUEUE_ADJUSTMENT**. Nonce `USER-GH-SH3-SH4-GPU-CAPS-1-3-20261007-R1-SERVER3`를 수락했다. 최신 authority `f0cd42a2821228b470a735386e08ae0340b0e0a1`의 envelope·policy를 FULL_READ하고 전달 SHA와 일치함을 확인했다. 정본은 S1=2/S2=2/S3=1/S4=3이며 SH3 로컬만 변경했다.

직전 main `b77927d3`의 cap2 보고는 과거 적용 이력으로 보존한다. **현재 SH3 유효 합산 project cap은1**이며 과학·GPU 준비·W0·pilot·평가를 모두 합산한다. task cap이나 STOP을 확대·해제하지 않는다.

| 적용 위치 | 이전 → 최신 |
|---|---|
| root ignored servers/local/gpu-caps.tsv | 2 → 1 |
| 직전 gpu-cap2 정책 worktree의 ignored cap | 2 → 1 |
| 새 gpu-cap-update 전용 worktree ignored cap | 새1 |

node ubuntu, 메모리121856MiB, job-name patterns, 다른 서버 행은 그대로다. 과거 과학 frozen source/설정과 정지된 task worktree는 보존했다. Root tracked checkout/dirty도 보존하므로 새 source/admission은 최신 정본이 있는 `/data/janghj/ODE-edit/local/gpu-cap-update-20261007/worktree` 또는 그 이후 최신 전용WT에서 수행한다.

2026-10-07 09:42:02 UTC(18:42:02 KST)에 owner janghj queue를 이름 필터 없이1회 조회했다. 반환17행은 모두 다른 node를 명시했으며 ubuntu 할당/요청 행과 node 미지정 PENDING은0이었다. 다른 서버 job의 추가 source·결과 조회나 변경은 하지 않았다.

| server3 한정 관측 | 값 |
|---|---:|
| RUNNING/COMPLETING/CONFIGURING 할당 GPU | 0 |
| released/admitted PENDING job | 0 |
| pending array/DAG 최대 동시 GPU | 0 |
| scheduler 변경/신규 제출/취소 | 0 |

따라서 조정할 own pending dependency/array가 없었다. 이름 기반 helper만으로0이라 주장하지 않았다. 향후 owner/task/source/receipt 확인 및 pending DAG 최대 폭은 running helper 검사와 별도로 수행한다. 이 snapshot은 예약이나 계속되는 모니터링이 아니다.

CPU mock19검사 PASS. 현재 로컬1·정본1 helper에 빈 own-node snapshot을 CPU 재현해 요청1 허용/요청2 거부를 확인했다. 실제 신규 admission·실험 허가가 아니다. task1 및 task2 모두 project1과 min을 취하며 더 엄격한 제한과 STOP을 유지한다. [적용 감사](../../../../audits/servers/server3/gpu-cap-update-20261007/application.json)에 SHA·before/after·검산을 기록했다.

Owner 검산만 수행했으며 독립 reviewer는 사용하지 않았다. 모델/GPU 계산·W&B 새run·새Slurmjob·running변경·자동재개는0이다. 보고 게시 뒤 반복 queue 조회 없이 종료한다.
