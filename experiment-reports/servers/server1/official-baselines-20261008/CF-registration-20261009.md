# CF checkpoint 전용 실제 등록

README 갱신용 단발 관측: **2026-10-09 03:06:48 KST** (`2026-10-08T18:06:48Z`).
61657/61658/61659 RUNNING, 61660/61661/61662/61663/61664 PENDING.
이는 scheduler 상태이며 native qualification/과학/online 완료 증명이 아니다.
정확 config SHA256:

- FT: `1008ab8f3861f15708ca262d1902514abef5819f58633270bdbd777aca17a86d`
- MEMIT: `0da941fe5903e1261ee28a0feb6ac930d8d049611c0e7b3fc9307f61a011ab9f`
- MEMIT_FE: `18f01afc0051787b44dd67b73c5af17e97ce42ea55aa89967177e1f0bb10097d`

공통 ordered CF stream SHA256:
`66edc483a8d4bcadedd479e4c36759a686ad61a38741d8870a9052b795710e37`.
stream bundle SHA256 `02f335a7e2005d62881aacf09fc3535a6b1a0cd73e3ce9ed0af3ba4ec288a9b1`.

사용자 직접 main 통합/즉시 제출 지시에 따라 main source
`ecc80fc06d1e20d1550cda2fbb1436fbe1059c27`, official tree
`c7ebf3eff6537aa969e71462543994eee4173d11`를 게시·봉인했다.
기존 dirty root/job/archive/raw 변경0. scope는 Llama3 CF FT/MEMIT/MEMIT_FE,
official existing_file_first2000/edit_seed0/100x20이다. W20 FLU/CON은 하지 않으며
W20 checkpoint 및 future consumer pending을 보존한다. 기존 별도 공통 W0 계약은 유지한다.

| 역할 | 실제 ID | dependency |
|---|---:|---|
| FT qualification | 61657 | 없음 |
| MEMIT qualification | 61658 | 없음 |
| MEMIT_FE qualification | 61659 | 없음 |
| 공통 base W0 | 61660 | afterok 61657/61658/61659 |
| CF FT | 61661 | afterok 61660 |
| CF MEMIT | 61662 | afterok 61660 |
| CF MEMIT_FE | 61663 | afterok 61660 |
| CPU collector | 61664 | afterany 61657..61663 |

등록 전 owner/devbox GPU 할당0/admitted DAG0. cap4, 신규 DAG폭3.
GPU 각1/CPU8/65536MiB/48h, collector GPU0/CPU8/24576MiB/4h.
512GiB output/reserve admission과 inode/node/QoS 검사를 통과했다.
전량 held owner/실제 batch script/argv/source/input/memory/wall/dependency 검산 후
후속부터 release했고 bounded snapshot 모두 PENDING이었다. GPU/과학/online PASS는 미관측.

첫 held 검사가 Slurm NumCPUs `8-14`를 정수로 읽어 중단됐다. actual ReqTRES cpu8,
CPUs/Task8/NumTasks1/MinCPUsNode8과 범위하한8을 함께 검증하도록 control parser만 수리.
CPU submit23 및 범위 양성/음성3 PASS. 기존8 ID를 그대로 검사·release했고 중복submit0,
runtime source/archive/config hotpatch0. 최초 failure receipt는 보존했다.

로컬 원본 receipt/log:
`/mnt/raid5/janghj/ODE-edit/local/official-baselines/server1/cf-checkpoint-r1/registration-r1/`
`submission.json`, `held-inspection.json`, `execution-lock.json`, `job-manifest.json`, `logs/`.
source/config 및 실제 ID/W&B run identity는 분리한다. W&B는 다음 실제 startup의
receipt/readback으로 확인하며 현재 remote delivery를 주장하지 않는다.
no recurring monitor/automatic retry. sealed runner가 진행한다.
