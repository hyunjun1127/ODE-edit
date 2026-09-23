# Alpha-key E0–E4 수리·재등록 인계

상태: **MAIN_GPU_RESOURCE_PENDING_HANDOFF / MONITORING_PAUSED_AWAITING_USER**.
Nonce `ODEEDIT-GH-SH4-ALPHA-KEY-AUTONOMOUS-RESUME-20260923-R1`.

원 native에 존재하는 BOS를 잘못 거부하던 검증을 수정했다. native tokenizer·target·solve·정밀도·수치 임계값은 불변이다. 실제 12시퀀스 token/mask/lookup과 원 tokenizer backend를 exact 결속하며 잘못된 토큰/lookup은 계속 차단한다. 수리 및 제어 경로 CPU143 PASS, 실제 GPU gate는 **NOT_OBSERVED**다.

## 실제 제출 및 마지막 한정 관측

2026-09-23 10:37:06–10:38:02 KST 관측 범위다. 마지막 scheduler 평가 시각은 10:37:08 KST다. 네 job 모두 owner/source/full argv/저장 script/자원/의존성을 held 상태에서 검사한 후 release했다. 즉시 PENDING(None)을 pause 근거로 쓰지 않았으며, 아래 Resources/Dependency가 확인된 뒤 조회를 중지했다.

| 역할 | Job | 마지막 상태 | 의존성 | 요청 GPU |
|---|---:|---|---|---:|
| G0/G1 준비 gate | 52563 | PENDING / Resources | 없음 | 1 |
| E1/E2 geometry | 52564 | PENDING / Dependency | afterok:52563 | 1 |
| E3/E4 writers, native100+SHAM 포함 | 52565 | PENDING / Dependency | afterok:52563 | 1 |
| CPU reducer | 52566 | PENDING / Dependency | afterany:52563:52564:52565 | 0 |

Server4 GPU는 **8/8 할당**, gate는 Priority=1/Dependency 없음/AllocTRES 없음이었다. 스케줄러 CPU 잔여76개, host memory 잔여86,016MiB로 이 job의 8CPU/60,416MiB는 들어가지만 GPU 잔여는0이었다. 실제 free host memory와 스케줄러 예약 메모리를 혼동하지 않는다. Project/task 최대2GPU이며 gate 이후 geometry/writers만 독립1GPU씩 가능하다. 제출 직전 본인 admission은0이었다. 다른 job의 설정/취소/hold 변경0.

E0+94 contrast family 전체가 자율 graph에 등록됐다. G1 READY 없이 dependent science는 시작하지 않는다. G2 W50→B51 native100+SHAM은 아직 관측하지 않았으며, graph 등록을 G3 또는 전체 initial gate PASS로 쓰지 않는다. SEQ/ORDER/FUTURE는 미제출이다.

## Frozen source와 저장

- 실행 source: `f9fbd56f31b0c520763ec9026e660a76cb3074ff`, tree `d7b824052de14288dec559f0cf235a59b8a81f12`.
- archive SHA `62f3495a6899b441db556f9c2cfc7893f4ae3d862b9b465646f4242f7b692ee0` / 495,225,396B.
- lock SHA `5c461288fc77ae7071e84b0264cc240cb845b8cb4862ac47ecf37e874fcbc38b`.
- control: `local/alpha-key-concentration-causal/20260923-r1/controls/attempt-r3/`.
- immutable executable: `local/alpha-key-concentration-causal/20260923-r1/execution-source-r3/`.
- output: `local/alpha-key-concentration-causal/20260923-r1/execution/attempt-r3/`; logs는 `logs/attempt-r3/`.
- autonomous factual report 목적지는 이 repair worktree의 같은 task 보고 경로 `generated-r3/`이다. 생성 여부/결과는 인계 후 조회하지 않았다.

각 GPU job은 1GPU/8CPU/60,416MiB/exportNONE/Requeue0, operational wall7일이다. CPU reducer는 4CPU/8,192MiB/2일이다. 이는 사용자가 지정한 GPU-hour hardcap이나 실측 소요시간이 아니다.

Freeze 시 free246,621,741,056B, 제출 직전 free241,854,758,912B / inode225,521,902. 기존 future 계획232,700,000,000B(공유 FS safety50GB 포함)를 유지했으며 waiver0/독점예약0이다. 실제 write 실패는 기술 실패로 남긴다. 다른 프로세스의 이후 disk 변화를 이 task가 통제하거나 모니터하지 않는다.

12 inputCP/23small input/1819 native-binding 검증은 exact 이전 증거로 재사용했다. 이번에12CP full rehash/retransfer0. 기존 원 source/raw/teacher/입력CP와 실패 archive를 보존하며, 신규 full-state W/M/optimizer resume CP0이다. K/R/Δ/target/timestamp bank만 원 명시 진단 예외다.

## 실패와 검증 범위

이전 GPU gate52527은 44 allocated GPU-sec 실패, 후속52528/52529는 미시작 CANCELLED, CPU52530은 COMPLETED였다. r2 source52dfb6a6의 CPU freeze 변수 충돌도 archive와 함께 보존했다(lock/Slurm/GPU0). 수정 후 source f9fbd56f/attempt-r3에만 새 등록했다. 새 비용은 마지막 관측까지 미할당이며 이후 비용/결과는 추정하지 않는다.

CPU143은 synthetic regression 및 실제 tokenizer 검사이고 Llama G1/G2 PASS가 아니다. 이번 수정은 owner 검토이며 별도 agent red를 수행했다고 하지 않는다. 수리 상세는 [readiness](readiness-ko.md), compact seal은 [handoff.json](handoff.json), [rooted receipt](rooted-receipt.json)에 있다.

이 시점부터 scheduler/log/result polling·초기 gate 추가 관측·heartbeat·자동 agent 재개·추가 scientific 제출을 중단한다. 등록된 프로그램/reducer는 자연 진행한다. Own branch만 nonforce 게시하며 main 통합은 GH 소유다. NO_BROADCAST_NOT_REQUIRED.
