# SH2 official 별도 GPU 검증 제거 및 cold 교체

권한: `USER-GH-SH1-SH2-OFFICIAL-NO-GPU-QUAL-RERUN-20261009-R1`.
현재 단계: 대상 취소·own caller CPU 검산·실제 held 등록/검사/release 완료.

- 정상 완료 CF FT **61650**은 W20 result/source/CP/raw를 보존하고 재실행하지 않는다.
- exact owner/source/Command/WorkDir/server2/state를 확인하고 pending을 먼저 hold한 뒤
  61672,61656,61671,61670,61669,61668,61667,61666,61655,61654,61653,61652,61651을 취소했다.
  accounting은 13개 CANCELLED; 후속 한정 squeue에서 active 대상0이다.
- MEMIT61651/AlphaEdit61652는 각 elapsed 00:11:44이며 main 이전 qualification 중이었다.
  MEMIT qualification W1 raw, 양쪽 latest checkpoint 등 부분 증거는 KEEP. main 재개에 사용하지 않는다.
- exact cancellation raw receipt:
  `/mnt/raid5/janghj/ODE-edit/local/official-baselines-server2/20261008-r1/no-gpu-qualification-stop-r2/receipt.json`.
  r1은 purged completed61650의 scontrol 부재에서 mutation 전에 중단했다. 그 오류도 보존했다.

새 경로는 CF5+zsRE6이며 원 native math/hparams/precision/seed/2k/BS100 및 factual 일정은 유지한다.
별도 qualification/smoke/native oracle forward/복원 비교 및 과거 proof gate만 제거했다.
명시 상태는 **NOT_RUN_USER_DISABLED**이며 가짜 READY/PASS가 아니다.
W0/shape/finite/native commit/nonedited guard와 latest1/W20 checkpoint, 공식 W&B는 유지한다.
CF FLU/CON은 deferred이며 zsRE에 생성 평가를 추가하지 않는다.

W0 입력은 실제 factual 관측이다. 기존 CF W0의 factual source SHA는
`2bc41883b9261d084a3b4b99e0920911789659401a6d9b2b0f0d801a446ab47b`,
현재는 `537d3d8d641b26075de6801dc459424fcb989c428973d62c5501a0e71809ad6f`로 다르다.
기존 source-bound external identity를 새 source로 relabel하지 않는다. 현재 exact reader 계약에는
호환하지 않아 원 승인된 factual W0만 dataset당1회 새 입력 노드로 계산한다. zsRE W0는 아직 없다.
생성/oracle/fit 검증을 이 입력에 추가하지 않는다.

CPU 검산: 새 overlay/DAG/route/tracking 포함45 tests와 기존 run/submit/collect/execution55 tests PASS.
cap4 DAG의 모든 도달 가능한 완료 집합에서 동시 runnable 수≤4를 CPU 검산했다.
source verify157 SHA/Python267/external-task-imports0 PASS. 실제 GPU qualification은 USER_DISABLED.
독립 reviewer는 사용하지 않았고 owner review다. 원 dirty root/타 job/공통 수학/README는 변경하지 않았다.
통합 후 source verify는157 SHA/Python280/external-task-imports0 PASS다.

## 실제 등록

Source/main `478464689231595eb63da368cc7c376e775eb559`,
official tree `41ed261d1fc8c7c0022366c50fa9cb1ae3c05320`.
Manifest `cf7619c3e47122d7b616253d08c76318287887dc43ef2b5cc9874d605d6aea3e`,
lock `ff7ee062dddb9ef2724eb0e527c46a1f1265c1f8cb56e7ad83330d1646217e07`.

| 방법 | CF | zsRE |
| --- | --- | --- |
| FT | 61650 완료 KEEP | 61726 |
| MEMIT | 61725 | 61728 |
| AlphaEdit | 61727 | 61730 |
| BLUE | 61729 | 61732 |
| FE | 61731 | 61734 |
| SPHERE | 61733 | 61735 |

필수 factual W0 입력 CF61723/zsRE61724는 별도 qualification이 아니다.
CPU collector61736은 새13 GPU 입력/chain job의 afterany다.
4lane resource afterany와 자기 dataset W0의 technical afterok만 사용하며 old cancelled ID는 없다.
전체 그룹 완료 barrier는 없다. 동시상한4, 실제 admission frontier는 비어 있었다.
각 GPU1/CPU6/59392MiB/48h, collectorGPU0/CPU6/24576MiB/4h,
exportNONE/Requeue0, node server2/QoS lab_gpu_s2다. Wall은 ETA가 아니다.

2026-10-09 04:32:25 KST 최초 bounded snapshot: 전부 PENDING, 전부 release 완료.
실제 W&B startup/remote readback은 NOT_OBSERVED이며 GPU 과학완료를 주장하지 않는다.
별도 GPU qualification 및 checkpoint-resume 동등성은 NOT_RUN_USER_DISABLED다.
실제 데이터·source·full argv·resource·dependency·checkpoint/W&B 결속은 held 검사로 확인했다.

원본 로그/receipt: `/mnt/raid5/janghj/ODE-edit/local/official-baselines-server2/20261008-r1/registration-no-gpu-qual-r1/`.
각 `<role>-<jobID>.out/.err`, `<role>/tracking/`과 checkpoint/raw는 local-only다.
작은 submission/cancellation audit만 Git 게시한다. NO_BROADCAST_NOT_REQUIRED.
GH direct nonce `SH2-GH-NO-GPU-QUAL-SUBMITTED-20261009-R1`은 accepted turn
`01a11d01-08e6-7ec2-9083-147d43afe367`로 전달했다. README 통합은 GH 소유다.
이후 장기 W20 대기/반복조회/자동재시도 없이 runner/collector가 자연 진행한다.
