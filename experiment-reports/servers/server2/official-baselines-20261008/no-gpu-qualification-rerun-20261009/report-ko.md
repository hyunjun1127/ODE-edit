# SH2 official 별도 GPU 검증 제거 및 cold 교체

권한: `USER-GH-SH1-SH2-OFFICIAL-NO-GPU-QUAL-RERUN-20261009-R1`.
현재 단계: 대상 취소 완료, own caller CPU 검산 완료, 실제 교체 제출 전.

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
실제 job/lock/source 및 초기 W&B 상태는 등록 후 별도 submission receipt로 갱신한다.
