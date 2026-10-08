# 최종 checkpoint 정책의 미래 job 한정 적용

server3는 `GH-SH3-FINAL-CHECKPOINT-FUTURE-ONLY-20261009-R1`을 수락했고, 적용 경계 관측 시각을 **2026-10-09 01:44:20 KST**(2026-10-08 16:44:20 UTC)로 기록했다. 최신 main `f9a6084a`의 정책 전체 SHA는 `6da0d1fc…cfd9da`다. 사용자 정정 “이건 앞으로의 job에 대한 정책이다”가 [이전 server3 수락 보고](report-ko.md)의 적용범위보다 우선한다.

**이 경계 뒤 새로 제출하는 job만** source/config 봉인 시 중앙 archive 정책을 결속한다. 이전에 등록된 PENDING·RUNNING·완료 job과 기존 checkpoint는 나중에 계산을 마치더라도 이번 정책의 전송·삭제 대상이 아니다. MEMIT-HJ의 1k 임시 파일을 포함한 과거 자료는 그대로 보존한다. 이번 정정 후 과거 checkpoint를 추가 inventory하거나 SHA를 다시 계산하지 않았다.

새 job이 실제 최종 checkpoint를 만들 때에만 마지막 계산과 모든 소비자 종료, 정확한 파일·full SHA·크기·identity를 확인한다. SH1이 server1의 용량·64GiB 여유와 unique 목적지를 승인하고 별도 사본을 독립 검증한 `VERIFIED_DESTINATION` 영수증을 발행해야 원본 불변과 소비자 종료를 다시 확인한 뒤 해당 payload만 개별 삭제한다. 실패·목적지 부족·같은 object이면 원본을 유지한다. 아직 SH1의 게시된 정확한 helper/API commit을 새 server3 caller에 결속하지 않았으며, 향후 미봉인 source에만 연결한다. 기존 official Qwen 장애·중단 실험·frozen job은 이 정책으로 재개하거나 수정하지 않는다.

정책 수락과 실제 archive를 구분한다. 이번 단계는 최종 manifest **0건**, 목적지 admission **0건**, 전송 **0건**, 독립 검증 **0건**, 삭제 **0건**이다. [정본·시각·범위 영수증](../../../../audits/servers/server3/final-checkpoint-archive-20261009/prospective-r1.json)에 정확한 상태를 남겼다. 새 GPU/Slurm job·모델 계산·장기 polling은 하지 않았다.
