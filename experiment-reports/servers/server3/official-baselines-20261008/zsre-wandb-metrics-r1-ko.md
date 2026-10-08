# server3 zsRE W&B 공통 API 채택

nonce `GH-SH3-ZSRE-WANDB-METRICS-READY-20261009-R1`. 전용 준비 WT를 main `a0408faada5d493f562331609b9d19d987bee3f4`로 ff-only 동기화했다. caller 구현 source `993f691703f2f5a71205bf739cc70839a35a07dd`는 공통 API source `21a78e27`를 포함한다. 상태는 **CALLER_ADOPTED_NOT_SUBMITTED**이며 실행 archive는 아직 생성하지 않았다.

server3 `_log_scalar_receipt`가 실제 tracker의 bound `config_values`에서 dataset=zsre를 확인하고 기존 reducer-summary scalar를 `official_zsre_metrics`에 전달한다. raw case/token은 전달하지 않는다. 기존 official namespace와 새 zsre namespace를 함께 기록하며 공통 validator가 값 일치를 검사한다. Score 계산·분모·state 축 검증은 공유 API 소유이고 서버3가 별도로 재구현하지 않는다. CF 경로는 변경하지 않았다.

CPU에서 W5 current requests100/all_seen requests500, E/G/W0 agreement의 조화평균 Score, 별도 loc_ans, 원 payload 비변이를 검사했다. server3 caller+tracking+SH2 checkpoint profile **55 tests PASS**, source157/Python257 SHA/AST 검사 PASS, externalimports0. 공통 검산은 실제 GPU/온라인 W&B 또는 SH2 전체 fixture PASS가 아니다. 새 forward/Slurm/온라인조회0, frozen job/archive와 기존 raw/CP 변경0이다.

SH2가 보고한 qwen25 saved view는 `https://forge.coreweave.com/wandb/wkdguswns2256/layer%20allocation?nw=zsreqwen25`이며 이번 turn에서 원격 조회하지 않았다. 향후 승인 zsRE run은 실제 source/config/model/job identity와 edits 축을 결속한다. 기존 서버3 자산·저장공간 blocker는 이번 코드 입력으로 해소됐다고 표시하지 않는다. README main table은 GH 소유이며 새 main job ID/name은 없다.
