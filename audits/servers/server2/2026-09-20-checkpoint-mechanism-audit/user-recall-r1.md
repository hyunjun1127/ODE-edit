# 명시 사용자 재개 및 admission 보완

사용자: “자리 비었으니 task 이어서 진행시켜”. 이번 checkpoint mechanism task만
재개한다. 다른 task의 STOP/PAUSE는 유지한다. 과거 구현-only/pause receipts는 보존했다.

- 기존 job51071은 실행 전 held 상태, allocated GPU=0을 확인하고 release했다.
- frozen execution source `3f65d1705ff69fe1d53c8b5e30313265889d5c2a` 및 lock
  `f06da6977695e69a90655db0315788d472369691524ccc04f1626d79540936de`를 변경하지 않았다.
- 새 제출0, 타 job 변경0. release 당시 노드 GPU6/8 할당, CPU51/64 할당이었다.
- 현재 정책은 common gate 뒤 독립 1GPU lane 두 개, project cap2/8CPU/60416M이다.
- `squeue -w server2`가 아직 NodeList 없는 pending 요청을 누락할 수 있으므로,
  후속 submit helper는 전체 project job 목록에서 실제 NodeList/ReqNodeList를 확인한다.
  미배정 pending의 ReqTRES도 admission에 포함하며 불명확한 node는 fail-closed한다.
  이는 자원 통제 보완이며 수식/precision/tolerance/sample 변경이 아니다.
- 원 gate source를 hot-patch하지 않았다. 후속 실행은 새 source freeze를 사용한다.

Local recall receipt:
`local/checkpoint-mechanism-audit/20260920-v1/attempt-v1/receipts/user-recall-20260920-r1.json`.
Model/tokenizer 원본10파일16,069,717,915B의 full SHA를14.033s에 확인했으며,
이후 동일 stat identity인 경우 이 task-local receipt를 재사용한다.
Receipt SHA `c614a829c941bdf1aa79469558e75ba3898f3ff7fa25995c5acf3009d5788438`.
이는 실제 모델 수치 parity와 별개다.
