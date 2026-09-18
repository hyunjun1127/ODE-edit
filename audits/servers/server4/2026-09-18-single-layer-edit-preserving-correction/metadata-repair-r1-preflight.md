# M metadata serialization 최소 수리 — SH 자체 preflight

사용자: “기술적 오류는 해당 오류 보고 이후 SH가 직접 수정후 재제출해”.
보고된 50021 N4 endpoint reload 오류 후 본 scope 기술 수리·재제출을 진행한다.

- 직접 원인: `torch.__version__`의 TorchVersion subclass가 endpoint metadata에 pickle되어 weights_only reload 거절. 실제 tiny CPU 재현 및 원 N4 파일의 scoped allowlist CPU 검사로 확인했다. weights_only=False를 사용하지 않는다.
- 수정: endpoint metadata의 torch/transformers 버전을 builtin str로 저장한다. 실제 tensor·수치·method guard·geometry·optimizer·threshold·원 runtime/native bytes는 변경하지 않는다.
- 기존 50021_0/1 FAILED, _2–9 exact 취소. 과거 자료 삭제0, cancel rollback NOT_VERIFIED. terminal allocation2301GPU-sec는 이전 T49928/M49973의3231sec와 분리한다.
- B2/B3 native capsules를 CPU weights_only/mmap/fileSHA/finite/weightSHA/요청순서/W0/M0/P4/receipt로 검사했다. B1/B2/B3 native REUSE, B4–10 최대7신규fit/700target. B4의 불완전 native 작업은 비용만 보존하고 완료로 재사용하지 않는다. full process/GPU resume 주장이 아니다.
- 새 좁은 CPU7검사: metadata 안전 reload, retained valid/order/warm-state/projector/weight corruption/history negative. 원 CPU82와 skip4/storage3은 변경 없는 범위의 이전 근거를 재사용한다. 새로운 T/FD/GPU pilot0.
- T=SKIPPED_USER_DIRECTED, full_numerical_validation=NOT_ESTABLISHED. M10 independent cold/8arm/80finalL4 보존, cap2, S/R/L0, initial M 후 pause 유지.
- 72GiB reserve는 USER_WAIVED_RESERVE_PENDING_USER_SPACE_CLEANUP이며 실제 IO/저장완결성 오류는 계속 실패다. 삭제/이동0.

별도 독립 red agent는 사용하지 않았다. SH 자체 source/state/reuse/resource/raw-free 검사이며 실제 M initial은 재제출 후 관측해야 한다.
Local evidence: `/data/janghj/ODE-edit/local/single-layer-edit-preserving-correction/20260918-v1/receipts/metadata-repair-r1/`.
