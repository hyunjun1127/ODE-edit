# Qwen zsRE SPHERE OOM 수리 및 cold 재등록

Nonce `USER-SH2-QWEN-ZSRE-SPHERE-OOM-RERUN-20261010-R1`.
이 문서는 최초 cold 등록 당시 기록이다. 이후 사용자 지시
`USER-SH2-QWEN-ZSRE-SPHERE-B9-RESUME-20261010-R1`가 이 한 건의 cold-only 조건을 대체했다.
62534는 실행 전에 취소됐고, 원62087 B9에서 B10–B20을 이어가는
[후속 재개 기록](../qwen-zsre-sphere-b9-resume-20261010/report-ko.md)을 따른다.
아래 최초 제출 snapshot/source/CPU 기록은 지우거나 새 재개 결과로 바꾸지 않는다.
같은 사용자 메시지를 부모가 SH2에 직접 전달했고, SH2는 기존 accepted turn
`01a122f4-e240-7532-91fd-5295daaa902a`에서 별도 수리 범위를 명시 수락했다.
GH는 추가 독립 배정/중복 제출을 하지 않았다. FE author 네 chain은 그대로 유지한다.

## 원 실패와 최소 변경

원 job62087은 corrected-context Qwen zsRE SPHERE이며 B1–B9 commit 후 B10의
`sparse_projection` GPU `eigh(C)`에서 OOM. W20 없음; 정상 CF62079 완료값과 구분한다.
원 B9 payload SHA `7e44f382ba1f3befcb9f4429f42fd328d55422306f9cef868a920c8bf7ec17b8`,
8,535,442,009 bytes와 raw/source/비용은 보존하고 새 cold run에 묵시 복원하지 않는다.

공통 수리 commit `c54779f2873b362497a5ba5d1fa4d9663815a9ec`은 weight 반영 후
`del upd_matrix_proj, P_soft, U`로 이전 layer의 불필요 반환물 참조를 종료한다.
U view가 유지하던 전체 eigenvector storage를 다음 layer RHS/eigh 전에 해제하는 변경이다.
FP32 GPU eigh, 투영식/순서/rank/eta/alpha/threshold/native hparams는 변경하지 않는다.
target fit 전체를 no_grad로 감싸지 않는다. source 변경은 실제 GPU OOM 해결 입증이 아니다.

## 실행 및 표 범위

기존 SPHERE hparams/seed0/first2000/BS100×20/FP32 eager/TF32off 유지, 별도 경로의 cold W0.
최신 public-query E/G/Loc(loc_ans request-macro), W0 및 batch/W20 checkpoint와 W&B identity를 유지한다.
FE author clamp/step을 SPHERE에 적용하지 않는다. 추가 GPU qualification/smoke 없음.
server2 현행 cap/메모리 내 1GPU chain 정확히1개, 기존 jobs의 resource dependency 뒤에 등록한다.
기존 job 취소·우선권 변경·checkpoint 삭제·무관 모델/CF 재실험 없음.

SH2 실제 **62534 `s2-qwen25-zsre-sphere-oom-r1`** held 검사/release 완료.
2026-10-10 08:33:20 KST 단발 snapshot은 **PENDING**, `afterany:62532`다.
따라서 FE author CF62531 → zsRE62532 → SPHERE62534 순서로 대기하며 기존 job은 변경하지 않았다.
1GPU/6CPU/59392MiB/48h ceiling, server2 cap4 유지.
README Qwen SPHERE zsRE 3셀만 이 실제 등록 상태로 갱신했다.
[통합 영수증](../../../audits/global/qwen-zsre-sphere-oom-rerun-20261010/coordination.json).
CF Score/E/G/Loc **83.73/99.40/97.70/64.37**은 그대로다. 실패를 점수0 또는 완료로 표시하지 않는다.

실행 source `9b70ecca6f445728542d7a55ab0014b01252ecc1`, config SHA
`dcc7018f54b1f63c8c295ebe5d82b32fb933856e0d0b94b8659c9238ef41a814`.
새 checkpoint 위치는 `/mnt/raid5/janghj/ODE-edit/local/qwen-zsre-sphere-oom-rerun-20261010/registration-r1/runs/qwen25-zsre-sphere/checkpoint/latest.json`이며
최종 batch20은 아직 미래 산출물이다. CPU4 검사는 실제 apply loop의 weakref 수명/old negative control,
bitwise weight parity, target-fit gradient 유지, native 식/호출 설정을 확인했다.
**실제 GPU OOM 해결/W20/온라인 run startup은 아직 미관측**이다.
[SH2 원 영수증](../../../audits/servers/server2/qwen-zsre-sphere-oom-rerun-20261010/receipt.json)과
[담당 보고서](../../servers/server2/qwen-zsre-sphere-oom-rerun-20261010/report-ko.md)를 보존한다.

실제 source/config/checkpoint 경로와 held 검사/release는 owner receipt를 근거로 구분하며,
GPU 실험 완료까지 기다리지 않고 등록 후 bounded snapshot에서 인계한다.
GH GPU/forward/job 변경0, 신규 recurring monitor/자동 retry0. 원 raw/가중치 Git 추가0.
