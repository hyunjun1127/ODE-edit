# SH1 JLZ efficiency 수신 / M0

Nonce: ODEEDIT-GH-SH1-JLZ-EFFICIENCY-20261001-R1.
RECEIVED / IMPLEMENTING / NOT_SUBMITTED (본 기록 시점).

SH1 session 01a04939-f93a-7b50-bca0-65438eab2062, server1/devbox,
root `/mnt/raid5/janghj/ODE-edit`, repository hyunjun1127/ODE-edit.
전용 `codex/server1-jlz-efficiency-20261001-v1` worktree에서 처리한다.
원 root dirty/공유 identity/기존 56684 source·config·job은 변경하지 않는다.

실행 envelope·design·contract 및 연구 원7파일 FULL_READ.
dispatch manifest7 SHA/size 및 reference10 source SHA MATCH.
Reference 7b4de31d4361caffbbb383fb0cadb8c5258e9cbe와 새 benchmark source를 분리한다.
`inspect_costs.py`는 실행하지 않았으며 옛 mutable log를 효율화 근거로 새 조회하지 않았다.
기존 task 권한의 B1→B2 관찰은 별도 유지한다.

구현은 새 `jlz_efficiency/`에만 둔다. 소형 first4 fixed≤32,
short≤8×12(각 final1 포함), standalone reserve≤32, 합≤160;
B100 fixed 동일 후보 paired oracle8, observer 별도 장부.
Native first4의 원 singleton/캐시/독립 batching, 요청당25candidate/24Adam 상한.
실제 모델/속도 PASS는 아직 없다. 실패한 효율화 경로는 제외하며 기준을 완화하지 않는다.

계획 자원: projectcap2, 새 taskcap1, 1GPU/8CPU/131072MiB/12h,
exportNONE/requeue0; CPU collector4CPU/16384MiB/2h.
8GiB scalar/temp 계획, noCP, exact_resume=NOT_AVAILABLE.
원56684 B1 commit/5append/observer→B2 entry 초기인계는 취소하지 않는다.
NO_BROADCAST_NOT_REQUIRED; Server3 조회/새 과학 chain 없음.
