# 명시 승인경로와 generic helper 차이

사용자 Instruction `ODEEDIT-S06-EN-ADAPTIVE-NULLSPACE-B300-SH3-V1` §7은 `tasks/status/server3-en-adaptive-nullspace-20260920-v1/**`와 해당 runs prefix를 명시 승인했다. Command-scoped head-server3/server-head/server3 identity로 staged access helper를 실행했으며, generic helper는 status.json을 미지원 경로로 거부했다(exit7). 나머지 staged 경로는 거부가 없었다.

동일 사용자 지시가 exact 승인경로의 예외 기록을 허용하므로 이 범위만 예외로 진행한다. 공용 helper/identity/config는 변경하지 않는다. 이는 자동 승인 review의 거절이 아니라 저장소의 일반 경로표 미지원이다.
