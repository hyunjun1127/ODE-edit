# R-GD/R-QP FULL_READ / M0

- instruction: ODEEDIT-S06-L4-PRESERVING-REPAIR-TWOARM-SH4-V1.
- main b70993d6ef8cb2cd37422f697f8c327d7e9562ae, tree660838de0da3b9213522fbb7ddc0c1940d6dc21a에서 새 clean branch/worktree로 착수했다.
- 설계 revision2·원 5-arm cells·CPU8/28·cold7 참고 prose·2-arm dispatch·native L4 source를 읽었다. 원 5-arm cells는 실행 목록이 아니다. 신규 R-GD/R-QP만 20 batches, unique1000/arm-request2000이다.
- 새 L8 response/GN/QP/runner 구현 중이며 actual Llama 기술 PASS·scientific submission은 아직 없다. 기존 N4/REFIT4/LD는 read-only 비교다.
- 최신 Server4 cap2를 적용한다. 공통 pilot 후 두 arm은 각각1GPU/8CPU/60416M. 시간은 pilot 실측 후 봉인하며 GPU-hour cap=null이다.
- 후속 사용자 `checkpoint는 저장하지 말고 진행하라`에 따라 W4/W8/M4/RNG disk checkpoint를 저장하지 않는다. In-memory rollback/history/평가/commit hash와 Q/response 진단은 유지한다. Exact restart는 미제공이다.
- 잠정 저장 reserve48GiB; 초기 확인 available136.7GB. 기존 artifact 삭제0. 신규 teacher는 현재 미요청·미실행이며 기존 cold7 effective teacher를 연결한다.
- session 환경/host/repo/registry 일치. Helper는 worktree local config 없음 및 root의 구세션 설정으로 BLOCK; helper PASS로 기록하지 않았다. 공유 설정 변경0.
- INITIAL_GATE_ONLY_OR_PENDING_HANDOFF. Pending이면 actual NOT_RUN과 미등록 science를 명시하고 모니터링하지 않는다. NO_BROADCAST_NOT_REQUIRED.
