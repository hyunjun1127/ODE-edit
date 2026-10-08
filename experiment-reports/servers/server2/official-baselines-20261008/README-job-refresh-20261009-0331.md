# Server2 README job 번호 최신화

USER 요청: “job 번호 올린거 readme에 최신화해”.
2026-10-09 03:31 KST `sacct`/`squeue` exact14 job snapshot이다.
조회·문서 갱신만 수행했고 job/source/config/dependency 변경은 없다.

| 방법 | CF job / 상태 | zsRE job / 상태 |
|---|---|---|
| FT |61650 RUNNING|61667 PENDING|
| MEMIT |61651 PENDING|61666 PENDING|
| AlphaEdit |61652 PENDING|61668 PENDING|
| BLUE |61653 PENDING|61669 PENDING|
| FE |61654 PENDING|61670 PENDING|
| SPHERE |61655 PENDING|61671 PENDING|
| GPU0 collector |61656 PENDING|61672 PENDING|

owner는 모두 janghj, RUNNING node는 server2. PENDING reason은 모두 Dependency.
FT의 Slurm RUNNING은 편집 chain 시작/qualification PASS/과학 완료를 뜻하지 않는다.
CF source6d35c65d, zsRE sourceccc1f5d6 및 원 submission/lock 보고는 그대로 유지한다.
합산 cap4, CF FLU/CON 연기·최종 W20 checkpoint 보존 불변.
과학 결과/온라인 W&B는 이번 문서 작업에서 재조회하지 않았다.
원 root dirty/타서버 README 변경 보존, 신규 모니터·자동 재시도 없음.
