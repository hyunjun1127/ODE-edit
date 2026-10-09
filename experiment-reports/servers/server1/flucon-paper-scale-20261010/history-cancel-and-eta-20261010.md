# 사용자 history Flu/Con 취소 및 잔여시간 추정

2026-10-10 약08:01 KST 직접 사용자 지시: MEMIT_FE_HISTORY Flu/Con은 수행하지 않는다.

own submission/config/source/script/owner/WorkDir/현재 상태를 취소 직전 대조했다. 후속62263→선행62262 순서로 scancel, 둘 다 PENDING→CANCELLED 확인. GPU 평가 시작 전 RunTime0이다. 원 CP/편집 결과/로그는 KEEP. 다른 새 history generation 제출 없음.

- GPT-J history Flu/Con 62262: CANCELLED.
- Llama history Flu/Con 62263: CANCELLED.
- FT62259/SPHERE62260/MEMIT-FE62261: 유지.
- 공유 GPU0 collector62264: 유지. 남은3개 afterany가 있어 조기 실행되지 않는다. frozen collector를 수정하지 않으며 취소된 history의 COMPLETE 부재는 사용자 취소/미관측으로 해석, 0 또는 평가완료로 쓰지 않는다.

원 취소 영수증: `/mnt/raid5/janghj/ODE-edit/local/official-baselines/server1/flucon-paper-scale-20261010/history-cancel-20261010.json`.

한 차례 accepted scalar 진행률을 읽은 실측 기반 추정(전체2000 요청). scheduler 48h 제한이나 과거 no-cache ETA를 사용하지 않았다.

| 방법 | job | 완료 요청 | generation 경과초 | 평균 requests/sec | 예상 잔여 | 예상 전체 generation |
|---|---:|---:|---:|---:|---|---|
| FT | 62259 | 825 | 4808.1524 | 0.1715836 | 약1시간55분 | 약3시간14분 |
| SPHERE | 62260 | 702 | 4073.1052 | 0.1723501 | 약2시간5분 | 약3시간13분 |
| MEMIT-FE | 62261 | 561 | 3273.7730 | 0.1713619 | 약2시간20분 | 약3시간15분 |

계산은 `(2000-completed)/cases_per_sec`. 최근60요청 구간 속도도 전체평균과 근접한다. 2026-10-10 약08:01 KST 기준 대략 FT09:55–10:00, SPHERE10:05–10:10, FE10:20–10:25 완료 예상이며, 자원 경합/마지막 저장·readback을 고려해 각각 약15–25분 변동 가능하다. 확정 종료시각이나 GPU 완료 주장이 아니다. 세 실행은 병렬이므로 세 잔여시간을 합산하지 않는다. 추가 반복 모니터링 없음.

GH README 단독 소유: history Flu/Con 두 셀은 USER_CANCELLED_NOT_MEASURED로, 남은 세 평가는 RUNNING으로 갱신할 입력이다. 이번 사용자 요청으로 factual 결과는 변경하지 않는다.
