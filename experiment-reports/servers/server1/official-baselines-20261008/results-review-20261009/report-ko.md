# Llama official 완료 결과 CPU 리뷰

지시 `USER-GH-S4-S2-QWEN-BASELINES-MIGRATION-RESULTS-20261009-R1`. SH1 결과 리뷰만 수행했으며 Qwen 이전/취소/등록에는 관여하지 않았다. README 편집은 GH 소유로 남겼다.

봉인된 각 실행 source에서 audit_factual/audit_commits를 읽기 전용으로 사용했다. 실제 config/source/model/stream/ordered occurrence/token 계획, raw NLL·정답 bit·request macro/토큰 분모를 재계산하고 20개 native commit 및 선택 가중치/history hash 연결, W20 latest pointer와 실제 checkpoint 파일 SHA를 확인했다. 모델/텐서 역직렬화·복원·forward·GPU·새 fit은 0. 기존 별도 GPU qualification은 NOT_RUN_USER_DISABLED 그대로다.

## 신규 W20 factual 검산 완료: 8개

| dataset | method | job | Score | Efficacy | Generalization | Specificity |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| CF | FT | 61771 | 56.586626 | 89.25 | 73.30 | 35.50 |
| CF | SPHERE | 61770 | 86.870970 | 99.50 | 94.95 | 71.675 |
| zsRE | FT | 61716 | — | 14.835139 | 11.991627 | 1.424254 |
| zsRE | MEMIT | 61717 | — | 46.707817 | 42.067381 | 37.248325 |
| zsRE | AlphaEdit | 61718 | — | 98.737778 | 94.596528 | 51.221902 |
| zsRE | AlphaEdit-BLUE | 61719 | — | 99.412738 | 95.848571 | 66.686652 |
| zsRE | MEMIT-FE | 61720 | — | 14.947381 | 13.711012 | 0.965650 |
| zsRE | SPHERE | 61721 | — | 98.517123 | 94.718710 | 50.220332 |

각 endpoint는 ordered first2000/20batch, 실측 requests=2000이다. CF Score는 원 E/G/S의 조화평균이며 AlphaEdit 표시용 반올림 Score는 JSON의 별도 필드다. CF Flu/Con은 미측정 DEFERRED이고 0으로 채우지 않는다. zsRE는 teacher-forced request macro이며 Specificity는 W0 예측 일치율이다. loc_ans 정답률은 JSON/CSV에 별도 저장하고 README Loc에 대입하지 않는다. 이 CPU 리뷰로 온라인 remote 검증을 대체했다고 주장하지 않는다.

## 유지/미완료

- CF MEMIT-FE 61773은 RUNNING, 리뷰 시점 commit14/20이며 W20 terminal이 없다. 완료값 없이 상태만 제출한다.
- CF AlphaEdit61769/MEMIT61772는 CANCELLED이며 과거42657/42658의 사용자 승인 표 예외를 유지한다. 이번 새 W20 결과로 표시하지 않는다.
- CF BLUE 역사39283_1 예외도 유지한다. 복원/추가 평가는 없다.
- 옛 failed/cancelled 시도·raw·CP·source·비용은 모두 KEEP. GPU0collector61774는 미완료 chain을 기다리므로 8건 검산을 전체 pipeline 완료로 표시하지 않는다.

CF 실행 source34e4d52d821113d56ee6c59b71d553dd01768e73, zsRE 실행 source94304dc93db928300a71d8dcd0f87199fdb93f1f. 각 config/source/endpoint/terminal/checkpoint metadata SHA와 실측 분모는 `audits/servers/server1/official-baselines-20261008/results-review-20261009/results.json`, README용 행은 `table-rows.csv`에 있다. raw/CP는 local-only, NO_BROADCAST_NOT_REQUIRED. 별도 독립 agent 없이 기존 독립 CPU reducer를 owner가 실행했다. 주기 monitor 없음.
