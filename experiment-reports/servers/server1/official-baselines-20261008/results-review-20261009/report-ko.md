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

各 endpoint は ordered first2000/20batch、実測 requests=2000。CF Scoreは生E/G/Sの調和平均で、AlphaEdit表示用round済ScoreはJSON内の別フィールド。CF Flu/Conは未測定DEFERRED、ゼロ補完なし。zsREはteacher-forced request macroで、SpecificityはW0予測一致率。loc_ans正答率は別フィールドとしてJSON/CSVに保存し、READMEのLocへ代入しない。オンラインremote検証をこのCPUレビューで代替したとは主張しない。

## 維持/未完了

- CF MEMIT-FE 61773はRUNNING、レビュー時点でcommit14/20、W20 terminalなし。完成値なし、状態のみ。
- CF AlphaEdit61769/MEMIT61772はCANCELLED、過去42657/42658のユーザー承認済み表例外を維持。今回新W20結果とはしない。
- CF BLUE歴史39283_1例外も維持。復元/追加評価なし。
- 旧failed/cancelled試行・raw・CP・source・費用は全てKEEP。現GPU0collector61774は未完了chainを待つため、この8件の検算を全pipeline完了とはしない。

CF実行source34e4d52d821113d56ee6c59b71d553dd01768e73、zsRE実行source94304dc93db928300a71d8dcd0f87199fdb93f1f。各config/source/endpoint/terminal/checkpoint metadata SHAと実測分母は `audits/servers/server1/official-baselines-20261008/results-review-20261009/results.json`、README用行は `table-rows.csv`。raw/CPはlocalのみ、NO_BROADCAST_NOT_REQUIRED。独立別agentなし、既存独立CPU reducerをownerが実行。周期monitorなし。
