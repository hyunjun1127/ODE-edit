# Qwen FE author hparams CF/zsRE 등록

`USER-SH-FE-AUTHOR-HPARAMS-2K-CF-ZSRE-20261010-R1`, accepted turn `01a122f4-e240-7532-91fd-5295daaa902a`.

| Dataset | 실제 job | 초기 상태 | dependency |
|---|---|---|---|
| CF | s2-qwen25-cf-fe-author-history (62531) | PENDING, released | afterany:62081:62083:62085 |
| zsRE | s2-qwen25-zsre-fe-author-history (62532) | PENDING, released | afterany:62531 |

실행 source `a8c4c611`의 main 게시 후 official 전체를 별도 archive로 봉인했다. SH1 shared runner/context API를 그대로 호출하며 scientific fork는 없다. 원 FE baseline은 변경하지 않고 author profile의 clamp=1/steps=35만 실제 registry override로 전달한다. lr=.5/decay=.001/loss27/KL=.0625/C0=15000/L4..8, FP32/eager/TF32off, native history/FP64 single-system/CPU rollback은 유지한다.

각 job GPU1/CPU6/59392MiB/48h. cap4, 기존 GPU DAG 폭3과 정확 Command/WorkDir/owner/resource를 확인했다. 새 두 chain은 기존 frontier 뒤 직렬이다. held fullargv/script/source/config/input/dependency/resource 검사를 마친 뒤 release했다. 실행 중 작업과 62262/62263 취소 이력은 변경하지 않았다.

모델/C0 전체 SHA, C0 shape/count/finite, context `5c01bc1a91c0890af2897f18b7a5f95de011badc1eca5e27f44199acc0cb6515` 및 producer READY/config/tokenizer 실제 token ID를 검산했다. zsRE 2000 요청, 24858 query(E/G/Loc token수 6691/6691/11476)의 공개 query CPU 비교 mismatch0. 좁은 CPU 16개(별도 SPHERE lifetime 3개 포함), source166 SHA PASS. 이는 실제 GPU/성능/온라인 PASS가 아니다. GPU qualification은 NOT_RUN_USER_DISABLED.

CF factual W0 및 W5/10/15/20, Flu/Con DEFERRED. zsRE는 `official.evaluation.zsre_paper` request-macro loc_ans, 생성 없음. W0부터 매 committed batch latest checkpoint, 최종 W20을 보존한다. 두 번째 chain atomic 저장 중 첫 W20 포함 CP peak 예산 25,807,552,512B. 기존 작업 성장48GiB/관측8GiB/파일시스템32GiB 여유까지 총120,296,833,024B를 요구하며 당시 free159,858,565,120B로 충족했다. 256GiB 임의 gate나 cleanup은 없다.

W20 경로: `/mnt/raid5/janghj/ODE-edit/local/fe-author-hparams-2k-20261010/configs-r1/runs/{cf,zsre}/checkpoint/`. `latest.json` batch20/final_W20=true를 실제 완료 후 확인해야 한다. 현재 CP/완료/온라인 관측은 없으며 W&B는 dependency 대기 단계다. future consumer pending 상태로 보존하며 자동 전송/삭제는 하지 않는다.

정확 config/profile/source tree/Slurm 초기 snapshot은 `audits/servers/server2/fe-author-hparams-2k-20261010/{submission,table-rows,preparation,source-members}.json` 참조. README는 GH 단독 통합. NO_BROADCAST_NOT_REQUIRED: raw/CP/model은 local, compact source/receipt만 Git. 장기 monitor/자동 retry 없음.
