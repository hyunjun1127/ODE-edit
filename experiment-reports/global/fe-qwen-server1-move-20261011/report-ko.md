# Qwen FE 두 조건 server1 재배정 — main 표 통합

Nonce `USER-FE-QWEN-SERVER1-MOVE-20261011-R1`, parent `USER-FE-ORIGINAL-W0-RESET-20261011-R1`. 사용자 원문 “server1에 올리자”. 기존6조건 중 미제출 Qwen2개의 host 변경이며 추가 과학조건/중복실험이 아니다. GH는 제출을 다시 지시하지 않고 SH1 main8953c047의 [실제 영수증](../../../audits/servers/server1/fe-qwen-server1-move-20261011/submission.json)과 [보고서](../../servers/server1/fe-qwen-server1-move-20261011/report-ko.md)를 확인해 통합했다.

| 조건 | owner | 실제 job | 전달된 관측 | dependency |
| --- | --- | --- | --- | --- |
| Qwen CF | SH1 | 63217 / official-s1-qwen25-cf-fe-original | PENDING / Dependency | afterany:63152:63154:63207 |
| Qwen zsRE | SH1 | 63218 / official-s1-qwen25-zsre-fe-original | PENDING / Dependency | afterany:63152:63154:63207 |

두 job 모두 held owner/source/config/full argv/자원/입력/dependency 검산 후 release했다. 요청자 후속 readback도 Dependency PENDING이며 정확 snapshot 시각은 전달되지 않았다. GH는 추가 scheduler polling을 하지 않았다. 각1GPU/8CPU/98304MiB/48h, QoS lab_gpu_s1, exportNONE/requeue0. 기존63125/63151/63207 GPU3와 역사 DAG 폭3은 유지되며 cap2로 재명명하지 않는다. 두 신규 head는 기존 전체 GPU 말단 종료 뒤 시작하여 새 실행 폭2를 보장한다. 기존63151–63154의 source/config와 비FE 작업 변경0.

실행 source `2bac5732fc594e0eaf0d200772a56a640e02bc21`, official tree `360eb1c8e419c52a8e871cfb0a52641de873aa16`. 공용03b21d40 bytes와 저자478134df/native YAML/BF16/W0-fixed firstforward/history 계약 유지. CF canonical configSHA `0e91094c3ea95949fcea63a1e3814690ded4cc9e58972a5d949a1cf47f5f8121`, zsRE `7680c5e6b8843795d2fac07d08c1800703c2f1661f6fa797c66d56afd106e86f`.

SH1 현물 model/tokenizer/C0/stream SHA는 SH2와 일치하며 대형전송/재계산0. storage available540,832,616,448B 대비 required135,741,833,216B(6latest+hostlock tmp1+z/raw+reserve64GiB)로 승인 계획을 충족한다. SH2의 과거 부족1,334,243,328B는 이 SH1 실행 blocker가 아니다. 실제 저장 시 여유 재검사와 atomic overwrite 유지.

출력 root `/mnt/raid5/janghj/ODE-edit/local/fe-original-w0-2k-20261011/qwen-server1-move-r1/runs/qwen25-{cf,zsre}`, checkpoint는 각 `checkpoint/latest.pt`. current100 매batch/누적500·1000·1500·2000 all_seen 및 최신 checkpoint overwrite, CF Flu/Con DEFERRED, zsRE public-query loc_ans requestmacro. CPU19 PASS/전체2000·24858 query input/target mismatch0는 GPU 또는 W20 성공 증명이 아니다. W&B startup/URL·성능은 미관측.

SH2 accepted turn `01a126eb-c792-7cd0-b7e3-d6550a4c4cea`의 독립 확인은 신규제출0/SH1 단독소유/자동재시도0이다. GH가 SH2 scheduler를 새로 확인했다고 주장하지 않는다. SH2 source-ready/assets/audit는 보존한다.

README Qwen FE9셀만 갱신(CF factual4: PENDING63217, generation2: DEFERRED, zsRE3: PENDING63218). 기존 FE 철회 수치·비FE·PRICE·W0·GPT-J W0 Flu/Con63219와 모든 다른 표 셀은 불변이다. 신규 GPU 제출/취소/삭제/전송/반복 monitor는 GH가 수행하지 않았다.
