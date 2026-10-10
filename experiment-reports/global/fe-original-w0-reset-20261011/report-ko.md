# FE 원저자 W0-fixed 전환: 실제 등록 및 정리 영수증 통합

Parent `USER-FE-ORIGINAL-W0-RESET-20261011-R1`, 후속 `ROOT-FE-ACTUAL4-QWEN-STORAGE-20261011-R1`.
GH는 기존 owner 영수증을 통합했으며 신규 dispatch/Slurm 제출·취소·삭제는 수행하지 않았다.

## 실제 실행과 미제출

| 서버 | 모델 / dataset | 실제 job name / ID | 관측 상태 | dependency |
| --- | --- | --- | --- | --- |
| SH1 | Llama CF | official-s1-llama3-cf-fe-original / 63151 | RUNNING | 없음 |
| SH1 | Llama zsRE | official-s1-llama3-zsre-fe-original / 63152 | PENDING / Dependency | afterany:63151 |
| SH1 | GPT-J CF | official-s1-gptj-cf-fe-original / 63153 | PENDING / Dependency | afterany:63125 |
| SH1 | GPT-J zsRE | official-s1-gptj-zsre-fe-original / 63154 | PENDING / Dependency | afterany:63153 |
| SH2 | Qwen CF | 없음 | RESOURCE_BLOCKED_STORAGE_KEEP_SOURCE / 미제출 | 미등록 |
| SH2 | Qwen zsRE | 없음 | RESOURCE_BLOCKED_STORAGE_KEEP_SOURCE / 미제출 | 미등록 |

SH1 등록 당시 네 job은 PENDING이었다. 위 RUNNING/Dependency는 2026-10-11 후속 nonce의 요청자 squeue readback이며 정확 관측 시각은 전달되지 않았다. GH의 신규 scheduler 조회나 실험 완료 주장이 아니다. SH1 [원 제출 영수증](../../../audits/servers/server1/fe-original-w0-2k-20261011/submission.json)의 실제 ID/name/config/dependency와 대조했다.

실행 source `03b21d404d6275b5f212974fe130ca98819789de`, official tree `7bdebfa8edcd812716a492160f8895b0d8865543`, 저자 commit `478134dfb24b43f4e18b47e8500893ce3f9cc50f`. GPU1/CPU8/98304MiB/48h 각 job, combined cap2. 비-FE63125는 보존한다. BF16/저자 기본 attention backend이며 Llama CF resolved backend는 요청자 관측상 sdpa다. 기존 FP32 실험과 동일 runtime이라고 표시하지 않는다.

GH는 Llama CF local `tracking/receipt.json`의 READY_ONLINE, job63151/source/config와 startup readback identity를 확인했다. [새 W&B run](https://wandb.ai/wkdguswns2256/layer%20allocation/runs/2dd26c6040cd43ae)은 startup 확인이며 W20/전체 history/최종 점수 검증이 아니다. 다른 세 job의 online 상태는 확인하지 않았다.

SH1 checkpoint 예정 경로는 `local/fe-original-w0-2k-20261011/runs/<llama3|gptj>-<cf|zsre>/checkpoint/latest.pt`. W0 전체2K z 선계산/고정, current100 매batch, 누적500/1000/1500/2000 평가와 실행별 latest atomic overwrite를 유지한다. host 공용 lock은 저장만 직렬화하며 GPU 두 lane을 직렬화하지 않는다. CF generation DEFERRED, zsRE public-query loc_ans requestmacro. CPU15/config 검산은 실제 GPU/W20 성공을 뜻하지 않는다.

## SH2 저장공간 blocker

[source-ready-r1](../../../audits/servers/server2/fe-original-w0-2k-20261011/source-ready-r1.json), main `a326582b`, own source `e64e557b`: 공통03b21d40 결속, CPU15 및 실제 Qwen config/schema2 PASS, 전체2000/24858 query input/target mismatch0. 이전 SOURCE_INPUT_PENDING은 해소됐다. 실제 실행 archive freeze/등록은 미완료다.

공용 host lock으로 two latest + one tmp + z/raw + 32GiB reserve 계획을 유지해도 필요60,363,309,056B 대비 admission 가용59,029,065,728B로 **1,334,243,328B 부족**하다. 요청자 후속 df 58,890,551,296B는 별도 후시점 관측이며 최초 admission에 대입하지 않았다. 신규 job/held/release0, reserve 축소·추가 삭제·자동 retry0. Qwen 두 표 셀 묶음은 미제출 공란으로 유지하고 숫자나 PENDING ID를 만들지 않았다.

## 삭제와 보존

- SH1: 11개 / 54,211,593,935B. `checkpoint-deleted.json` 6개 + `native-checkpoint-deleted.json` 4개 + `corrected-qwen-deleted.json` 1개를 합산했다.
- SH2: `delete-after.json` 6개 / 23,008,049,946B.
- 합계 **17개 / 77,219,643,881B**는 삭제 payload 논리 bytes다. filesystem 가용 증가량과 동일하다고 주장하지 않는다. 지정 파일 부재/backup 없음은 owner 영수증이며 이 작업 내 복구사본은 없다. raw/log/config/source/소형 과거 수치는 보존했다.
- SH3 main `ddd2e99d`, SH4 main `f7ab8015`: 등록 metadata/지정 root의 bounded inventory에서 FE-bound replica0. 전체 filesystem 부재 증명이 아니며 추가 대형 검색/전송/삭제를 하지 않았다.

[SH1 보고](../../servers/server1/fe-original-w0-2k-20261011/report-ko.md) · [SH2 보고](../../servers/server2/fe-original-w0-2k-20261011/report-ko.md) · [SH3 제한 조사](../../servers/server3/fe-original-w0-2k-20261011/replica-report-ko.md) · [SH4 제한 조사](../../servers/server4/fe-original-w0-2k-20261011/replica-report-ko.md).

README 수정은 FE 행의 실제 등록 상태와 관련 설명만이며 비-FE/PRICE/W0 셀은 불변이다. 철회된 FE 수치는 [기존 이력](withdrawn-results.md)에 남기고 새 결과로 복사하지 않았다. 완료 성능0, 신규 반복 monitor0.
