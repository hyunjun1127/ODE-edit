# SH1 FE author 본표 통합 입력

nonce `USER-GH-S1-S2-FE-AUTHOR-MAIN-REFRESH-20261010-R1` 수신 turn에서 OWNER_ACK 후 수행. 별도 accepted turn UUID는 추정하지 않았다.
정본 main8d717f7b 및 envelope FULL_READ/SHA `0fedadab3c031f82efd3237ffc48ed0d21125a7e2269fb896004e4ee6b7aa187` 일치.
own session01a04939-f93a-7b50-bca0-65438eab2062, origin hyunjun1127/ODE-edit, root/mnt/raid5/janghj/ODE-edit. 전용 codex/server1-author-main-refresh-20261010에서 작은 audit/report만 게시. README는 GH sole writer.

## 새 완료 / 진행 상태

| 모델·방법 | dataset / job | 상태 | Eff | Gen | Loc | Score |
|---|---|---|---:|---:|---:|---:|
| Llama MEMIT-FE (FE author hparams + history) | CF 62529 | W20 CPU 검산 완료 | 99.7 | 95.55 | 79.215 | 90.58056682126457 |
| 같은 author variant | zsRE 62530 | RUNNING, 14 commits 관측, W20 미완료 | — | — | — | — |

CF source9c3fe23282bcaf2e353b1494913a323e2f037d39, profile SHA1ffbf9adeec9525b88db3d86a6dc4ace8b98bdaff1d5eb4e4ee064c24b35b125. clamp.75/steps35 등의 실제 config/hparams를 compact 행에 보존했다.
20×100 원 request hash, 각 commit identity/history append 및 연속 state, final cursor, ordered stream, 원 factual raw의 query/token/NLL/strict success/분모를 CPU 검산했다. 최종 W20 checkpoint fullSHA423b5dadc3027c73d642caa47eaae30c45859f676a55c69edf5f9cccea347e83 일치. 모델/CP 역직렬화·forward 없음.
author CF Flu/Con은 **DEFERRED**이며 future consumer pending/CP KEEP. 기존 native FE 61773 및 generation62261의 점수를 author로 relabel하지 않는다. 기존 native FE CF/zsRE 행에는 LEGACY_NATIVE_FE_NOT_AUTHOR를 표시해 별도 보존한다. GPT-J author 생성 없음.

Qwen FE_HISTORY의 기존 generation62583도 완료/collector62584 완료 확인. raw Flu5.323883976741921 bits / Con0.0041069385104137695 cosine, 표시 **532.39 / 0.41**. 각각 valid2000, planned2000. exact endpoint/COMPLETE/config/source/원CP fullSHA 및 reference-bound CPU collector 검산 증거를 재사용했다. 이는 native FE/author와 별도 history 행이다.

## 기존 결과 검산

기존 완료12행의 endpoint/config/terminal SHA 불변 확인. 공개-query zsRE61932..37은 frozen evaluator/query/public lock/stream 증거를 결속하고 저장 predicted-target correctness로 요청별 평균→전체2000요청 평균을 다시 계산했다. E/G/Loc token분모6035/6035/12465, 기존 수치와 일치. W0agreement/tokenmicro로 대체하지 않았다. CPU query 검산은 실제 pretrained forward parity 주장이 아니다.
기존 FT/SPHERE/native FE 및 GPTJ/Llama history generation raw는 기존 검산과 SHA 불변으로 재사용. generation raw/W&B 변경0, 표시만 반올림 전 평균×100 후 half-up2. 전체14행 중 완료13행/author zsRE 미완료1행이다.

한 차례 accounting에서 own 기존완료와62529/62583/62584 COMPLETED,62530 RUNNING을 확인했다. 실제 actor source/config/raw로 완료를 검산하여 scheduler COMPLETED만으로 값을 만들지 않았다. 이후 반복조회/완료대기 없음. 관측은 2026-10-11 KST 수신 turn 초반이며 JSON at는 CPU 검산 완료 영수증 시각이다.

`audits/servers/server1/author-main-refresh-20261010/table-rows.json`, CSV, inventory.json과 reduce.py에 exact source/config/profile/cohort/raw/CP 증거 및 분모를 제공한다. 신규 GPU/제출/취소/hold/dep/복원/전송/삭제/online backfill0; cap2/기존 jobs/dirty root/frozen source KEEP. NO_BROADCAST_NOT_REQUIRED.
