# 미제출 Qwen 원저자 FE 두 조건의 SH1 재배정

USER-FE-QWEN-SERVER1-MOVE-20261011-R1. parent USER-FE-ORIGINAL-W0-RESET-20261011-R1.

| 조건 | 실제 job | 등록 직후 | resource dependency |
|---|---:|---|---|
| Qwen CF | 63217 (`official-s1-qwen25-cf-fe-original`) | PENDING | afterany63152:63154:63207 |
| Qwen zsRE | 63218 (`official-s1-qwen25-zsre-fe-original`) | PENDING | afterany63152:63154:63207 |

실제 held owner/Command/WorkDir/full argv/source/config/자원/의존성/script를 검사하고 두 job을 release했다. 각 GPU1/CPU8/98304MiB/48h, devbox/gpu/lab_gpu_s1, exportNONE/requeue0. 48h는 상한이며 ETA가 아니다. source `2bac5732fc594e0eaf0d200772a56a640e02bc21`, official tree `360eb1c8e419c52a8e871cfb0a52641de873aa16`. `03b21d40` 공용 runner/author patch는 byte 불변이며, 병행 main W0 eval-only tracking 확장은 검토·caller 재검사했다. 기존 63151–63154 frozen source/config는 수정하지 않았다.

Fresh admission 때 기존 GPU 3개(63125/63151/63207)와 기존 admitted DAG 폭3을 확인했다. 기존 할당/대기 의존성은 변경하지 않는다. 새 두 head 모두 기존 전체 GPU 말단63152/63154/63207 뒤로 연결해 기존 초과할당 구간에 추가 실행하지 않으며, 이후 새 실행 폭은2이다. 전체 역사 DAG 폭3을 cap2 PASS로 거짓 표기하지 않는다.

디스크 available540,832,616,448B, 요구135,741,833,216B. 요구에는 기존4+신규2 latest 추정39,804,993,536B, host 공용 lock 하의 최대 tmp1개7,890,010,112B, z2GiB/raw16GiB/reserve64GiB가 포함된다. 실제 serialize 시 공용 writer가 크기·여유를 다시 검사한다. reserve 축소/다른 파일 삭제 없음.

기존 SH1 Qwen HF snapshot revision `a09a35458c702b33eeacc393d103063234e8bc28`, tokenizer·C0 L4..8·CF/zsRE stream을 read-only fullSHA 검산했다. SH2의 소형 config/asset manifest만 읽어 model/tokenizer members·C0 및 stream SHA를 대조했다. fullmodel/C0 전송·다운로드·재추정·old CP 복원 없음. CF fileSHA `66edc483a8d4bcadedd479e4c36759a686ad61a38741d8870a9052b795710e37`, zsRE `f42ee4bc6e98b1133e48dc81102201ccd85ccc917a97160ee3c6a0f3a38f378c`.

Qwen 원저자 YAML BF16/default attention/L4..8/clamp1/steps35/lr.5/decay.001, add_old_keys=True/L2=0. 독립 cold W0/H0, native W0 contexts·2000 fixed z, BS100x20/seed0. 공용 원저자 clone/전용 venv 재사용; runtime 차이는 기존 author-lock에 기록한다. CF current100 R/P/N 및 누적500/1000/1500/2000, zsRE public-query requestmacro loc_ans. CF FLUCON deferred, zsRE 생성 없음.

CPU19 PASS(공용15+재배정4), 실제 Qwen 두 caller/schema 검사 포함. zsRE2000/24858 query input/target mismatch0, token분모 E6691/G6691/Loc11476. 이는 CPU query 검산이지 실제 pretrained/GPU 출력 동일성 증명이 아니다. 별도 GPU smoke/qualification 없음. 등록 당시 startup/새 W&B URL/과학 완료 미관측이다.

configSHA CF `0e91094c3ea95949fcea63a1e3814690ded4cc9e58972a5d949a1cf47f5f8121`, zsRE `7680c5e6b8843795d2fac07d08c1800703c2f1661f6fa797c66d56afd106e86f`.
경로 root `/mnt/raid5/janghj/ODE-edit/local/fe-original-w0-2k-20261011/qwen-server1-move-r1/`: preparation/configs, registration-r1, runs/qwen25-{cf,zsre}. 각 run의 checkpoint/latest.pt는500단위 overwrite, z-cache/factual/commits는 별도보존. host lock은 기존4와 동일 `/mnt/raid5/janghj/ODE-edit/local/fe-original-w0-2k-20261011/checkpoint-serialization.lock`.

SH2 직접 OWNER_ACK(2026-10-10T17:46:15.480435+00:00): task submission receipts0/queue0/accounting0, SH1 단독 소유, 자동재시도0. 이는 SH2 제공 독립 확인이며 SH1이 원격 scheduler를 다시 검증했다고 주장하지 않는다. 이전6조건 중 미제출2의 이전이며 신규8조건이 아니다. README는 GH sole writer. NO_BROADCAST_NOT_REQUIRED, raw/weights Git0, 새 삭제0. 직접 인계 상태는 delivery.json에 별도 기록한다.
