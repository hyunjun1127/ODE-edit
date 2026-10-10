# 원저자 FE W0-fixed-z 전환 / SH1

인계 상태: main 보고 게시 `18508a72`. SH2 source/API 전달은 공식 transport 접수, 요청자 최종 인계도 turn/steer 접수(`01a1268d-44aa-7e80-aa1e-6342ad7ebd0b`). 별도 owner ACK/과학완료를 의미하지 않는다. GH 최종 direct 전달은 bounded timeout으로 COMMUNICATION_HOLD/수신 미확인이며 자동 재전송하지 않았다. GH가 읽을 exact 제출 영수증은 아래 main 게시 경로에 보존했다.

USER-FE-ORIGINAL-W0-RESET-20261011-R1. 실행 source `03b21d404d6275b5f212974fe130ca98819789de`, official tree `7bdebfa8edcd812716a492160f8895b0d8865543`. README 통합은 GH 소유이며 이 보고는 완료 성능이 아니다.

| 조건 | 실제 job | 등록 직후 상태 | resource dependency |
|---|---:|---|---|
| Llama CF | 63151 | PENDING | 없음 |
| Llama zsRE | 63152 | PENDING | afterany63151 |
| GPT-J CF | 63153 | PENDING | afterany63125 |
| GPT-J zsRE | 63154 | PENDING | afterany63153 |

네 job 전량 held 상태에서 owner/Command/source/full argv/config/자원/의존성/script를 확인한 후 release했다. 기존 비-FE 63125를 보존하며 총 DAG 폭2. GPU1/CPU8/98304MiB/48h, devbox/gpu/lab_gpu_s1, exportNONE/requeue0. 48h는 요청 상한이지 ETA가 아니다. 실제 GPU·W&B online·W20 완료는 등록 영수증에서 미관측으로 표시한다.

저자 clone은 `/mnt/raid5/janghj/ODE-edit/local/fe-original-w0-2k-20261011/author`, commit `478134dfb24b43f4e18b47e8500893ce3f9cc50f`. 공식 patch는 미사용 missing import/lazy CLI import/재생 hook Tensor·tuple·list 호환만 변경한다. native fit/key/residual/solve/H/write는 저자 `FE-memit_main.batch_edit`를 그대로 호출한다. 저자 W0에서 native contexts와 전체2000 layer-wise z를 편집 전에 생성하여 고정한다. 기존 FE_HISTORY writer/old context/old checkpoint를 사용하지 않는다.

BF16 raw-byte 해시, 저자 기본 attention backend 유지 및 resolved backend 기록, authoritative latest.pt 자체 identity/cursor/integrity 검증, advisory latest.json crash-gap 복구, resume별 별도 raw 경로를 구현했다. 실제 current100 CF raw를 공통 reducer로 집계해 current/post R/P/N 및 official summary를 기록한다. all_seen과 current의 분모/관측은 섞지 않는다.

CPU 15 tests PASS, 추가 실제 payload 경로 fixture current100/all_seen500/all_seen2000 schema PASS. BF16 hash, 고정 YAML/author patch, W0 z 단일호출, host lock 직렬화, overwrite/CPU restore, stale/missing metadata 및 손상 거절, resume raw 충돌 방지, cap2 DAG를 포함한다. 별도 GPU qualification은 NOT_RUN_USER_DISABLED이며 CPU 검산을 pretrained/GPU 동일성 증명으로 표시하지 않는다.

Runtime는 전용 venv: Python3.12, 기존 torch2.9.1+cu128 read-only 재사용, transformers4.51.3/tokenizers0.21.4/OmegaConf2.3.0. Python3.12 호환상 Hydra1.3.2/NumPy1.26.4 사용은 저자 pin 대비 공개한 차이이며 수학 수정이 아니다. 기존 model/tokenizer/C0/stream fullSHA를 결속했고 재다운로드/통계 재계산은 없다.

저장: `/mnt/raid5/janghj/ODE-edit/local/fe-original-w0-2k-20261011/runs/<llama3|gptj>-<cf|zsre>/checkpoint/latest.pt`. 500/1000/1500/2000마다 원자적 overwrite, W0 CP 없음. host 공용 `checkpoint-serialization.lock`을 tmp 생성 전 획득하여 fsync/replace/directory fsync까지 보호한다. 각 run latest는 별도로 유지한다. CF generation은 DEFERRED, zsRE는 public-query loc_ans request-macro이며 generation 없음.

기존 FE writer62530을 정확 취소했다. 소유 config/source에 결속한 이전 FE checkpoint **11개 / 54,211,593,935 bytes**를 개별 unlink하고 부재를 확인했다. 백업/이름변경 보존 없음; 삭제 파일의 이 작업 내 복구본은 없다. raw/log/config/source는 보존했다. 정확 삭제 전 manifest와 후 receipt는 동명 own audit에 있다. 다른 archive/results/source-handoff의 소형 metadata 점검에서 추가 FE-bound replica는 식별하지 못했으며, 이름만으로 무관한 파일을 삭제하지 않았다.

등록 상세: `local/fe-original-w0-2k-20261011/registration-r1/{submission,held-inspection,admission,execution-lock}.json`. compact 수치는 `audits/servers/server1/fe-original-w0-2k-20261011/submission.json`. 공용 source/API는 SH2에 direct 전달했으며 SH2의 실제 제출은 SH2 소유로 별도 확인한다. NO_BROADCAST_NOT_REQUIRED: 모델/raw/CP 전송 없이 소형 source와 영수증만 Git 공유. 장기 polling/자동재시도 없음.
