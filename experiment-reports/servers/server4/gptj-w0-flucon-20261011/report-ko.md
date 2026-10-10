# GPT-J W0 Flu/Con 등록 영수증

- 사용자 nonce: `USER-GH-GPTJ-W0-FLUCON-SERVER4-20261011-R1`.
- accepted turn: `01a126e6-c815-7ba1-8a01-f186b8f18c4e`.
- SH2 bounded inventory 결과 동일 native full2000 W0 완료/등록 없음. 61723은 factual만 있고 generation은 DEFERRED. Legacy 396건 EOS-corrected 및 edited-state 부분 결과는 재사용하지 않았다. 전체 파일시스템 부재 주장은 아니다.
- 실행 source: `9d2420afc9dfa3574973109a9de3b3518ee8b1ee`. 공통 tracking READY `884c913e` 포함. CPU 9 tests PASS, source166/Python365/import0 PASS. 실제 GPU/온라인 검증은 아직 아니다.
- 설정 canonical SHA: `4bc1905379d3a972734b47f677bf7228e75354ceeccbf32fdcab6ae4ee71020e`.
- 실제 job **63219** (`s4-gptj-W0-flucon-2k-r1`): owner/argv/source/config/입력/resource/tracking/noCP/의존성 held 검사 PASS 후 release. 단발 상태 PENDING, `afterany:63144` 미충족. 정확 UTC 관측시각은 submission.json에 기록.
- GPU1/CPU8/59392MiB/48h/QoS lab_gpu_s4/exportNONE/Requeue0. 기존 63144·63145 KEEP. 두 lane `[63144 → 63219]`, `[63145]`로 combined cap2. 추가로 관측된 63220은 CPU allocation이며 GPU0. node-filter만으로 held job이 빠지는 것을 확인해 전체 own pending의 required_nodes까지 검산했다.
- model revision/tokenizer/ordered stream은 61723과 동일; reference identity도 동일. 기존 base weight manifest를 결속하고 실제 fullSHA 검사는 runtime model load 전에 강제한다. 모델 bytes 검산을 이미 완료했다고 주장하지 않는다.
- native casebatch/global endpoint RNG/noEOS/total100/seed20261007. Cold base W0 생성 1회로 두 지표 공동 산출; edit/fit/factual/zsRE/CP 복원·저장/별도 GPU qualification 없음.
- W&B 공통 logger의 W0-only schema 사용. 실제 시작 전 run ID/URL/online readback **NOT_OBSERVED**. SDK 접수와 원격 확인은 구분하며 startup/finish의 기존 bounded readback을 사용한다.
- output: `/data/janghj/ODE-edit/local/gptj-w0-flucon-20261011/execution-r1/output`; log: 같은 execution-r1의 `logs/63219.out`, `logs/63219.err`.
- GH README sole writer에게 두 W0 generation 셀만 `PENDING: 63219`로 전달. factual/zsRE/다른 모델 불변. 완료 점수는 없음; raw bits/cosine 및 최종 half-up x100 표시를 분리한다.
- 반복 모니터/자동 retry/기존 job mutation0. NO_BROADCAST_NOT_REQUIRED: 원자료 local KEEP, 작은 source/receipt만 Git 게시.
