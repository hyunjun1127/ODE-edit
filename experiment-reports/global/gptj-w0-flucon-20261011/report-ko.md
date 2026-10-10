# GPT-J W0 Flu/Con 확인 및 server4 평가

Nonce `USER-GH-GPTJ-W0-FLUCON-SERVER4-20261011-R1`. 사용자는 기존 결과/등록이 없으면 server4에서 평가하고 main 표에는 job name이 아닌 **job 번호**를 표시하도록 지시했다.

## 기존 결과 확인

SH2 accepted turn `01a126e6-c989-7583-91bf-f7bb3bb6e493`, main `b12be3a9`의 [한정 inventory](../../servers/server2/gptj-w0-flucon-20261011/report-ko.md)를 확인했다. 기존61723은 cold W0 factual READY이며 generation=null/DEFERRED_NOT_MEASURED다. 이전396개 cold 관측은 EOS-corrected/no-cache 프로토콜이고98/558개는 부분 edited-state다. 동일 native full2000 W0로 합치지 않았다. 현재 해당 owner queue의 동일 W0 등록0; 기존62864..76은 edited W20 평가라 제외했다. 전체 filesystem 부재 증명은 아니다.

기준 revision `47e169305d2e8376be1d31e765533382721b2cc1`, tokenizer identity `a435eae8f10bebbd17d64ab60736c80dfe41c4946a2407dc89539c59a45ea7e5`, CF stream `66edc483a8d4bcadedd479e4c36759a686ad61a38741d8870a9052b795710e37`, reference identity `75e595c7f26ec334830e9bb9ca6028098c19ea84a9509a5713985847683f8ea6`.

## 공통 기록 경로 및 담당 수락

SH4 accepted turn `01a126e6-c815-7ba1-8a01-f186b8f18c4e`, OWNER_ACK 확인. SH2는 read-only inventory만, SH4는 조건부 generation-only 1개 실행을 소유한다. GH는 README sole writer다.

기존 W&B schema가 W20 edited-checkpoint eval-only만 허용하여 main `884c913e`에서 이번 server4/GPT-J/CF W0-only authority를 좁게 추가했다. profile `cf-native-generation-W0-only-v1`, schedule `W0_ONLY_FIRST2000`, `base_model_sha256` 필수/edited checkpoint SHA 금지. W0 generation·진행률·edits0만 기록하고 factual/fit/W20/zsRE는 거절한다. SH4의 같은 accepted turn에 exact source를 steer했고 READY 수락을 읽기 전용 확인했다. transport 연결의 ACK 이벤트 미수신과 이후 명시 owner 수락 확인을 구분한다.

공통/새 W0 schema/기존 SH1·SH2 W20 eval tests **CPU56 PASS**, source166 SHA/Python363/import0 PASS. production worker→fake SDK 검사를 포함하며 실제 모델/GPU/온라인 검증이 아니다. 공통 native case-batch KV/total100/global-RNG/noEOS/seed20261007 프로토콜을 유지한다. 최초 envelope의 row RNG 표현은 canonical native profile과 혼동되지 않도록 정정했다. 원 bits/cosine 및 표의 반올림 전 평균×100/half-up2 분리는 그대로다.

## 실제 등록

SH4 main `2b58c6fe`의 [실제 제출 영수증](../../../audits/servers/server4/gptj-w0-flucon-20261011/submission.json)과 [owner 보고서](../../servers/server4/gptj-w0-flucon-20261011/report-ko.md)를 확인했다.

- **63219**, `s4-gptj-W0-flucon-2k-r1`: held owner/source/config/argv/입력/resources/tracking/noCP/dependency 검사 후 release. **2026-10-11 02:49:05 KST** 단발 관측 PENDING, `afterany:63144` 미충족.
- 실행 source `9d2420afc9dfa3574973109a9de3b3518ee8b1ee`, canonical config SHA `4bc1905379d3a972734b47f677bf7228e75354ceeccbf32fdcab6ae4ee71020e`.
- 1GPU/8CPU/59392MiB/48h, QoS lab_gpu_s4, exportNONE/requeue0. lane `[63144→63219]`와 `[63145]`로 cap2, 기존 job 변경0. 관측된63220은 GPU0 CPU allocation으로 별도 구분했다.
- SH4 CPU9/source166/Python365/import0 PASS. base model manifest SHA `40fbd3f57fadd84f104c40fe2b33b5ee188149470b34c854f252ab06960b55a1` 및 tokenizer/stream/reference는 위 SH2 identity와 결속. 모델 payload fullSHA는 **실행 시 model load 전에 강제 검사 예정**으로, 등록 전 완료된 검산으로 표시하지 않는다.
- 출력 `/data/janghj/ODE-edit/local/gptj-w0-flucon-20261011/execution-r1/output`. W&B run ID/online startup·최종 값은 아직 미관측. noCP, raw local, tensor/raw upload0, NO_BROADCAST_NOT_REQUIRED.

README GPT-J W0 CF Flu/Con 두 셀만 **`PENDING: 63219`**로 갱신했다. job name은 상세 보고서에 남기고 표에서는 사용자 정정대로 번호만 사용한다. 기존 W0 factual/zsRE 및 모든 다른 표 셀은 불변이다. 완료값0개, 모델 편집/새 checkpoint 삭제/새 반복 monitor/자동 retry0. GPU 평가 완료까지 기다리지 않고 실제 등록·표 게시로 인계한다.
