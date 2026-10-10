# Native context 순차 생성 등록

nonce `USER-GH-SH4-NATIVE-CONTEXT-ORDER-20261010-R1`, accepted turn `01a12156-29cb-7bc3-9bc1-75e0d132de43`.
main066395e1 정본 전체를 읽고 exact session/CWD/origin 및 기존 own nonce 기록 부재를 확인했다. dirty root는 변경하지 않았으며 `codex/server4-native-context-order-20261010` 전용 WT에서 준비했다.

| 순서 | model | job | dependency | 관측 단계 |
|---|---|---:|---|---|
| 1 | qwen25 | 62090 | afterany:62064 | RELEASED / PENDING |
| 2 | gptj | 62091 | afterok:62090 | RELEASED / PENDING |
| 3 | llama3 | 62092 | afterok:62091 | RELEASED / PENDING |

실행 source는 `a8a68592` (freeze.json의 full commit), 구현 `0f7a75da`.
root: `/data/janghj/ODE-edit/local/native-context-order-20261010/execution-r1`.
각 모델 config/source/model manifest SHA, 실제 job name/argv와 held inspection은 `audits/servers/server4/native-context-order-20261010/submission.json`에 있다.

산출물은 `outputs/{qwen25,gptj,llama3}/contexts.json`, `context-token-ids.json`, `READY.json`이다. 현재 실제 생성 및 context SHA는 **NOT_OBSERVED**이며 등록을 생성 완료로 표현하지 않는다. READY는 model manifest/원 weight fullSHA 검증/source/native module/generator SHA/token IDs/context SHA/실제 job/runtime를 결속한다. context/token 원문은 ignored local에만 둔다.

## 생성 정의

official native MEMIT registry 경로 및 `official/hparams/contract.json`의 seed **0**을 사용한다. context profile은 native 다섯 prompt(The/Therefore/Because/I/You), top-k5, total10, 5개 template와 bare `{}`이다. Qwen/GPT-J는 corrected EasyEdit generator SHA `35506690c41ecb7d59f11660da41dde50338f5a2ba9613735e045a8c7ab98db4`, Llama는 기존 정상 sphere/MEMIT generator를 readonly 사용한다. 모델별 native hparams SHA를 별도 결속하며 모델 간 token ID/context를 공유하지 않는다.

이 seed0/profile은 새 입력 자산의 명시적 identity이다. 기존 heldout62063/62064 seed20261002 context와 같은 것으로 relabel하거나 기존 job에 주입하지 않는다. 다른 writer/runtime와의 무조건 호환 또는 pretrained numerical parity를 주장하지 않는다.

- Qwen revision `a09a35458c702b33eeacc393d103063234e8bc28`
- GPT-J revision `47e169305d2e8376be1d31e765533382721b2cc1`
- Llama revision `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`

모두 기존 local HF snapshot 사용. 모델 shard SHA는 content-addressed blob identity로 봉인하고 **runtime fullSHA 검증 후** 로드한다. 준비 단계 모델 weight fullSHA 검증 완료로 과장하지 않는다. 작은 tokenizer/config 파일은 CPU fullSHA로 결속했다. 새 다운로드/asset 전송 없음.

## 검산·자원·보존

CPU syntax/order PASS, official source166/Python326/externalimports0 PASS. 새 GPU qualification/fit/edit/평가/W0 forward/FluCon은 0. 실제 model load와 forward는 승인된 context 생성에만 사용한다. FP32/eager/TF32off, 모델 비편집 version/pointer guard 포함. model/effort 사용자 설정 변경 없음.

각 GPU1/CPU8/59392MiB/2h/QoS lab_gpu_s4/exportNONE/Requeue0. Context는 단일 lane이며 기존 62037→62063→62064 resource tail 뒤에 이어진다. 보호 HELD62038이 나중에 사용자에 의해 풀려도 context1+OURS1≤cap2다. 기존 62063/62064/62038 및 다른 job의 hold/source/config/dependency를 바꾸지 않았다. GPU가 비기를 agent가 기다리지 않고 정상 dependency로 인계한다. 앞 모델 생성 실패 시 다음 afterok는 풀리지 않으며 자동 retry 없음.

생성 원문 출력은 local job log/JSON에만 보존한다. 이는 input preparation이며 새 scientific W&B run이나 performance metric을 만들지 않았다. 실제 생성 성공은 READY와 terminal job 상태를 나중에 함께 확인해야 한다. 새로운 monitor/heartbeat는 없다. NO_BROADCAST_NOT_REQUIRED: compact source/receipt만 Git 게시; 기존 source/context/pack/lock/model/CP 보존.
