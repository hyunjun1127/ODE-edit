# Llama·Qwen native baseline 2k checkpoint 확인

사용자 지시 `USER-GH-LLAMA-QWEN-BASELINE-CHECKPOINT-REUSE-20261009-R1`에 따라 2026-10-09 GH와 두 read-only explorer가 과거 inventory와 현재 server2 파일을 대조했다. 아래 다섯 파일은 현재 regular file 존재·정확한 크기·**전체 SHA256 일치**를 확인했다. 모델/tensor 역직렬화·GPU/Slurm·전송·삭제·generation 실행은 0이다. 전체 SHA는 파일 보존의 증거이지 새 runtime 복원 또는 GPU parity 증거가 아니다.

| 모델 | 방법 | endpoint | bytes | 현재 보존 |
| --- | --- | --- | ---: | --- |
| Llama3-8B-Instruct | MEMIT | W2000, B20 | 1,174,421,709 | server2, full SHA 일치 |
| Llama3-8B-Instruct | AlphaEdit | W2000, B20 | 5,284,839,565 | server2, full SHA 일치 |
| Qwen2.5-7B-Instruct | MEMIT | W2000, B20 | 1,357,922,509 | server2, full SHA 일치 |
| Qwen2.5-7B-Instruct | AlphaEdit | W2000, B20 | 8,535,425,249 | server2, full SHA 일치 |
| Llama3-8B-Instruct | AlphaEdit-BLUE | W2000, B20 | 2,113,956,769 | server2, full SHA 일치 |

## 네 native checkpoint의 정확한 위치

공통 server2 root는 `/mnt/raid5/janghj/ODE-edit/local/results/official-layer-realization-debt-lifelong-b100-v1/`이다. 원 server4 경로와 옛 `.incoming-server4-20260918-r1` 경로는 현재 없고, 아래 최종 경로가 보존돼 있다. 원본 경로 부재를 checkpoint 소실로 판단하지 않았다.

| 모델/방법 | root 아래 상대 경로 | SHA256 |
| --- | --- | --- |
| Llama/MEMIT | `campaign-20260902-four-arm-completion-tech-r1/llama3-8b-inst-memit-lifelong-b100x100/checkpoints/state-02000.pt` | `600e2260953e2f048df84f6b2d23705eeb111e8d288c28bc1eb1760b246ff26e` |
| Llama/AlphaEdit | `campaign-20260902-four-arm-completion-tech-r1/llama3-8b-inst-alphaedit-lifelong-b100x100/checkpoints/state-02000.pt` | `23a9af78e8f236cca1e076843a50d67bcb854733242114d8eb008eb978d92708` |
| Qwen/MEMIT | `campaign-20260901-tech-r1/qwen2.5-7b-inst-memit-lifelong-b100x100/checkpoints/state-02000.pt` | `0fd28a6b952a0ff71d5802a529ee7473eee0d9fcde08a5a93776dcc519d566b8` |
| Qwen/AlphaEdit | `campaign-20260901-v1/qwen2.5-7b-inst-alphaedit-lifelong-b100x100/checkpoints/state-02000.pt` | `62a574c5ec949741d369e6133a9b38e7c983c7a7928cebec1f3dafa40a954058` |

각 run의 `result.json`과 `journals/checkpoint-02000.json`도 현재 존재한다. 과거 native EasyEdit 계보의 source는 Llama 두 방법과 Qwen MEMIT `85a05d0a`, Qwen AlphaEdit `583aaa3f`다. PRICE/ours writer checkpoint가 아니다. 기록상 선택된 다섯 W, native cache/history, RNG, cursor와 source/stream identity를 저장한다. full pretrained model 전체를 담는 형식은 아니므로 원 base revision와 원 cohort/reference 자산을 함께 결속해야 한다. 이번에는 payload를 load하지 않았다.

Base revision은 Llama `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`, Qwen `a09a35458c702b33eeacc393d103063234e8bc28`이다. 과거 hparams는 Llama LR .1/norm .5/clamp .75/25평가/MEMIT15000/AlphaL2=1, Qwen LR .5/norm .001/clamp4/25평가/MEMIT15000/AlphaL2=1이다. 현재 실행 계약에 조용히 덮어쓰지 않는다.

**Cohort 차이가 있다.** 과거 CF 원본은 21,919 rows, SHA `d017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f`, native lifelong stream은 salted-hash 선택·기존 요청/충돌 배제·order seed index1이다. 현재 official `existing_file_first2000/edit_seed0` 또는 PRICE의 fixed10k first2000과 같다고 하지 않는다. 과거 first2000 ordered request hash는 `e220355ad9419ea93942be368804bd26b2a04747454076940206e8e063a7fad7`, 현재 존재하는 원 stream seal full SHA는 `5848dca7323335483039ec125b644dadfb87b0996133f9825f951088cf3a7861`이다. 이후 fluency/consistency는 원 cohort에 대한 checkpoint 평가로 원 provenance를 보존하고, 새 평가 코드·reference·generation 정책은 별도 기록한다.

## BLUE

Llama AlphaEdit-BLUE의 정확한 현재 경로:

`/mnt/raid5/janghj/ODE-edit/local/blue-checkpoint-downstream/20260909-v1/imports/initial6-v1/payload/local/blue-lifelong-b100x100/attempt-checkpoint-r2/output/main-cell-1/B020/W-method-state.pt`

SHA256 `268cf596e963b3455a694ccb2b7599b97ee483ad091cadcc051c185358fcc675`. 원 job `39283_1`, L4+L8, L2=1, seed20260907, transformers4.44.2. 선택 W2개와 `[2,14336,14336]` history/context/RNG를 저장한 과거 BLUE baseline이다. 원 fixed10k SHA `3d7f5e31…`, order root `5b013569…`이며 위 native 네 checkpoint와도 cohort를 무조건 합치지 않는다. 정확한 복원/GPU 검증은 아직 수행하지 않았다.

Qwen BLUE는 지정 BLUE/migration inventory에서 확인하지 못했다. 과거 [BLUE 1k 보고서](../../servers/server4/blue-alphaedit-fivearm-sequential1000-review-2026-09-07-v1/factual-report-ko.md)는 Qwen model config 미지원으로 미실행을 명시한다. Qwen JVP/L8-only/PRICE checkpoint는 BLUE가 아니다. 이는 전체 filesystem에 Qwen BLUE가 없다는 전역 부재 증명이 아니다. 현재 Qwen BLUE L2/clamp 비교를 이 Llama checkpoint로 대체할 수 없다.

## 계획 반영

사용자 지시대로 Llama·Qwen × MEMIT/AlphaEdit의 **신규 CF 편집 네 chain을 제외**하고, 기존 checkpoint 기반 fluency/consistency 후속 관측으로 분리한다. 이번 확인에서 관측 job을 제출하지 않았다. CF checkpoint로 zsRE 편집을 대체하지 않으며 FT/FE/SPHERE/GPT-J 및 무관한 job은 변경하지 않는다. BLUE는 보존 확인 결과를 제공한 것이며 신규 BLUE chain을 자동 취소하지 않았다. 기존 checkpoint 중앙 보존 정책은 forward-only이므로 이 파일을 이동·삭제하지 않았다.

원 보존 근거: [Sep18 checkpoint index](../checkpoint-location-index-2026-09-18-v1/checkpoint-index.csv), [BLUE inventory](../../../audits/servers/server4/2026-09-09-blue-downstream-transfer/checkpoint-inventory.csv). 새 현재 파일 검산은 위 정확한 최종 server2 경로의 stat와 sha256sum이다. 실제 복원·GPU parity·fluency/consistency 완료는 모두 NOT_OBSERVED다.

정본 게시 commit `348d6fb3`. SH1은 Llama MEMIT CF 제외, SH3는 Qwen MEMIT/AlphaEdit CF 제외를 직접 nonce ACK했다. SH4는 Qwen PRICE Tier2의 무관 active turn이어서 직접 전달을 보류했으며, 정본 게시를 수신 ACK로 간주하지 않는다. [전달 영수증](../../../audits/global/native-baseline-checkpoint-reuse-20261009/direct-delivery.json)에 accepted turn과 보류 사유를 분리했다. 신규 GPU 관측 job은 제출하지 않았다.
