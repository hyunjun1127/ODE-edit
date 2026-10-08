# Generation 수리 및 GPT2XL 입력 보완

사용자의 2026-10-08 수리·재등록 지시에 따른 기록이다. 이전 frozen source/raw는 보존하고 새 attempt로 cold2k를 시작한다. NoCP이므로 이전 prefix에서 재개하지 않는다.

## 확인과 변경

기존61207/61208 caller는 `UNPADDED_FULL_PREFIX_NO_CACHE`였으며 요청별 prompt를 하나씩 생성했다. CAKE 고정 source는 KV cache와 prompt batch를 사용한다([generate.py](https://github.com/zjh-vinky/CAKE/blob/c8243e1d7e43ca9cf64d552f96221fcb9561aac2/util/generate.py), [evaluate.py](https://github.com/zjh-vinky/CAKE/blob/c8243e1d7e43ca9cf64d552f96221fcb9561aac2/experiments/evaluate.py)). 기존 schedule은 current20×100에서 네 milestone을 누적으로 대체하여 6,600 request-endpoint이고 final-only는 2,000이다. 이는 평가 건수3.3배이지 실제 시간 배수 측정이 아니다.

이번에는 기존 generation 관측 일정/current·milestone 분모를 유지한다. 새로운 caller는 고정 MB4 KV 경로를 연결한다. B1 편집 뒤 기존 shared reference/singleton/batch 비교를 수행한다(최대8 prompt, 추가 fit0, W0아님). batch 검산 통과 시 batching, singleton만 통과하면 KV singleton을 명시한다. KV 자체가 불일치하면 느린 reference를 몰래 재사용하지 않고 기술 차단한다. seed/topk5/temperature1/max-total100/수치 허용오차는 변경하지 않았다.

공통 KV 구현은 GPT2/GPTJ로 model-family guard가 제한돼 있다. 공유 소스는 수정하지 않았고 task-private adapter에서 실제 Llama family만 추가했다. 모델 타입을 속이지 않는다. native cache/attention/positions/sampling을 재사용하며 task-private observer import만 결속했다. shared source SHA와 diff 범위는 [owner audit](../../../../../audits/servers/server4/price-ridge-m1-m3-2k-20261008/generation-repair.json)에 명시했다. 실제 Llama GPU 정합은 아직 NOT_OBSERVED다.

## 기존 job 보존 및 취소

61209 collector를 먼저, 이어61207/61208을 exact ID로 취소했다. 당초 취소 응답은 collector CANCELLED/GPU COMPLETING이었다. 새 admission 시 해당 GPU allocations가 없어야 등록하며, 기존 baseline60917–60923 hold는 유지한다. 별도 모니터/자동 retry는 없다.

## GPT2XL

L14–L17 C0 4개(655,364,920 bytes)는 server1의 기존 EasyEdit 파일을 exact SHA 검산해 복사했다. L13/model/projector/tokenizer/context/20pack은 기존 server4 자산을 재사용했다. 새 통계 계산이나 model 다운로드는 없다.

기존 취소된 L13 attempt에 같은 server4 GPU/runtime의 완료 W0 2,000개·26,000행이 있었다. 원 전체 모델은 아직 편집되지 않은 상태였고 선택층 W/H 표기만 L13이었다. 동일 full-model asset과 모든 원 편집층 cold W hash를 결속하고, 새 L13–L17 editor state와 원 관측 state를 별도 기록한다. 원 raw는 변경하지 않는다. CPU 독립 reducer 통과; 실제 startup의 device/runtime/cold hash 불일치면 새 W0를 실행하지 않고 차단한다.

M1+M2는 L13–L17/anchor17/lr.5/native ridgeλ20000/grace9로 준비했다. M3 두 arm은 저장 B1 K가 없고 본 B1 RAM K 사용에 대한 추가 답변을 기다리는 상태다. 수리 요청만으로 ‘저장 K만 사용’ 과학 제약을 임의 해제하지 않았다.

## 검산과 제출 경계

좁은 CPU 검사8개, AST/import, 세 모델 MB4 token coverage, GPT2 raw reducer가 PASS다. 실제 모델/GPU parity·속도·W&B remote 전달 PASS는 아니다. 사전 고정3셀의 저장량 계획41,716,285,440 bytes, 준비 시 free69,280,104,448 bytes다. 동시 GPU cap2, job당59392MiB/hard60416MiB, CPU8, wall48h 상한. 신규 W0/qualification job0, post-B1 검산은 각 main 안에 포함한다.

실행·source·job IDs는 제출 후 별도 receipt로 추가한다. 기존 source8f39b226와 이후 수리 source를 구분한다. own branch 게시만 허용하며 main 통합은 GH 절차다.
