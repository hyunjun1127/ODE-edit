# RTE scoring 교정 및 체크포인트 인수

2026-09-09 KST. GH `ODEEDIT-GH-RTE-LABEL-FIX-20260909-R1` 승인.

원본 BLUE 311b076a 및 AlphaEdit dataset bytes를 변경하지 않았다. RTE의
semantic True(1)를 GLUE entailment(0), False(0)를 not_entailment(1)로
**scoring boundary에서만** 변환한다. 생성/alternative branch를 각각 기록하며
invalid는 유지한다. raw prediction, semantic prediction, canonical prediction,
raw gold 및 mapping version을 별도 저장한다. 원본 역매핑 metric은 bug diagnostic이며
corrected main과 섞지 않는다. 이는 method 성능 개선 claim이 아니다.

CPU RTE fixture 6/6 PASS: 양 gold의 True/False 정오판정, invalid 보존,
perfect/inverted ACC/F1/MCC, 두 branch 일치, 입력 불변 및 unknown alternative.
독립 재해시: source 100, evaluator 13, import 보완 2, dataset 10 PASS.
72 checkpoint 파일 62,011,141,768 bytes의 크기/SHA도 SH2 독립 PASS.
물리적 GPU 복원/평가 validity는 아직 주장하지 않는다.

Receipt (local-only):
`/mnt/raid5/janghj/ODE-edit/local/blue-checkpoint-downstream/20260909-v1/rte-corrected-payload-verification-v1.json`

MMLU parser는 원본 그대로이다. bare `A`와 `A\n`의 차이 및 invalid를 보존한다.
generation weighted-F1과 alternative weighted-F1을 분리한다. 각 task
rows[10:110] 100개, fewshot0, greedy gen_len5이며 공식 full benchmark가 아니다.

전송 대상은 완료 6 chains × 12 checkpoints이다. L567 NOT_READY와 native
baseline NOT_AVAILABLE은 별도 유지한다. downstream W0는 한 번 평가하며
CounterFact PRE_EDIT 평가를 대체로 사용하지 않는다. SH4는 sole transfer writer,
SH2 rsync=0. PRE_EDIT 42673 조회/변경/모니터링 재개=0.

전용 adapter는 원본 6 class를 직접 import한다. 사용하지 않는 GLUEEval wrapper와
util import는 호출하지 않으며 13+2 원본 파일을 모두 provenance lock에 포함한다.
원본 helper의 상대 dataset 경로만 안전한 primitive loader/절대 allowlist로 bind한다.
모델의 `_name_or_path`는 canonical Llama 이름으로 metadata binding하여 원본
BOS 제거와 4096 context lookup을 유지하고, 실제 snapshot 경로/revision을 별도 기록한다.

선택 weight만 W0에 덮어쓰고 전체 parameter의 pointer/version/byte identity를 검사한다.
각 CP 후 전체 W0를 검증하고 다음 CP를 적용한다. cache_c/RNG/context는 추론에 적용하지 않는다.
수학적/성능 promotion은 false. cap2, 1GPU/process, 60416MiB, 기존 job 변경0.
