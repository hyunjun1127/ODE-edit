# 지정 dataset + BLUE evaluator 검산

2026-09-09 KST, `ODEEDIT-S06-BLUE-CHECKPOINT-DOWNSTREAM-S2-S4-V1`.
이 기록은 최초 source 부재 기록의 후속이며 이전 기록을 덮어쓰지 않는다.

## 확인 결과

- 데이터10개 파일: SH2 독립 SHA/schema/label/order 검산 PASS. member root `e9328a5d351816cb9ba89454d228f7ade841c526313dae9c0d1a0e72a8ab00fc`.
- SH4 source100members 및 evaluator13members: 전체 크기/SHA 대조 PASS. checkpoint payload는 이 gate에서 열지 않았으며 독립 검증 완료를 주장하지 않는다.
- BLUE source HEAD/tree: `311b076a92e4ed0f14f5c8b4909732da781bc5f7` / `f3c933c31cba2fe979c5c34546a99a72e6beb763`.
- 원 AlphaEdit HEAD/tree: `b84624f44dfe8fc6cd9e41df916c44124a0c46dc` / `d853eeb4bc59a6ad32758cfb7d6eee40c21586ff`.
- 양 repo의 고정 Git object를 read-only로 대조했다. BLUE를 EasyEdit upstream이라고 표기하지 않는다.
- 상태: **EVALUATOR_SEMANTIC_HOLD — RTE 라벨 매핑**. 데이터 전송 부재 HOLD가 아니다. downstream 제출0.

## 데이터 범위

SST2 856, MRPC258, CoLA644, RTE262, MMLU1516, NLI2489 records.
각 primary 평가는 원본 배열 `[10:110]`의100개, fewshot0, gen_len5다. `[0:10]`은 reserved이며 이번에 사용하지 않는다. 이100개를 full split으로 부르지 않는다. 추가 dialogue/sentiment 데이터는 실행하지 않는다.
NLI label은 entailment/not_entailment 두 가지다. MMLU는 지정 subset이지 full57subject benchmark가 아니다. CounterFact fixed10k는 edit provenance일 뿐 downstream test data가 아니다.

## metric 및 파서

6개 task의 `f1`은 생성문 parser의 prediction으로 계산한 sklearn weighted F1이다. `f1_new`는 각 답 선택지의 teacher-forced mean token NLL에 exp(-NLL)을 적용한 확률 비교 prediction으로 계산한다. `mcc`는 generation prediction 기준이다. accuracy는 correct/total로 계산 가능하지만 F1과 같은 이름으로 혼합하지 않는다. invalid generation=-1을 제외하지 않는다.

| Task | label / prediction | 확인 사항 |
| --- | --- | --- |
| SST2 | negative0 / positive1 | 생성문 positive 우선 substring, alternative 동점은0 |
| MRPC | No0 / Yes1 | 생성문 yes 우선 substring, alternative 동점은0 |
| CoLA | No0 / Yes1 | 생성문 yes 우선 substring, alternative 동점은0 |
| RTE | source False0 / True1, raw integer label 직접 비교 | GLUE RTE entailment0/not_entailment1 의미와 반대 |
| NLI | entailment1 / not_entailment0 | 지정 binary 데이터와 내부 boolean 비교가 일치; MNLI3class 아님 |
| MMLU | A0/B1/C2/D3 | generation과 alternative prediction을 별도 유지; alternative 최대확률 동점=-1 |

GLUE RTE 공식 dataset card는 entailment=0, not_entailment=1을 명시한다: [NYU GLUE dataset card](https://huggingface.co/datasets/nyu-mll/glue/blob/main/README.md). 지정 파일의 공식 전체split 일치까지 이 사실로 증명하는 것은 아니다.

CPU에서 실제 source의 pure parser method를 추출해 실행한 결과:

- RTE `answer: True` →1, `answer: False` →0. source는 raw label과 직접 비교한다. entailment 답이 True인 경우 raw0과 비교되어 잘못 실패로 세어진다. alternative도 True 우세→1로 같은 매핑이다. 원 AlphaEdit도 동일 매핑이며 BLUE만의 차이가 아니다.
- NLI entailment→1 / not_entailment→0.
- MMLU `A` →-1, `A\n` →0, `The answer is B.\n` →-1. 줄바꿈을 요구하는 기존 parser의 invalid 경계를 기록한다. 이를 임의로 관대하게 수정하지 않는다.

필요한 결정은 RTE의 평가용 label/prediction 매핑 교정이다. raw pickle bytes/label은 보존하고, raw label과 scoring label을 별도 기록하는 좁은 adapter 변경이 가능하지만 GH 결정 전 실행하지 않는다. MMLU `f1`과 `f1_new`를 혼동하지 않으며 파서 변경 권한을 추정하지 않는다.

## 원 AlphaEdit 대비 byte/semantic diff

| 파일 | BLUE SHA256 | 원 AlphaEdit SHA256 |
| --- | --- | --- |
| glue_eval.py | 732020d450a40960efd206f9dfe7610229838d740c9f4995f8cea4b408400e58 | 82b78a82eb011caec34bb9b316dd3d10e35ad7c4238ac7dc78594cca4597b9b8 |
| sst_eval.py | 57f1c033f257128f4253f5e417938e10be40f1223c4536afad448dcc733f53cf | 5e141538122c61971af9cefd02bae89721f1c5d7ce0bbef3745cd5f08e579f2d |
| mrpc_eval.py | 5c1e5852bdcad3dc37023a03122a33de9fb7b02c906b386abbfac2fab6227ff4 | c198cf37b665d3877996a843276f6ca8ec3ec4a6f781f247e74336b207eeeddb |
| cola_eval.py | 8b9ba18b3dcb4b5c7583ea4812ff26636c22f8340143b2b2168b97c3a158550f | 966aa9ea2538f1b92e39b1629a635ec3c8a436f91d1b7335011b926fdca2e604 |
| rte_eval.py | 851e0be4314f49cc9d48d2a43f2371857011c737ab5b875ae374408022440a34 | 9784545b77724ffdb6954f76e4b7ee6b464acb09c9c7972df6464f9dedb90089 |
| mmlu_eval.py | e34dba1ceb7c281094f45c0679523131f11f8f9523439cf5edfcdd125835e854 | f18e66ff55b7d1a7fb5004a15e2c385f1a3148e100af88e65f7e3da6adf2fdc4 |
| nli_eval.py | 7207a3bdbad6c4820c60e7de92cd1de50e203bad4b0d5d4270b4a88b17d26fa9 | d6184b2065fb59d057ede1f83f41cd3dd50658168cf9e5481ad0449f39c1b6ee |
| useful_functions.py | ec16120cde06cd853681545300a84776987c4811f9eec837b324c7db854e89d4 | 8197b38318318d13e97ee67f4c3d796d1784d35fe028999c098545618a91c1e9 |

8/8 byte non-identical. 실제 diff는 wrapper의 개인 sys.path 추가 및 print 배너, 6task의 print_logs 조건/추가 lower(), helper의 Meta-Llama/GPT-J context-name alias 추가다. 비교한6task의 label/prompt/scoring 식 차이는 관측되지 않았다. 같은 RTE 매핑 문제를 물려받았다.

## portability / 복원 준비

Llama pinned tokenizer CPU 검사에서 ` True/False/A/B/positive/negative/Yes/No`는 BOS128000+answer token이었다. 기존 Llama BOS제거 경로와 일치한다. 다만 local snapshot의 basename은 revision hash이므로 native context lookup 및 Llama 분기가 canonical model identifier를 사용하도록 runtime metadata binding이 필요하다. numerical context limit4096은 임의 확대하지 않는다.

GLUE wrapper는 비실행 dialogue/sentiment/perplexity도 import한다. 제공 `util/__init__.py`는 미포함 `util.logit_lens`를 import하므로13file identity PASS를 완전 import-closure PASS로 부르지 않는다. 승인6task class만 직접 import하는 얇은 adapter로 이 비실행 의존성을 피할 수 있으며 evaluator 수식 수정과 구분한다.

72CP manifest의 key inventory는 physical layer map 및 FP32 `[4096,14336]`와 일치한다. source serializer/SH4 loader는 weights/cache_c/metadata schema를 사용하고 `weights_only=True`, selected key overwrite를 명시한다. actual payload 및 model restore는 아직 검증하지 않았다. M은 forward에 넣지 않는다.

## 재현 / 경계

CPU audit source: `project/run_scripts/blue_checkpoint_downstream/audit_inputs.py`.
실행: `python3 -B project/run_scripts/blue_checkpoint_downstream/audit_inputs.py --imports /mnt/raid5/janghj/ODE-edit/local/blue-checkpoint-downstream/20260909-v1/imports/initial6-v1 --dataset-receipts /mnt/raid5/janghj/ODE-edit/local/state/downstream-dataset-20260909-v1 --output <new-private-create-once-json>`.
실제 receipt: `/mnt/raid5/janghj/ODE-edit/local/blue-checkpoint-downstream/20260909-v1/source-data-semantic-audit-v1.json`, SHA256 `ace8ee88adf4626c6ee9443b8750c65edbc5a22337756694a93fd49e882013f9`.

신규 model/GPU/Slurm0, raw dataset mutation0, SH2 rsync0, checkpoint partial read0, PRE_EDIT42673 재조회/변경0. scientific_promotion=false.
