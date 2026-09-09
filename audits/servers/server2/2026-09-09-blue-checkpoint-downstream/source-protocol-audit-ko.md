# BLUE checkpoint downstream — source 확인 기록

- instruction: `ODEEDIT-S06-BLUE-CHECKPOINT-DOWNSTREAM-S2-S4-V1`
- 날짜: 2026-09-09 KST
- 상태: `SOURCE_PROTOCOL_UNRESOLVED`; downstream 제출 0, 측정 0.
- 사용자 원문: “server4에 존재하는 각 checkpoint를 바탕으로 server2에서 downstream task 측정을 진행하고 싶다. easyedit에 존재하는 sst, mrpc,cola,rte,mmlu,nli 측정 하는 것 진행해라. checkpoint 자체는 server2로 rsync로 전달해서 task 진행하자”

## source / 환경 경계

origin/main을 fetch하여 확인한 HEAD는 `6fd7f1482c395b9ea7271c15c94967120dffca7e`, tree는 `474e7afba1c809bc59ba0935425e1033c678deda`다.
전용 branch는 `codex/server2-blue-checkpoint-downstream-v1`, worktree는 `/mnt/raid5/janghj/.codex/worktrees/odeeditsh2-blue-checkpoint-downstream-v1`다.
최신 PROTOCOL은 1269 lines / 60382 bytes / SHA256 `af806a449be800251393bfcd81b2dfa5689ee34305f3bf1323e0fae82f16c87b`다.
기존 root의 untracked `agents/server2/`와 EasyEdit 사용자 변경은 보존했다.

실제 EasyEdit root는 `/mnt/raid5/janghj/EasyEdit`이며 HEAD `3488a66ee988d83ee7891a8abbbe6bcb24a77daf`, tree `1f6d5e9a4a95daa15a4b5a8963dcd133876e47b3`다. HEAD/tree는 Git 기준이며 현재 dirty 파일 bytes와 동일하다고 주장하지 않는다. 선택한 파일은 아래 actual SHA로 구분한다.

| 관측 파일 | SHA256 | 관측 내용 |
| --- | --- | --- |
| `EasyEdit/README_2.md` | `abc6913a0144148b519d2f3116f47ba3f7c8cb3b212a438a4004dc2f34c3828f` | SST2 2-label 및 MMLU 57-subject 설명; 이 설명만으로 평가 계약을 확정하지 않음 |
| `EasyEdit/hparams/Steer/dataset.md` | `2511d7cc5363e7475295c0f966f2bcf910f0ddb410499879e82fa537aa01cc29` | steering 학습/생성 데이터 형식 안내 |
| `EasyEdit/hparams/Steer/dataset_format.yaml` | `4ff0839b2758011cf7fe315a3f12784566241f873fe0f761d1092a708f1736b6` | `train.sst2_pair/label`, `generation.mmlu` 경로 선언 |
| `EasyEdit/examples/run_LLM_evaluation.py` | `1ce249a68f8feb367b67316fa24579f16aaff11b93f9ed7c0497abe4c8b88fbb` | KnowEdit 및 editor.edit 실행 경로; 이번 downstream-only 실행에 사용하지 않음 |

`easyeditor/` 및 `examples/`의 Python/shell에서 정확한 SST/SST2/MRPC/CoLA/RTE/MMLU/NLI/SNLI/MNLI/ANLI/GLUE 단어 검색으로 요청된 6-task launcher를 확인하지 못했다. `data/mmlu`, `data/sst2*`, `steer/evaluate/evaluate.py`는 현재 filesystem에 없다. 파일 부재를 임의로 복원하거나 사용자 삭제를 되돌리지 않았다.

## 미결정 평가 계약

| 요청 task | 상태 | 미확정 항목 |
| --- | --- | --- |
| SST | SOURCE_PROTOCOL_UNRESOLVED | SST2 채택 여부, dataset/config/split, prompt/label/extraction, metric |
| MRPC | SOURCE_PROTOCOL_UNRESOLVED | accepted evaluator, dataset/split, prompt/label/extraction, F1 averaging/accuracy |
| CoLA | SOURCE_PROTOCOL_UNRESOLVED | accepted evaluator, dataset/split, prompt/label/extraction, MCC/F1/accuracy |
| RTE | SOURCE_PROTOCOL_UNRESOLVED | accepted evaluator, dataset/split, label mapping, metric |
| MMLU | SOURCE_PROTOCOL_UNRESOLVED | accepted evaluator, subjects/split, few-shot/example/order, aggregation/extraction |
| NLI | SOURCE_PROTOCOL_UNRESOLVED | SNLI/MNLI/ANLI 등 정확한 dataset/config/split, label mapping, metric |

모든 task에서 seed, decoding, max tokens, batching, dtype/backend 및 invalid/tie 처리는 source-backed lock 확정 전 미결정이다. Figure6의 F1 표기를 수식으로 간주하지 않는다. 외부 harness/BLUE harness로 대체하지 않는다. 정확한 source 또는 명시적 protocol 선택을 GH에 peer-direct 요청했다.

## checkpoint 전달 경계

SH4 peer 보고: 완료6chain(39307 및 39283_1..5)의72CP를 우선 inventory한다. 각 method MEMIT/AlphaEdit × BLUE/L4/L8 × edits100/500/1000/2000/3000/4000/5000/6000/7000/8000/9000/10000이다. 이는 SH4의 관측으로서 SH2 payload 독립 검증 완료를 뜻하지 않는다.

SH4 보고상 L56740441/40442는 RUNNING,40443..40446은 PENDING이므로 이번 initial inventory에서 NOT_READY; native42657/42658은 NOT_AVAILABLE다. SH2는 해당 job이나 live scientific output을 조회하지 않았다.

SH4가 단독 transfer writer다. SH2 landing `/mnt/raid5/janghj/ODE-edit/local/blue-checkpoint-downstream/20260909-v1/imports/`는0700/empty로 확인했고 available1690470416384 bytes /445705926 inodes다. 기존 동일 identity는 없다. 고유 partial staging→전체 해시 검증→create-once seal에 동의했다. SH2 중복 rsync writer0, 원본 W0 반복 전송0. 전송 완료/CP SHA 검증은 아직 주장하지 않는다.

원본 checkpoint는 full model이 아니다. 실제 serializer와 tensor key/shape/dtype/SHA를 읽은 뒤 동일 W0에 selected weights만 overwrite하는 복원 adapter를 작성한다. method-state M은 provenance이며 forward에 적용하지 않는다. restore correctness와 평가 지표 확정 전 downstream GPU 제출0.

## 기존 PRE_EDIT 유지

별도 task의42673은 첫 B100 검증 후 `MONITORING_PAUSED_AWAITING_GH`다. W0 guard PASS,166 forward,2600 raw target rows,edit/backward0. 전체10000 완료를 뜻하지 않는다. 실행 설정/모델/평가 hook 변경0, 취소0.

신규 downstream admission은 기존 allocation을 포함한 server2 cap2,1GPU/process,60416M/GPU를 적용한다. 현재 downstream queue는 미제출이다. 확인되지 않은 실행 시간·점수는 추정하지 않는다. 총 기본 범위는72CP×6tasks+공유W0×6tasks=438 checkpoint-task evaluations이며 L567 추가는 봉인 inventory가 준비된 경우만 별도 결속한다.

scientific_promotion=false. 실제 downstream data는 CounterFact edit provenance와 별개이며, CounterFact PRE_EDIT RS/PS/NS를 downstream W0 결과로 대체하지 않는다.
