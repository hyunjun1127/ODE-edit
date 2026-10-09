# 저장된 zsRE W20의 최종 평가 전용 API

권한: `USER-GH-ZSRE-SAVED-WEIGHTS-2K-REEVAL-20261009-R1`.
기존 실행의 가중치·frozen source·raw를 변경하지 않는다. 새 편집, W0,
중간 milestone, generation은 실행하지 않는다. 평가 전용 source/identity를 봉인한다.

```python
from official.evaluation.zsre_query_parity import compare_queries
from official.evaluation.zsre_paper import build_queries, evaluate
proof = compare_queries(tokenizer, records, model_family="gptj")  # CPU, 전체2K
result = evaluate(model, tokenizer, records, model_family="gptj",
                  batch_size=16, device="cuda:0", identity=binding,
                  progress=callback)
```

family는 `llama3`, `gptj`, `qwen25`이다. tokenizer는 봉인된 revision, right padding,
native pad/eos 설정을 사용한다. records는 공식 **미확장** stream의 전체2K이며
neighborhood 항목 하나가 `loc + '?'`와 전체 `loc_ans`를 갖는다. 이미 token별로
확장된 public loader 결과를 다시 넣지 않는다. 원 stream 파일 SHA와 query의
정규 JSON SHA는 다른 항목이며 둘 다 기록한다.

`build_queries`는 local-only case/query 문자열·ID와 query SHA를 반환한다.
`compare_queries`는 봉인된 공개 dataset loader AST의 neighborhood 구성과 공개
evaluator 함수 AST를 독립 실행한다. prompt/target 문자열, batch tokenizer 입력
ID 및 실제 채점 target ID를 모두 exact 비교하며 LM forward는 없다.
Llama의 Eff/Gen i>0 공백과 decode-retokenize, Loc loader의 BOS 포함 prefix도
공개 코드 그대로 적용한다. Llama BOS는 명시 검증 후 처리하고 Qwen에는 slice를
하지 않는다. Qwen은 공개 non-Llama 분기를 적용한 정의이지 논문 Qwen 재현 주장이 아니다.
원문 파일/URL/SHA는 `zsre_public_sources/lock.json`에 고정했다.

`evaluate`는 복원 완료한 모델의 next-token logits를 right-pad full-prefix로
계산한다. batch_size는 평가 물리 microbatch이며 fit/데이터는 바뀌지 않는다.
summary의 Efficacy/Generalization/Specificity는 요청 내부 token 평균 후 요청 간
평균 ×100, `Specificity_loc_ans`는 같은 값 alias다. W0agreement는 산출하지 않는다.
cases는 predicted/target ID와 correctness를 포함한 local-only 자료다.
token_denominators는 세 그룹의 실제 token 수다. 부분 진행은 완료 점수로 쓰지 않는다.

모델 tensor/version/hook/config/training/RNG guard를 적용한다. caller는 별도로
base/CP/stream/tokenizer full identity, W20/2000 완료, 적용한 tensor manifest,
checkpoint 파일 불변을 검증한다. CP를 GPU로 전부 적재하거나 H/C0/P/편집 engine을
로드할 필요는 없으며 manifest의 edited tensors만 정확 복원한다. 제한된 모델
파라미터만 저장한 CP를 전체 모델로 오인하면 안 된다. 원 CP는 수정·삭제하지 않는다.

CPU query parity와 CPU fixture는 실제 pretrained forward 결과의 bitwise parity를
의미하지 않는다. 평가 batch 연산 순서에 따른 수치 차이 가능성을 기록하며, 입력
mismatch 개수를 성능 차이로 쓰지 않는다. 새 실제 관측만 최종 표에 반영한다.

## W&B

`official.tracking` 하나를 사용한다. 새 config는 기존 공통 필수 필드 외에:

```text
instruction_id=USER-GH-ZSRE-SAVED-WEIGHTS-2K-REEVAL-20261009-R1
dataset=zsre
role=eval_only
metric_schema=official-baselines-scalar-v1
evaluation_profile=zsre-public-query-W20-only-v1
checkpoint_sha256/evaluator_sha256/stream_sha256/tokenizer_sha256=<봉인 SHA>
source_run_id=<원 job/run 식별자>
```

원 metadata를 새 실행으로 relabel하지 않는다. model/family/writer/실제 Slurm job ID,
새 run UUID/name 및 bounded finish/readback은 기존 transport 규칙을 유지한다.
generation 설정은 금지. 새 관측은 `official_zsre_metrics(result['summary'],
config_values=cfg, endpoint='all_seen/post', edits=2000, post_state_edits=2000)`로 기록한다.
진행률 callback은 `eval_progress/<completed_queries|total_queries|physical_forward_calls|elapsed_seconds>`
숫자만 별도 log한다. 이는 endpoint 성능이 아니며 final scalar와 섞지 않는다.
평가 전용 config는 W0/current/intermediate/fit/generation 지표를 거부한다.
