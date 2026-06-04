# Superseded Prior Reflective Knowledge Editing Proposal

## 목적

이 문서는 새로운 repository의 `global-head`가 바로 연구를 이어받을 수
있도록 작성한 인수인계용 proposal이다. 이전 repository의 세부 실험 로그를
전부 읽지 않아도, 왜 새 방향으로 pivot하는지, 어떤 가설을 검증해야 하는지,
어떤 실험부터 시작해야 하는지, 어떤 결과가 나오면 중단해야 하는지를 한 번에
파악할 수 있게 하는 것이 목적이다.

이 proposal의 핵심은 다음 한 문장이다.

```text
Shallow parameter edit 이후에도 모델 내부에는 old prior와 그 주변 knowledge
structure가 남아 있다. 이 old prior를 단순 오류가 아니라 superseded prior,
즉 "새 지식에 의해 대체된 이전 지식"으로 명시적으로 활성화하면, 모델은 old
knowledge graph를 reasoning substrate로 사용하여 new fact의 ripple
consequence를 더 잘 구성할 수 있는가?
```

여기서 중요한 점은 이 방향이 최종적으로는 **in-context editing paper가
아니라 parameter-modifying knowledge editing paper**를 목표로 한다는 점이다.
In-context setting은 새 가설을 빠르게 검증하기 위한 diagnostic testbed일 뿐,
최종 방법의 identity가 아니다.

## 배경

초기 연구 목표는 reflection 기반 knowledge editing이었다. 기존
knowledge editing method가 `(s, r, o_old) -> (s, r, o_new)`를 삽입한 뒤,
모델이 reasoning 중 old prior를 꺼내면 스스로 reflection을 통해 new fact로
route를 바꾸게 만들고자 했다. 즉, knowledge edit 이후의 long trajectory가
단순 output이 아니라 edit utilization과 ripple reasoning을 보강하는 runtime
mechanism이 될 수 있는지 확인하려 했다.

이 과정에서 세 가지 방향을 실험하거나 검토했다.

첫째, E3-style self-validation과 GRPO 방향이다. Verification-rich trace를
SFT하거나 RL로 강화하면 모델이 첫 answer가 틀렸을 때 `Correct? No` 형태로
switching할 수 있을 것이라고 기대했다. 그러나 기존 결과는 generation path와
validation path가 잘 align되지 않는다는 쪽에 가까웠다. 모델은 edit query에서
new fact를 생성할 수 있어도, self-validation으로 그 fact를 일관되게 검증하지
못했다. 일부 oracle signal은 있었지만 Qwen 모델 및 제한된 조건에 의존했고,
method spine으로 삼기에는 약했다.

둘째, ROME/MEMIT/AlphaEdit 이후 long free generation에서 old -> new
switching이 자연스럽게 나타나는지 확인하려 했다. 이 방향의 병목은 기존
ROME-family method가 대체로 old fact를 new fact로 overwrite하는 방식이라는
점이다. Causal tracing 또는 middle-layer low-rank update는 new object를
강하게 만들 수 있지만, 모델에게 "이 old fact는 이제 outdated이고 new fact가
current belief다"라는 provenance structure를 명시적으로 주지 않는다. 따라서
reflection이 trigger되더라도 모델은 무엇이 old이고 무엇이 new인지 안정적으로
판단하기 어렵다.

셋째, in-context editing으로 전환하여 visible update가 있을 때 detector와
reflection을 runtime에 걸어보는 방향을 검토했다. 이 실험은 useful했다.
In-context setting에서는 retrieved fact가 prompt에 보이므로 edit scope와
utilization을 관찰하기 쉽고, ceiling도 빠르게 볼 수 있다. 하지만 이 방향을
그 자체로 method paper로 가져가려면 novelty가 runtime detector에 지나치게
의존한다. Multihop에서 decomposition 없이 hidden runtime state만으로
intermediate edit scope를 탐지하는 문제가 너무 어렵거나 선행 연구와 겹칠
수 있다는 위험이 컸다.

따라서 새 방향은 다음과 같이 정리한다.

```text
In-context editing을 최종 연구 주제로 삼지 않는다.
In-context setting은 "old prior를 superseded prior로 명시하면 ripple
reasoning이 좋아지는가"를 빠르게 확인하는 diagnostic stage로만 사용한다.

최종 연구 주제는 parameter 내부에 old/new dual state를 의도적으로 만들고,
runtime reflective exploration으로 shallow edit을 deeper belief
consolidation처럼 사용하게 만드는 knowledge editing method다.
```

## 핵심 문제의식

현실의 pretraining corpus에는 서로 충돌하는 지식이 존재한다. 예를 들어
"지구는 평평하다"와 "지구는 둥글다"라는 문장이 모두 corpus에 등장할 수 있다.
하지만 모델은 두 문장을 단순한 동일 confidence fact로만 저장하지 않는다.
많은 surrounding context가 "flat earth"를 outdated, historical, false
belief, ancient belief 쪽과 연결하고, "round earth"를 current scientific
belief 쪽과 연결한다. 즉, pretraining은 단순 `(s, r, o)` 삽입이 아니라,
그 fact가 속한 conceptual manifold와 provenance를 함께 학습한다.

현재의 lightweight KE method는 이 점을 거의 다루지 않는다. ROME-family는
shallow factual slot을 수정하는 데는 강하지만, old fact를 outdated prior로
남기고 new fact를 current belief로 배치하는 구조적 update에는 약하다. Anthropic
`Believe It or Not`/SDF 계열 논의에서 지적된 shallow edit 문제도 이 지점과
맞닿아 있다. 그들은 단일 prompt overwrite가 아니라 fact universe 자체를
증강하여 belief를 움직이려 했다. 그러나 SDF-style tuning은 KE의 원래 장점인
lightweight fact insertion과 거리가 멀다.

이 연구는 그 사이의 공간을 노린다.

```text
Parameter edit은 lightweight해야 한다.
하지만 edit된 fact가 기존 conceptual structure와 연결되지 않으면 ripple
reasoning이 약하다.
따라서 old prior를 지워버리는 대신 superseded prior로 남기고,
runtime reflection trajectory가 old graph를 재료로 new fact의 ripple
consequence를 구성하게 만든다.
```

## 핵심 가설

### H1. Superseded prior substrate hypothesis

New fact만 주는 것보다, old fact를 "이전에는 맞았지만 이제 대체된 prior"로
명시하고 new fact와 함께 제공하면 multihop/ripple reasoning이 좋아질 수 있다.

비교해야 할 조건은 다음이다.

| 조건 | 설명 | 목적 |
| --- | --- | --- |
| `new_only` | new fact만 제공 | 기본 visible update baseline |
| `old_only` | old fact만 제공 | old prior graph가 어떤 답을 만드는지 확인 |
| `old_new_no_label` | old/new 둘 다 제공하지만 provenance 없음 | 단순 conflict 제공 효과 |
| `superseded_prior` | old는 superseded, new는 current라고 명시 | 핵심 가설 |
| `scrambled_old_control` | old fact의 graph structure를 섞어서 제공 | old graph 자체의 구조적 기여 검증 |
| `explicit_graph_positive` | old graph와 new fact의 연결을 직접 자세히 제공 | upper bound |

핵심 성공 조건은 다음이다.

```text
superseded_prior > new_only
superseded_prior > old_new_no_label
superseded_prior > scrambled_old_control
```

여기서 `superseded_prior > new_only`만으로는 부족하다. 단순히 더 긴 prompt,
더 강한 instruction, 더 많은 token budget 때문에 좋아졌을 수 있다.
`scrambled_old_control`과 length-matched control을 반드시 둬야 한다.

### H2. Belief switching feasibility hypothesis

모델이 old prior를 먼저 활성화한 뒤 reflection을 통해 new belief로 switch할
수 있어야 한다. 만약 모델이 old answer를 한 번 말한 뒤 거의 절대 수정하지
못한다면, reflective exploration은 method로 성립하기 어렵다.

이 가설은 다음 형태로 확인한다.

```text
old prior trace prefix:
  The answer follows from old entity/path ...

reflection trigger:
  This was the superseded prior. Re-evaluate using the current update.

expected:
  model revises intermediate and recomputes downstream answer from new entity/path
```

측정 지표는 final answer만 보지 말고 `first_entity`, `final_entity`,
`first_old_to_final_new`, `downstream_recompute`, `repeated_final`,
`hit_limit`, `visible_tokens`를 함께 본다.

### H3. Parameter dual-state feasibility hypothesis

최종 method가 되려면 old/new dual state가 parameter 내부에 만들어질 수 있어야
한다. 단순 prompt context가 아니라, parameter edit 이후에도 old prior가
outdated/superseded 쪽으로 남아 있고 new fact가 current belief 쪽으로
활성화되는 상태가 필요하다.

초기에는 full method를 만들지 않는다. 먼저 다음 질문을 probing한다.

- Base model은 pretraining knowledge, finetuning knowledge, prompt knowledge의
  우선순위를 어떻게 처리하는가?
- Contradictory facts가 있을 때 모델 내부 hidden/logit/attention/probability
  dynamics에서 outdated/current 구분 signal이 존재하는가?
- ROME/MEMIT/AlphaEdit edit 이후 old/new가 동시에 존재하는 흔적이 있는가?
- Old fact를 "former/outdated/superseded" context로 넣을 때 hidden/logit
  trajectory가 일반 wrong answer context와 달라지는가?

### H4. Trajectory distillation hypothesis

In-context reflective exploration으로 효과가 확인된다면, 그 trajectory를
후처리 학습 데이터로 사용하여 context가 사라진 뒤에도 deep-edit-like behavior를
일부 유지할 수 있는지 확인한다. 후보는 SFT, DPO, GRPO 또는 lightweight
adapter distillation이다.

이 단계는 H1/H2가 통과된 뒤에만 진행한다. 초기부터 RL을 method spine에 넣으면
가설 의존성이 너무 커진다.

## Stage 0. Synthetic Diagnostic

가장 먼저 synthetic graph에서 가설을 검증한다. Natural benchmark는 entity
alias, dataset artifact, base model prior, retrieval/decomposition 문제가
섞이므로 첫 단계로 부적절하다.

### 데이터

Synthetic graph를 만든다.

```text
entity: E0001 ... E0100
relation: r_city, r_company, r_parent, r_teacher, r_award, r_country ...
old edge: (s, r, o_old)
new edge: (s, r, o_new)
ripple query: old/new object에서 이어지는 2-hop, 3-hop question
```

예시:

```text
Old graph:
  A's director is B.
  B's birthplace is C.
  C's capital landmark is D.

Update:
  A's director is now X. B is the superseded director.

New graph:
  X's birthplace is Y.
  Y's capital landmark is Z.

Question:
  What is the capital landmark of the birthplace of A's director?

Expected:
  Z
```

이때 `new_only`는 `A's director is X`만 제공한다. `superseded_prior`는
`A's director was B, but this is superseded; A's current director is X`와
old graph 주변 구조를 함께 제공한다. `scrambled_old_control`은 old graph를
동일 길이로 제공하되 relation/entity 연결을 섞어 구조적 도움을 제거한다.

### Prompt 조건

각 condition은 token length를 최대한 맞춘다. 모든 prompt는 동일한 answer
format을 사용한다.

권장 output schema:

```json
{
  "recognized_superseded_prior": "...",
  "recognized_current_update": "...",
  "active_intermediate": "...",
  "used_old_structure": true,
  "reasoning_path": ["...", "..."],
  "final_answer": "..."
}
```

초기에는 자연어 free generation도 함께 저장한다. Strict parser가 실패해도
trace를 수동 분석할 수 있어야 한다.

### Metric

필수 metric:

- `final_accuracy`
- `edited_intermediate_accuracy`
- `downstream_recompute_accuracy`
- `old_path_rate`
- `new_path_rate`
- `first_old_to_final_new`
- `superseded_prior_recognition_rate`
- `used_old_structure_rate`
- `format_valid_rate`
- `hit_limit_rate`
- `median_visible_tokens`
- `token_overhead`

분석은 hop별로 나눈다.

- 1-hop direct
- 2-hop ripple
- 3-hop ripple
- 4-hop ripple

### Stage 0 success gate

최소 성공 조건:

```text
2/3-hop에서 superseded_prior가 new_only보다 유의하게 높다.
scrambled_old_control은 superseded_prior보다 낮다.
old_new_no_label은 superseded_prior보다 낮거나 불안정하다.
hit_limit과 format failure가 결과를 설명하지 않는다.
```

### Stage 0 kill gate

치명적인 중단 조건:

```text
superseded_prior <= length_matched_new_only
superseded_prior <= scrambled_old_control
superseded_prior improvement가 token length 또는 explicit answer leakage로만 설명됨
old prior trace를 준 뒤 reflection해도 first_old_to_final_new가 거의 발생하지 않음
```

이 결과가 나오면 "old prior as substrate" 가설은 최소한 prompt/trajectory 수준에서
성립하지 않는다고 보고 방향을 접거나 완전히 재정의한다.

## Stage 1. Natural Benchmark Diagnostic

Stage 0이 통과되면 natural benchmark로 확장한다.

후보:

- MQuAKE
- RippleEdits
- CounterFact 기반 synthetic ripple 확장
- lab 내부 EAIR-style multihop set

Natural benchmark에서는 retrieval 문제와 reasoning 문제를 분리한다.

### Oracle-retrieved setting

각 case의 relevant edit fact와 필요한 surrounding old/new graph를 직접 넣는다.
이는 retrieval을 완벽하다고 가정하는 ceiling이다. 여기서도 `superseded_prior`가
`new_only`를 이기지 못하면, retrieval detector를 만들 이유가 없다.

### Retrieved setting

EAIR/MeLLo/PokeMQA류처럼 query에서 relevant edit fact를 retrieve하거나
decompose한 뒤 prompt에 넣는다. 이 단계는 final method가 아니라 deployment
방향 검토다.

### 비교 대상

- `new_only direct context`
- `superseded_prior direct context`
- `strong instruction`
- `always reflect`
- `reflection after old prefix`
- RippleCOT-style static CoT
- EAIR-style decomposition/retrieval if code is available

## Stage 2. Parameter Modification Transition

이 단계부터 최종 research identity에 들어간다. 목표는 old/new dual state를
parameter 내부에 만들 수 있는지 보는 것이다.

ROME-family를 그대로 쓰면 overwrite 성격이 강하다. 따라서 먼저 ROME-family를
baseline으로만 사용한다.

필수 비교:

| 조건 | 설명 |
| --- | --- |
| Base | edit 없음 |
| ROME/MEMIT/AlphaEdit | 기존 shallow parameter edit |
| LoRA new-only | `(s,r)->o_new`만 학습 |
| LoRA superseded-current | old는 superseded, new는 current 형태로 소량 학습 |
| dual-adapter | old-status adapter와 new-current adapter를 분리하는 실험적 구조 |

처음부터 새 editor를 만들지 않는다. 먼저 다음이 가능한지 본다.

```text
1. old fact를 direct answer로 계속 꺼낼 수 있는가?
2. new fact를 current answer로 꺼낼 수 있는가?
3. "former/outdated/superseded" context에서는 old가 활성화되는가?
4. "current/updated" context에서는 new가 활성화되는가?
5. long reflection trajectory에서 old graph를 재료로 new downstream answer를 만드는가?
```

## Stage 3. Probing And Mechanistic Analysis

이 연구는 "old/new dual state"라는 강한 주장을 하므로 probing이 필수다.

수집할 feature:

- prompt-end hidden states, all layers
- generation-token hidden states, selected layers
- top-k probability vector
- top-k mass
- max probability
- entropy
- p(EOS)
- old/new candidate rank if answer memory가 허용되는 offline analysis
- JS divergence to successful edited-utilization prototype
- rank churn
- attention or representation similarity to superseded/current spans

주의:

Runtime method에서 old/new answer를 직접 쓰지 않는 claim을 할 수는 있다.
하지만 offline analysis에서는 old/new candidate를 사용해 signal을 검증할 수
있다. 이 둘을 report에서 명확히 분리해야 한다.

## Stage 4. Trajectory Distillation

H1/H2/H3가 통과된 뒤에만 진행한다. 목표는 reflective exploration trajectory를
학습 데이터로 만들어 context 없이도 deep-edit-like ripple을 유지하는 것이다.

가능한 방식:

- SFT on successful reflective trajectories
- DPO: successful switch trace vs old-path trace
- GRPO: reward based on edited intermediate, downstream recompute, locality
- lightweight adapter distillation

처음부터 RL을 사용하지 않는다. Reward 설계가 불안정하면 이전 E3-style 실패를
반복할 수 있다.

## Red Team Kill Tests

Red team은 다음 질문을 항상 우선한다.

```text
이 결과는 진짜 old prior substrate 때문인가,
아니면 prompt 길이, answer leakage, stronger instruction, CoT budget,
retrieval oracle, parser artifact, dataset leakage 때문인가?
```

필수 kill/control:

- length-matched `new_only`
- CoT-length-matched `new_only`
- `scrambled_old_control`
- random old graph control
- temporal label only control
- explicit answer leakage check
- swapped `o_new` object test
- same subject different relation negative
- same relation different subject negative
- locality query false improvement check
- parser-free manual sample audit
- no-context/post-context retention check if distillation을 주장하는 경우

Red team은 논문 novelty도 매번 다시 확인해야 한다. 단순히 관련 연구 이름만
나열하면 안 된다. 각 related work의 actual method, supervision, runtime
input, retrieval assumption, metric, intervention timing을 비교해야 한다.

## Blue Team Minimum Checklist

Blue team은 첫 1주 안에 다음을 끝내는 것을 목표로 한다.

1. Synthetic graph generator 작성
2. 100 edit case x 2/3-hop ripple query 생성
3. `new_only`, `old_new_no_label`, `superseded_prior`,
   `scrambled_old_control` 네 조건 실행
4. JSON trace + raw text 저장
5. strict parser와 loose parser 작성
6. hop별 metric table 생성
7. 20개 sample 수동 audit
8. Red team kill test 결과를 같은 report에 병합

각 실험은 `local/`에 raw를 저장하고, Git에는 다음만 남긴다.

- run config
- Slurm job id
- output path
- compact metrics JSON
- Korean report
- red-team audit
- artifact broadcast verification

## 새 Repository 운영 원칙

새 repository는 `agent-control-template` 기반으로 만든다.

역할 분리:

- `global-head`: 연구 방향, canonical proposal, task 승인, final synthesis
- `server-head`: 자기 서버 실험 preflight, job submit, monitoring, artifact
  broadcast, report integration
- `blue-team`: 실행 계획과 결과 분석
- `red-team`: kill test, novelty audit, data/eval/protocol audit

Global-head가 직접 다른 서버에서 Slurm job을 올리는 것은 emergency 또는 명시적
사용자 지시가 있을 때만 한다. 기본은 server-head가 자신의 inbox/task를 읽고
preflight 후 제출하는 방식이다.

모든 보고서는 한글로 작성한다. Metric 이름, command, path, error snippet은
원문 그대로 둘 수 있다.

## 첫 달 실행 계획

### Week 1

- 새 repo 생성 및 protocol 설치
- synthetic diagnostic 구현
- Qwen/Llama에서 Stage 0 pilot 실행
- prompt condition별 token length와 output format 안정화
- red-team kill test 1차 수행

### Week 2

- Stage 0 확장: 100-300 edits, hop 2/3/4 분리
- manual audit와 parser 개선
- natural benchmark oracle-retrieved pilot
- 관련 연구 table 업데이트

### Week 3

- parameter baseline 준비: ROME/MEMIT/AlphaEdit/LoRA new-only
- old/current/superseded probing prompt suite 작성
- all-layer prompt-end hidden/logit/prob collection
- old/new dual-state 가능성 preliminary report

### Week 4

- 결과에 따라 두 갈래 중 하나 선택
  - H1/H2 통과: parameter dual-state editor 설계로 이동
  - H1/H2 실패: 연구 종료 보고서 작성, insight만 다른 continual learning
    연구로 이전
- 가능하면 workshop-style method sketch 작성

## 가장 중요한 판단 기준

이 연구는 "reflection을 하면 좋아진다"가 아니다. 다음을 보여야 한다.

```text
Old prior를 superseded prior로 구조화하면,
new fact만 주는 것보다 ripple reasoning이 좋아지고,
그 improvement가 prompt 길이/CoT budget/answer leakage가 아니라
old knowledge graph를 reasoning substrate로 쓴 결과임을 보인다.
```

그 다음에야 parameter modification으로 넘어갈 수 있다.

```text
Parameter edit가 old를 지우지 않고 superseded prior로 유지하며,
new를 current belief로 삽입할 수 있는가?
그 dual state가 runtime reflection trajectory에서 실제로 활용되는가?
```

이 두 질문이 새 repository의 중심이다.
