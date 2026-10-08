# GPT2-XL baseline F/C native-like 생성 수리 및 서버2 인계

작성일 KST 2026-10-08, version1. 직접 승인 `USER-DIRECT-NATIVE-FLUCON-REPAIR-20261008-R1`. **SOURCE_IMPLEMENTED · CPU178 PASS · INDEPENDENT_REVIEW_DONE.** source 게시·신규 등록 결과는 root의 별도 실제 receipt로 결속한다. 이 보고서는 구현/CPU 검산을 GPU 인증·과학 완료로 표현하지 않는다. 담당 문서 worker는 모델/GPU/Slurm/W&B/API/네트워크 작업, commit/push를 수행하지 않았다.

## 결론과 범위

사용자가 승인한 방향은 기존 `cf-cake-prompt-inclusive-total100-eos-corrected-v1`의 unpadded row/full-prefix reference 및 equal-length KV qualification 선택 경로를 그대로 강화하는 것이 아니라, 원본에 가까운 **case별 padded prompt batch + call-local KV cache + total100 + no EOS early stop + global batched sampling**을 별도 profile로 구현하는 것이다. F/C 계산식과 고정 reference는 유지한다. 생성 profile의 EOS·RNG·batch 의미가 달라지므로 예전 raw/점수를 새 profile로 재표시하거나 원본·구 profile과 bitwise 동일하다고 주장하지 않는다.

신규 server1 우선 대상은 MEMIT, stock AlphaEdit, AlphaEdit-BLUE 세 baseline이다. 실제 edited W20/edits2000의 ordered CounterFact first2000에서 generation 한 번만 수행한다. W0/current/pre/중간 W5·10·15 generation, 추가 fit, pilot, 성능 gate는 0이다. R/P/N W0/current pre/post·W5/10/15/20 all-seen·retention의 기존 정의·분모·일정, native editing/solver/history/hparams는 변경하지 않는다. 승인된 server1 취소범위는 기존 baseline `61436–61438`와 collector `61439`이며 실제 취소·교체 결과는 exact receipt로만 확정한다. OURS·PRICE·다른 owner/서버 job에 확대하지 않는다.

## 직접 확인한 원본 provenance

세 로컬 repository의 HEAD와 아래 관련 파일 bytes를 읽고 확인했다. 관련 generator/metric 파일은 해당 HEAD 대비 clean이다. repository의 다른 dirty 파일이 있다는 사실은 upstream source 증거로 승계하지 않는다.

| 원본 | 확인한 commit | generator SHA256 | metric SHA256 |
| --- | --- | --- | --- |
| CAKE | `0b378234862bd76c69f58404ef84c27d5f4bf9ef` | `43d219ecc1955a4e4f82ea22edc286b4774e52e917d4b375021b8d7329b0adf3` | `ee00ee9cfb1be1cb1e5d1495fd15b2d1bfef9426af2b159095b924e6bdd0da26` |
| BLUE | `311b076a92e4ed0f14f5c8b4909732da781bc5f7` | `51f41871d750a1fe86f39d4142fe8a5c12a41086c6d4a2cee98b72f809ff0366` | `b521945531a2b95b95c412b69e0c713ce67897af33a5ece9c7e608108b2ddb33` |
| EasyEdit | `3488a66ee988d83ee7891a8abbbe6bcb24a77daf` | `13d28184bd215ea7d51bef091fcd3624271d93f4c3df32fdbf2e97cc15581b45` | `f5973dedb4e1b811ebec02133a1bec4caf6d64d39feeefd4b2456c72379518b3` |

CAKE 정본은 [generate_fast](/mnt/raid5/janghj/CAKE/util/generate.py:77), [reference 선택](/mnt/raid5/janghj/CAKE/experiments/py/eval_utils_counterfact.py:99), [한 생성의 F/C 공동 사용](/mnt/raid5/janghj/CAKE/experiments/py/eval_utils_counterfact.py:194), [entropy·TF-IDF cosine](/mnt/raid5/janghj/CAKE/experiments/py/eval_utils_counterfact.py:228)이다. BLUE의 대응 경로는 [generator](/mnt/raid5/janghj/BLUE/util/generate.py:77), [CounterFact F/C](/mnt/raid5/janghj/BLUE/experiments/py/eval_utils_counterfact.py:194)이다. EasyEdit는 [generator](/mnt/raid5/janghj/EasyEdit/easyeditor/util/generate.py:77)와 [F utility](/mnt/raid5/janghj/EasyEdit/easyeditor/evaluate/evaluate_utils.py:344)를 확인했다.

CAKE/BLUE는 raw CounterFact `generation_prompts`를 한 case 단위로 생성하고 `snips[relation_id][target_new.id]`의 모든 text와 비교한다. EasyEdit의 inspected native 평가 경로는 rewrite prompt로 F만 산출하며 `reference_score` C를 구현·발행하지 않는다. 따라서 이번 공통 F/C 의미는 CAKE/BLUE CounterFact 정의를 따른 것이며, EasyEdit native가 C를 내는 것으로 설명하지 않는다.

원본 generator는 `padding=True`, `past_key_values`, `use_cache=True`, full-vocabulary softmax → top5 → 재정규화 → 단일 `torch.multinomial([batch,5],1)`를 사용한다. `max_out_len=100`은 새 토큰100이 아니라 prompt를 포함한 padded token-array width100이며 EOS early stop이 없다. case 안의 longest prompt가 이미100 이상이면 원본 while loop는 모든 row에 continuation0을 준다. shortest prompt 길이까지 공통 prefill한 뒤 긴 prompt의 기존 토큰을 overwrite하지 않고 각 row가 catch-up하도록 진행한다. 원본은 prompt를 제거하지 않고 전체 decoded text를 점수화한다.

CAKE는 accumulated cache+query mask `attention_mask[:, :current_pos]`를 이미 사용한다. BLUE와 EasyEdit custom branch는 current query mask slice를 넘긴다. CAKE decode는 `tok.decode(x)` 뒤 GPT2 end-of-text 문자열 제거, BLUE/EasyEdit는 `skip_special_tokens=True`를 사용한다. 셋 모두 NFKD와 double-newline-to-space 정규화를 한다. EasyEdit의 별도 `vanilla_generation=True`는 `model.generate(max_new_tokens=...)`이므로 이번 custom/native-like 경로의 근거가 아니다.

## 이전 경로 → 새 경로: 변경 의미

| 항목 | 이전 파생 profile의 실제 구현 | 신규 실제 구현 / source 봉인·GPU 실행은 별개 |
| --- | --- | --- |
| 생성 단위 | row별 unpadded reference; qualification 성공 시 exact equal-length bucket KV | case 내부 원래 prompt 순서의 padded batch, case 간 합치기 없음 |
| KV/route | MB8 등 사전 PLAN과 logits/token/EOS/RNG exact gate로 route 선택; no-cache reference 가능 | call-local KV를 기본 사용, equal-length MB8 gate·no-cache fallback 제거 |
| 길이 | row별 input+continuation 총100; 긴 row는 다른 row를 막지 않음 | 원본 padded-array total100 의미와 긴 prompt 처리 차이를 새 profile에 명시 |
| EOS | native EOS ID에서 row 종료·survivor gather | no EOS early stop; EOS도 length cap까지 sampling에 남음 |
| RNG | model+ordered occurrence+prompt index seed, row별 독립 RNG | endpoint에서 eval seed 한 번 설정 + global batched multinomial; case/prompt별 reseed·독립 stream 아님; endpoint 종료 시 caller RNG finally restore |
| decode | full prompt+continuation, skip special IDs, NFKD·double newline 정규화 | CAKE 원본처럼 padded token array에 `decode`(skip flag 없음), NFKD·double newline·literal endoftext 제거 |
| 과학 점수 | 한 observation을 F/C가 공동 사용 | 동일 원칙·동일 고정 TF-IDF/snippet 자산; 재생성0 |
| 과거 자료 | 기존 source/profile/observation identity | 모두 보존; 새 profile·source·seed·case order로 별도 identity |

이전 경로의 직접 근거는 신규 worktree의 `project/run_scripts/experiment_generation_eval/{common,generator,observer,kv_qualification,compatibility}.py`이다. reference는 `use_cache=False`로 매 step 전체 prefix를 다시 처리하며 EOS에서 종료한다. cached route는 singleton 또는 equal-length KV bucket과 row별 `torch.Generator`를 사용한다. 이 구조에서 full-prefix token work가 누적되거나 equal-length bucket 폭이 작을 수 있다는 것은 소스 설명이지 실제 GPU 병목·속도 측정은 아니다. 실제 route, peak, time, speedup은 아직 주장하지 않는다.

## 변하지 않는 F/C 수학과 privacy

F는 NLTK word 2/3-gram Shannon entropy(bits)의 `H2/3 + 2*H3/3`, case 안 generated text들의 arithmetic mean이다. C는 `vec.transform([" ".join(full_generated_texts), " ".join(all new-target snippets)])`의 cosine이다. relation/new-target reference 전량이며 subject-name essence subset이나 essence perplexity가 아니다. fixed vocab/idf를 사용하고 생성 결과에 TF-IDF를 refit하지 않는다. 유효 case의 equal-occurrence macro·sum/count와 planned2000, prompt 수, continuation 수, typed missing 이유를 구분한다. 결측을0/NaN/Inf로 올리지 않고 진짜 측정 entropy/cosine0은 유효0으로 유지한다. F는 문법 정확도, C는 사실 정합성 인증이 아니라 lexical proxy다. 기존 R/P/N harmonic에 F/C를 합치지 않는다.

raw prompt/text/token/case ID/reference, 전체 stdout, secret, model tensor는 ignored local에만 보존한다. Git/W&B에는 compact source/manifest/report와 허용 scalar만 게시한다. KV는 함수 호출 RAM에서만 살아 있고 model config/hook/H/edit state를 지속 변경하지 않는다. durable model/W/H/optimizer/checkpoint0인 NoCP를 유지한다. SDK acceptance·remote readback·native/W20 scientific completion은 별개 상태다.

## 구현·검산 상태: 계획과 실제를 분리

원본 source/metric 비교와 이전 경로 확인은 이 문서 담당자가 실제 수행했다. 현재 파일로 확인된 API는 `native_profile.PROFILE='cf-cake-native-casebatch-kv-total100-globalrng-v1'`, `ROUTE='NATIVE_CASE_PADDED_KV_GLOBAL_RNG'`, `runtime_identity(config, assets_sha)`, `source_identity()`와 `native_generator.generate_case(model, tokenizer, prompts, *, occurrence, eval_seed=20261007, trace=None)`다. generator/profile 두 파일을 원본과 독립 대조했고 padded catch-up·cumulative mask·단일 batch multinomial·noEOS·정확한 CAKE decode를 확인했다. CAKE mask 자체가 cache+query를 누적하므로 BLUE current-query mask를 새 기준으로 승계하지 않는다. FP32/eval/finite/right-padding/cache-return guard와 local work metadata는 compatibility/evidence 보완이며 extra forward0이다. RNG 격리·restore는 endpoint observer의 책임이며 generator 함수는 caller의 global stream을 실제 진전시킨다.

`NativeGenerationObserver`는 이전 observer와 같은 constructor·`observe`·`subset`·`read_observed` API를 제공하되 profile과 raw 완결성 validator는 분리한다. endpoint 전체 RNG stream을 순서대로 사용하고 caller RNG·model/native state를 finally 검산하며 정확히 완료된 endpoint만 재사용한다. 부분 case raw를 건너뛰면 global sampling stream이 달라지므로 resume/reuse하지 않는다. 모든 ordered record identity를 `sampling_stream_sha256`에 결속한다. 독립 source 대조의 두 지적은 수리했다: generator row에는 거짓 `RNG_restored` 대신 endpoint restore guard 필요성을 기록하고, CPU subset은 parent endpoint member/identity·stream SHA를 보존하며 readback이 완료된 원 native execution receipt와 runtime/state/exact row 포함을 재귀 검산한다. 기존 raw를 새 cohort에서 재표시하거나 parent 없는 subset을 수락하지 않는다.

`repo_native_run`은 `NativeGenerationObserver`로 actual20 native commits 뒤 W20 first2000만 관측하고 `native_execution_member`를 final receipt에 넣는다. old qualification PLAN/binding/reuse prerequisite를 읽거나 실행하지 않는다. `repo_native_collect.native_generation_endpoint(reader, receipt, c, records, physical_state)`는 `native_observer.read_observed`/`verify_native_raw`를 사용하며 expected runtime·actual W20 state·ordered first2000의 record identity에서 stream SHA를 재계산한다. raw/endpoint/execution/final receipt의 physical/logical forward·prefill/decode/token/case counters를 교차검산한다. 기존 old-profile validator로 native raw를 판정하지 않는다. task-private `repo_native_{common,prepare,preflight,run,collect,submit,transition}` 경로를 사용하며 기존 `native_w20_*`/`w20_*` frozen profile·canonical policy bytes는 보존한다. 새 direct authority가 이번 profile만 override한다.

원본은 앞선 context-template 생성·편집 호출과 같은 global RNG를 소비할 수 있다. 새 endpoint는 `eval_seed=20261007`로 한 번 reseed하고 격리하므로 batching/noEOS가 원본에 가까워도 upstream의 호출 이력까지 재현하는 것은 아니다. 이 RNG provenance와 이전 prompt별 seed와의 차이를 모두 새 profile에 명시한다.

기존 작업의 CPU PASS141이나 타 profile의 qualification receipt를 이번 수리 PASS로 승계하지 않는다. 최초 통합177 PASS 뒤 별도 caller 독립 reviewer가 발견한 collector의 exact stream 재계산과 raw/execution/receipt work counter 교차검산 두 proof gap을 수리하고 tamper 회귀를 추가했다. 실제 science/helper 최종178 PASS 뒤 운영 relay readback mode까지 포함한 exact source를 다시 검사했다. 최신 [CPU receipt](/mnt/raid5/janghj/ODE-edit/local/gpt2xl-baselines-fluency-consistency-2k/20261007-v1/native-repo-repair-r1/cpu-source-final-r1/cpu-preflight.json)는 **178 PASS, failure/error/skip0**, SHA `f6f31faf7da03007d053f7f49bf29c88a30d41543ff60a7b3bf0213bdbdca8bf`다. 최초·중간 receipt와 source bytes도 보존했다.

문서 worker의 독립 common-native source review와 별도 native implementation worker의 caller integration review가 완료됐고 남은 source blocker는 없다. root owner CPU178검산과 독립 read-only source 검토의 역할을 구분한다. AlphaEdit-BLUE cold two-layer history와 current R/P/N 유지, W20 once-after-commit20 guard도 caller seam에서 확인했다. native fixture는 원본 oracle와 mixed-length batch·noEOS·total100/긴 prompt·global stream·mask/cache/decode·raw/readback·subset/privacy·RNG/state restore를 확인한다. `CUDA_VISIBLE_DEVICES=''`의 fake-model/software checks이며 pretrained/GPU/online/model native apply 검증은0이다. actual GPU native generation/W20 completion은 문서 worker가 미관측했고 speedup/ETA는 `NOT_MEASURED`다. 새 profile은 old MB8 route qualification을 수행·필수화하지 않으며 native execution receipt와 실제 work/RNG/state guard를 기록한다. 이를 qualifier PASS로 표현하지 않는다.

sourcefreeze/publication commit·tree/archive/config SHA, exact 신규 job ID·owner/fullargv/WorkDir/source/dependency·release snapshot은 root 담당자가 실제 receipt 확인 후 별도 execution 보고에 기록한다. version1에 아직 없는 commit/job 값을 추측하지 않는다. root의 admission 검산은 input lock 전량 PASS, server1 own-project GPU allocation0/DAG width0이며 새 계획은 MEMIT+AlphaEdit parallel, AlphaEdit-BLUE after MEMIT인 cap2다. 이는 제출·GPU 가용·실행/완료의 선행 선언이 아니다.

root 확인 실제 취소 결과: `61436` AlphaEdit-BLUE, `61437` PRUNE, `61438` RECT, `61439` collector 모두 terminal `CANCELLED`다. 취소 직전 확인 snapshot의 authoritative native batch commits는 BLUE4, PRUNE0이며 BLUE의 기록된2240 GPU-sec와 PRUNE980 GPU-sec를 보존했다. RECT는 미시작/0sec, collector0sec였다. old source/raw/부분 science/비용은 KEEP이며 취소가 실패·비용·부분 편집 기록을 지우거나 W20 완료를 의미하지 않는다. 새 우선3개 외 CAKE/PRUNE/RECT는 새 제출대상이 아니다.

## 서버2 실제 취소 ACK와 source 재사용 인계

root의 official `REGISTERED_SSH_UNIX_WEBSOCKET` `turn/start` accepted 뒤 같은 turn `01a11b80-fce1-75b1-9553-632eb5dd2800`의 exact readback에서 full `OWNER_ACK nonce=SH1-SH2-NATIVE-FLUCON-REPAIR-20261008-R1`과 실제 취소 보고를 확인했다. 최초 fragmented text는 ACK 증거로 사용하지 않았으며 후속 [readback receipt](/mnt/raid5/janghj/ODE-edit/local/gpt2xl-baselines-fluency-consistency-2k/20261007-v1/native-repo-repair-r1/server2-direction-readback.json), SHA `fee44a863a537f7ee2c391698950a01820b70e916bdcee0e54ea1139c04439b1`에 결속한다.

서버2 owner는 fresh owner/source/argv/state 대조로 `61429–61434`가 모두 미시작 PENDING인 것을 확인한 뒤 후속 collector부터 취소했다. 결과는 여섯 job 전부 `CANCELLED`, elapsed0/실행·GPU allocation0이며 `61428 MEMIT`는 RUNNING으로 보존했다. source/raw/log·원 실행 worktree·frozen running source는 KEEP, 새 server2 제출0·공통 구현 수정0이다. 불명확한 pending 범위를 추측하거나 RUNNING으로 바뀐 대상에 취소를 확대하지 않는 경계를 owner가 수락했다.

source/API/report 최종 게시·전달은 version1 시점에 아직 미완료이며 source 재사용 완료로 표현하지 않는다. 서버2 인계할 재사용 경로는 `native_profile`, `native_generator.generate_case`, `NativeGenerationObserver`, `native_observer.read_observed`/`verify_native_raw`와 strict collector contract다. owner는 자신의 native editing/TF/RPN/precision/hparams·source closure를 보존하고 새 namespace에서 연결하며 profile/seed/reference/source/runtime identity를 봉인한다. old EOS/row-seed에서 noEOS/endpoint-global-seed/batched sampling으로의 차이는 science profile revision이고 bitwise 동일성은 주장하지 않는다. old raw를 새 profile로 재사용·backfill하지 않는다. 이번 방향은 구현·보고 전달과 승인된 not-started baseline 취소이며 새 server2 제출을 별도 승인하지 않는다. server1 신규등록·cap2/stricter admission·최종 source 전달은 root의 별도 실제 receipt로 결속한다.
