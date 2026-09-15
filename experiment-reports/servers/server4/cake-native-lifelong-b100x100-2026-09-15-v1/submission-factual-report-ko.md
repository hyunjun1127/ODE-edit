# CAKE native lifelong 제출 사실 보고

상태: **48101 제출·held 검사·release, 마지막 관측 PENDING; MONITORING_PAUSED_AWAITING_USER**.
GPU 실제 native batch/성능/전체 완료를 아직 관측하지 않았다. 다른 EP/BLUE/REFIT4 작업은 재개하지 않았다.

## 최신 사용자 저장 override

“CAKE 부분은 weight 저장 하지 말고 그냥 올려라”를 적용했다.
W/M tensor checkpoint를 디스크에 저장하지 않는다. 5개 down_proj weight와 5개 cache_c는
단일 process 메모리에서 자기 이전 batch state를 이어받으며, 열린 batch rollback도 메모리에서만 수행한다.
원래12개 checkpoint 시점은 **평가 시점으로 유지**한다. Hash/context/RNG/commit 기록은 남지만
가중치/history 자체가 없으므로 재구성 가능한 checkpoint·정확 restart·continuation PASS가 아니다.
기존 CP 삭제·이관·공유자산 변경은0이다. 원래63,417,876,480B tensor floor 디스크 부족 관측은 보존했다.
최신 admission available=256,463,474,688B, raw/source/log reserve=17,179,869,184B(16GiB, 추정여유)이다.
그 사이 filesystem 가용량 증가는 SH4가 삭제한 결과가 아니다. 이후 공간이 늘어도 최신 no-checkpoint 지시는 유지한다.

## 원본 method와 source

- 원본 CAKE c8243e1d7e43ca9cf64d552f96221fcb9561aac2 / tree4f59249bb23c7cacf0f6490bce8127c74ff7b111 / MIT.
- 실행 ODE hook 7884aeb6000f8343139172825ec6c4ca24357fc0 / tree4dabdb1e408726974ac0f91285ad35bac698b88e. 분석·제출 publication commit과 구분한다.
- Archive983589f0b6ca5dcca85508bdaa2d5a3a1d38128697ec89dedf54f5ede1f16ebf (4,003,840B).
- Lockf2ade3ee9dbdabdc6a1ac00a9d36b0e902710a44cf5e2a6a49c6d861e18eb401.
- 원 `Cake_main.apply_Cake_to_model`을 batch당1회 직접 호출한다. 최종L8 target100→L4..8 현재 residual/native directsolve→최종 각층 history1.
  wrapper의 추가 finalize/history append, 다른 method solver, target cache 재사용은0이다.
- 원본에 없는 notebooks.util의 미사용 import로 CPU import가 실패했다. 실행 복사본에서 그 한 줄 제거와 EOF LF만 변경했다.
  원본 clone은 clean 보존했고 함수/나머지AST 동일을 확인했다. 호환 patch는 audit에 별도 봉인했다.
- Original layers4..8/L2=10/decay.4/clamp.5/temperature.1와 원 causal_scores0..4를 유지한다.
  Physical4..8→P asset0..4→local0..4. 점수 key를 physical layer로 재색인하지 않았다.

원본 README 환경torch2.6.0/transformers4.51.3과 실제 재사용환경torch2.9.1+cu128/transformers4.44.2는 다르다.
기존 baseline과 FP32/eager, matmulTF32false/cudnnTF32true, seed20260907을 결속했다.
Writer는 기존 add_bos_token=false/right padding; evaluator는 기존 별도 tokenizer 및 kernel의 manual left-padding/MB16 그대로다.
CAKE 원 context 생성함수와 기존 native 생성함수AST가 같아 기존 context bytes를 제공한다(재생성0).
이는 CAKE README 환경 전체와 같다는 주장이 아니다. BaseAlphaEdit clamp.75/decay.5와도 다르다.
Layer weighting 하나만 바꾼 통제 비교로 해석하지 않는다.

## 고정 stream·평가·저장

Llama revision8afb486c1db24fe5011ec46dfbe5b5dccdb575c2, freshW0/coldcache_c0.
고정 counterfact SHA3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1,
whole orderedroot5b01356928ad2e2e46effad5a2c5621c87746b610992b7b92bf3ace16b6f5729를 공식loader로 검증했다.
B1=[0,100), …, B100=[9900,10000);10000unique,100×B100. 각batch case/request/target/prompt/order hash를 lock에 저장했다.

매batch actual editedW에서 CurrentR/P/N 및 기존 baseline의 all-seen rewrite를 유지한다.
B1/5/10/20/30/40/50/60/70/80/90/100에서 actual full-seenR/P/N.
최종W100 분모10000/20000/100000을 실제cardinality로 확인하고 한번만 평가한다.
Current와 fullseen의 같은state rows는 identity로 합쳐 중복forward/분모를 피한다.
RS/PS newNLL<trueNLL, NS trueNLL<newNLL, ties=failure; TFstrict/token은별도다.
평가결과는 target/causal allocation 입력이 아니다. W0는 기존42673 publication 재사용, 새GPU W0평가0.
GLUE/MMLU/downstream/다른모델/다른baseline/새 stats 생성은0이다.

## 최소 준비·자원·미검증

CPU9 fixtures, Python AST/shellsyntax/import/config/unused-import native AST, 공식data/order,
선택closure1959 members/8,287,575,787B fullSHA 검산 PASS. Import 중 CUDA initialized=false.
모델전체shard는 기존 revision/closure와 가독성·size를 재사용했으며 새 전체content rehash는 하지 않았다.
P/stats fileSHA는 이번선택검산에 포함, P tensor schema/identity 및 실제 native write/history 검사는 제출 프로그램의 첫batch에서 확인한다.
별도GPU smoke/FD/ULP/gradient parity gate는 실행하지 않았다. 기존타task 수치PASS를 CAKE로 전용하지 않는다.

요청1GPU/8CPU/60416M/exportNONE/Requeue0/48h; GPUhour hardcap=null, cap2.
Fresh admission의 다른 project active+admitted pending=0. 하나의 chain만 등록했다.
Held 상태에서 owner/job/command/source/args/memory/GPU/48h/no-requeue를 확인해 release했다.
마지막 관측은 2026-09-15T08:52:49.137129+00:00 / PENDING; 이후 scheduler/log/output을 다시조회하지 않았다.

계획상10000 request-z, 최대250000 loss/240000 Adam,500solve,100whole-history passes=500layerappends.
실제조기종료/시간/메모리/성능은 아직미측정이다. 총 wall은48h 제한으로 요청했으나 CAKE speedup 추정이나48GPUh 예산으로 부르지 않는다.
Target/keys/solve는 전체write wall 내부 nested component이며 단순합산하지 않는다.
평가계획의 repeated state-population 관측은 RS505000/PS128800/NS644000 rows, NLL pair2배=2,555,600 teacherforced sequences이며
unique sample수가 아니다. CAKE 실제초기실행/완료시 비용계측으로 확인해야 한다.

## 경로·인계

Local `/data/janghj/ODE-edit/local/cake-native-lifelong/20260915-v1/attempt-v1`. Output `output/main/`, logs `logs/48101.out` 및`.err`.
예상 terminal `output/main/terminal.json`; 최초 marker `output/main/initial-execution.json`.
현재 산출물 존재/commit은 pending인계 후 재조회하지 않았다.
`resume-manifest.json`: monitoring_active=false, automatic_resume=false, explicit_user_call만 재개.
100batch 프로그램/평가/기록은 자연진행하며 agent heartbeat/callback/자동분석/추가submit0.
Git에는 source/raw-free metadata만, rawNLL/로그는local. NO_BROADCAST_NOT_REQUIRED.

후속비교는 `blue-native-lifelong-comprehensive-review-2026-09-11-v1/diagnostic-report-ko.md`와
W0/BASE_ALPHAEDIT/BASE_MEMIT/BLUE/가용single-layer를 동일10kW100·호환성 범위에서 연결할 예정이며
사용자 완료recall 전에는 결과분석을 시작하지 않는다. Scientific promotion=false.
