# GPT2-XL MEMIT·AlphaEdit·CAKE W20-only 신규 실행

승인 nonce `USER-GH-SH1-GPT2XL-MEMIT-ALPHAEDIT-CAKE-W20-GENERATION-20261008-R1`, authority `d4838fa83124356cbac74605178196dd98476c6f`. 구현과 CPU 검산을 완료했으며 실제 신규 job은 아직 등록하지 않았다. 이 문서는 담당 수락/구현·등록 상태와 실제 GPU/W&B/과학 완료를 구별한다.

## 실행 범위와 경계

각 native method 독립 cold W0/history, fixed CounterFact first2000, seed20261002, BS100×20. generation은 실제 edited W20에서 ordered first2000 한 번만 수행하며 W0/중간 generation·B21·별도 fit/pilot은 0. 기존 RPN W0/current pre/post와 W5/10/15/20 all-seen, TF/NLL/paired/retention 일정·분모는 유지한다. MEMIT20000, stock AlphaEdit L2=10, CAKE L2=40 및 기존 native lr.5/20평가/19update/readout47을 재사용한다.

App 실제 root CWD는 `/mnt/raid5/janghj/ODE-edit`, session `01a04939-f93a-7b50-bca0-65438eab2062`, repository `hyunjun1127/ODE-edit`, host `devbox`. 전용 non-main WT는 `/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-gpt2xl-memit-alphaedit-cake-w20-generation-20261008`. registry29e4는 역사 경로이며 강제 이동하지 않았다. 원 root dirty와 타 agent/source/raw는 보존했다. 사용자 model/effort를 변경하지 않았다.

## 이전 실행 대조

`attempt-cache-repair-r2` source `bb86a6ca514a47bbad372efd9032f9a5962f0140` 및 actual sacct owner/WorkDir/SubmitLine/node/source를 대조했다. MEMIT61167은 FAILED, `IMMUTABLE_GENERATION_IDENTITY_CONFLICT`, authoritative commit0/native z100/5solve/5write 및 allocatedGPU28015sec를 보존한다. AlphaEdit61168와 CAKE61169는 이미 CANCELLED, elapsed0/미할당이다. shared collector61173은 실패·부분 보고를 작성하고 COMPLETED이며 과학 성공이 아니다. 이번 지시의 실제 취소는 0건이다.

보호한 새 BLUE61436/PRUNE61437/RECT61438/collector61439는 해당 nonce/source/Command/fullargv/WorkDir를 확인하고 변경하지 않았다. GPU admission은 기존61436→61438 lane와61437 lane를 유지하고 새 root가61437·61438 두 말단 모두의 afterany를 기다리도록 한다. CPU61439를 GPU frontier로 사용하지 않는다. PRICE OURS·W0/FE/준비·다른 서버 변경0.

## 구현·검산 및 실제 등록

새 task-private profile/entrypoint와 nonce·source/config/member·old-ID guard를 결속한다. 기존 w20_common/prepare/submit의 세 target 기본값/봉인 source는 변경하지 않는다. stock AlphaEdit cold native H={}와 B1 뒤 H5를 구별하며 MEMIT H없음, CAKE H5를 검산한다. 이전 observer member_name 수정과 endpoint/window별 immutable identity를 재사용한다.

공통 W&B nonce의 좁은 allowlist 확장과 CPU/fake SDK 7개 검사 PASS: 실제 job/arrayindex0/signedstep/name·writer·개별 UUID·별도 generation_progress 축·privacy 및 SDK접수≠remote 확인. 실제 online run은 생성하지 않았다. 사전 고정 qualification PLAN은 유지하며 actual receipt는 새 arm runtime에서 기록하고 CPU검사를 GPU PASS로 표현하지 않는다.

최종 통합 CPU 회귀검사 141 PASS, failure/error/skip0. 최초 묶음 검사는 142개 중 존재하지 않는 test_kv_qualification 모듈명 1건만 import 실패했으며 원 FAIL receipt를 보존했다. 실제 qualification 회귀검사는 기존 test_kv_generator/test_compatibility에 포함돼 있다. private preflight 목록의 잘못된 한 항목만 제거한 뒤 새 create-once r2 receipt로 재검산했다. 생산 native/run/generation 수학과 허용오차 변경0. 독립 읽기전용 reviewer는 신규 run/collector/control/tracking 28개와 qualification 회귀검사17개 PASS, skip0 및 source/frontier/H lifecycle을 검토했다. Full20 CPU fake-model fixture의 heavy native fit/RPN 일부는 mock이므로 실제 모델/GPU PASS가 아니다.

최종 CPU receipt SHA `94228f319d112c40462b950755a39969a16d74898349e1af2bc1b4cc80b8a954`, sealed config SHA `909548b5d9ed4b77a1e1a3808968ddc76d19c63325c53c63de43616221908f95`. 두 신규 파일 EOF 여분 공백을 제거해 diffcheck를 통과시킨 뒤 source bytes에 맞춰 r3 141개를 다시 검산했으며 이전 receipt도 보존했다. Runtime Python `/mnt/raid5/janghj/EasyEdit/.venv/bin/python`, torch2.9.1+cu128/transformers4.57.1, model revision `15ea56dee5df4983c59b2538573817e1667135e2`. 기존 native closure99개·source config3개·20 packs·자산16개·evaluator4개를 결속했다. Qualification PLAN digest `83cee93d1b462f8e0b503f4692dd537430ee9084971aaa3a9150bcf4a2a4590b`, reference manifest SHA `6d9a713ab7eaa10871f277e10a3974e0bd5b140c265be80052c279f258876ca8`를 재사용한다. 준비 CPU PASS와 runtime 실제 route qualification/remote readback을 분리한다.

## 추가 사용자 첨부 대조

참조한 [CAKE 평가 코드](https://github.com/zjh-vinky/CAKE/blob/c8243e1d7e43ca9cf64d552f96221fcb9561aac2/experiments/py/eval_utils_counterfact.py#L194-L269)와 현재 pinned local CAKE/BLUE 코드는 한 번의 gen_texts를 entropy와 TF-IDF cosine 모두에 사용한다. 원 entropy는 가중치[2/3,4/3]의 arithmetic mean이라 H2/3+2H3/3이다. [BLUE의 native 평가 루프](https://github.com/xpq-tech/BLUE/blob/311b076a92e4ed0f14f5c8b4909732da781bc5f7/experiments/evaluate.py#L249-L423)는 generation을 편집 루프 뒤에서 수행하고 downstream_eval_steps는 별도 GLUE 평가다. [native 생성](https://github.com/xpq-tech/BLUE/blob/311b076a92e4ed0f14f5c8b4909732da781bc5f7/util/generate.py#L77-L155)은 KV cache와 top-k sampling을 사용한다. 링크된 CAKE revision과 승인된 실행 native revision은 구분하고 실행 source를 변경하지 않았다.

이번 구현은 RPN 중간 평가를 유지하면서 generation 요청-시점을 W20 first2000 하나로 제한한다. 두 점수용 이중 생성0, CPU subset만 허용, partial은 최종점수로 발행하지 않는다. 생성 microbatch/route는 사전 고정 PLAN과 실제 모델 qualification 결과로 선택하며 배치 확대/허용오차 완화/생산 OOM 자동 retry0. EOS·row RNG 보완을 포함한 기존 파생 profile을 유지하므로 upstream bitwise 동일성을 주장하지 않는다. 첨부의 Llama759초·14h/4.2h·3.7배는 이번 GPT2-XL 측정이나 ETA/원본 대비 속도 증거로 승계하지 않았다. 실행 중인 PRICE OURS/다른 서버 source·job의 일정은 변경하지 않았다.

현재 신규 제출 ID 없음; sourcefreeze/최종 CPU 검산/held등록·검사·release 결과를 후속 기록한다. 합산 cap2, perGPU1/CPU8/65536MiB/48h 요청상한, GPU0collector CPU8/24576MiB/4h, 물리/QoS/메모리 stricter 적용. 물리 GPU 부족은 scheduler PENDING으로 인계한다. 장기 완료 대기/recurring monitor/자동 retry 없음.

## 산출물

새 attempt 예정 경로 `/mnt/raid5/janghj/ODE-edit/local/gpt2xl-baselines-fluency-consistency-2k/20261007-v1/attempt-w20-only-native-three-r1`. 원 source/raw/취소/실패/비용 KEEP, durable model/W/H/optimizer/checkpoint0, rawtext/tokens/caseID/reference 대형자산/secret Git/W&B0. `NO_BROADCAST_NOT_REQUIRED`: 동일 host 원자산/raw 보존, compact source/manifest/report만 Git 게시. 참조 정본은 `messages/head/2026-10-08-gpt2xl-memit-alphaedit-cake-w20-generation.json`, method/native 및 generation policy를 직접 결속한다.
